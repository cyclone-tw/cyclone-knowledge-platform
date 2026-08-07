"""Issue #25 scope, offline-boundary, and base-dependency tripwires.

Every test in this file runs without the pinned model weights present --
they check structure, not inference output. The weight-gated behavior tests
(golden digests, determinism, descriptor honesty against a real loaded
model) live in ``tests/test_semantic_embedding_provider.py`` and follow the
same skip/``CKP_REQUIRE_SEMANTIC=1`` pattern ``tests/test_index_qdrant.py``
uses for Qdrant. Keeping the two apart is what lets
``scripts/test-c25-semantic-mutations.sh`` run everywhere, including a
laptop with no model cache and no network.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from ckp.semantic.assets import resolve_semantic_assets
from ckp.semantic.errors import SemanticErrorCode, SemanticRefusal
from ckp.semantic.manifest import SEMANTIC_MODEL_DIMENSION, SEMANTIC_MODEL_FILES
from conftest import REPO_ROOT

PACKAGE = REPO_ROOT / "src/ckp/semantic"

#: Tokens that must never appear in this package's source, mirroring
#: ``tests/test_c5_contract.py::test_index_package_stays_inside_the_offline_boundary``.
#: A substring scan (not an AST import allowlist like C4's) because this
#: package legitimately imports a broader third-party surface
#: (``onnxruntime``, ``tokenizers``, ``numpy``) that an allowlist would have
#: to special-case anyway -- what actually matters is that none of these
#: network-capable or credential-adjacent names ever show up.
FORBIDDEN_TOKENS = (
    "httpx",
    "requests",
    "urllib",
    "socket",
    "huggingface_hub",
    "hf_hub",
    "openai",
    "cohere",
    "voyage",
    "torch",
    "transformers",
    "sentence_transformers",
    "ftplib",
    "smtplib",
    "subprocess",
    "api_key",
)


def _module_paths() -> list[Path]:
    # rglob, not glob: R1 review flagged that a one-level-deep scan would
    # miss a future submodule under this package entirely. The package is
    # flat today, so this is currently equivalent to glob("*.py") -- the
    # point is that it stays correct if that ever changes.
    return sorted(PACKAGE.rglob("*.py"))


def test_semantic_expected_modules_are_present() -> None:
    assert {path.name for path in _module_paths()} == {
        "__init__.py",
        "assets.py",
        "composition.py",
        "errors.py",
        "manifest.py",
        "provider.py",
    }


def test_the_package_stays_inside_the_offline_boundary() -> None:
    """No HTTP client, no cloud SDK, no hub client -- anywhere in the tree.

    This is issue #25's non-goal made mechanical, the same way C4's
    ``test_the_package_imports_nothing_that_could_reach_a_network_or_a_secret``
    and C5's index equivalent are: a network path cannot be added to this
    package without this test going red first.

    R1 review flagged that a one-level-deep scan would miss a future
    submodule under this package entirely; ``_module_paths()`` uses
    ``rglob`` for that reason, so this check covers any nested file, not
    just the current flat layout. What remains a known, accepted scope
    limit is that the check itself is a *substring* scan over each file's
    text -- not an AST import allowlist like C4's own contract test -- so
    it would not catch, say, a network call built up from string
    concatenation rather than a literal token. That is the same tradeoff
    C5's equivalent test already makes; an AST-based rewrite is a larger
    change than this PR's scope.
    """
    for path in _module_paths():
        source = path.read_text(encoding="utf-8").lower()
        for forbidden in FORBIDDEN_TOKENS:
            assert forbidden not in source, (path.name, forbidden)


def test_the_package_opens_no_file_by_path_outside_assets_module() -> None:
    """Only ``assets.py`` is allowed to touch the filesystem directly.

    ``provider.py`` receives already-verified paths from ``assets.py``; it
    must never re-derive its own path into the cache (that is how a digest
    check gets silently bypassed on one of two call sites, AGENTS.md §9
    question 3).
    """
    for path in _module_paths():
        if path.name in ("assets.py",):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"open", "eval", "exec", "compile"}, (
                    path.name,
                    node.func.id,
                )


def test_onnxruntime_and_tokenizers_are_only_imported_by_the_provider_module() -> None:
    """Every other module -- including the rest of this package, and
    everything outside it (see the C4 contract's sanctioned-consumer test)
    -- must import cleanly without the ``semantic`` extra installed.

    AST-based, not a substring scan, because docstrings in this package
    (``manifest.py``, ``__init__.py``, this file) legitimately *name*
    ``onnxruntime``/``tokenizers`` in prose while explaining why they are
    not imported there; only an actual ``import`` statement should trip
    this guard.
    """
    for path in sorted((REPO_ROOT / "src/ckp").rglob("*.py")):
        if path.name == "provider.py" and path.parent == PACKAGE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "onnxruntime" not in imported, path
        assert "tokenizers" not in imported, path


def test_onnxruntime_and_tokenizers_are_not_base_dependencies() -> None:
    """Mirrors C5's equivalent test for the Qdrant client: the inference
    stack lives behind the ``semantic`` extra, never in base
    ``dependencies``, and therefore never in the runtime image."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    base = pyproject.split("[project.optional-dependencies]")[0]
    assert "onnxruntime" not in base.lower()
    assert "tokenizers" not in base.lower()
    assert 'semantic = [\n  "onnxruntime' in pyproject


def test_the_semantic_extra_is_excluded_from_the_runtime_image() -> None:
    """The Dockerfile's shipped runtime stage installs bare ``.`` -- no
    extras at all -- and only the disposable C7 smoke stage adds ``dev`` and
    ``index``. ``semantic`` must never appear in either ``pip install``
    line in the image that actually ships."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "semantic" not in dockerfile.lower()


def test_asset_manifest_pins_a_sha256_digest_and_a_positive_size_per_file() -> None:
    assert len(SEMANTIC_MODEL_FILES) > 0
    seen_paths = set()
    for spec in SEMANTIC_MODEL_FILES:
        assert spec.relative_path not in seen_paths, spec.relative_path
        seen_paths.add(spec.relative_path)
        assert len(spec.sha256) == 64
        int(spec.sha256, 16)  # raises ValueError if it is not hex
        assert spec.size_bytes > 0
        # The ≤150 MB weight budget (D4 amendment), enforced per file so a
        # future manifest edit that swaps in the fp32/fp16 variant (or a
        # bigger model entirely) fails here instead of silently blowing the
        # cold-rebuild budget.
        assert spec.size_bytes <= 150 * 1024 * 1024, spec.relative_path


def test_missing_model_directory_refuses_with_a_coded_error() -> None:
    with pytest.raises(SemanticRefusal) as refusal:
        resolve_semantic_assets(model_dir=REPO_ROOT / "does-not-exist-25")
    assert refusal.value.code is SemanticErrorCode.ASSET_DIR_INVALID


def test_missing_asset_file_refuses_with_a_coded_error(tmp_path: Path) -> None:
    # An existing, empty directory: no files at all.
    with pytest.raises(SemanticRefusal) as refusal:
        resolve_semantic_assets(model_dir=tmp_path)
    assert refusal.value.code is SemanticErrorCode.ASSETS_MISSING


def test_a_tampered_asset_file_refuses_with_a_coded_error(tmp_path: Path) -> None:
    """A digest mismatch must never be silently accepted -- this is the
    guard that stands between a corrupted or swapped-out model file and a
    provider that starts producing vectors from something nobody pinned."""
    for spec in SEMANTIC_MODEL_FILES:
        dest = tmp_path / spec.relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Right size, wrong content -- the size check alone must not be
        # enough to pass.
        dest.write_bytes(b"\x00" * spec.size_bytes)

    with pytest.raises(SemanticRefusal) as refusal:
        resolve_semantic_assets(model_dir=tmp_path)
    assert refusal.value.code is SemanticErrorCode.ASSET_DIGEST_MISMATCH


def test_a_truncated_asset_file_refuses_before_hashing(tmp_path: Path) -> None:
    for spec in SEMANTIC_MODEL_FILES:
        dest = tmp_path / spec.relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * min(16, spec.size_bytes))

    with pytest.raises(SemanticRefusal) as refusal:
        resolve_semantic_assets(model_dir=tmp_path)
    assert refusal.value.code is SemanticErrorCode.ASSET_DIGEST_MISMATCH


def test_asset_resolution_reads_every_byte_it_pins(tmp_path: Path) -> None:
    """A sanity check on the test fixtures above: the manifest's digest
    really is a whole-file sha256, not e.g. a digest of the first chunk --
    otherwise a mutant that only checks the first 1 MiB would still pass."""
    for spec in SEMANTIC_MODEL_FILES:
        dest = tmp_path / spec.relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = bytes((i % 256) for i in range(spec.size_bytes))
        dest.write_bytes(payload)
        assert hashlib.sha256(payload).hexdigest() != spec.sha256


def test_semantic_error_codes_are_prefixed_and_unique() -> None:
    from ckp.semantic.errors import SEMANTIC_ERROR_PREFIX, SemanticErrorCode

    values = [code.value for code in SemanticErrorCode]
    assert len(values) == len(set(values))
    assert all(value.startswith(f"{SEMANTIC_ERROR_PREFIX}/") for value in values)


def test_composition_builders_have_no_permissive_defaults() -> None:
    import inspect

    from ckp.semantic.composition import build_semantic_registry, build_semantic_stack

    expected = {"model_dir", "embedding_name", "reranker_name"}
    for builder in (build_semantic_registry, build_semantic_stack):
        signature = inspect.signature(builder)
        assert set(signature.parameters) == expected, builder.__name__
        for parameter in signature.parameters.values():
            assert parameter.default is inspect.Parameter.empty, parameter.name
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, parameter.name


def test_stack_validates_itself_rather_than_trusting_its_builder() -> None:
    """Mirrors ``tests/test_embedding_registry.py``'s equivalent test for
    ``EmbeddingStack`` -- ``SemanticEmbeddingStack`` re-derives its own
    guarantee instead of trusting whatever ``build_semantic_stack`` handed
    it, so a hand-assembled stack around an unadmitted provider, or one
    carrying a revision it cannot reproduce, refuses at construction."""
    from ckp.embedding.errors import EmbeddingErrorCode, EmbeddingRefusal
    from ckp.embedding.hashing import CosineReranker, HashEmbeddingProvider
    from ckp.embedding.models import compute_provider_revision
    from ckp.semantic.composition import SemanticEmbeddingStack
    from embedding_fixtures import DescriptorOnlyEmbedding, descriptor_with

    embedding = HashEmbeddingProvider(dimension=16)
    reranker = CosineReranker(embedder=embedding)
    good_revision = compute_provider_revision(embedding.descriptor)
    reranker_revision = compute_provider_revision(reranker.descriptor)

    # A real (if not actually semantic) provider pair with a tampered
    # revision string must still be refused.
    with pytest.raises(EmbeddingRefusal) as refusal:
        SemanticEmbeddingStack(
            embedding=embedding,
            reranker=reranker,
            embedding_revision="sha256:" + "0" * 64,
            reranker_revision=reranker_revision,
        )
    assert refusal.value.code is EmbeddingErrorCode.DESCRIPTOR_INVALID

    # An embedding provider that is not itself admissible (here: it claims
    # to require the network) must be refused by the stack's own
    # ``__post_init__`` -- not merely by whatever composed it. This is the
    # case a mutant that replaces ``require_provider(...)`` with a bare
    # ``self.embedding.descriptor`` read would let through.
    network_embedder = DescriptorOnlyEmbedding(descriptor_with(requires_network=True))
    with pytest.raises(EmbeddingRefusal) as refusal:
        SemanticEmbeddingStack(
            embedding=network_embedder,
            reranker=reranker,
            embedding_revision=compute_provider_revision(network_embedder.descriptor),
            reranker_revision=reranker_revision,
        )
    assert refusal.value.code is EmbeddingErrorCode.NETWORK_PROVIDER_DENIED

    # Sanity: the matching revision is accepted.
    stack = SemanticEmbeddingStack(
        embedding=embedding,
        reranker=reranker,
        embedding_revision=good_revision,
        reranker_revision=reranker_revision,
    )
    assert stack.embedding is embedding


# --- R1 review: provider_version must fingerprint everything that can
# --- change the output vector (weight digest, runtime package versions,
# --- implementation version), not a truncated model-revision hash. These
# --- run without the pinned weights -- ``_compute_fingerprinted_provider_version``
# --- only touches the manifest constants and whatever ``runtime_versions``
# --- tuple it is given.


def test_provider_version_is_a_full_64_char_digest_not_a_truncated_one() -> None:
    from ckp.semantic.provider import _compute_fingerprinted_provider_version

    version = _compute_fingerprinted_provider_version(
        runtime_versions=("onnxruntime==1.28.0", "tokenizers==0.23.1", "numpy==2.5.1")
    )
    assert len(version) == 64
    int(version, 16)  # raises ValueError if it is not hex


def test_provider_version_changes_when_a_runtime_package_version_changes() -> None:
    """R1 review finding 2: ``onnxruntime>=1.18,<2`` etc. are wide ranges in
    ``pyproject.toml``; two different resolved installs must not collapse
    onto the same ``provider_version``, or a stale index could be reused
    across an environment change nothing here would ever detect."""
    from ckp.semantic.provider import _compute_fingerprinted_provider_version

    baseline = ("onnxruntime==1.28.0", "tokenizers==0.23.1", "numpy==2.5.1")
    for index in range(len(baseline)):
        changed = list(baseline)
        changed[index] = changed[index].rsplit("==", 1)[0] + "==999.0.0"
        version_baseline = _compute_fingerprinted_provider_version(
            runtime_versions=baseline
        )
        version_changed = _compute_fingerprinted_provider_version(
            runtime_versions=tuple(changed)
        )
        assert version_baseline != version_changed, baseline[index]


def test_provider_version_changes_when_a_pinned_asset_digest_changes() -> None:
    """Same reasoning, for the model weights themselves: two different
    weight files must never fingerprint the same."""
    from ckp.semantic.manifest import SemanticAssetFile
    from ckp.semantic.provider import _compute_fingerprinted_provider_version

    runtime_versions = ("onnxruntime==1.28.0", "tokenizers==0.23.1", "numpy==2.5.1")
    original = tuple(SEMANTIC_MODEL_FILES)
    tampered = (
        SemanticAssetFile(
            relative_path=original[0].relative_path,
            sha256="0" * 64,
            size_bytes=original[0].size_bytes,
        ),
        *original[1:],
    )
    version_original = _compute_fingerprinted_provider_version(
        runtime_versions=runtime_versions, asset_files=original
    )
    version_tampered = _compute_fingerprinted_provider_version(
        runtime_versions=runtime_versions, asset_files=tampered
    )
    assert version_original != version_tampered


def test_provider_version_does_not_collide_on_a_shared_digest_prefix() -> None:
    """R1 review finding 2, the precise failure mode: the pre-review
    implementation used ``MODEL_REVISION[:16]`` -- the first 16 characters
    of a commit hash. Two pinned weight files whose full sha256 digests
    happen to share the same first 16 hex characters (entirely plausible;
    16 hex chars is only 64 bits) but differ afterward must still produce
    *different* fingerprints. A fingerprint built from a truncated digest
    would collapse these two onto the same ``provider_version`` -- a stale
    index silently reused for genuinely different weights."""
    from ckp.semantic.manifest import SemanticAssetFile
    from ckp.semantic.provider import _compute_fingerprinted_provider_version

    runtime_versions = ("onnxruntime==1.28.0", "tokenizers==0.23.1", "numpy==2.5.1")
    shared_prefix = "abcdabcdabcdabcd"  # 16 hex chars
    files_a = (
        SemanticAssetFile(
            relative_path="onnx/model_quantized.onnx",
            sha256=shared_prefix + "1" * 48,
            size_bytes=1,
        ),
    )
    files_b = (
        SemanticAssetFile(
            relative_path="onnx/model_quantized.onnx",
            sha256=shared_prefix + "2" * 48,
            size_bytes=1,
        ),
    )
    version_a = _compute_fingerprinted_provider_version(
        runtime_versions=runtime_versions, asset_files=files_a
    )
    version_b = _compute_fingerprinted_provider_version(
        runtime_versions=runtime_versions, asset_files=files_b
    )
    assert version_a != version_b


def test_provider_version_is_stable_for_the_same_inputs() -> None:
    from ckp.semantic.provider import _compute_fingerprinted_provider_version

    runtime_versions = ("onnxruntime==1.28.0", "tokenizers==0.23.1", "numpy==2.5.1")
    first = _compute_fingerprinted_provider_version(runtime_versions=runtime_versions)
    second = _compute_fingerprinted_provider_version(runtime_versions=runtime_versions)
    assert first == second


def test_installed_runtime_versions_reads_resolved_versions_not_a_range_string() -> (
    None
):
    """R1 review finding 2, checked mechanically: the fingerprint input must
    come from ``importlib.metadata`` (what is actually installed), not from
    re-parsing ``pyproject.toml``'s ``>=``/``<`` range string."""
    source = (PACKAGE / "provider.py").read_text(encoding="utf-8")
    assert "importlib" in source
    assert "metadata.version(" in source


def test_deterministic_claim_is_scoped_to_the_provider_revision_in_the_docstring() -> (
    None
):
    """R1 review finding 1: single-threaded ONNX execution proves
    same-host, same-process-family reproducibility -- it does not prove
    bit-identical output across different CPU microarchitectures or
    ``onnxruntime`` builds. The module docstring must say so explicitly
    rather than imply an unqualified cross-host guarantee; this pins the
    words that say so instead of letting a rewrite quietly drop them."""
    import ckp.semantic.provider as provider_module

    docstring = provider_module.__doc__ or ""
    assert "same provider revision" in docstring
    assert "not" in docstring and "bit-identical" in docstring
    assert "CPU" in docstring


# --- R1 review, round 2: the pure fingerprint function was well tested, but
# --- nothing weight-free exercised whether ``__init__`` actually *wires*
# --- ``_installed_runtime_versions()`` into it, rather than a mutant
# --- swapping in ``()`` at the call site. ``_build_descriptor`` isolates
# --- that wiring from session/tokenizer loading (see its docstring), so it
# --- is checkable here without the pinned weights on disk -- building a
# --- ``ProviderDescriptor`` needs no model bytes. The complementary
# --- weight-gated check that ``__init__`` actually *calls*
# --- ``_build_descriptor`` with the real installed versions lives in
# --- ``tests/test_semantic_embedding_provider.py``
# --- (``test_descriptor_provider_version_matches_the_installed_runtime_fingerprint``).


def test_build_descriptor_feeds_the_given_runtime_versions_into_the_fingerprint() -> (
    None
):
    from ckp.semantic.provider import (
        _build_descriptor,
        _compute_fingerprinted_provider_version,
    )

    runtime_versions = ("onnxruntime==1.28.0", "tokenizers==0.23.1", "numpy==2.5.1")
    descriptor = _build_descriptor(runtime_versions=runtime_versions)
    expected = _compute_fingerprinted_provider_version(
        runtime_versions=runtime_versions
    )
    assert descriptor.provider_version == expected

    # The mutation this test exists to catch: swapping in an empty tuple
    # (Codex R1 Finding 2's exact shape) must not produce the same
    # fingerprint as the real installed versions.
    empty_version = _build_descriptor(runtime_versions=()).provider_version
    assert empty_version != descriptor.provider_version


def test_build_descriptor_is_otherwise_honest() -> None:
    from ckp.semantic.provider import SEMANTIC_EMBEDDING_ID, _build_descriptor

    descriptor = _build_descriptor(runtime_versions=("onnxruntime==1.28.0",))
    assert descriptor.provider_id == SEMANTIC_EMBEDDING_ID
    assert descriptor.dimension == SEMANTIC_MODEL_DIMENSION
    assert descriptor.normalized is True
    assert descriptor.deterministic is True
    assert descriptor.requires_network is False
    assert descriptor.semantic is True
