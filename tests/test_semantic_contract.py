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
from ckp.semantic.manifest import SEMANTIC_MODEL_FILES
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
    return sorted(PACKAGE.glob("*.py"))


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
