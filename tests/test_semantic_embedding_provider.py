"""The real semantic provider, against the frozen ``EmbeddingProvider``
invariants -- and against the pinned model weights themselves.

Needs the local semantic model cache (``CKP_SEMANTIC_MODEL_DIR``, populated
by ``scripts/fetch-semantic-model.sh``). Locally the suite skips when the
cache is absent or fails digest verification; in CI
``CKP_REQUIRE_SEMANTIC=1`` turns that skip into a failure so this path can
never silently stop running -- mirroring ``tests/test_index_qdrant.py``'s
``CKP_REQUIRE_QDRANT`` exactly.

**Round 3 (CI, real failure, not a flake).** This file used to pin a
``GOLDEN`` dict of sha256 digests over packed embedding vectors, frozen on
one MacBook (arm64). CI (ubuntu x86_64) failed every one of them the first
time it ran this suite: ``onnxruntime``'s kernel selection is not
bit-identical across CPU microarchitectures, even with thread count pinned
and the same package versions -- exactly what ``ckp.semantic.provider``'s
module docstring already said ``deterministic=True`` does *not* cover. The
docstring was right; the tests contradicted it. Nothing here pins an exact
vector to a cross-machine constant anymore. What is checked instead:
semantic *relationships* between freshly computed vectors (cosine
similarity, with real margin -- stable across hosts because small
floating-point drift moves a cosine value by far less than the thresholds'
margins), within-host reproducibility (compared fresh against itself on
whichever host is running the suite, never a frozen value), and structural
properties (dimension, dtype, unit norm) that genuinely are stable across
hosts. Detecting a swapped model is still covered -- just by the correct
mechanism, ``ckp.semantic.assets.resolve_semantic_assets``'s content-digest
verification (``tests/test_semantic_contract.py``), not by comparing
inference output to a frozen value.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from ckp.embedding.errors import EmbeddingErrorCode, EmbeddingRefusal
from ckp.embedding.hashing import CosineReranker
from ckp.embedding.models import ProviderKind
from ckp.embedding.provider import EmbeddingProvider, RerankerProvider
from ckp.semantic.assets import resolve_semantic_assets
from ckp.semantic.errors import SemanticRefusal
from ckp.semantic.manifest import SEMANTIC_MODEL_DIMENSION
from ckp.semantic.provider import (
    SEMANTIC_EMBEDDING_ID,
    SemanticEmbeddingProvider,
    _compute_fingerprinted_provider_version,
    _installed_runtime_versions,
)
from semantic_fixtures import RELATED_MIN, UNRELATED_MAX, cosine

_MODEL_DIR_RAW = os.environ.get("CKP_SEMANTIC_MODEL_DIR", "")
_REQUIRED = os.environ.get("CKP_REQUIRE_SEMANTIC") == "1"


def _assets_available() -> Path | None:
    if not _MODEL_DIR_RAW:
        return None
    candidate = Path(_MODEL_DIR_RAW)
    try:
        resolve_semantic_assets(model_dir=candidate)
    except SemanticRefusal:
        return None
    return candidate


_MODEL_DIR = _assets_available()
if _REQUIRED and _MODEL_DIR is None:
    pytest.fail(
        "CKP_REQUIRE_SEMANTIC=1 but no verified semantic model cache is "
        "reachable at CKP_SEMANTIC_MODEL_DIR"
    )

needs_semantic = pytest.mark.skipif(
    _MODEL_DIR is None,
    reason="no local semantic model cache reachable (CKP_SEMANTIC_MODEL_DIR)",
)

#: Representative sample texts -- Chinese-majority, matching the pilot
#: corpus this benchmark exists to evaluate (``tests/embedding_fixtures.py``
#: ``SAMPLE_TEXTS`` uses the same mix). Used across several structural
#: tests below. Deliberately *not* paired with frozen output digests -- see
#: ``test_vectors_are_identical_across_processes_and_thread_counts`` for
#: why: this repo's own CI caught the reason. Golden digests over the
#: packed vectors were frozen on a MacBook (arm64) during development; CI
#: runs ubuntu x86_64, and every single one of them failed there --
#: ``onnxruntime``'s kernel selection is not bit-identical across CPU
#: microarchitectures even with thread count pinned and the same package
#: versions, exactly as ``ckp.semantic.provider``'s module docstring
#: (written *before* this was caught) already said it would not be. That
#: docstring was honest; the tests were not yet consistent with it. Fixed
#: here by testing what is actually true across machines: the semantic
#: relationships between vectors, and within-host reproducibility checked
#: freshly on whichever host is running the suite, never against a value
#: baked in from a different one.
SAMPLE_TEXTS = (
    "kettle temperature control",
    "手沖 咖啡 水溫 控制",
    "特教 個別化 教育 計畫",
    "個別化教育計畫要怎麼撰寫",
)

_EMBED_SCRIPT = """
import hashlib, json, os, struct, sys
from pathlib import Path
from ckp.semantic.provider import SemanticEmbeddingProvider

model_dir = Path(os.environ["CKP_SEMANTIC_MODEL_DIR"])
provider = SemanticEmbeddingProvider(model_dir=model_dir)
out = {}
for text in json.loads(sys.stdin.read()):
    values = provider.embed_query(text).values
    packed = struct.pack("<%dd" % len(values), *values)
    out[text] = hashlib.sha256(packed).hexdigest()
sys.stdout.write(json.dumps(out))
"""


def _digest(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


def _digests_in_subprocess(extra_env: dict[str, str]) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["CKP_SEMANTIC_MODEL_DIR"] = str(_MODEL_DIR)
    environment.update(extra_env)
    completed = subprocess.run(
        [sys.executable, "-c", _EMBED_SCRIPT],
        input=json.dumps(list(SAMPLE_TEXTS)),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@needs_semantic
def test_vectors_are_identical_across_processes_and_thread_counts() -> None:
    """Within-host determinism, checked freshly against itself -- not
    against a value frozen on a different machine (round 3: CI proved that
    was wrong; see the ``SAMPLE_TEXTS`` docstring above).

    The oracle still lives outside the unit, same reasoning as the hash
    provider's equivalent test: nothing inside one process proves the ONNX
    session is actually pinned to single-threaded execution. Two
    subprocesses, one of them with OMP_NUM_THREADS forced to 4, can -- and
    this is exactly the claim ``ckp.semantic.provider``'s module docstring
    scopes ``deterministic=True`` to: reproducible *on a given host*, which
    this test proves by comparing two runs on whichever host is actually
    running it, never against a cross-machine constant.
    """
    first = _digests_in_subprocess({})
    second = _digests_in_subprocess({"OMP_NUM_THREADS": "4"})
    assert first == second


@needs_semantic
def test_in_process_vectors_match_a_fresh_subprocess() -> None:
    """Same within-host reproducibility claim, in-process vs. subprocess
    this time -- still nothing frozen from a different machine. This is
    what replaces the old
    ``test_the_in_process_provider_matches_the_frozen_golden_digests``: it
    keeps the "does the in-process provider agree with a clean interpreter"
    check without pinning to a value that cannot survive a different CPU.
    """
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    in_process = {
        text: _digest(provider.embed_query(text).values) for text in SAMPLE_TEXTS
    }
    assert in_process == _digests_in_subprocess({})


@needs_semantic
def test_query_and_document_paths_are_bit_identical() -> None:
    """Frozen ``EmbeddingProvider`` invariant #2. Also the reason this
    provider carries no query-side instruction prefix -- see the module
    docstring in ``ckp.semantic.provider``."""
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    for text in SAMPLE_TEXTS:
        assert (
            provider.embed_query(text).values
            == provider.embed_documents([text]).vectors[0]
        )


@needs_semantic
def test_a_batch_preserves_length_and_order() -> None:
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    texts = list(SAMPLE_TEXTS)
    batch = provider.embed_documents(texts)
    assert len(batch.vectors) == len(texts)
    for index, text in enumerate(texts):
        assert batch.vectors[index] == provider.embed_query(text).values
    assert provider.embed_documents([]).vectors == ()


@needs_semantic
def test_vectors_have_the_declared_dimension_and_dtype() -> None:
    """Structural properties, unlike exact values, are stable across hosts
    -- these keep being asserted precisely (round 3 guidance point 4)."""
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    vector = provider.embed_query(SAMPLE_TEXTS[0]).values
    assert len(vector) == SEMANTIC_MODEL_DIMENSION
    assert all(isinstance(value, float) for value in vector)
    assert all(value == value for value in vector)  # no NaN
    norm = sum(value * value for value in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-9


@needs_semantic
def test_the_shipped_descriptor_is_honest() -> None:
    """RP3: ``semantic`` must reflect reality, not a hardcoded claim. This
    is checked against the *actually loaded* provider, backed by the
    pinned weights -- not a fixture that merely asserts the field's type."""
    descriptor = SemanticEmbeddingProvider(model_dir=_MODEL_DIR).descriptor
    assert descriptor.provider_id == SEMANTIC_EMBEDDING_ID
    assert descriptor.kind is ProviderKind.EMBEDDING
    assert descriptor.dimension == SEMANTIC_MODEL_DIMENSION
    assert descriptor.normalized is True
    assert descriptor.deterministic is True
    assert descriptor.requires_network is False
    assert descriptor.semantic is True


#: Thresholds and the cosine helper live in ``semantic_fixtures.py``, not
#: here -- ``tests/test_semantic_contract.py`` sanity-checks the threshold
#: *values* themselves (real margin, not tuned to the boundary) without
#: needing the pinned weights; see that module's docstring for the measured
#: numbers behind them.


@needs_semantic
def test_related_chinese_phrases_are_closer_than_unrelated_ones() -> None:
    """A quality sanity check, not just a shape check: this is the whole
    point of issue #25 -- the hash baseline cannot tell these apart at all
    beyond token overlap, and these two share zero substring overlap."""
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    related_a = provider.embed_query("特教 個別化 教育 計畫").values
    related_b = provider.embed_query("個別化教育計畫要怎麼撰寫").values
    # Same topic (coffee brewing), unrelated to the pair above -- not the
    # same sentence pair used in the English case below, so this also
    # checks a genuinely different pairing crosses the "unrelated" bar.
    unrelated = provider.embed_query("手沖 咖啡 水溫 控制").values

    sim_related = cosine(related_a, related_b)
    sim_unrelated = cosine(related_a, unrelated)
    assert sim_related > RELATED_MIN, sim_related
    assert sim_unrelated < UNRELATED_MAX, sim_unrelated
    assert sim_related > sim_unrelated + 0.2, (sim_related, sim_unrelated)


@needs_semantic
def test_related_english_phrases_are_closer_than_unrelated_ones() -> None:
    """English case for the same check (round 3 guidance: both languages
    need a case). Uses a near-paraphrase as the related pair rather than a
    loosely topic-associated one -- see the threshold docstring above for
    the measured reason."""
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    related_a = provider.embed_query("kettle temperature control").values
    related_b = provider.embed_query("controlling the temperature of the kettle").values
    unrelated = provider.embed_query(
        "individualized education plan for special needs students"
    ).values

    sim_related = cosine(related_a, related_b)
    sim_unrelated = cosine(related_a, unrelated)
    assert sim_related > RELATED_MIN, sim_related
    assert sim_unrelated < UNRELATED_MAX, sim_unrelated
    assert sim_related > sim_unrelated + 0.2, (sim_related, sim_unrelated)


@needs_semantic
def test_empty_and_non_text_inputs_fail_closed() -> None:
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    for empty in ("", "   ", "\n\t"):
        with pytest.raises(EmbeddingRefusal) as refusal:
            provider.embed_query(empty)
        assert refusal.value.code is EmbeddingErrorCode.TEXT_EMPTY
    for bad in (None, 7, b"bytes"):
        with pytest.raises(EmbeddingRefusal) as refusal:
            provider.embed_query(bad)  # type: ignore[arg-type]
        assert refusal.value.code is EmbeddingErrorCode.TEXT_INVALID


@needs_semantic
def test_the_provider_satisfies_the_protocol_and_composes_with_the_reranker() -> None:
    """No new reranker was written for issue #25 -- ``CosineReranker`` from
    C4's ``hashing.py`` is reused unmodified, proof that swapping the
    embedding provider changes no caller, not even the one immediately
    downstream of it."""
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    assert isinstance(provider, EmbeddingProvider)
    reranker = CosineReranker(embedder=provider)
    assert isinstance(reranker, RerankerProvider)
    assert reranker.descriptor.deterministic is True
    assert reranker.descriptor.requires_network is False


@needs_semantic
def test_offline_admission_still_holds_for_this_provider() -> None:
    """Same fence C4 has always enforced -- constructing this provider does
    not create a way around ``require_offline_provider``. Registration is
    exercised end to end via ``build_semantic_stack`` in
    ``tests/test_semantic_contract.py``; this test pins the descriptor
    flags the fence actually reads."""
    from ckp.embedding.provider import require_offline_provider

    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    require_offline_provider(provider.descriptor)  # must not raise


@needs_semantic
def test_descriptor_provider_version_matches_the_installed_runtime_fingerprint() -> (
    None
):
    """R1 review, round 2 (Codex Finding 2 alive at the wiring layer): the
    seven pure-function tests in ``tests/test_semantic_contract.py`` prove
    ``_compute_fingerprinted_provider_version`` reacts to a changed
    ``runtime_versions`` argument. None of them prove ``__init__`` actually
    *passes* the real ``_installed_runtime_versions()`` into it rather than,
    say, an empty tuple -- that call only happens inside a real
    construction, which needs the pinned weights. This is the integration
    check: build a real provider against the real cache, and assert its
    descriptor's ``provider_version`` equals the fingerprint independently
    recomputed from the actually-installed runtime versions. A mutant that
    breaks the wiring at the ``__init__`` call site (e.g. hardcoding
    ``runtime_versions=()``) makes this fail while every weight-free test
    stays green.
    """
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    expected = _compute_fingerprinted_provider_version(
        runtime_versions=_installed_runtime_versions()
    )
    assert provider.descriptor.provider_version == expected
