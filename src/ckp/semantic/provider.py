"""ONNX-backed offline semantic embedding provider (issue #25).

Real BGE-family sentence embeddings (see ``ckp.semantic.manifest`` for the
exact checkpoint and why it was chosen), CLS-pooled and L2-normalized,
running on CPU through ``onnxruntime``.

**Determinism, scoped honestly (R1 review finding, AGENTS.md §8).** The ONNX
session is pinned to single-threaded, sequential execution
(``intra_op_num_threads=1`` / ``inter_op_num_threads=1``), which is
sufficient to make output independent of thread count and of
``OMP_NUM_THREADS`` -- verified empirically across two separate interpreter
processes on this host, one of them with ``OMP_NUM_THREADS=4`` forced
(``tests/test_semantic_embedding_provider.py``). It is **not** sufficient to
guarantee bit-identical output across different CPU instruction sets or
different ``onnxruntime`` builds: kernel selection inside ``onnxruntime`` can
vary with detected CPU features (e.g. AVX2 vs AVX-512 vs ARM NEON)
independently of the package version string, and this repo's own
portability plan means the exact host changes (MacBook to CI to Mac
Studio). Pinning thread count does not close that gap, and nothing in this
module claims it does.

What *is* claimed, and what ``deterministic=True`` on the descriptor means
here: **the same provider revision reproduces the same vector for the same
text on a given host.** ``compute_provider_revision`` (folded from
``provider_version``, see below) is the unit of reproducibility -- it
changes whenever the pinned model weights, any runtime package version that
can affect inference output, or this module's own implementation change.
Two installs that report the *same* ``compute_provider_revision`` are the
only two this provider claims agree; a revision match across genuinely
different CPU microarchitectures is not independently verified by anything
in this repo and is a known residual gap, not a solved problem. A rebuild
that crosses host architectures should still re-verify its ``payload_digest``
rather than assume byte-for-byte inheritance from a snapshot taken
elsewhere.

``embed_query`` and ``embed_documents`` share one code path
(``_embed_one``), matching the frozen ``EmbeddingProvider`` invariant that
the two cannot silently diverge. This is also why this provider does *not*
use BGE's asymmetric query-instruction prefix ("为这个句子生成表示以用于检索
相关文章：") even though the upstream model card recommends it for queries:
prefixing only the query path would violate that invariant. Retrieval
quality is left on the table here in exchange for a guarantee C4 will not
let this provider break silently.

Heavy dependencies (``onnxruntime``, ``tokenizers``, ``numpy``) are imported
inside ``__init__``/``_embed_one`` rather than at module scope, so
``import ckp.semantic`` still succeeds without the ``semantic`` extra
installed -- only constructing the provider requires it. This mirrors how
C5's Qdrant index module defers importing its network client library.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

from ckp.embedding.errors import EmbeddingErrorCode, EmbeddingRefusal
from ckp.embedding.models import (
    EMBEDDING_CONTRACT,
    EmbeddingBatch,
    EmbeddingVector,
    ProviderDescriptor,
    ProviderKind,
    SimilarityMetric,
    require_text,
)
from ckp.semantic.assets import resolve_semantic_assets
from ckp.semantic.errors import SemanticErrorCode, SemanticRefusal
from ckp.semantic.manifest import (
    SEMANTIC_MODEL_DIMENSION,
    SEMANTIC_MODEL_FILES,
    SEMANTIC_MODEL_MAX_SEQUENCE_LENGTH,
)

SEMANTIC_EMBEDDING_ID = "bge-small-zh-v1.5-onnx-int8"

#: Bump when the inference algorithm itself changes -- pooling strategy,
#: normalization precision, truncation length, tokenizer feed shape -- even
#: if the pinned weights and the installed runtime package versions do not.
#: Folded into ``_compute_fingerprinted_provider_version`` below, so a code
#: change that alters output without touching the manifest or the
#: environment still bumps ``compute_provider_revision``.
PROVIDER_IMPLEMENTATION_VERSION = "1"

#: Every installed package whose behavior can change the output vector.
#: ``numpy`` only does float64 widening and division here, but a numpy
#: build change (BLAS backend) is exactly the kind of thing R1 review
#: flagged as unaccounted for, so it is fingerprinted too even though it is
#: imported lazily inside ``_embed_one``, not here.
_RUNTIME_PACKAGES: tuple[str, ...] = ("onnxruntime", "tokenizers", "numpy")

_REVISION_FINGERPRINT_DOMAIN = b"ckp-semantic-provider-fingerprint-v1"


def _installed_runtime_versions() -> tuple[str, ...]:
    """The *resolved* installed version of every package that can change
    inference output -- read at construction time via package metadata, not
    the ``>=``/``<`` range string in ``pyproject.toml``.

    ``pyproject.toml`` pins ``onnxruntime>=1.18,<2`` etc.: a wide range that
    covers many installable versions, none of which are guaranteed to
    produce identical inference output (R1 review finding 2). Reading the
    version actually resolved by this environment's package installer, and
    folding it into the provider revision, is what makes
    ``compute_provider_revision`` change when the environment does instead
    of silently reusing a stale identity for a different install.

    Uses ``importlib.metadata`` rather than each package's own
    ``__version__`` attribute so this works without importing ``numpy`` at
    construction time (it stays a lazy import inside ``_embed_one``).
    """
    versions: list[str] = []
    for name in _RUNTIME_PACKAGES:
        try:
            versions.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError as error:
            raise SemanticRefusal(
                SemanticErrorCode.INFERENCE_BACKEND_UNAVAILABLE
            ) from error
    return tuple(versions)


def _length_framed_hash(*values: bytes) -> str:
    """Length-prefix every field before hashing, exactly like C4's
    ``compute_provider_revision`` -- concatenating unframed byte strings
    would let two different input sets collide at a field boundary (e.g.
    ``"ab" + "c"`` vs ``"a" + "bc"``)."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _compute_fingerprinted_provider_version(
    *,
    runtime_versions: tuple[str, ...],
    asset_files: tuple[object, ...] = SEMANTIC_MODEL_FILES,
) -> str:
    """Fold everything that can change the output vector into one
    64-hex-char fingerprint, used as ``provider_version``.

    Inputs, each length-framed: the full (untruncated) sha256 of every
    pinned model asset -- not a truncated commit hash, which would let two
    different weight swaps that happen to share a revision prefix collide
    -- the resolved version of every runtime package that can affect
    inference (``_installed_runtime_versions``), and
    ``PROVIDER_IMPLEMENTATION_VERSION``. Two installs that produce the same
    fingerprint are the only two this provider claims will produce the same
    vectors (see the module docstring for the scope this does *not* cover:
    cross-CPU-microarchitecture kernel selection inside ``onnxruntime``).

    ``asset_files`` defaults to the real pinned manifest but takes an
    explicit override so ``tests/test_semantic_contract.py`` can prove the
    fingerprint actually moves when a digest changes, without needing the
    real weights on disk.
    """
    parts: list[bytes] = [
        _REVISION_FINGERPRINT_DOMAIN,
        PROVIDER_IMPLEMENTATION_VERSION.encode("ascii"),
    ]
    for spec in asset_files:
        parts.append(spec.relative_path.encode("utf-8"))
        parts.append(spec.sha256.encode("ascii"))
    for version in runtime_versions:
        parts.append(version.encode("utf-8"))
    return _length_framed_hash(*parts)


def _build_descriptor(*, runtime_versions: tuple[str, ...]) -> ProviderDescriptor:
    """The descriptor computation, isolated from session/tokenizer loading.

    R1 review round 2: the previous shape inlined ``ProviderDescriptor(...)``
    directly inside ``__init__``, so the only thing exercising "did the
    installed runtime versions actually reach the fingerprint" was
    constructing a real provider against real weights -- something no
    weight-free test could reach, and the seven pure-function tests in
    ``tests/test_semantic_contract.py`` could not catch a mutant that
    swapped ``runtime_versions=runtime_versions`` for
    ``runtime_versions=()`` at the call site inside ``__init__``.

    Extracting this function does not, by itself, prove ``__init__`` calls
    it correctly -- that still needs a weight-gated integration test
    (``tests/test_semantic_embedding_provider.py``, which asserts a really
    constructed provider's ``descriptor.provider_version`` against this same
    function called independently). What it *does* buy: this function's own
    correctness -- does it actually feed ``runtime_versions`` into the
    fingerprint rather than a hardcoded stand-in -- is now checkable by
    ``tests/test_semantic_contract.py`` without the pinned weights on disk,
    since building a ``ProviderDescriptor`` needs no model bytes at all.
    """
    return ProviderDescriptor(
        contract_version=EMBEDDING_CONTRACT,
        provider_id=SEMANTIC_EMBEDDING_ID,
        provider_version=_compute_fingerprinted_provider_version(
            runtime_versions=runtime_versions
        ),
        kind=ProviderKind.EMBEDDING,
        dimension=SEMANTIC_MODEL_DIMENSION,
        metric=SimilarityMetric.COSINE,
        normalized=True,
        # Scoped claim -- see the module docstring: reproducible for the
        # same provider revision on a given host, not independently
        # verified bit-identical across different CPU microarchitectures
        # even at the same revision.
        deterministic=True,
        requires_network=False,
        # Backed by a real pretrained sentence-embedding checkpoint (see the
        # manifest docstring), not hashed term overlap -- the honesty field
        # this whole issue exists to earn.
        semantic=True,
    )


class SemanticEmbeddingProvider:
    """Loads once from a caller-verified local cache; opens no connection of
    any kind."""

    def __init__(self, *, model_dir: Path) -> None:
        assets = resolve_semantic_assets(model_dir=model_dir)
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as error:
            # The ``semantic`` extra was never installed. A bare
            # ImportError three frames into construction is not a coded
            # refusal a caller can branch on.
            raise SemanticRefusal(
                SemanticErrorCode.INFERENCE_BACKEND_UNAVAILABLE
            ) from error

        runtime_versions = _installed_runtime_versions()

        session_options = ort.SessionOptions()
        # Single-threaded, sequential: floating-point reduction order (and
        # therefore the output vector) must not depend on how many cores
        # the host happens to have, or on whatever OMP_NUM_THREADS the
        # caller's environment sets. This closes the *within-host* source
        # of nondeterminism; see the module docstring for what it does not
        # close (cross-host kernel selection).
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Loaded from the verified in-memory bytes ``resolve_semantic_assets``
        # already hashed -- not re-opened by path -- so there is no window
        # between "digest verified" and "loader reads the file" for a
        # swapped file to land in (R1 review, non-blocking TOCTOU finding).
        self._session = ort.InferenceSession(
            assets.model_bytes,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_str(assets.tokenizer_json)
        self._tokenizer.enable_truncation(max_length=SEMANTIC_MODEL_MAX_SEQUENCE_LENGTH)
        self._input_names = frozenset(item.name for item in self._session.get_inputs())

        self._descriptor = _build_descriptor(runtime_versions=runtime_versions)

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        if isinstance(texts, str) or not isinstance(texts, Sequence):
            raise EmbeddingRefusal(EmbeddingErrorCode.BATCH_INVALID)
        return EmbeddingBatch(
            descriptor=self._descriptor,
            vectors=tuple(self._embed_one(text) for text in texts),
        )

    def embed_query(self, text: str) -> EmbeddingVector:
        # Deliberately the same code path as a one-element batch -- see the
        # module docstring on why this provider carries no query-side
        # instruction prefix.
        return EmbeddingVector(
            descriptor=self._descriptor,
            values=self._embed_one(text),
        )

    def _embed_one(self, text: str) -> tuple[float, ...]:
        import numpy as np

        encoded = self._tokenizer.encode(require_text(text))
        input_ids = np.asarray([encoded.ids], dtype=np.int64)
        attention_mask = np.asarray([encoded.attention_mask], dtype=np.int64)
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.asarray([encoded.type_ids], dtype=np.int64)

        outputs = self._session.run(None, feed)
        # CLS-token pooling, per the upstream model's
        # ``1_Pooling/config.json`` (``pooling_mode_cls_token: true``) --
        # not mean pooling over the sequence.
        # The ONNX graph outputs float32. Normalizing in that precision and
        # then widening to Python's double-precision float (what
        # EmbeddingVector stores) leaves a residual L2 error well past
        # NORM_TOLERANCE (1e-12) -- observed ~1e-3 empirically. Widening to
        # float64 *before* computing the norm and dividing, mirroring
        # ``HashEmbeddingProvider``'s "normalize exactly once, in the
        # precision the descriptor promises" rule, keeps the residual near
        # float64 epsilon instead.
        cls_vector = outputs[0][0, 0, :].astype(np.float64)
        norm = float(np.linalg.norm(cls_vector))
        if norm == 0.0:
            # Never observed against the pinned checkpoint's real output,
            # but a zero vector has no direction to normalize -- refusing
            # is honest, inventing a fallback vector is not (AGENTS.md §8).
            raise EmbeddingRefusal(EmbeddingErrorCode.TEXT_EMPTY)
        return tuple((cls_vector / norm).tolist())


__all__ = [
    "PROVIDER_IMPLEMENTATION_VERSION",
    "SEMANTIC_EMBEDDING_ID",
    "SemanticEmbeddingProvider",
]
