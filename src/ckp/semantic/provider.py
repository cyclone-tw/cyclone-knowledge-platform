"""ONNX-backed offline semantic embedding provider (issue #25).

Real BGE-family sentence embeddings (see ``ckp.semantic.manifest`` for the
exact checkpoint and why it was chosen), CLS-pooled and L2-normalized,
running on CPU through ``onnxruntime``. The session is pinned to
single-threaded, sequential execution so a vector is bit-identical across
processes, hosts, and thread counts -- required both by the offline
admission fence (``NONDETERMINISTIC_PROVIDER_DENIED``) and by
``compute_provider_revision`` staying reproducible (AGENTS.md §8). Verified
empirically (three separate interpreter invocations, one with
``OMP_NUM_THREADS=4`` set, all producing the identical packed-vector SHA256)
before being pinned into the golden digests in
``tests/test_semantic_embedding_provider.py``.

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

from collections.abc import Sequence
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
    MODEL_REVISION,
    SEMANTIC_MODEL_DIMENSION,
    SEMANTIC_MODEL_MAX_SEQUENCE_LENGTH,
)

SEMANTIC_EMBEDDING_ID = "bge-small-zh-v1.5-onnx-int8"

#: Tied to the pinned manifest revision (not a hand-picked "1") so a model
#: swap or asset update forces a new ``provider_version`` -- and therefore a
#: new ``compute_provider_revision`` -- rather than silently reusing the old
#: provider identity for different weights. Matches the descriptor's
#: ``provider_version`` shape (``^[a-z0-9][a-z0-9._-]*$``, <=64 chars): a
#: commit hash prefix already satisfies it.
PROVIDER_VERSION = MODEL_REVISION[:16]


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

        session_options = ort.SessionOptions()
        # Single-threaded, sequential: floating-point reduction order (and
        # therefore the output vector, down to the last bit) must not
        # depend on how many cores the host happens to have, or on whatever
        # OMP_NUM_THREADS the caller's environment sets.
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(assets.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(assets.tokenizer_path))
        self._tokenizer.enable_truncation(max_length=SEMANTIC_MODEL_MAX_SEQUENCE_LENGTH)
        self._input_names = frozenset(item.name for item in self._session.get_inputs())

        self._descriptor = ProviderDescriptor(
            contract_version=EMBEDDING_CONTRACT,
            provider_id=SEMANTIC_EMBEDDING_ID,
            provider_version=PROVIDER_VERSION,
            kind=ProviderKind.EMBEDDING,
            dimension=SEMANTIC_MODEL_DIMENSION,
            metric=SimilarityMetric.COSINE,
            normalized=True,
            deterministic=True,
            requires_network=False,
            # Backed by a real pretrained sentence-embedding checkpoint
            # (see the manifest docstring), not hashed term overlap -- the
            # honesty field this whole issue exists to earn.
            semantic=True,
        )

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
    "PROVIDER_VERSION",
    "SEMANTIC_EMBEDDING_ID",
    "SemanticEmbeddingProvider",
]
