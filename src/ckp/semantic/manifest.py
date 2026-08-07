"""Pinned semantic model assets: identity, revision, and content digest.

Issue #25 / D4 (amended): a real local model is allowed to download its
*weights* during install/build, never at query time. This module is the
single source of truth for exactly what gets fetched and exactly what it
must hash to -- ``scripts/fetch-semantic-model.sh`` reads it to download and
verify, and ``ckp.semantic.assets`` reads it again at load time to refuse
anything that does not match. Bumping ``MODEL_REVISION`` or swapping the
model is a change to this file plus a re-run of the fetch script; it is
never a silent runtime download, because nothing on the query path reads
this file's revision/repo fields, only the per-file digests.

Model choice, recorded for the PR / issue trail:

* Upstream weights: ``BAAI/bge-small-zh-v1.5`` (MIT), a small (512-dim,
  4-layer) Chinese sentence embedding model -- chosen because the pilot
  corpus this benchmark exists to evaluate is Traditional-Chinese-majority
  special-education content (see ``tests/embedding_fixtures.py``
  ``SAMPLE_TEXTS``), and English-only small models cannot embed it at all.
* Distribution used here: ``Xenova/bge-small-zh-v1.5``, a same-weights ONNX
  re-export (the ``model_quantized.onnx`` int8 graph). This keeps the
  inference dependency to ``onnxruntime`` + ``tokenizers`` + ``numpy`` only
  -- no full deep-learning framework and no Python bindings for one -- and
  keeps the pinned payload at ~24 MB total, inside the ≤150 MB budget with
  headroom for the ``fp32``/``fp16`` variants this repo does *not* use.
* Pooling is CLS-token, per the upstream ``1_Pooling/config.json``
  (``pooling_mode_cls_token: true``, ``pooling_mode_mean_tokens: false``) --
  not mean pooling. Output is L2-normalized (upstream ``2_Normalize``
  module), reproduced in ``ckp.semantic.provider``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Hugging Face repo that hosts the pinned ONNX export. Not read by anything
#: on the query path -- only ``scripts/fetch-semantic-model.sh`` resolves it
#: into a download URL.
MODEL_REPO_ID = "Xenova/bge-small-zh-v1.5"

#: A commit, not a moving ref (``main`` would silently swap weights under a
#: cold rebuild). Recorded here as the provenance trail for what
#: ``SEMANTIC_MODEL_FILES`` below was fetched from; the content digests are
#: what actually gets enforced at load time.
MODEL_REVISION = "75c43b069aac4d136ba6bc1122f995fedcfd2781"

#: Upstream weights this ONNX export was converted from -- documentation
#: only, not fetched or verified by this package.
UPSTREAM_MODEL_REPO_ID = "BAAI/bge-small-zh-v1.5"
UPSTREAM_MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"

#: Fixed by the upstream architecture (``hidden_size`` in its
#: ``config.json``); not a tunable here.
SEMANTIC_MODEL_DIMENSION = 512

#: BERT's own position-embedding ceiling for this checkpoint
#: (``max_position_embeddings`` / ``sentence_bert_config.json``).
SEMANTIC_MODEL_MAX_SEQUENCE_LENGTH = 512


@dataclass(frozen=True)
class SemanticAssetFile:
    """One pinned file: where it lives under the model cache root, and what
    it must hash to. ``size_bytes`` is a cheap first check before spending
    time on the digest."""

    relative_path: str
    sha256: str
    size_bytes: int


#: Only the two files actually loaded at inference time. ``config.json`` and
#: the fp32/fp16 ONNX variants exist upstream but are never fetched --
#: fewer files to download and verify, and no unused multi-hundred-MB graph
#: sitting in the cache.
SEMANTIC_MODEL_FILES: tuple[SemanticAssetFile, ...] = (
    SemanticAssetFile(
        relative_path="onnx/model_quantized.onnx",
        sha256="15b717c382bcb518ba457b93ea6850ede7f4f1cd8937454aa06972366cd19bcc",
        size_bytes=24010842,
    ),
    SemanticAssetFile(
        relative_path="tokenizer.json",
        sha256="48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26",
        size_bytes=439125,
    ),
)


__all__ = [
    "MODEL_REPO_ID",
    "MODEL_REVISION",
    "SEMANTIC_MODEL_DIMENSION",
    "SEMANTIC_MODEL_FILES",
    "SEMANTIC_MODEL_MAX_SEQUENCE_LENGTH",
    "UPSTREAM_MODEL_REPO_ID",
    "UPSTREAM_MODEL_REVISION",
    "SemanticAssetFile",
]
