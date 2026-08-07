"""Resolve and verify the local semantic model cache. No network here.

This module only ever reads local files (``pathlib``, ``hashlib``) -- it is
the piece of issue #25 that must be provably incapable of a network call
regardless of what ``scripts/fetch-semantic-model.sh`` did or did not do
first. ``tests/test_semantic_contract.py`` pins the import surface.

The caller (``ckp.semantic.composition``) resolves ``model_dir`` from an
explicit argument, never a guessed default -- Phase 8 portability forbids a
baked-in host path, and a silently-wrong default cache directory is exactly
the kind of thing that should fail loud instead of loading nothing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ckp.semantic.errors import SemanticErrorCode, SemanticRefusal
from ckp.semantic.manifest import SEMANTIC_MODEL_FILES, SemanticAssetFile

_CHUNK_SIZE = 1 << 20


@dataclass(frozen=True)
class SemanticModelAssets:
    """Digest-verified *content*, in memory -- not paths.

    ``ckp.semantic.provider`` loads directly from ``model_bytes`` /
    ``tokenizer_json`` rather than re-opening a path after verification.
    Handing back a path here would leave a TOCTOU window between "this file
    hashed correctly" and "the loader opened this file": on a shared host,
    something else could in principle replace the file on disk in between
    (R1 review, non-blocking). Returning the exact bytes that were hashed
    closes that window structurally -- there is no second read for a
    swapped file to land in.
    """

    model_bytes: bytes
    tokenizer_json: str


def _read_and_verify_file(model_dir: Path, spec: SemanticAssetFile) -> bytes:
    path = model_dir / spec.relative_path
    if not path.is_file():
        raise SemanticRefusal(SemanticErrorCode.ASSETS_MISSING)
    if path.stat().st_size != spec.size_bytes:
        # Cheaper than hashing a wrong-sized file, and a distinct enough
        # symptom (partial download, wrong variant) that the same code as a
        # digest mismatch is still the right call -- both mean "this is not
        # the pinned file".
        raise SemanticRefusal(SemanticErrorCode.ASSET_DIGEST_MISMATCH)

    hasher = hashlib.sha256()
    chunks: list[bytes] = []
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
            chunks.append(chunk)

    if hasher.hexdigest() != spec.sha256:
        raise SemanticRefusal(SemanticErrorCode.ASSET_DIGEST_MISMATCH)
    # Only reached once the digest of exactly these bytes has matched --
    # nothing downstream re-reads the path.
    return b"".join(chunks)


def resolve_semantic_assets(*, model_dir: Path) -> SemanticModelAssets:
    """Verify every pinned file under ``model_dir`` and return its content.

    Fails closed and by name: a missing directory, a missing file, a
    truncated download, or a bit-flipped one all raise a coded
    ``SemanticRefusal`` rather than the provider silently loading whatever
    it finds (or a crash three layers into ``onnxruntime``).
    """
    if not isinstance(model_dir, Path) or not model_dir.is_dir():
        raise SemanticRefusal(SemanticErrorCode.ASSET_DIR_INVALID)

    verified = {
        spec.relative_path: _read_and_verify_file(model_dir, spec)
        for spec in SEMANTIC_MODEL_FILES
    }
    return SemanticModelAssets(
        model_bytes=verified["onnx/model_quantized.onnx"],
        tokenizer_json=verified["tokenizer.json"].decode("utf-8"),
    )


__all__ = [
    "SemanticModelAssets",
    "resolve_semantic_assets",
]
