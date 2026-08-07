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
    """Local, digest-verified paths to every file the provider loads."""

    model_path: Path
    tokenizer_path: Path


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_file(model_dir: Path, spec: SemanticAssetFile) -> Path:
    path = model_dir / spec.relative_path
    if not path.is_file():
        raise SemanticRefusal(SemanticErrorCode.ASSETS_MISSING)
    if path.stat().st_size != spec.size_bytes:
        # Cheaper than hashing a wrong-sized file, and a distinct enough
        # symptom (partial download, wrong variant) that the same code as a
        # digest mismatch is still the right call -- both mean "this is not
        # the pinned file".
        raise SemanticRefusal(SemanticErrorCode.ASSET_DIGEST_MISMATCH)
    if _digest(path) != spec.sha256:
        raise SemanticRefusal(SemanticErrorCode.ASSET_DIGEST_MISMATCH)
    return path


def resolve_semantic_assets(*, model_dir: Path) -> SemanticModelAssets:
    """Verify every pinned file under ``model_dir`` and return their paths.

    Fails closed and by name: a missing directory, a missing file, a
    truncated download, or a bit-flipped one all raise a coded
    ``SemanticRefusal`` rather than the provider silently loading whatever
    it finds (or a crash three layers into ``onnxruntime``).
    """
    if not isinstance(model_dir, Path) or not model_dir.is_dir():
        raise SemanticRefusal(SemanticErrorCode.ASSET_DIR_INVALID)

    verified = {
        spec.relative_path: _verify_file(model_dir, spec)
        for spec in SEMANTIC_MODEL_FILES
    }
    return SemanticModelAssets(
        model_path=verified["onnx/model_quantized.onnx"],
        tokenizer_path=verified["tokenizer.json"],
    )


__all__ = [
    "SemanticModelAssets",
    "resolve_semantic_assets",
]
