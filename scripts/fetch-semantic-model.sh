#!/usr/bin/env bash
# Build-time fetch for the pinned semantic embedding model (issue #25, D4
# amendment). Downloads exactly the files listed in
# ``src/ckp/semantic/manifest.py`` from the pinned commit, verifies each
# against its frozen sha256, and writes them under a local cache directory.
#
# This is the *only* place in the whole feature that is allowed to touch a
# network. ``ckp.semantic.provider`` and ``ckp.semantic.assets`` load only
# from the local files this script produces; neither imports anything
# network-capable (``tests/test_semantic_contract.py`` pins that).
# Idempotent: re-running against a cache that already verifies does nothing.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
SOURCE_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON:-python3}"

TARGET_DIR="${1:-${CKP_SEMANTIC_MODEL_DIR:-}}"
if [[ -z "$TARGET_DIR" ]]; then
  echo "usage: $0 <target-dir>  (or set CKP_SEMANTIC_MODEL_DIR)" >&2
  exit 2
fi
mkdir -p "$TARGET_DIR"

cd "$SOURCE_ROOT"

# The manifest (repo, revision, per-file path/digest/size) is the single
# source of truth -- read it from Python rather than duplicating those
# constants in shell, which is exactly how a pin and its enforcement drift
# apart.
FETCH_TARGET_DIR="$TARGET_DIR" PYTHONPATH="$SOURCE_ROOT/src" "$PYTHON_BIN" <<'PYEOF'
import hashlib
import os
import sys
import urllib.request

from ckp.semantic.manifest import MODEL_REPO_ID, MODEL_REVISION, SEMANTIC_MODEL_FILES

target_dir = os.environ["FETCH_TARGET_DIR"]


def digest(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


for spec in SEMANTIC_MODEL_FILES:
    dest = os.path.join(target_dir, spec.relative_path)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    if (
        os.path.isfile(dest)
        and os.path.getsize(dest) == spec.size_bytes
        and digest(dest) == spec.sha256
    ):
        print(f"==> {spec.relative_path}: already cached and verified")
        continue

    url = f"https://huggingface.co/{MODEL_REPO_ID}/resolve/{MODEL_REVISION}/{spec.relative_path}"
    print(f"==> fetching {spec.relative_path} from {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)

    actual_size = os.path.getsize(tmp)
    actual_sha256 = digest(tmp)
    if actual_size != spec.size_bytes or actual_sha256 != spec.sha256:
        os.remove(tmp)
        sys.exit(
            f"digest/size mismatch for {spec.relative_path}: "
            f"expected {spec.size_bytes}b sha256:{spec.sha256}, "
            f"got {actual_size}b sha256:{actual_sha256}"
        )
    os.replace(tmp, dest)
    print(f"==> {spec.relative_path}: fetched and verified ({actual_size} bytes)")

print(f"==> semantic model assets verified under {target_dir} "
      f"(repo={MODEL_REPO_ID} revision={MODEL_REVISION})")
PYEOF
