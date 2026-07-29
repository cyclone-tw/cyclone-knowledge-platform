#!/usr/bin/env bash
# Container runtime smoke for the walking skeleton.
#
# Builds the image from a clean context, runs it, and asserts that /revision
# reports *evidence* rather than placeholders. The assertion that matters is
# the last one: the image has no git, so bundle_commit can only be right if
# the build-time stamp actually reached the service. A hardcoded field would
# pass every other check here and fail that one.
#
# Overridable: IMAGE, PORT, NAME, BUNDLE_COMMIT.
set -euo pipefail

IMAGE="${IMAGE:-ckp:smoke}"
PORT="${PORT:-8080}"
NAME="${NAME:-ckp-smoke}"
# `-` not `:-` on purpose: an explicitly empty BUNDLE_COMMIT means "build
# without provenance" and must reach the unknown-commit assertion below. With
# `:-` an empty value would silently fall back to git HEAD and that branch
# would never be exercised.
BUNDLE_COMMIT="${BUNDLE_COMMIT-$(git rev-parse HEAD 2>/dev/null || true)}"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

echo "==> building $IMAGE (BUNDLE_COMMIT=${BUNDLE_COMMIT:-<none>})"
docker build --quiet --build-arg BUNDLE_COMMIT="$BUNDLE_COMMIT" -t "$IMAGE" .

echo "==> starting $NAME on port $PORT"
docker run -d --name "$NAME" -p "$PORT:8080" "$IMAGE" >/dev/null

ready=0
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.5
done
if [ "$ready" -ne 1 ]; then
  echo "smoke: service never became ready; container logs follow" >&2
  docker logs "$NAME" >&2 || true
  exit 1
fi

health="$(curl -fsS "http://127.0.0.1:$PORT/health")"
revision="$(curl -fsS "http://127.0.0.1:$PORT/revision")"
echo "==> health:   $health"
echo "==> revision: $revision"

BUNDLE_COMMIT="$BUNDLE_COMMIT" python3 - "$health" "$revision" <<'PY'
import json
import os
import sys

health = json.loads(sys.argv[1])
revision = json.loads(sys.argv[2])
failures = []

if health.get("status") != "ok":
    failures.append(f"health status is {health.get('status')!r}, expected 'ok'")
if health.get("checks", {}).get("bundle_readable") is not True:
    failures.append("bundle is not readable inside the container")

for field in ("profile_version", "api_version", "bundle_commit", "index_revision"):
    if field not in revision:
        failures.append(f"contract field {field} missing from /revision")

if not revision.get("api_version"):
    failures.append("api_version is empty")
if revision.get("profile_version") != "cyclone-profile-v1":
    failures.append(f"unexpected profile_version {revision.get('profile_version')!r}")
if revision.get("sources", {}).get("profile_version") != "bundle-descriptor":
    failures.append("profile_version did not come from the bundle descriptor")
if not str(revision.get("index_revision") or "").startswith("sha256:"):
    failures.append(f"index_revision is not a digest: {revision.get('index_revision')!r}")

expected_commit = os.environ.get("BUNDLE_COMMIT", "").strip()
if expected_commit:
    # The image ships no git, so "stamp" is the only honest source here. A
    # value that appeared any other way is not evidence.
    if revision.get("bundle_commit") != expected_commit:
        failures.append(
            f"bundle_commit is {revision.get('bundle_commit')!r}, "
            f"expected the build arg {expected_commit!r}"
        )
    if revision.get("sources", {}).get("bundle_commit") != "stamp":
        failures.append(
            "bundle_commit source is "
            f"{revision.get('sources', {}).get('bundle_commit')!r}, expected 'stamp'"
        )
else:
    # No build arg supplied: the correct answer is an honest unknown.
    if revision.get("bundle_commit") is not None:
        failures.append("bundle_commit was invented without a build-time stamp")

if failures:
    for line in failures:
        print(f"smoke: {line}", file=sys.stderr)
    sys.exit(1)
print("smoke: container revision reporting looks right")
PY
