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

# The fixture digest, written out rather than computed by the code under test.
# Deriving it here with ckp.revision would make the oracle agree with any
# algorithm the implementation happened to use -- swap sha256 for sha1, keep
# the prefix, and both sides would move together. Kept in step with
# GOLDEN_FIXTURE_DIGEST in tests/test_revision.py, which fails if the fixture
# or the algorithm changes.
expected_index="sha256:754d8514119194d5cb3dbb906ac852282e2e318f3aeae2d5ca5dcd490280509d"

BUNDLE_COMMIT="$BUNDLE_COMMIT" EXPECTED_INDEX="$expected_index" \
  python3 - "$health" "$revision" <<'PY'
import json
import os
import sys

# Written out rather than imported from ckp.revision: an oracle that reads the
# value it is checking agrees with any value, including a wrong one.
EXPECTED_API_VERSION = "0.1"
EXPECTED_PROFILE_VERSION = "cyclone-profile-v1"

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

if revision.get("api_version") != EXPECTED_API_VERSION:
    failures.append(
        f"api_version is {revision.get('api_version')!r}, "
        f"expected {EXPECTED_API_VERSION!r}"
    )
if revision.get("profile_version") != EXPECTED_PROFILE_VERSION:
    failures.append(f"unexpected profile_version {revision.get('profile_version')!r}")

expected_index = os.environ.get("EXPECTED_INDEX", "").strip()
if not expected_index or not expected_index.startswith("sha256:"):
    failures.append(f"host-side digest did not compute: {expected_index!r}")
elif revision.get("index_revision") != expected_index:
    # The whole portability claim in one assertion: the container and the host
    # must derive the same revision from the same bundle.
    failures.append(
        f"index_revision is {revision.get('index_revision')!r}, "
        f"but this checkout computes {expected_index!r}"
    )

sources = revision.get("sources", {})
for field, expected_source in (
    ("profile_version", "bundle-descriptor"),
    ("api_version", "code"),
    ("index_revision", "computed"),
):
    if sources.get(field) != expected_source:
        failures.append(
            f"sources[{field}] is {sources.get(field)!r}, "
            f"expected {expected_source!r}"
        )

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

# Privacy gate smoke (Epic #1 / C2): prove the module ships in the image and
# behaves on the deployment interpreter, not just in the dev venv. Notes are
# synthesised inside the container -- no tracked fixture Markdown carries a
# non-public class (tests/test_no_wiki_content.py stays at full strength).
echo "==> privacy gate smoke"
docker run --rm -i --entrypoint python "$IMAGE" - <<'PY'
import sys
import tempfile
from pathlib import Path

from ckp.privacy import (
    Admitted,
    FrontmatterClassifier,
    PrivacyClass,
    PrivacyGate,
    Refused,
)
from ckp.privacy.gate import REASON_STUDENT_PRIVATE_SINK

failures = []
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    gate = PrivacyGate(
        FrontmatterClassifier(root),
        frozenset({PrivacyClass.PUBLIC, PrivacyClass.INTERNAL}),
    )
    ok = root / "ok.md"
    ok.write_bytes(b"---\nprivacy: public\n---\n\n# synthetic\n")
    undetermined = root / "undetermined.md"
    undetermined.write_bytes(b"---\ntitle: no declaration\n---\n\n# synthetic\n")
    student = root / "student.md"
    student.write_bytes(
        b"---\nprivacy: student-private\n---\n\n"
        b"# SYNTHETIC FIXTURE, invented student Zaphod Example-Student\n"
    )

    if not isinstance(gate.admit(ok), Admitted):
        failures.append("a public note was not admitted")
    verdict = gate.admit(undetermined)
    if not isinstance(verdict, Refused):
        failures.append("an undetermined note passed the gate")
    verdict = gate.admit(student)
    if not isinstance(verdict, Refused) or verdict.reason != REASON_STUDENT_PRIVATE_SINK:
        failures.append("student-private was not refused with its dedicated code")

if failures:
    for line in failures:
        print(f"smoke: {line}", file=sys.stderr)
    sys.exit(1)
print("smoke: privacy gate refuses what it must on the deployment interpreter")
PY
