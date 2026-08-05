#!/usr/bin/env bash
# Container runtime smoke for revision, privacy, Catalog, and Gateway.
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
WRITER_IMAGE="${WRITER_IMAGE:-ckp:c7-writer-smoke}"
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
EXPECTED_API_VERSION = "0.2"
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

# C3 Gateway smoke through the production HTTP surface. Add synthetic notes
# only inside this disposable container: the tracked bundle stays public-only,
# while the already-running process must notice the mutation and rebuild its
# app-owned snapshot cache.
echo "==> Catalog and Gateway HTTP smoke"
initial_catalog="$(curl -fsS "http://127.0.0.1:$PORT/catalog?limit=100")"
docker exec -i "$NAME" python - <<'PY'
from pathlib import Path

root = Path("/app/fixtures/synthetic-bundle")
(root / "c3-public.md").write_text(
    "---\n"
    "privacy: public\n"
    "title: C3 Public Needle\n"
    "type: Concept\n"
    "content_category: development\n"
    "---\n\n"
    "# C3 Public Needle\n\n"
    "gatewayneedle synthetic public body.\n",
    encoding="utf-8",
)
(root / "c3-undetermined.md").write_text(
    "---\n"
    "title: C3 Missing Needle\n"
    "type: Concept\n"
    "content_category: development\n"
    "---\n\n"
    "gatewayneedle privacy is deliberately absent.\n",
    encoding="utf-8",
)
(root / "c3-student.md").write_text(
    "---\n"
    "privacy: student-private\n"
    "title: Invented Learner Needle\n"
    "type: Concept\n"
    "content_category: development\n"
    "---\n\n"
    "gatewayneedle entirely synthetic student fixture.\n",
    encoding="utf-8",
)
PY

catalog="$(curl -fsS "http://127.0.0.1:$PORT/catalog?limit=100")"
query="$(
  curl -fsS \
    -H 'content-type: application/json' \
    --data '{"query":"gatewayneedle","limit":20}' \
    "http://127.0.0.1:$PORT/query"
)"
gateway_revision="$(curl -fsS "http://127.0.0.1:$PORT/revision")"

python3 - "$initial_catalog" "$catalog" "$query" "$gateway_revision" <<'PY'
import json
import sys

initial = json.loads(sys.argv[1])
catalog = json.loads(sys.argv[2])
query = json.loads(sys.argv[3])
revision = json.loads(sys.argv[4])
failures = []

if catalog.get("total") != initial.get("total", -1) + 1:
    failures.append(
        "Catalog total did not increase by exactly the one public synthetic note"
    )

def facet_count(payload, facet, value):
    buckets = payload.get("facets", {}).get(facet, [])
    return next((bucket.get("count") for bucket in buckets if bucket.get("value") == value), 0)

if facet_count(catalog, "types", "Concept") != facet_count(
    initial, "types", "Concept"
) + 1:
    failures.append("Concept facet counted a refused synthetic note")

results = query.get("results", [])
if query.get("total") != 1 or len(results) != 1:
    failures.append("bounded lexical query did not return exactly one public result")
elif results[0].get("title") != "C3 Public Needle":
    failures.append("query returned the wrong result")
else:
    citation = results[0].get("citation")
    if not citation:
        failures.append("Gateway result has no citation")
    else:
        if citation.get("path") != "c3-public.md":
            failures.append(f"citation path is {citation.get('path')!r}")
        if citation.get("concept_id") != results[0].get("concept_id"):
            failures.append("citation points at a different result")
        if citation.get("index_revision") != query.get("revision", {}).get(
            "index_revision"
        ):
            failures.append("citation and Gateway result use different revisions")

for name, payload in (("Catalog", catalog), ("query", query)):
    rendered = json.dumps(payload, sort_keys=True)
    for forbidden in ("C3 Missing Needle", "Invented Learner Needle"):
        if forbidden in rendered:
            failures.append(f"{name} leaked refused payload {forbidden!r}")

served_revision = revision.get("index_revision")
if query.get("revision", {}).get("index_revision") != served_revision:
    failures.append("Gateway and /revision do not use the same current snapshot")
if catalog.get("revision", {}).get("index_revision") != served_revision:
    failures.append("Catalog and /revision do not use the same current snapshot")

if failures:
    for line in failures:
        print(f"smoke: {line}", file=sys.stderr)
    sys.exit(1)
print("smoke: public-only Catalog and cited Gateway query look right")
PY

# C6 scoped-read smoke inside the production image. Credentials, notes and
# domain bindings exist only in this disposable process; the shipped app keeps
# its deny-all default and no production Private bundle is connected.
echo "==> C6 auth scope and bounded context smoke"
docker run --rm -i --entrypoint python "$IMAGE" - <<'PY'
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ckp.auth import (
    AccessActor,
    AccessDenied,
    ActorCredential,
    Capability,
    CredentialRegistry,
    DomainBinding,
    DomainRegistry,
    GrantCredential,
    GrantPolicy,
    KnowledgeDomain,
    ScopeGrant,
)
from ckp.bundle import AnchoredBundleReader, SnapshotCache
from ckp.gateway.models import CatalogRequest, QueryRequest
from ckp.gateway.scoped import OfflineQmdScope, ScopedCatalogSelector, ScopedKnowledgeGateway
from ckp.packer import BoundedContextPacker, ContextRequest
from ckp.privacy import FrontmatterClassifier, PrivacyClass

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
ACTOR_TOKEN = "synthetic-container-actor-codex"
OTHER_ACTOR_TOKEN = "synthetic-container-actor-claude"
GRANT_TOKEN = "synthetic-container-grant-finance"
COMMIT = "c" * 40
FINANCE_INTERNAL = "00000000-0000-7000-8000-000000000101"
FINANCE_SENSITIVE = "00000000-0000-7000-8000-000000000102"
CALENDAR_SENSITIVE = "00000000-0000-7000-8000-000000000103"
STUDENT = "00000000-0000-7000-8000-000000000104"


def note(root, name, *, privacy, note_id, title, body):
    (root / name).write_text(
        "---\n"
        f"privacy: {privacy}\n"
        f"id: {note_id}\n"
        f"title: {title}\n"
        "type: Concept\n"
        "content_category: life\n"
        "---\n\n"
        f"# {title}\n\n{body}\n",
        encoding="utf-8",
    )


def grant(*, expires_at):
    return ScopeGrant(
        grant_id="grant-synthetic-container-finance",
        actor=AccessActor.CODEX,
        task_id="task-synthetic-container-finance",
        domains=frozenset({KnowledgeDomain.FINANCE}),
        capabilities=frozenset({Capability.READ, Capability.REPORT_GENERATION}),
        privacy_classes=frozenset(
            {PrivacyClass.PUBLIC, PrivacyClass.INTERNAL, PrivacyClass.SENSITIVE}
        ),
        issued_by="human:cyclone",
        issued_at=NOW - timedelta(minutes=5),
        expires_at=expires_at,
        max_items=2,
        max_token_upper_bound=1024,
    )


def registry(scope):
    return CredentialRegistry(
        policy=GrantPolicy(),
        actor_credentials=(
            ActorCredential.from_plaintext(ACTOR_TOKEN, AccessActor.CODEX),
            ActorCredential.from_plaintext(OTHER_ACTOR_TOKEN, AccessActor.CLAUDE_CODE),
        ),
        grant_credentials=(GrantCredential.from_plaintext(GRANT_TOKEN, scope),),
        clock=lambda: NOW,
    )


failures = []
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / ".bundle-commit").write_text(COMMIT + "\n", encoding="utf-8")
    note(
        root,
        "finance-internal.md",
        privacy="internal",
        note_id=FINANCE_INTERNAL,
        title="Synthetic Finance Internal",
        body="c6needle synthetic amount TWD 1200 category utilities.",
    )
    note(
        root,
        "finance-sensitive.md",
        privacy="sensitive",
        note_id=FINANCE_SENSITIVE,
        title="Synthetic Finance Sensitive",
        body="c6needle synthetic allocation summary.",
    )
    note(
        root,
        "calendar-sensitive.md",
        privacy="sensitive",
        note_id=CALENDAR_SENSITIVE,
        title="Synthetic Calendar Hidden",
        body="c6needle CALENDAR-CROSS-DOMAIN-SENTINEL event synthetic-42.",
    )
    note(
        root,
        "student.md",
        privacy="student-private",
        note_id=STUDENT,
        title="Invented Student Hidden",
        body="c6needle STUDENT-PRIVATE-SENTINEL entirely invented fixture.",
    )

    reader = AnchoredBundleReader(root)
    selector = ScopedCatalogSelector(
        SnapshotCache(reader, "**/*.md", expected_profile_version=None),
        FrontmatterClassifier(reader),
        DomainRegistry(
            (
                DomainBinding(FINANCE_INTERNAL, KnowledgeDomain.FINANCE),
                DomainBinding(FINANCE_SENSITIVE, KnowledgeDomain.FINANCE),
                DomainBinding(
                    CALENDAR_SENSITIVE, KnowledgeDomain.CALENDAR_CONTEXT
                ),
                DomainBinding(STUDENT, KnowledgeDomain.FINANCE),
            )
        ),
    )
    gateway = ScopedKnowledgeGateway(selector, BoundedContextPacker)
    credentials = registry(grant(expires_at=NOW + timedelta(hours=1)))
    offline = OfflineQmdScope(selector, credentials)
    access = credentials.resolve(
        actor_credential=ACTOR_TOKEN,
        grant_credential=GRANT_TOKEN,
        requested_domain=KnowledgeDomain.FINANCE,
        required_capabilities=frozenset(
            {Capability.READ, Capability.REPORT_GENERATION}
        ),
    )
    catalog = gateway.catalog(access, CatalogRequest(limit=100))
    query = gateway.query(access, QueryRequest(query="c6needle", limit=20))
    context = gateway.context(access, ContextRequest(query="c6needle"))
    plan = offline.plan(
        actor_credential=ACTOR_TOKEN,
        grant_credential=GRANT_TOKEN,
        domain=KnowledgeDomain.FINANCE,
    )

    if catalog.total != 2 or query.total != 2:
        failures.append("scoped counts did not contain exactly two finance notes")
    rendered = context.context + repr(catalog) + repr(query)
    for forbidden in (
        "CALENDAR-CROSS-DOMAIN-SENTINEL",
        "STUDENT-PRIVATE-SENTINEL",
    ):
        if forbidden in rendered:
            failures.append(f"scoped output leaked {forbidden}")
    if context.token_upper_bound_used > context.token_upper_bound_limit:
        failures.append("context exceeded its token upper bound")
    if len(context.items) > context.item_limit:
        failures.append("context exceeded its item bound")
    for item in context.items:
        if item.citation.bundle_commit != COMMIT:
            failures.append("context item lacks the synthetic bundle commit")
        if item.citation.index_revision != context.revision.index_revision:
            failures.append("context citation revision drifted")
    if set(plan.paths) != {item.citation.path for item in catalog.items}:
        failures.append("offline QMD scope differs from Gateway scope")

    for actor_token, domain, expected in (
        (OTHER_ACTOR_TOKEN, KnowledgeDomain.FINANCE, "wrong-actor"),
        (ACTOR_TOKEN, KnowledgeDomain.CALENDAR_CONTEXT, "domain-denied"),
    ):
        try:
            credentials.resolve(
                actor_credential=actor_token,
                grant_credential=GRANT_TOKEN,
                requested_domain=domain,
                required_capabilities=frozenset({Capability.READ}),
            )
        except AccessDenied as error:
            if error.code != expected:
                failures.append(f"expected {expected}, got {error.code}")
        else:
            failures.append(f"{expected} case was admitted")

    expired = registry(
        ScopeGrant(
            grant_id="grant-synthetic-container-expired",
            actor=AccessActor.CODEX,
            task_id="task-synthetic-container-expired",
            domains=frozenset({KnowledgeDomain.FINANCE}),
            capabilities=frozenset({Capability.READ}),
            privacy_classes=frozenset({PrivacyClass.INTERNAL}),
            issued_by="human:cyclone",
            issued_at=NOW - timedelta(hours=2),
            expires_at=NOW - timedelta(hours=1),
            max_items=1,
            max_token_upper_bound=512,
        )
    )
    try:
        expired.resolve(
            actor_credential=ACTOR_TOKEN,
            grant_credential=GRANT_TOKEN,
            requested_domain=KnowledgeDomain.FINANCE,
            required_capabilities=frozenset({Capability.READ}),
        )
    except AccessDenied as error:
        if error.code != "scope-expired":
            failures.append(f"expected scope-expired, got {error.code}")
    else:
        failures.append("expired scope was admitted")

if failures:
    for line in failures:
        print(f"smoke: {line}", file=sys.stderr)
    sys.exit(1)
print("smoke: C6 scope, privacy, context bounds, citations and QMD parity look right")
PY

# The runtime path above must work from production dependencies alone.
docker exec -i "$NAME" python - <<'PY'
import importlib.util
import sys

unexpected = [
    name for name in ("pytest", "ruff", "httpx") if importlib.util.find_spec(name)
]
if unexpected:
    print(f"smoke: dev dependencies present in runtime image: {unexpected}", file=sys.stderr)
    sys.exit(1)
print("smoke: Gateway runtime uses no dev dependency")
PY

# C7 Writer smoke runs in a separate disposable stage. It contains Git and
# the synthetic tests, while the final runtime image above remains read-only
# and free of dev dependencies. Every fixture is generated under container
# TMPDIR; no Wiki, Private, runtime state, or production credential is mounted.
echo "==> C7 synthetic Git Writer smoke"
docker build \
  --quiet \
  --target c7-writer-smoke \
  --build-arg BUNDLE_COMMIT="$BUNDLE_COMMIT" \
  -t "$WRITER_IMAGE" \
  .
docker run --rm --entrypoint python "$WRITER_IMAGE" \
  -m pytest -q \
  tests/test_writer_models.py \
  tests/test_writer_errors.py \
  tests/test_writer_identity.py \
  tests/test_writer_target.py \
  tests/test_writer_validation.py \
  tests/test_writer_transaction.py \
  tests/test_c7_contract.py
echo "smoke: C7 synthetic Writer transaction matrix looks right"

# C8 outbox smoke reuses the same disposable stage: the deterministic matrix
# builds its synthetic Git fixture and encrypted outbox store under container
# TMPDIR, enqueues, replays, and asserts exactly one logical commit with no
# plaintext at rest -- on the deployment interpreter, not the dev venv.
echo "==> C8 synthetic outbox enqueue and replay smoke"
docker run --rm --entrypoint python "$WRITER_IMAGE" \
  -m pytest -q \
  tests/test_outbox_errors.py \
  tests/test_outbox_models.py \
  tests/test_outbox_crypto.py \
  tests/test_outbox_store.py \
  tests/test_outbox_enqueue_gate.py \
  tests/test_outbox_privacy_routing.py \
  tests/test_outbox_replay.py \
  tests/test_outbox_recovery.py \
  tests/test_c8_contract.py
echo "smoke: C8 synthetic outbox enqueue and replay matrix looks right"

# C4 embedding contract on the deployment interpreter. Pure computation, so
# it needs neither Git nor a running service -- only the same disposable
# stage that already carries the dev dependencies.
echo "==> C4 embedding and reranker interface smoke"
docker run --rm --entrypoint python "$WRITER_IMAGE" \
  -m pytest -q \
  tests/test_embedding_models.py \
  tests/test_embedding_provider.py \
  tests/test_embedding_registry.py \
  tests/test_embedding_neutrality.py \
  tests/test_c4_contract.py
echo "smoke: C4 embedding and reranker interfaces look right"

# The strongest available proof that the shipped provider downloads no model
# and calls no API: run it in the production image with the network removed
# and require the frozen golden digest. A provider that reached for a network
# would fail here rather than silently degrade.
echo "==> C4 offline determinism in the runtime image (no network)"
# `-i` is load-bearing: without it `docker run ... python -` gets an empty
# stdin, prints nothing, and exits 0 -- a check that always passes. The
# verdict line is required below so that failure mode cannot come back.
offline_verdict="$(docker run --rm -i --network none --entrypoint python "$IMAGE" - <<'PY'
import hashlib
import struct
import sys

from ckp.embedding import HashEmbeddingProvider

# Same constant as tests/test_embedding_provider.py, deliberately duplicated:
# an independent copy is what makes this an outside check rather than an echo.
GOLDEN = "67d72b59a5d87529b7a81e6a8ee7751c5c27f507774a6a59d91f475241408912"

provider = HashEmbeddingProvider(dimension=64)
values = provider.embed_query("手沖 咖啡 水溫 控制").values
digest = hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()
if digest != GOLDEN:
    print(f"smoke: offline embedding digest drifted: {digest}", file=sys.stderr)
    sys.exit(1)
if provider.descriptor.requires_network or not provider.descriptor.deterministic:
    print("smoke: shipped provider descriptor is not offline-deterministic", file=sys.stderr)
    sys.exit(1)
print("smoke: C4 hash provider is byte-identical with no network at all")
PY
)"
printf '%s\n' "$offline_verdict"
case "$offline_verdict" in
  *"byte-identical with no network at all"*) ;;
  *)
    echo "smoke: offline embedding check produced no verdict" >&2
    exit 1
    ;;
esac
