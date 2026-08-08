"""P6 resilience benchmark dimensions (contract §Phase 4, issue #27).

Three §5.3 degradation scenarios that Phase 3's shadow harness never
measured -- Gateway down, stale snapshot, and MacBook (disconnected) offline
-- turned into standalone, independently runnable observations. Each one
answers a yes/no question with evidence, never a "probably falls back":

* **Gateway down** (:func:`probe_gateway_down`): the bundle root the
  Gateway reads from is entirely unreachable. ``/health``, the anonymous C3
  routes (``/catalog``, ``/query``), and the authenticated C6 routes
  (``/scoped/catalog``, ``/scoped/query``, ``/context``) must each answer
  with an explicit >=500 error, never a 200 whose body quietly looks like
  "the corpus has zero notes". This list is deliberately exhaustive of the
  routes ``ckp/app.py`` exposes minus ``/revision`` -- ``/revision`` is
  intentionally excluded, not overlooked: its own docstring in
  ``ckp/app.py`` frames a 200 with null fields as the honest answer to "what
  exactly is being served" (a metadata surface), with ``/health`` as the
  separate "am I serving" signal this scenario checks instead.
* **Stale snapshot** (:func:`check_stale_snapshot`): a bundle changes on
  disk after a request has already been served from it. Contract §5.5 names
  "Dashboard shows a revision mismatch without flagging it stale" as a
  rollback trigger. As of #39, ``GET /revision`` (``ckp/app.py``) flags
  exactly this: it compares the snapshot other routes were just serving
  (``SnapshotCache.peek()``, no filesystem probe) against a fresh
  recomputation from the bundle root, and reports the mismatch as
  ``"stale": true``. This scenario drives the real FastAPI app -- not a
  benchmark-only comparator -- through the sequence unedited-call,
  edit-on-disk, next-call, next-call-again, and asserts the flag is
  ``False``, ``False``, ``True``, ``False`` in that order (the last ``False``
  because ``/revision``'s own call already re-verified and healed the
  cache). See "Findings" below for what changed from the pre-#39 version of
  this module.
* **MacBook offline** (:func:`check_offline_scope_fail_closed`): C6's
  :class:`~ckp.gateway.scoped.OfflineQmdScope` is the permission projection
  meant to run disconnected. This dimension checks it stays fail-closed --
  wrong actor, wrong domain, and an expired grant are all refused, and the
  emitted allowlist never contains a path outside the granted scope.

This module is deliberately independent of ``benchmarks/shadow.py`` (owned
by #22/#24/#26/#29 -- see AGENTS.md file ownership) and of
``benchmarks/questions.py``'s frozen corpus: it builds its own tiny,
synthetic, in-memory fixtures per scenario so it never has to touch either
file. It is a standalone module with its own entry point
(``python -m benchmarks.resilience``), not a code path inside
``run_shadow_benchmark``.

**Non-goal** (issue #27): this module measures existing behavior. It does
not add new fallback logic to ``src/ckp``. Where a scenario's assertion
would require behavior the platform does not yet have, that gap belongs in
a new issue, not a silent patch here -- see "Findings" below.

Findings from building this benchmark, and how #39 changed it:

* Before #39, no module in ``src/ckp`` compared a served revision against a
  fresh recomputation from the live bundle and flagged a mismatch as stale.
  This module carried its own ``_is_snapshot_stale`` comparator so the
  dimension could be observed at all, and ``stale_snapshot.passed`` was
  hardcoded ``False`` -- correct behavior in a benchmark-only comparator
  must never be presented as a production capability that does not exist
  (Codex Round 1 review, #27).
* #39 added that comparison to production: ``GET /revision`` now flags
  ``stale`` by comparing ``SnapshotCache.peek()`` (what was just served)
  against a fresh recomputation. This module's own comparator is gone --
  the whole reason it existed was that production had no equivalent path.
  ``check_stale_snapshot`` now drives the real endpoint through an
  edit-on-disk sequence instead, and ``stale_snapshot.passed`` reflects
  what that endpoint actually did, not a hardcoded value.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from ckp.app import create_app
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
from ckp.config import load_config
from ckp.privacy import PrivacyClass

REPORT_SCHEMA = "ckp-resilience-report/1"

#: A fixed clock, mirroring the C6 test suite's pattern, so every run of
#: this module is byte-for-byte reproducible except for the FastAPI/HTTP
#: plumbing timings we do not measure here.
NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)

_ACTOR_TOKEN = "resilience-actor-codex"
_OTHER_ACTOR_TOKEN = "resilience-actor-claude"
_GRANT_TOKEN = "resilience-grant-finance"
_BUNDLE_COMMIT = "b" * 40

_FINANCE_ID = "00000000-0000-7000-8000-0000000000a1"
_CALENDAR_ID = "00000000-0000-7000-8000-0000000000a2"
_STALE_NOTE_ID = "00000000-0000-7000-8000-0000000000b1"


@dataclass(frozen=True)
class ScenarioOutcome:
    """One resilience dimension's verdict plus the evidence behind it."""

    scenario: str
    passed: bool
    assertion: str
    observed: dict


# --------------------------------------------------------------------------
# Shared synthetic fixtures (deliberately not imported from tests/ or from
# benchmarks/questions.py -- this module owns its own tiny fixtures so it
# never collides with #22/#24/#26/#29's files).
# --------------------------------------------------------------------------


def _write_note(
    root: Path, relative_path: str, *, privacy: str, note_id: str, body: str
) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(
            [
                "---",
                f"privacy: {privacy}",
                "title: Resilience fixture",
                "type: Concept",
                f"id: {note_id}",
                "---",
                "",
                body,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _grant(
    *,
    domains: frozenset[KnowledgeDomain] = frozenset({KnowledgeDomain.FINANCE}),
    expires_at: datetime = NOW + timedelta(hours=1),
) -> ScopeGrant:
    return ScopeGrant(
        grant_id="grant-resilience-finance",
        actor=AccessActor.CODEX,
        task_id="task-resilience-finance",
        domains=domains,
        capabilities=frozenset({Capability.READ, Capability.REPORT_GENERATION}),
        privacy_classes=frozenset({PrivacyClass.PUBLIC, PrivacyClass.INTERNAL}),
        issued_by="human:cyclone",
        issued_at=NOW - timedelta(minutes=1),
        expires_at=expires_at,
        max_items=5,
        max_token_upper_bound=2_048,
    )


def _headers(*, actor: str = _ACTOR_TOKEN, grant: str = _GRANT_TOKEN) -> dict[str, str]:
    return {"X-CKP-Actor-Credential": actor, "X-CKP-Grant-Credential": grant}


def _registry(*, grant: ScopeGrant | None = None) -> CredentialRegistry:
    return CredentialRegistry(
        policy=GrantPolicy(),
        actor_credentials=(
            ActorCredential.from_plaintext(_ACTOR_TOKEN, AccessActor.CODEX),
            ActorCredential.from_plaintext(_OTHER_ACTOR_TOKEN, AccessActor.CLAUDE_CODE),
        ),
        grant_credentials=(
            GrantCredential.from_plaintext(_GRANT_TOKEN, grant or _grant()),
        ),
        clock=lambda: NOW,
    )


# --------------------------------------------------------------------------
# Scenario 1: Gateway down.
# --------------------------------------------------------------------------


#: A response body carrying any of these keys has the shape of a normal
#: listing/answer payload (``CatalogResponse``, ``QueryResponse``, ...),
#: never of this repo's rejection receipt (``request_id``/``status``/
#: ``code``, or FastAPI's bare ``detail``). A 5xx wearing this shape would
#: be "zero results" dressed up as an explicit error -- the same disguise
#: this scenario exists to catch, just pointed the other direction.
_LISTING_RESPONSE_KEYS = frozenset({"total", "items", "results", "revision"})


def _looks_like_listing_schema(body: object) -> bool:
    return isinstance(body, dict) and bool(_LISTING_RESPONSE_KEYS & body.keys())


def _is_explicit_error(response) -> bool:
    """True only for a real failure signal, never a disguised empty success.

    A ``>=500`` status is required -- the caller cannot mistake it for a
    valid answer. Two more checks, both belt-and-suspenders against the same
    failure mode from opposite directions: a response is never accepted as
    an "explicit error" if its body claims ``status == "ok"`` (a 5xx that
    still says everything is fine), nor if its body has the shape of a
    normal listing/answer payload -- e.g. ``{"total": 0, "items": []}`` --
    under a 5xx status, because a caller that only skims for "did I get an
    empty list" would read that as "zero results", not as "the source is
    down".
    """
    if response.status_code < 500:
        return False
    try:
        body = response.json()
    except ValueError:
        return True
    if isinstance(body, dict) and body.get("status") == "ok":
        return False
    return not _looks_like_listing_schema(body)


def _error_evidence(response) -> dict:
    try:
        body = response.json()
    except ValueError:
        body = None
    return {
        "status_code": response.status_code,
        "is_explicit_error": _is_explicit_error(response),
        "body_keys": sorted(body.keys()) if isinstance(body, dict) else None,
    }


def probe_gateway_down() -> ScenarioOutcome:
    """The bundle root the Gateway reads from is entirely unreachable.

    This is a stronger condition than "the commit stamp file is missing"
    (which C6 already treats as a public-but-unattributed corpus, see
    ``tests/test_scoped_gateway.py::test_scoped_read_requires_bundle_commit_but_public_c3_remains_available``)
    -- here the whole directory the bundle is supposed to live in was never
    created, i.e. the backing store itself is down.
    """
    with tempfile.TemporaryDirectory() as tmp:
        missing_root = Path(tmp) / "never-created"
        config = load_config(env={"CKP_BUNDLE_ROOT": str(missing_root)})
        client = TestClient(
            create_app(
                config,
                credential_registry=_registry(),
                domain_registry=DomainRegistry(()),
            )
        )

        health = client.get("/health")
        anonymous_catalog = client.get("/catalog")
        anonymous_query = client.post("/query", json={"query": "resilience-needle"})
        scoped_catalog = client.get("/scoped/catalog/finance", headers=_headers())
        scoped_query = client.post(
            "/scoped/query/finance",
            json={"query": "resilience-needle"},
            headers=_headers(),
        )
        context = client.post(
            "/context/finance",
            json={"query": "resilience-needle"},
            headers=_headers(),
        )

        surfaces = {
            "anonymous_catalog": anonymous_catalog,
            "anonymous_query": anonymous_query,
            "scoped_catalog": scoped_catalog,
            "scoped_query": scoped_query,
            "context": context,
        }
        observed = {name: _error_evidence(resp) for name, resp in surfaces.items()}
        observed["health"] = {
            "status_code": health.status_code,
            "status": health.json().get("status"),
        }

        health_reports_down = (
            health.status_code == 503 and health.json().get("status") == "degraded"
        )
        every_surface_explicit = all(
            entry["is_explicit_error"]
            for entry in observed.values()
            if "is_explicit_error" in entry
        )

        return ScenarioOutcome(
            scenario="gateway_down",
            passed=health_reports_down and every_surface_explicit,
            assertion=(
                "when the bundle root is unreachable, /health reports "
                "status=degraded with 503, and every read surface (public "
                "and scoped) answers with an explicit >=500 rejection -- "
                "none returns 200 with an empty-looking success payload"
            ),
            observed=observed,
        )


# --------------------------------------------------------------------------
# Scenario 2: stale snapshot.
# --------------------------------------------------------------------------


#: Production now has a real revision-staleness comparison (#39):
#: ``GET /revision`` in ``ckp/app.py`` compares ``SnapshotCache.peek()``
#: against a fresh recomputation and reports the mismatch as
#: ``"stale": true``. Naming the literal here means a caller reading the
#: report sees the same fixed string every time, and the mutation test can
#: pin its exact value rather than "any non-empty string that isn't the gap
#: case".
PRODUCTION_STALE_DETECTION_STATUS = "implemented"


def check_stale_snapshot() -> ScenarioOutcome:
    """Drive the real ``/revision`` endpoint through an edit-on-disk sequence.

    This is deliberately end to end: a real bundle directory, a real
    ``create_app`` instance, real HTTP calls through ``TestClient`` -- no
    benchmark-only comparator standing in for production. The sequence:

    1. First call ever made to this app: nothing was served before it, so
       ``SnapshotCache.peek()`` is ``None`` and ``stale`` must read
       ``False`` -- "nothing served yet" is not evidence of drift, and must
       never be misread as "stale" the way an unreadable live bundle would
       be (AGENTS.md §8 governs that distinct case, not this one).
    2. A second call with nothing changed on disk: negative control, must
       stay ``False``.
    3. The bundle is edited on disk. A third call must read ``True`` --
       the snapshot the second call served no longer matches a fresh
       recomputation.
    4. A fourth call, still after the edit: ``/revision`` always
       re-verifies before answering (contract: ``test_app.py::
       test_revision_is_recomputed_per_request``), so this call already
       observed and healed the drift while producing call 3's answer --
       the flag must have dropped back to ``False``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "bundle"
        root.mkdir(parents=True)
        _write_note(
            root,
            "resilience-note.md",
            privacy="public",
            note_id=_STALE_NOTE_ID,
            body="alpha content before the edit",
        )
        config = load_config(env={"CKP_BUNDLE_ROOT": str(root)})
        client = TestClient(create_app(config))

        first_call = client.get("/revision").json()
        unedited_call = client.get("/revision").json()

        _write_note(
            root,
            "resilience-note.md",
            privacy="public",
            note_id=_STALE_NOTE_ID,
            body="alpha content after the edit",
        )
        after_edit_call = client.get("/revision").json()
        healed_call = client.get("/revision").json()

    passed = (
        first_call["stale"] is False
        and unedited_call["stale"] is False
        and after_edit_call["stale"] is True
        and healed_call["stale"] is False
        and after_edit_call["index_revision"] != unedited_call["index_revision"]
    )

    return ScenarioOutcome(
        scenario="stale_snapshot",
        passed=passed,
        assertion=(
            "GET /revision flags stale=true exactly on the first call after "
            "the bundle changed underneath a previously-served snapshot, "
            "and stale=false before the edit, immediately after it heals on "
            "the next call, and on the very first call an app ever serves "
            "-- exercised against the real FastAPI app (#39), not a "
            "benchmark-only comparator"
        ),
        observed={
            "first_call_stale": first_call["stale"],
            "unedited_call_stale": unedited_call["stale"],
            "after_edit_call_stale": after_edit_call["stale"],
            "healed_call_stale": healed_call["stale"],
            "index_revision_before_edit": unedited_call["index_revision"],
            "index_revision_after_edit": after_edit_call["index_revision"],
            "production_detection": PRODUCTION_STALE_DETECTION_STATUS,
        },
    )


# --------------------------------------------------------------------------
# Scenario 3: MacBook offline (OfflineQmdScope fail-closed).
# --------------------------------------------------------------------------


def _denied_with(action: Callable[[], object], expected_code: str) -> bool:
    """True only when ``action`` raises :class:`AccessDenied` with exactly
    ``expected_code`` -- a wrong code or no exception at all is a failure,
    never silently treated as "close enough"."""
    try:
        action()
    except AccessDenied as error:
        return error.code == expected_code
    return False


def check_offline_scope_fail_closed() -> ScenarioOutcome:
    """``OfflineQmdScope``'s permission projection must fail closed offline.

    Wrong actor, wrong domain, and an expired grant must all be denied, and
    the emitted allowlist must never contain a path outside the granted
    domain -- "disconnected" must never widen into "no standing bypass".
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "bundle"
        root.mkdir(parents=True)
        (root / ".bundle-commit").write_text(_BUNDLE_COMMIT + "\n", encoding="utf-8")
        _write_note(
            root,
            "finance-in-scope.md",
            privacy="public",
            note_id=_FINANCE_ID,
            body="scope-needle finance in-scope body",
        )
        _write_note(
            root,
            "calendar-out-of-scope.md",
            privacy="public",
            note_id=_CALENDAR_ID,
            body="scope-needle calendar out-of-scope body",
        )
        domains = DomainRegistry(
            (
                DomainBinding(_FINANCE_ID, KnowledgeDomain.FINANCE),
                DomainBinding(_CALENDAR_ID, KnowledgeDomain.CALENDAR_CONTEXT),
            )
        )
        config = load_config(env={"CKP_BUNDLE_ROOT": str(root)})
        app = create_app(
            config, credential_registry=_registry(), domain_registry=domains
        )
        offline = app.state.offline_qmd_scope

        plan = offline.plan(
            actor_credential=_ACTOR_TOKEN,
            grant_credential=_GRANT_TOKEN,
            domain=KnowledgeDomain.FINANCE,
        )
        in_scope_only = plan.paths == ("finance-in-scope.md",)

        domain_outside_scope_denied = _denied_with(
            lambda: offline.plan(
                actor_credential=_ACTOR_TOKEN,
                grant_credential=_GRANT_TOKEN,
                domain=KnowledgeDomain.CALENDAR_CONTEXT,
            ),
            "domain-denied",
        )
        wrong_actor_denied = _denied_with(
            lambda: offline.plan(
                actor_credential=_OTHER_ACTOR_TOKEN,
                grant_credential=_GRANT_TOKEN,
                domain=KnowledgeDomain.FINANCE,
            ),
            "wrong-actor",
        )

        expired_app = create_app(
            load_config(env={"CKP_BUNDLE_ROOT": str(root)}),
            credential_registry=_registry(grant=_grant(expires_at=NOW)),
            domain_registry=domains,
        )
        expired_grant_denied = _denied_with(
            lambda: expired_app.state.offline_qmd_scope.plan(
                actor_credential=_ACTOR_TOKEN,
                grant_credential=_GRANT_TOKEN,
                domain=KnowledgeDomain.FINANCE,
            ),
            "scope-expired",
        )

        passed = (
            in_scope_only
            and domain_outside_scope_denied
            and wrong_actor_denied
            and expired_grant_denied
        )

        return ScenarioOutcome(
            scenario="macbook_offline",
            passed=passed,
            assertion=(
                "OfflineQmdScope.plan() emits only paths inside the granted "
                "domain, and raises AccessDenied (never a narrowed or "
                "silently-widened plan) for a domain outside the grant, a "
                "mismatched actor, or an expired grant"
            ),
            observed={
                "in_scope_paths": list(plan.paths),
                "domain_outside_scope_denied": domain_outside_scope_denied,
                "wrong_actor_denied": wrong_actor_denied,
                "expired_grant_denied": expired_grant_denied,
            },
        )


# --------------------------------------------------------------------------
# Report assembly and standalone entry point.
# --------------------------------------------------------------------------


def run_resilience_benchmark() -> dict:
    """Run all three scenarios and assemble one report, schema-versioned
    like ``benchmarks/shadow.py``'s report so both can sit side by side in
    the same Phase 4 report bundle.

    ``all_passed`` is a plain ``all(...)`` over every scenario's ``passed``
    field. As of #39, ``stale_snapshot.passed`` is a real measurement of
    ``GET /revision`` (see :func:`check_stale_snapshot`), not a hardcoded
    value, so ``all_passed`` can now legitimately reach ``True``.
    """
    outcomes = (
        probe_gateway_down(),
        check_stale_snapshot(),
        check_offline_scope_fail_closed(),
    )
    scenarios = {
        outcome.scenario: {
            "passed": outcome.passed,
            "assertion": outcome.assertion,
            "observed": outcome.observed,
        }
        for outcome in outcomes
    }
    return {
        "schema": REPORT_SCHEMA,
        "scenarios": scenarios,
        "all_passed": all(outcome.passed for outcome in outcomes),
    }


def _main() -> int:
    report = run_resilience_benchmark()
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "PRODUCTION_STALE_DETECTION_STATUS",
    "REPORT_SCHEMA",
    "ScenarioOutcome",
    "check_offline_scope_fail_closed",
    "check_stale_snapshot",
    "probe_gateway_down",
    "run_resilience_benchmark",
]
