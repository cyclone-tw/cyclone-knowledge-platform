"""P6 resilience benchmark dimensions (contract §Phase 4, issue #27).

Three §5.3 degradation scenarios that Phase 3's shadow harness never
measured -- Gateway down, stale snapshot, and MacBook (disconnected) offline
-- turned into standalone, independently runnable observations. Each one
answers a yes/no question with evidence, never a "probably falls back":

* **Gateway down** (:func:`probe_gateway_down`): the bundle root the
  Gateway reads from is entirely unreachable. Every read surface -- the
  anonymous C3 routes and the authenticated C6 routes -- must answer with an
  explicit >=500 error, never a 200 whose body quietly looks like "the
  corpus has zero notes".
* **Stale snapshot** (:func:`check_stale_snapshot`): a vector index built
  from one bundle content is still being served after the bundle changed
  underneath it. Contract §5.5 names "Dashboard shows a revision mismatch
  without flagging it stale" as a rollback trigger; this dimension is the
  benchmark-side comparison that makes the mismatch observable at all --
  nothing in ``src/ckp`` computes it today; see the module docstring's
  "Findings" note below.
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

Findings from building this benchmark (reported, not fixed here):

* No module in ``src/ckp`` currently compares a served/composed index
  revision against a fresh recomputation from the live bundle and flags a
  mismatch as stale. :func:`_is_snapshot_stale` is this benchmark's own
  comparison, not a production code path. Contract §5.5 lists "revision
  mismatch shown without a stale flag" as a rollback trigger for Phase 5;
  today there is no production stale flag to fail. Recommend a follow-up
  issue once P9 (#30) needs a real stale gate rather than a benchmark-only
  one.
"""

from __future__ import annotations

import hashlib
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
from ckp.bundle import BundleMember, compute_index_revision_from_members
from ckp.config import load_config
from ckp.embedding.composition import build_c4_deterministic_stack
from ckp.index.memory import InMemoryVectorIndex
from ckp.index.models import plan_rebuild
from ckp.privacy import FrontmatterClassifier, PrivacyClass, PrivacyGate

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


def _member(relative_path: str, body: str) -> BundleMember:
    content = "\n".join(
        ["---", "privacy: public", "title: Resilience fixture", "---", "", body, ""]
    ).encode("utf-8")
    return BundleMember(
        relative_path=relative_path,
        digest_key=relative_path,
        content=content,
        content_sha256=hashlib.sha256(content).hexdigest(),
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


def _is_explicit_error(response) -> bool:
    """True only for a real failure signal, never a disguised empty success.

    A ``>=500`` status is required -- the caller cannot mistake it for a
    valid answer -- and, belt and suspenders, a response is never accepted
    as an "explicit error" if its body claims ``status == "ok"``: "zero
    items" and "the source is unreachable" must never look alike to a
    caller that only checks the status field.
    """
    if response.status_code < 500:
        return False
    try:
        body = response.json()
    except ValueError:
        return True
    return not (isinstance(body, dict) and body.get("status") == "ok")


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
        context = client.post(
            "/context/finance",
            json={"query": "resilience-needle"},
            headers=_headers(),
        )

        surfaces = {
            "anonymous_catalog": anonymous_catalog,
            "anonymous_query": anonymous_query,
            "scoped_catalog": scoped_catalog,
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


def _is_snapshot_stale(
    served_bundle_index_revision: str, live_bundle_index_revision: str | None
) -> bool:
    """The one comparison this benchmark treats as the stale signal.

    AGENTS.md §8: a derived revision that cannot be recomputed from its
    commit is itself a rollback trigger, so an unreadable live bundle
    (``None``) counts as stale rather than as "cannot tell" -- the caller
    must never read "unknown" as "current".
    """
    if live_bundle_index_revision is None:
        return True
    return served_bundle_index_revision != live_bundle_index_revision


def check_stale_snapshot() -> ScenarioOutcome:
    """A vector index built at one bundle content, served after an edit.

    ``plan_rebuild`` folds the bundle's content into
    ``plan.bundle_index_revision`` at build time (``src/ckp/index/models.py``).
    Nothing in this repo re-checks that binding against the *current* bundle
    on every serve -- this scenario builds that missing comparison as a
    benchmark-only dimension (see the module docstring's Findings note) and
    asserts it actually distinguishes "unedited" from "edited-since-build".
    """
    stack = build_c4_deterministic_stack(
        dimension=8,
        embedding_name="resilience-hash",
        reranker_name="resilience-cosine",
    )
    # ``classify_member`` never touches the filesystem (it classifies bytes
    # already captured in a BundleMember), so the classifier's bundle_root
    # is unused for this in-memory scenario.
    classifier = FrontmatterClassifier(Path("unused-by-classify-member"))
    gate = PrivacyGate(classifier, frozenset({PrivacyClass.PUBLIC}))

    built_members = (_member("resilience-note.md", "alpha content before the edit"),)
    plan = plan_rebuild(members=built_members, stack=stack, gate=gate)
    provider = InMemoryVectorIndex()
    provider.rebuild(plan)

    # Negative control: nothing changed -- must NOT be flagged stale.
    unedited_live_revision = compute_index_revision_from_members(built_members)
    flagged_when_unedited = _is_snapshot_stale(
        plan.bundle_index_revision, unedited_live_revision
    )

    # Positive case: the Wiki changed after the index was last built and
    # nobody rebuilt it -- the realistic §5.3 disconnected/stale scenario.
    edited_members = (_member("resilience-note.md", "alpha content after the edit"),)
    edited_live_revision = compute_index_revision_from_members(edited_members)
    flagged_after_edit = _is_snapshot_stale(
        plan.bundle_index_revision, edited_live_revision
    )

    passed = (flagged_when_unedited is False) and (flagged_after_edit is True)

    return ScenarioOutcome(
        scenario="stale_snapshot",
        passed=passed,
        assertion=(
            "bundle_index_revision recorded at index-build time must "
            "compare equal to a fresh recomputation while the bundle has "
            "not moved, and must be flagged stale the moment it diverges "
            "from that recomputation"
        ),
        observed={
            "served_composed_revision": plan.composed_revision,
            "served_bundle_index_revision": plan.bundle_index_revision,
            "live_bundle_index_revision_unedited": unedited_live_revision,
            "live_bundle_index_revision_after_edit": edited_live_revision,
            "flagged_stale_when_unedited": flagged_when_unedited,
            "flagged_stale_after_edit": flagged_after_edit,
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
    the same Phase 4 report bundle."""
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
    "REPORT_SCHEMA",
    "ScenarioOutcome",
    "check_offline_scope_fail_closed",
    "check_stale_snapshot",
    "probe_gateway_down",
    "run_resilience_benchmark",
]
