"""P6 resilience benchmark (issue #27): Gateway down, stale snapshot, and
MacBook offline as observable, independently asserted dimensions.

Every scenario function under test is exercised twice: once through the
full scenario (real FastAPI app / real ``plan_rebuild`` / real
``OfflineQmdScope``) and once through the small pure evaluator it is built
on (``_is_explicit_error``, ``_is_snapshot_stale``, ``_denied_with``). The
evaluator-level tests are what catches the three regressions the issue
calls out explicitly:

* stale snapshot not flagged as stale,
* Gateway down answered with an empty 200 instead of an explicit error,
* an offline scope check that lets an out-of-scope request through.

Each of those three is reproduced here as a direct "would this evaluator
have caught it" case, and was also hand-verified red/green against a
temporarily mutated copy of ``benchmarks/resilience.py`` during development
(see the coding-session report for the mutation log); the mutation itself
is not committed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from benchmarks.resilience import (
    REPORT_SCHEMA,
    ScenarioOutcome,
    _denied_with,
    _is_explicit_error,
    _is_snapshot_stale,
    check_offline_scope_fail_closed,
    check_stale_snapshot,
    probe_gateway_down,
    run_resilience_benchmark,
)

from ckp.auth import AccessDenied


class _FakeResponse:
    """A minimal stand-in for ``httpx.Response`` (status_code + json())."""

    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> object:
        if self._body is _NO_BODY:
            raise ValueError("no JSON body")
        return self._body


_NO_BODY = object()


# --------------------------------------------------------------------------
# _is_explicit_error: the Gateway-down evaluator.
# --------------------------------------------------------------------------


def test_explicit_error_requires_5xx_status() -> None:
    assert _is_explicit_error(_FakeResponse(503, {"status": "rejected"})) is True
    assert (
        _is_explicit_error(_FakeResponse(500, {"detail": "bundle-unavailable"})) is True
    )


def test_explicit_error_rejects_200_even_with_empty_body() -> None:
    """The literal regression named in the issue: a 200 with an
    empty-looking body must never count as an explicit error."""
    assert _is_explicit_error(_FakeResponse(200, {"total": 0, "items": []})) is False


def test_explicit_error_rejects_5xx_that_still_claims_ok_status() -> None:
    """Belt-and-suspenders: even a 5xx must not be trusted if its own body
    claims success -- a caller checking the body's status field alone must
    not be fooled either."""
    assert _is_explicit_error(_FakeResponse(503, {"status": "ok"})) is False


def test_explicit_error_accepts_5xx_with_no_json_body() -> None:
    assert _is_explicit_error(_FakeResponse(503, _NO_BODY)) is True


# --------------------------------------------------------------------------
# _is_snapshot_stale: the stale-snapshot evaluator.
# --------------------------------------------------------------------------


def test_snapshot_stale_when_revisions_diverge() -> None:
    assert _is_snapshot_stale("sha256:aaa", "sha256:bbb") is True


def test_snapshot_not_stale_when_revisions_match() -> None:
    assert _is_snapshot_stale("sha256:aaa", "sha256:aaa") is False


def test_snapshot_stale_when_live_revision_unreadable() -> None:
    """An unrecomputable live revision must read as stale, not as
    "cannot tell" -- AGENTS.md §8's honest-metadata rule."""
    assert _is_snapshot_stale("sha256:aaa", None) is True


# --------------------------------------------------------------------------
# _denied_with: the offline fail-closed evaluator.
# --------------------------------------------------------------------------


def test_denied_with_true_only_for_the_expected_code() -> None:
    def _raise_expected() -> None:
        raise AccessDenied("domain-denied")

    assert _denied_with(_raise_expected, "domain-denied") is True


def test_denied_with_false_for_a_different_code() -> None:
    def _raise_other() -> None:
        raise AccessDenied("wrong-actor")

    assert _denied_with(_raise_other, "domain-denied") is False


def test_denied_with_false_when_the_check_lets_the_request_through() -> None:
    """The literal regression named in the issue: an offline scope check
    that stops denying (raises nothing) must read as "not fail-closed",
    never as a pass by default."""

    def _permissive_no_op() -> str:
        return "silently allowed"

    assert _denied_with(_permissive_no_op, "domain-denied") is False


# --------------------------------------------------------------------------
# The three scenarios end to end.
# --------------------------------------------------------------------------


def test_gateway_down_fails_closed_on_every_surface() -> None:
    outcome = probe_gateway_down()

    assert isinstance(outcome, ScenarioOutcome)
    assert outcome.scenario == "gateway_down"
    assert outcome.passed is True
    assert outcome.observed["health"]["status_code"] == 503
    assert outcome.observed["health"]["status"] == "degraded"
    for name in (
        "anonymous_catalog",
        "anonymous_query",
        "scoped_catalog",
        "context",
    ):
        entry = outcome.observed[name]
        assert entry["status_code"] >= 500
        assert entry["is_explicit_error"] is True


def test_stale_snapshot_distinguishes_unedited_from_edited() -> None:
    outcome = check_stale_snapshot()

    assert outcome.scenario == "stale_snapshot"
    assert outcome.passed is True
    assert outcome.observed["flagged_stale_when_unedited"] is False
    assert outcome.observed["flagged_stale_after_edit"] is True
    assert (
        outcome.observed["live_bundle_index_revision_unedited"]
        == outcome.observed["served_bundle_index_revision"]
    )
    assert (
        outcome.observed["live_bundle_index_revision_after_edit"]
        != outcome.observed["served_bundle_index_revision"]
    )


def test_offline_scope_stays_fail_closed() -> None:
    outcome = check_offline_scope_fail_closed()

    assert outcome.scenario == "macbook_offline"
    assert outcome.passed is True
    assert outcome.observed["in_scope_paths"] == ["finance-in-scope.md"]
    assert outcome.observed["domain_outside_scope_denied"] is True
    assert outcome.observed["wrong_actor_denied"] is True
    assert outcome.observed["expired_grant_denied"] is True


# --------------------------------------------------------------------------
# The assembled report.
# --------------------------------------------------------------------------


def test_run_resilience_benchmark_reports_all_three_dimensions() -> None:
    report = run_resilience_benchmark()

    assert report["schema"] == REPORT_SCHEMA
    assert report["all_passed"] is True
    assert set(report["scenarios"]) == {
        "gateway_down",
        "stale_snapshot",
        "macbook_offline",
    }
    for entry in report["scenarios"].values():
        assert entry["passed"] is True
        assert isinstance(entry["assertion"], str) and entry["assertion"]


def test_report_is_json_serializable_including_timestamps() -> None:
    import json

    report = run_resilience_benchmark()
    # default=str covers any stray datetime; nothing here should need it,
    # but the module's own _main() relies on that fallback -- pin it directly
    # rather than only through the CLI path.
    serialized = json.dumps(report, default=str)
    assert json.loads(serialized)["all_passed"] is True


def test_now_constant_is_timezone_aware() -> None:
    from benchmarks.resilience import NOW

    assert NOW.tzinfo is UTC
    assert datetime(2026, 8, 7, 12, 0, tzinfo=UTC) == NOW


@pytest.mark.parametrize(
    "scenario_fn",
    [probe_gateway_down, check_stale_snapshot, check_offline_scope_fail_closed],
)
def test_every_scenario_is_independently_callable(scenario_fn) -> None:
    """Each scenario is its own function with no shared mutable state --
    calling one twice, or calling them out of order, must not change the
    outcome (guards against accidental cross-scenario coupling)."""
    first = scenario_fn()
    second = scenario_fn()
    assert first.passed == second.passed is True
