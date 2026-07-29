"""AGENTS.md is a governance artifact, so its substantive clauses are pinned.

Prose drifts silently. These assertions pin the clauses that other agents rely
on -- the review contract, the loop cap, the hard stops, and the Phase 3 red
lines -- so that removing one shows up as a red test rather than as a quiet
edit inside a docs commit.

Wording that is not pinned here is free to be rewritten.
"""

from __future__ import annotations

import pytest

from conftest import REPO_ROOT

AGENTS = REPO_ROOT / "AGENTS.md"

REQUIRED_SECTIONS = (
    "## 1. What this repo owns",
    "## 2. Issue-first",
    "## 3. Worktree isolation",
    "## 4. Review contract",
    "## 5. Hard stops",
    "## 6. Phase 3 red lines",
    "## 7. Portability",
    "## 8. Honest metadata",
    "## 9. Guard self-verification",
)

# (clause id, substring that must survive any rewrite)
REVIEW_CONTRACT_CLAUSES = (
    ("coder-not-reviewer", "Coder ≠ Reviewer"),
    ("codex-primary", "scripts/codex-review.sh"),
    ("verdict-line", "VERDICT: approved | nits-only | changes-requested"),
    ("loop-cap-2", "exceeded **2**"),
    ("no-fallback-tier", "no fallback reviewer tier"),
    ("codex-serial", "single-instance"),
    ("markers-review-status", "Review-Status:"),
    ("markers-reviewed-commit", "Reviewed-Commit:"),
    ("merge-without-asking", "do not ask permission to merge"),
)

HARD_STOPS = (
    ("secrets", "Secrets, tokens, or `.env`"),
    ("student-data", "Student-identifiable personal data"),
    ("force-push", "Force push"),
    ("prod-destructive", "Destructive production operations"),
    ("self-merge", "self-approving and auto-merging"),
)

# Contract §3 Phase 3. These are product invariants, not style.
RED_LINES = (
    ("privacy-before-enqueue", "Privacy routing completes before enqueue"),
    ("no-student-private-in-outbox", "encryption does not make it acceptable"),
    ("replay-before-freeze", "has not passed its replay test"),
    ("direct-formal-closed", "direct-formal templates stay closed"),
    ("manifest-frozen", "7278252"),
)


@pytest.fixture(scope="module")
def agents_text() -> str:
    assert AGENTS.is_file(), "AGENTS.md is missing"
    return AGENTS.read_text(encoding="utf-8")


@pytest.mark.parametrize("heading", REQUIRED_SECTIONS)
def test_required_sections_present(agents_text: str, heading: str) -> None:
    assert heading in agents_text, f"AGENTS.md lost section {heading!r}"


@pytest.mark.parametrize(("clause", "needle"), REVIEW_CONTRACT_CLAUSES)
def test_review_contract_clauses(agents_text: str, clause: str, needle: str) -> None:
    assert needle in agents_text, f"review contract clause {clause!r} was dropped"


@pytest.mark.parametrize(("clause", "needle"), HARD_STOPS)
def test_hard_stops(agents_text: str, clause: str, needle: str) -> None:
    assert needle in agents_text, f"hard stop {clause!r} was dropped"


@pytest.mark.parametrize(("clause", "needle"), RED_LINES)
def test_phase3_red_lines(agents_text: str, clause: str, needle: str) -> None:
    assert needle in agents_text, f"Phase 3 red line {clause!r} was dropped"


def test_worktree_convention_is_concrete(agents_text: str) -> None:
    """A convention nobody can follow from the text is not a convention."""
    assert "~/Cyclone-System/worktrees/ckp-<issue>-<slug>" in agents_text
    assert "<type>/<issue>-<slug>" in agents_text


def test_issue_first_names_the_pr_linkage(agents_text: str) -> None:
    for needle in ("Closes #NN", "Ref #NN"):
        assert needle in agents_text, f"issue linkage rule {needle!r} was dropped"


def test_readme_points_agents_at_the_rules() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "AGENTS.md" in readme
