"""P8 (issue #29): current export vs generated Catalog.

Round 2 (Codex changes-requested): "structural absence vs stale absence"
must be data the report carries, not prose in a docstring. This file tests
three layers:

* pure-function tests for :func:`zone_eligibility`, :func:`freshness_verdict`,
  and :func:`diff_export_against_catalog` -- no filesystem beyond a JSON
  file, no privacy gate, no pilot binding, no real git;
* a :class:`SubprocessGitOps` wiring test against a real temporary git
  repository -- the actual subprocess code path, not a fake standing in
  for it (AGENTS.md §9: "純函式測得好不等於接線有測");
* an end-to-end test for :func:`compare_export_to_catalog` against a
  synthetic pilot corpus (a ``tmp_path`` wiki root plus an injected frozen
  manifest, same pattern as ``tests/test_pilot_no_vendoring.py``).

Never touches the real Cyclone-Wiki checkout: every note here is synthetic
text written under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import pytest
from benchmarks.export_compare import (
    EXPORT_PATH_ENV,
    ExportEntry,
    ExportUnavailable,
    NoteFrontmatterFacts,
    NullGitOps,
    SubprocessGitOps,
    compare_export_to_catalog,
    default_export_path,
    diff_export_against_catalog,
    freshness_verdict,
    load_current_export,
    resolve_export_path,
    zone_eligibility,
)

from ckp.catalog.models import CatalogEntry, CatalogSnapshot, Citation
from ckp.config import load_config
from ckp.pilot.manifest import PILOT_NOTE_PATHS, PilotManifest, PilotManifestEntry

# --- helpers -----------------------------------------------------------


def _catalog_entry(
    path: str, *, title: str = "Title", status: str = "active"
) -> CatalogEntry:
    concept_id = path[:-3]
    return CatalogEntry(
        concept_id=concept_id,
        id=None,
        title=title,
        type="Procedure",
        description=None,
        status=status,
        content_category="meta",
        topic_id=None,
        module_id=None,
        tags=None,
        citation=Citation(
            concept_id=concept_id,
            path=path,
            index_revision="sha256:test",
            bundle_commit=None,
        ),
        body="body text never leaves this synthetic fixture",
    )


def _full_catalog(**overrides: dict[str, str]) -> CatalogSnapshot:
    entries = tuple(
        _catalog_entry(path, **overrides.get(path, {})) for path in PILOT_NOTE_PATHS
    )
    return CatalogSnapshot(
        entries=entries, index_revision="sha256:test", bundle_commit=None
    )


def _facts(
    path: str,
    *,
    status: str | None = "active",
    development_candidate: bool = False,
    library_id: str | None = None,
    privacy: str | None = "internal",
) -> NoteFrontmatterFacts:
    return NoteFrontmatterFacts(
        path=path,
        status=status,
        development_candidate=development_candidate,
        library_id=library_id,
        privacy=privacy,
    )


def _facts_all(**per_path_overrides: dict) -> dict[str, NoteFrontmatterFacts]:
    """Facts for all six pilot paths, all zone-ineligible by default (no
    project- filename, not under Core/_inbox/, no library_id) -- every
    unmatched path defaults to `reason: structural` unless overridden."""
    return {
        path: _facts(path, **per_path_overrides.get(path, {}))
        for path in PILOT_NOTE_PATHS
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _note(
    *, note_type: str = "Procedure", title: str = "Title", status: str = "active"
) -> str:
    return (
        "---\n"
        f"type: {note_type}\n"
        "privacy: internal\n"
        f"title: {title}\n"
        f"status: {status}\n"
        "---\n\n# note\n\nbody\n"
    )


def _write_synthetic_pilot_corpus(
    root, *, title: str = "Title", status: str = "active"
):
    """Write all six pilot notes under ``root`` and return a matching frozen
    manifest, mirroring ``tests/test_pilot_no_vendoring.py``'s pattern."""
    texts = {path: _note(title=title, status=status) for path in PILOT_NOTE_PATHS}
    for path, text in texts.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    manifest = PilotManifest(
        entries=tuple(
            PilotManifestEntry(relative_path=path, content_sha256=_sha256(texts[path]))
            for path in PILOT_NOTE_PATHS
        )
    )
    return manifest


def _write_export(
    path, zones: dict, *, schema: str = "wiki-export.v1", commit: str = "abc1234"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": schema,
        "exported_at": "2026-08-01T00:00:00Z",
        "commit": commit,
        "zones": zones,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class _FakeGitOps:
    """Deterministic stand-in for git, for pure classification tests.

    ``resolve`` defaults to identity (``resolve_commit(x) == x``) so every
    pre-round-4 fake-based test keeps working unchanged: those tests use
    arbitrary distinct strings ("oldcommit", "newcommit", ...), and
    resolving each string to itself is a no-op for them. Tests that
    specifically exercise resolution -- abbreviated-vs-full identity,
    ambiguity, unknown refs -- pass an explicit ``resolve`` mapping.
    """

    def __init__(
        self,
        *,
        last_touch: dict[str, str] | None = None,
        ancestry: dict[tuple[str, str], bool] | None = None,
        resolve: dict[str, str | None] | None = None,
    ) -> None:
        self._last_touch = last_touch or {}
        self._ancestry = ancestry or {}
        self._resolve = resolve

    def last_touch_commit(self, relative_path: str) -> str | None:
        return self._last_touch.get(relative_path)

    def is_ancestor(self, ancestor: str, descendant: str) -> bool | None:
        return self._ancestry.get((ancestor, descendant))

    def resolve_commit(self, ref: str) -> str | None:
        if self._resolve is None:
            return ref
        return self._resolve.get(ref)


def _git(*args: str, cwd) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


# --- resolve_export_path -------------------------------------------------


def test_default_export_path_matches_dashboard_wiki_adapter_default(tmp_path) -> None:
    assert default_export_path(tmp_path) == tmp_path / ".cache" / "wiki-export.v1.json"


def test_resolve_export_path_uses_env_override(tmp_path, monkeypatch) -> None:
    override = tmp_path / "elsewhere.json"
    monkeypatch.setenv(EXPORT_PATH_ENV, str(override))
    assert resolve_export_path(tmp_path) == override


def test_resolve_export_path_falls_back_to_default_when_env_unset(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv(EXPORT_PATH_ENV, raising=False)
    assert resolve_export_path(tmp_path) == default_export_path(tmp_path)


# --- load_current_export: never silently "empty" --------------------------


def test_missing_export_file_raises_export_unavailable(tmp_path) -> None:
    with pytest.raises(ExportUnavailable, match="unreadable"):
        load_current_export(tmp_path / "does-not-exist.json")


def test_invalid_json_raises_export_unavailable(tmp_path) -> None:
    bad = tmp_path / "wiki-export.v1.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ExportUnavailable, match="not valid JSON"):
        load_current_export(bad)


def test_wrong_schema_raises_export_unavailable(tmp_path) -> None:
    bad = tmp_path / "wiki-export.v1.json"
    _write_export(bad, zones={}, schema="something-else.v9")
    with pytest.raises(ExportUnavailable, match="schema"):
        load_current_export(bad)


def test_missing_zones_object_raises_export_unavailable(tmp_path) -> None:
    bad = tmp_path / "wiki-export.v1.json"
    bad.write_text(
        json.dumps({"schema": "wiki-export.v1", "exported_at": "x", "commit": "y"}),
        encoding="utf-8",
    )
    with pytest.raises(ExportUnavailable, match="zones"):
        load_current_export(bad)


def test_load_current_export_only_reads_path_addressable_zones(tmp_path) -> None:
    """agent_activity / shared_now carry no per-item path and must be
    ignored, not crash and not silently included as unmatched noise."""
    export_file = tmp_path / "wiki-export.v1.json"
    _write_export(
        export_file,
        zones={
            "shared_now": {"preferences": ["should be ignored"]},
            "agent_activity": [{"agent_id": "X", "active_focus": ["ignored"]}],
            "projects": [
                {
                    "path": "Core/project-cyclone-okf-knowledge-contract.md",
                    "title": "T",
                    "status": "active",
                }
            ],
        },
    )
    result = load_current_export(export_file)
    assert len(result.entries) == 1
    assert result.entries[0].zone == "projects"
    assert result.entries[0].path == "Core/project-cyclone-okf-knowledge-contract.md"


def test_load_current_export_ignores_items_without_a_path(tmp_path) -> None:
    export_file = tmp_path / "wiki-export.v1.json"
    _write_export(export_file, zones={"projects": [{"title": "no path here"}]})
    result = load_current_export(export_file)
    assert result.entries == ()


# --- zone_eligibility: wiki_dashboard_export.py's rules, as data -----------


def test_procedure_and_decision_notes_are_zone_ineligible() -> None:
    """Neither PILOT_NOTE_PATHS procedure/decision note matches any zone's
    path shape at all -- this is the exact structural gap Round 1 found."""
    for path in (
        "Core/procedure-agent-wiki-note-retrieval.md",
        "Core/procedure-agent-memory-read-scopes.md",
        "Core/decision-cyclone-wiki-openwiki-role-boundary.md",
    ):
        assert zone_eligibility(_facts(path, development_candidate=False)) == ()


def test_project_note_with_active_status_is_projects_eligible() -> None:
    path = "Core/project-cyclone-okf-knowledge-contract.md"
    assert zone_eligibility(_facts(path, status="active")) == ("projects",)


def test_project_note_without_active_status_is_zone_ineligible() -> None:
    path = "Core/project-cyclone-okf-knowledge-contract.md"
    assert zone_eligibility(_facts(path, status="parked")) == ()


def test_agent_capture_with_development_candidate_flag_is_eligible() -> None:
    path = "Core/_inbox/agent-captures/2026-07-04-relayapi-repo-analysis.md"
    assert zone_eligibility(_facts(path, development_candidate=True)) == (
        "development_candidates",
    )


def test_agent_capture_without_development_candidate_flag_is_ineligible() -> None:
    path = "Core/_inbox/agent-captures/2026-07-04-relayapi-repo-analysis.md"
    assert zone_eligibility(_facts(path, development_candidate=False)) == ()


def test_note_with_library_id_and_allowed_privacy_is_topics_eligible() -> None:
    path = "Core/some-course-note.md"
    assert zone_eligibility(
        _facts(path, library_id="2026-lib", privacy="internal")
    ) == ("topics",)


def test_knowledge_feed_and_life_domains_shapes() -> None:
    assert zone_eligibility(
        _facts("Private/_inbox/info-collect/2026-08-01-note.md")
    ) == ("knowledge_feed",)
    assert zone_eligibility(
        _facts("Private/Life/profile/reports/2026-08-01-report.md")
    ) == ("life_domains",)


def test_note_can_be_eligible_for_more_than_one_zone() -> None:
    """A project note that also carries a library_id is legitimately
    eligible for both projects and topics -- multi-zone is not a bug."""
    path = "Core/project-x.md"
    facts = _facts(path, status="active", library_id="lib-1", privacy="internal")
    assert set(zone_eligibility(facts)) == {"projects", "topics"}


# --- freshness_verdict: never guesses ---------------------------------


def test_freshness_unknown_when_export_commit_missing() -> None:
    assert freshness_verdict(_FakeGitOps(), None, "Core/x.md") == "unknown"


def test_freshness_unknown_when_git_has_no_last_touch_commit() -> None:
    git_ops = _FakeGitOps(last_touch={})
    assert freshness_verdict(git_ops, "abc123", "Core/x.md") == "unknown"


def test_freshness_current_when_last_touch_equals_export_commit() -> None:
    git_ops = _FakeGitOps(last_touch={"Core/x.md": "abc123"})
    assert freshness_verdict(git_ops, "abc123", "Core/x.md") == "current"


def test_freshness_stale_when_export_commit_is_ancestor_of_last_touch() -> None:
    git_ops = _FakeGitOps(
        last_touch={"Core/x.md": "newcommit"},
        ancestry={
            ("oldcommit", "newcommit"): True,
            ("newcommit", "oldcommit"): False,  # reverse direction, checked too
        },
    )
    assert freshness_verdict(git_ops, "oldcommit", "Core/x.md") == "stale"


def test_freshness_current_when_export_commit_is_not_an_ancestor() -> None:
    """The note's last-touch commit already predates (or equals a sibling
    of) the export commit -- the export is not stale relative to it. Both
    directions must be supplied (round 3): a single `False` alone is not
    enough to conclude "current"."""
    git_ops = _FakeGitOps(
        last_touch={"Core/x.md": "oldcommit"},
        ancestry={
            ("newcommit", "oldcommit"): False,
            ("oldcommit", "newcommit"): True,  # reverse direction, checked too
        },
    )
    assert freshness_verdict(git_ops, "newcommit", "Core/x.md") == "current"


def test_freshness_unknown_when_ancestry_is_ambiguous() -> None:
    git_ops = _FakeGitOps(last_touch={"Core/x.md": "newcommit"}, ancestry={})
    assert freshness_verdict(git_ops, "oldcommit", "Core/x.md") == "unknown"


def test_null_git_ops_always_reports_unknown() -> None:
    assert freshness_verdict(NullGitOps(), "abc123", "Core/x.md") == "unknown"


# --- SubprocessGitOps: real git wiring, not a fake standing in for it -----


def test_subprocess_git_ops_against_a_real_temporary_repo(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    (repo / "Core").mkdir()
    (repo / "Core" / "a.md").write_text("v1\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "first", cwd=repo)
    first_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    (repo / "Core" / "a.md").write_text("v2\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "second", cwd=repo)
    second_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    git_ops = SubprocessGitOps(wiki_root=repo)
    assert git_ops.last_touch_commit("Core/a.md") == second_commit
    assert git_ops.is_ancestor(first_commit, second_commit) is True
    assert git_ops.is_ancestor(second_commit, first_commit) is False
    assert git_ops.last_touch_commit("Core/does-not-exist.md") is None
    assert git_ops.is_ancestor("not-a-real-commit", second_commit) is None

    assert freshness_verdict(git_ops, first_commit, "Core/a.md") == "stale"
    assert freshness_verdict(git_ops, second_commit, "Core/a.md") == "current"


def test_resolve_commit_default_is_identity_for_fake_git_ops() -> None:
    assert _FakeGitOps().resolve_commit("anything") == "anything"


def test_freshness_unknown_when_export_commit_does_not_resolve() -> None:
    """An export commit that resolve_commit cannot find (bad ref, or
    ambiguous short SHA -- git itself detects ambiguity, this module never
    picks a candidate) must be unknown, not fall through to a string
    comparison or an ancestry check with garbage input."""
    git_ops = _FakeGitOps(
        last_touch={"Core/x.md": "full-commit"},
        resolve={"full-commit": "full-commit"},  # export_commit not in this map -> None
    )
    assert freshness_verdict(git_ops, "unresolvable-ref", "Core/x.md") == "unknown"


def test_freshness_unknown_when_last_touch_does_not_resolve() -> None:
    git_ops = _FakeGitOps(
        last_touch={"Core/x.md": "unresolvable-touch"},
        resolve={
            "export-commit": "export-commit"
        },  # last_touch not in this map -> None
    )
    assert freshness_verdict(git_ops, "export-commit", "Core/x.md") == "unknown"


def test_freshness_current_when_abbreviated_and_full_sha_resolve_to_same_commit() -> (
    None
):
    """The exact round 4 bug, reproduced with a fake: export_commit is an
    abbreviated form and last_touch is the full form of the *same* commit
    -- the two strings never match directly, but both resolve to one
    canonical form."""
    git_ops = _FakeGitOps(
        last_touch={"Core/x.md": "abc1234full567890"},
        resolve={
            "abc1234": "abc1234full567890",  # export_commit, abbreviated
            "abc1234full567890": "abc1234full567890",  # last_touch, already full
        },
    )
    assert freshness_verdict(git_ops, "abc1234", "Core/x.md") == "current"


def test_subprocess_git_ops_diverged_history_is_unknown_not_current(tmp_path) -> None:
    """Round 3 blocking finding: a single ``is_ancestor(export, note)``
    call returning False used to be read as "current". Real diverged
    history (a "人造分歧歷史", per the round 3 feedback's own suggested
    alternative to a shallow clone) proves that reading wrong: neither
    commit is an ancestor of the other here, so the correct answer is
    "unknown", not "current"."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    (repo / "Core").mkdir()
    (repo / "Core" / "a.md").write_text("base\n", encoding="utf-8")
    (repo / "Core" / "b.md").write_text("base\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)

    # Branch "feature": the only branch that touches Core/a.md again.
    _git("checkout", "-q", "-b", "feature", cwd=repo)
    (repo / "Core" / "a.md").write_text("feature change\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "feature touches a.md", cwd=repo)
    feature_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # Back on the original branch, an unrelated commit that never touches
    # Core/a.md -- diverges from "feature" at the shared "base" commit;
    # neither is an ancestor of the other.
    _git("checkout", "-q", "-", cwd=repo)  # back to the branch before "feature"
    (repo / "Core" / "b.md").write_text("trunk change\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "trunk touches b.md", cwd=repo)
    trunk_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # last_touch_commit must see feature_commit for Core/a.md, so HEAD has
    # to be on "feature" when queried -- git objects are content-addressed
    # and reachable by hash regardless of what is checked out, but `git
    # log -1 -- <path>` without an explicit revision walks from HEAD.
    _git("checkout", "-q", "feature", cwd=repo)
    git_ops = SubprocessGitOps(wiki_root=repo)

    assert git_ops.last_touch_commit("Core/a.md") == feature_commit
    assert git_ops.is_ancestor(trunk_commit, feature_commit) is False
    assert git_ops.is_ancestor(feature_commit, trunk_commit) is False

    assert freshness_verdict(git_ops, trunk_commit, "Core/a.md") == "unknown"


def test_freshness_unknown_when_both_ancestry_directions_report_true() -> None:
    """Impossible in a real acyclic git history for two distinct commits,
    but the function must still refuse to guess rather than trust it."""
    git_ops = _FakeGitOps(
        last_touch={"Core/x.md": "commit-b"},
        ancestry={("commit-a", "commit-b"): True, ("commit-b", "commit-a"): True},
    )
    assert freshness_verdict(git_ops, "commit-a", "Core/x.md") == "unknown"


def test_freshness_unknown_when_one_direction_is_none_even_if_other_is_false() -> None:
    """A prior, single-direction check would have read `False` alone as
    "current" -- must stay unknown when the other direction cannot be
    determined at all."""
    git_ops = _FakeGitOps(
        last_touch={"Core/x.md": "commit-b"},
        ancestry={("commit-a", "commit-b"): False},  # reverse direction absent -> None
    )
    assert freshness_verdict(git_ops, "commit-a", "Core/x.md") == "unknown"


def test_subprocess_git_ops_resolves_abbreviated_and_full_sha_to_the_same_commit(
    tmp_path,
) -> None:
    """Round 4 blocking finding, reproduced against real git, the same way
    the diverged-history test above reproduces round 3's: this is exactly
    the shape of the bug Codex caught -- the export producer
    (wiki_dashboard_export.py) writes `git rev-parse --short HEAD` while
    this module's own last_touch_commit() always returns the full SHA."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)

    (repo / "Core").mkdir()
    (repo / "Core" / "a.md").write_text("v1\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "only commit", cwd=repo)

    full_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    short_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert short_commit != full_commit  # the whole premise of the bug

    git_ops = SubprocessGitOps(wiki_root=repo)
    assert git_ops.resolve_commit(full_commit) == full_commit
    assert git_ops.resolve_commit(short_commit) == full_commit
    assert git_ops.last_touch_commit("Core/a.md") == full_commit

    # export_commit is the *abbreviated* form -- exactly what
    # wiki_dashboard_export.py writes -- for the very commit that last
    # touched Core/a.md. Before the round 4 fix this fell through the
    # string-equality shortcut, hit is_ancestor(X, X) both ways (True,
    # True -- a commit is trivially its own ancestor), and was refused
    # into "unknown". It must be "current".
    assert freshness_verdict(git_ops, short_commit, "Core/a.md") == "current"


def test_subprocess_git_ops_resolve_commit_unknown_ref_is_none(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "a.md").write_text("v1\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "only commit", cwd=repo)

    git_ops = SubprocessGitOps(wiki_root=repo)
    assert git_ops.resolve_commit("not-a-real-commit") is None
    assert git_ops.resolve_commit("deadbeef") is None


def test_subprocess_git_ops_on_a_non_git_directory_is_unknown_not_a_crash(
    tmp_path,
) -> None:
    git_ops = SubprocessGitOps(wiki_root=tmp_path)
    assert git_ops.last_touch_commit("Core/a.md") is None
    assert git_ops.is_ancestor("a", "b") is None
    assert git_ops.resolve_commit("a") is None


# --- diff_export_against_catalog: pure classification ----------------------


def test_all_six_match_cleanly_produces_no_diffs() -> None:
    catalog = _full_catalog()
    entries = tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in PILOT_NOTE_PATHS
    )
    diffs, summary = diff_export_against_catalog(
        entries,
        catalog,
        facts_by_path=_facts_all(),
        export_commit="abc1234",
        git_ops=NullGitOps(),
    )
    assert diffs == []
    assert summary["matched_clean"] == 6
    assert summary["missing_from_export"] == 0


def test_structural_absence_and_freshness_absence_are_different_categories() -> None:
    """The exact Round 1 finding, made into a test: a procedure note (zone-
    ineligible) and a project note (zone-eligible, export stale) must land
    in different reasons even though both are simply absent from export."""
    catalog = _full_catalog()
    procedure_path = "Core/procedure-agent-wiki-note-retrieval.md"
    project_path = "Core/project-cyclone-okf-knowledge-contract.md"

    facts = _facts_all()
    facts[procedure_path] = _facts(procedure_path, development_candidate=False)
    facts[project_path] = _facts(project_path, status="active")

    git_ops = _FakeGitOps(
        last_touch={project_path: "newcommit"},
        ancestry={
            ("export-commit", "newcommit"): True,
            ("newcommit", "export-commit"): False,
        },
    )

    diffs, summary = diff_export_against_catalog(
        (),  # nothing in the export at all
        catalog,
        facts_by_path=facts,
        export_commit="export-commit",
        git_ops=git_ops,
    )

    missing = {d["path"]: d for d in diffs if d["category"] == "missing_from_export"}
    assert missing[procedure_path]["reason"] == "structural"
    assert missing[project_path]["reason"] == "freshness"
    assert summary["missing_from_export_structural"] >= 1
    assert summary["missing_from_export_freshness"] == 1
    assert (
        summary["missing_from_export_structural"]
        + summary["missing_from_export_freshness"]
        <= summary["missing_from_export"]
    )


def test_missing_reason_is_unknown_when_git_signal_is_ambiguous() -> None:
    project_path = "Core/project-cyclone-okf-knowledge-contract.md"
    facts = {project_path: _facts(project_path, status="active")}
    git_ops = _FakeGitOps()  # no last-touch data at all

    diffs, summary = diff_export_against_catalog(
        (),
        CatalogSnapshot(
            entries=(_catalog_entry(project_path),),
            index_revision="sha256:test",
            bundle_commit=None,
        ),
        facts_by_path=facts,
        export_commit="export-commit",
        git_ops=git_ops,
    )
    missing = next(d for d in diffs if d["category"] == "missing_from_export")
    assert missing["reason"] == "unknown"
    assert summary["missing_from_export_unknown"] == 1


def test_missing_reason_structural_ignores_freshness_even_when_commit_present() -> None:
    """A structurally-ineligible note must never fall into freshness/unknown
    just because export_commit happens to be set -- eligibility is checked
    first and short-circuits."""
    catalog = _full_catalog()
    procedure_path = PILOT_NOTE_PATHS[0]
    git_ops = _FakeGitOps(
        last_touch={procedure_path: "some-commit"},
        ancestry={("export-commit", "some-commit"): True},
    )
    diffs, _ = diff_export_against_catalog(
        (),
        catalog,
        facts_by_path=_facts_all(),
        export_commit="export-commit",
        git_ops=git_ops,
    )
    missing = next(d for d in diffs if d["path"] == procedure_path)
    assert missing["reason"] == "structural"


def test_path_absent_from_export_is_missing_from_export() -> None:
    catalog = _full_catalog()
    entries = tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in PILOT_NOTE_PATHS[1:]
    )
    diffs, summary = diff_export_against_catalog(
        entries,
        catalog,
        facts_by_path=_facts_all(),
        export_commit="abc1234",
        git_ops=NullGitOps(),
    )
    assert summary["missing_from_export"] == 1
    missing = [d for d in diffs if d["category"] == "missing_from_export"]
    assert missing[0]["path"] == PILOT_NOTE_PATHS[0]


def test_out_of_scope_export_entry_is_silently_excluded_not_reported() -> None:
    """Scope is the pilot manifest -- an export path outside it is out of
    scope, same reasoning D3 applies to the QMD baseline (different scope
    is not a comparison)."""
    catalog = _full_catalog()
    entries = tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in PILOT_NOTE_PATHS
    ) + (
        ExportEntry(
            zone="projects", path="Core/some-other-note.md", title="X", status=None
        ),
    )
    diffs, summary = diff_export_against_catalog(
        entries,
        catalog,
        facts_by_path=_facts_all(),
        export_commit="abc1234",
        git_ops=NullGitOps(),
    )
    assert diffs == []
    assert summary["matched_clean"] == 6


def test_duplicate_zone_entries_are_extra_in_export_not_merged_into_missing() -> None:
    catalog = _full_catalog()
    dup_path = PILOT_NOTE_PATHS[0]
    entries = tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in PILOT_NOTE_PATHS
    ) + (
        ExportEntry(
            zone="knowledge_feed", path=dup_path, title="Title", status="active"
        ),
    )
    diffs, summary = diff_export_against_catalog(
        entries,
        catalog,
        facts_by_path=_facts_all(),
        export_commit="abc1234",
        git_ops=NullGitOps(),
    )
    assert summary["extra_in_export"] == 1
    assert summary["missing_from_export"] == 0
    extra = [d for d in diffs if d["category"] == "extra_in_export"]
    assert extra[0]["path"] == dup_path
    # dup_path is a procedure note: not a "project-*.md" filename, so
    # neither "projects" (the primary/first-seen zone) nor "knowledge_feed"
    # (the duplicate) is a zone this note is eligible for -- round 3 fix:
    # *both* representations are checked, including the primary one, not
    # just the ones beyond the first.
    assert extra[0]["zones"] == ["projects", "knowledge_feed"]
    assert extra[0]["reasons"] == ["zone_not_eligible", "zone_not_eligible"]


def test_duplicate_zone_entry_in_a_legitimately_eligible_zone_is_labeled_as_such() -> (
    None
):
    dup_path = "Core/project-x.md"
    catalog = CatalogSnapshot(
        entries=(_catalog_entry(dup_path),)
        + tuple(_catalog_entry(p) for p in PILOT_NOTE_PATHS[1:]),
        index_revision="sha256:test",
        bundle_commit=None,
    )
    facts = _facts_all()
    facts[dup_path] = _facts(dup_path, status="active", library_id="lib-1")
    pilot_paths_patched = (dup_path,) + PILOT_NOTE_PATHS[1:]
    entries = tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in pilot_paths_patched
    ) + (ExportEntry(zone="topics", path=dup_path, title="Title", status=None),)

    import benchmarks.export_compare as export_compare_module

    original = export_compare_module.PILOT_NOTE_PATHS
    export_compare_module.PILOT_NOTE_PATHS = pilot_paths_patched
    try:
        diffs, _ = diff_export_against_catalog(
            entries,
            catalog,
            facts_by_path=facts,
            export_commit="abc1234",
            git_ops=NullGitOps(),
        )
    finally:
        export_compare_module.PILOT_NOTE_PATHS = original

    extra = next(d for d in diffs if d["category"] == "extra_in_export")
    assert extra["zones"] == ["projects", "topics"]
    assert extra["reasons"] == [
        "duplicate_in_eligible_zone",
        "duplicate_in_eligible_zone",
    ]


def test_extra_in_export_flags_an_ineligible_primary_zone_not_just_the_duplicates() -> (
    None
):
    """Round 3, non-blocking finding #1: a prior version only validated
    matches[1:], so an ineligible *primary* (first-seen) representation
    went unflagged. Here the first-seen zone (knowledge_feed) is the
    illegitimate one and the second (projects) is legitimate -- the fix
    must flag the first, not just the second."""
    dup_path = "Core/project-y.md"
    catalog = CatalogSnapshot(
        entries=(_catalog_entry(dup_path),)
        + tuple(_catalog_entry(p) for p in PILOT_NOTE_PATHS[1:]),
        index_revision="sha256:test",
        bundle_commit=None,
    )
    facts = _facts_all()
    facts[dup_path] = _facts(dup_path, status="active", library_id="lib-2")
    pilot_paths_patched = (dup_path,) + PILOT_NOTE_PATHS[1:]
    entries = (
        ExportEntry(zone="knowledge_feed", path=dup_path, title="Title", status=None),
    ) + tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in pilot_paths_patched
    )

    import benchmarks.export_compare as export_compare_module

    original = export_compare_module.PILOT_NOTE_PATHS
    export_compare_module.PILOT_NOTE_PATHS = pilot_paths_patched
    try:
        diffs, _ = diff_export_against_catalog(
            entries,
            catalog,
            facts_by_path=facts,
            export_commit="abc1234",
            git_ops=NullGitOps(),
        )
    finally:
        export_compare_module.PILOT_NOTE_PATHS = original

    extra = next(d for d in diffs if d["category"] == "extra_in_export")
    assert extra["zones"] == ["knowledge_feed", "projects"]
    assert extra["reasons"] == ["zone_not_eligible", "duplicate_in_eligible_zone"]


def test_title_mismatch_is_metadata_mismatch_with_field_name_only() -> None:
    catalog = _full_catalog()
    mismatched_path = PILOT_NOTE_PATHS[0]
    entries = tuple(
        ExportEntry(
            zone="projects",
            path=path,
            title=(
                "A Completely Different Title" if path == mismatched_path else "Title"
            ),
            status="active",
        )
        for path in PILOT_NOTE_PATHS
    )
    diffs, summary = diff_export_against_catalog(
        entries,
        catalog,
        facts_by_path=_facts_all(),
        export_commit="abc1234",
        git_ops=NullGitOps(),
    )
    assert summary["metadata_mismatch"] == 1
    mismatch = next(d for d in diffs if d["category"] == "metadata_mismatch")
    assert mismatch["path"] == mismatched_path
    assert mismatch["fields"] == ["title"]
    # Privacy boundary: only the field *name* is recorded, never the value.
    serialized = json.dumps(diffs)
    assert "A Completely Different Title" not in serialized
    assert serialized.count("title") >= 1  # only the field *name* survives


def test_status_mismatch_is_reported_by_field_name() -> None:
    catalog = _full_catalog()
    mismatched_path = PILOT_NOTE_PATHS[1]
    entries = tuple(
        ExportEntry(
            zone="projects",
            path=path,
            title="Title",
            status=("deprecated" if path == mismatched_path else "active"),
        )
        for path in PILOT_NOTE_PATHS
    )
    diffs, _ = diff_export_against_catalog(
        entries,
        catalog,
        facts_by_path=_facts_all(),
        export_commit="abc1234",
        git_ops=NullGitOps(),
    )
    mismatch = next(d for d in diffs if d["category"] == "metadata_mismatch")
    assert mismatch["fields"] == ["status"]
    assert "deprecated" not in json.dumps(diffs)


def test_export_field_none_is_not_compared_not_a_mismatch() -> None:
    """An export zone that never carries `status` (e.g. knowledge_feed) must
    not manufacture a mismatch against Catalog's status."""
    catalog = _full_catalog()
    entries = tuple(
        ExportEntry(zone="knowledge_feed", path=path, title="Title", status=None)
        for path in PILOT_NOTE_PATHS
    )
    diffs, summary = diff_export_against_catalog(
        entries,
        catalog,
        facts_by_path=_facts_all(),
        export_commit="abc1234",
        git_ops=NullGitOps(),
    )
    assert diffs == []
    assert summary["matched_clean"] == 6


def test_path_mismatch_when_raw_path_differs_but_normalizes_the_same() -> None:
    catalog = _full_catalog()
    odd_path = PILOT_NOTE_PATHS[0]
    entries = (
        ExportEntry(
            zone="projects", path="./" + odd_path, title="Title", status="active"
        ),
    ) + tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in PILOT_NOTE_PATHS[1:]
    )
    diffs, summary = diff_export_against_catalog(
        entries,
        catalog,
        facts_by_path=_facts_all(),
        export_commit="abc1234",
        git_ops=NullGitOps(),
    )
    assert summary["path_mismatch"] == 1
    mismatch = next(d for d in diffs if d["category"] == "path_mismatch")
    assert mismatch["path"] == odd_path


def test_missing_from_catalog_defensive_branch() -> None:
    """Not a normal outcome (bind_pilot_corpus fails loudly first in the
    real wiring) -- exercised directly via a synthetic incomplete Catalog."""
    incomplete = CatalogSnapshot(
        entries=(_catalog_entry(PILOT_NOTE_PATHS[0]),),
        index_revision="sha256:test",
        bundle_commit=None,
    )
    entries = tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in PILOT_NOTE_PATHS
    )
    diffs, _ = diff_export_against_catalog(
        entries,
        incomplete,
        facts_by_path=_facts_all(),
        export_commit="abc1234",
        git_ops=NullGitOps(),
    )
    missing_catalog = [d for d in diffs if d["category"] == "missing_from_catalog"]
    assert len(missing_catalog) == len(PILOT_NOTE_PATHS) - 1


# --- summary categories must stay separate (acceptance: 分類, not 一坨) ----


def test_summary_has_all_categories_as_distinct_keys() -> None:
    catalog = _full_catalog()
    diffs, summary = diff_export_against_catalog(
        (),
        catalog,
        facts_by_path=_facts_all(),
        export_commit="abc1234",
        git_ops=NullGitOps(),
    )
    assert set(summary) == {
        "pilot_scope_count",
        "missing_from_export",
        "missing_from_export_structural",
        "missing_from_export_freshness",
        "missing_from_export_unknown",
        "extra_in_export",
        "metadata_mismatch",
        "metadata_mismatch_reliable",
        "metadata_mismatch_caveat",
        "path_mismatch",
        "matched_clean",
    }
    assert summary["missing_from_export"] == 6
    # No mismatches in this fixture -> nothing to flag as unreliable.
    assert summary["metadata_mismatch_reliable"] is True
    assert summary["metadata_mismatch_caveat"] is None


# --- compare_export_to_catalog: end-to-end wiring --------------------------


def test_compared_false_when_export_unavailable_has_no_diffs_key(tmp_path) -> None:
    """The single most important guard in this module: 'could not compare'
    must be structurally distinguishable from 'compared, found nothing'."""
    wiki_root = tmp_path / "wiki"
    manifest = _write_synthetic_pilot_corpus(wiki_root)
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(wiki_root)})

    report = compare_export_to_catalog(
        config,
        export_path=tmp_path / "no-such-export.json",
        frozen_manifest=manifest,
        git_ops=NullGitOps(),
    )

    assert report["compared"] is False
    assert "diffs" not in report
    assert "summary" not in report
    assert report["export"]["status"] == "unavailable"
    assert "reason" in report["export"]


def test_compared_true_end_to_end_against_synthetic_corpus(tmp_path) -> None:
    wiki_root = tmp_path / "wiki"
    manifest = _write_synthetic_pilot_corpus(wiki_root, title="Title", status="active")
    export_path = tmp_path / "wiki-export.v1.json"
    _write_export(
        export_path,
        zones={
            "projects": [
                {"path": path, "title": "Title", "status": "active"}
                for path in PILOT_NOTE_PATHS
            ]
        },
    )
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(wiki_root)})

    report = compare_export_to_catalog(
        config,
        export_path=export_path,
        frozen_manifest=manifest,
        git_ops=NullGitOps(),
    )

    assert report["compared"] is True
    assert report["export"]["status"] == "ok"
    assert report["diffs"] == []
    assert report["summary"]["matched_clean"] == 6
    assert report["catalog"]["entry_count"] == 6
    assert report["summary"]["metadata_mismatch_reliable"] is True
    assert report["summary"]["metadata_mismatch_caveat"] is None


def test_compared_true_with_metadata_mismatch_flags_report_as_unreliable(
    tmp_path,
) -> None:
    """Round 4: #48 established every metadata_mismatch this module can
    currently produce is a definitional false positive, not confirmed
    drift. A downstream reader of the *report* (not this module's
    docstring, #42's exact lesson) must see that unreliability as a field,
    not have to know to distrust the number."""
    wiki_root = tmp_path / "wiki"
    manifest = _write_synthetic_pilot_corpus(wiki_root, title="Title", status="active")
    export_path = tmp_path / "wiki-export.v1.json"
    entries = [
        {"path": path, "title": "Title", "status": "active"}
        for path in PILOT_NOTE_PATHS
    ]
    entries[0]["title"] = "A Totally Different Title"
    _write_export(export_path, zones={"projects": entries})
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(wiki_root)})

    report = compare_export_to_catalog(
        config,
        export_path=export_path,
        frozen_manifest=manifest,
        git_ops=NullGitOps(),
    )

    assert report["summary"]["metadata_mismatch"] == 1
    assert report["summary"]["metadata_mismatch_reliable"] is False
    assert "#48" in report["summary"]["metadata_mismatch_caveat"]


def test_compared_true_but_partial_coverage_reports_missing_with_reasons(
    tmp_path,
) -> None:
    """The realistic shape of the real wiki-export.v1.json today: it has no
    zone for Procedure/Decision notes at all, so most of the pilot corpus
    is structurally missing_from_export -- this must show up, split by
    reason, not merged into one undifferentiated bucket."""
    wiki_root = tmp_path / "wiki"
    manifest = _write_synthetic_pilot_corpus(wiki_root)
    export_path = tmp_path / "wiki-export.v1.json"
    covered = PILOT_NOTE_PATHS[-1]
    _write_export(
        export_path,
        zones={
            "development_candidates": [
                {"path": covered, "title": "Title", "status": "needs-validation"}
            ]
        },
    )
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(wiki_root)})

    # None of these synthetic notes set development_candidate: true, and
    # none are project-*.md files -- every unmatched one is structurally
    # ineligible for every zone, which NullGitOps cannot override (freshness
    # is never even consulted for a structurally-ineligible note).
    report = compare_export_to_catalog(
        config,
        export_path=export_path,
        frozen_manifest=manifest,
        git_ops=NullGitOps(),
    )

    assert report["compared"] is True
    missing = {
        d["path"]: d for d in report["diffs"] if d["category"] == "missing_from_export"
    }
    assert len(missing) == len(PILOT_NOTE_PATHS) - 1
    # Indices 0,1,2 (two procedures + one decision) and 4 (an agent-capture
    # whose synthetic body never sets development_candidate: true) are
    # structurally ineligible for every zone. Index 3 is
    # Core/project-cyclone-okf-knowledge-contract.md -- its filename and
    # `status: active` make it *eligible* for the projects zone, so its
    # absence is not structural; with NullGitOps supplying no freshness
    # signal at all, it must land in "unknown", never guessed as either of
    # the other two reasons.
    structural_paths = {PILOT_NOTE_PATHS[i] for i in (0, 1, 2, 4)}
    project_path = PILOT_NOTE_PATHS[3]
    assert {p: d["reason"] for p, d in missing.items() if p in structural_paths} == {
        p: "structural" for p in structural_paths
    }
    assert missing[project_path]["reason"] == "unknown"
    assert report["summary"]["missing_from_export_structural"] == len(structural_paths)
    assert report["summary"]["missing_from_export_unknown"] == 1


def test_result_is_reproducible_run_twice_same_inputs(tmp_path) -> None:
    wiki_root = tmp_path / "wiki"
    manifest = _write_synthetic_pilot_corpus(wiki_root)
    export_path = tmp_path / "wiki-export.v1.json"
    _write_export(
        export_path,
        zones={
            "projects": [
                {"path": path, "title": "Title", "status": "active"}
                for path in PILOT_NOTE_PATHS
            ]
        },
    )
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(wiki_root)})

    first = compare_export_to_catalog(
        config, export_path=export_path, frozen_manifest=manifest, git_ops=NullGitOps()
    )
    second = compare_export_to_catalog(
        config, export_path=export_path, frozen_manifest=manifest, git_ops=NullGitOps()
    )
    assert first == second


def test_pilot_binding_failure_propagates_not_swallowed(tmp_path) -> None:
    """No CKP_PILOT_WIKI_ROOT -> PilotBindingError, not an empty report."""
    from ckp.pilot.manifest import PilotBindingError

    config = load_config(env={})
    with pytest.raises(PilotBindingError, match="CKP_PILOT_WIKI_ROOT"):
        compare_export_to_catalog(config)


def test_compare_export_to_catalog_uses_real_subprocess_git_ops_by_default(
    tmp_path,
) -> None:
    """No git_ops override -> SubprocessGitOps against a real (non-git)
    tmp_path -- must degrade to "unknown", not crash."""
    wiki_root = tmp_path / "wiki"
    manifest = _write_synthetic_pilot_corpus(wiki_root)
    export_path = tmp_path / "wiki-export.v1.json"
    covered = PILOT_NOTE_PATHS[-1]
    _write_export(
        export_path,
        zones={
            "development_candidates": [
                {"path": covered, "title": "Title", "status": "needs-validation"}
            ]
        },
    )
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(wiki_root)})

    report = compare_export_to_catalog(
        config, export_path=export_path, frozen_manifest=manifest
    )
    assert report["compared"] is True
    # tmp_path is not a git repository, so SubprocessGitOps degrades to
    # "unknown" for every query -- this must not crash, and a
    # structurally-ineligible note must still report "structural" (freshness
    # is never even consulted for those), while the one zone-eligible note
    # (the project path, index 3) reports "unknown" rather than a guess.
    missing = {
        d["path"]: d for d in report["diffs"] if d["category"] == "missing_from_export"
    }
    structural_paths = {PILOT_NOTE_PATHS[i] for i in (0, 1, 2, 4)}
    assert {p: d["reason"] for p, d in missing.items() if p in structural_paths} == {
        p: "structural" for p in structural_paths
    }
    assert missing[PILOT_NOTE_PATHS[3]]["reason"] == "unknown"


# --- CLI entrypoint ----------------------------------------------------


def test_main_returns_1_when_pilot_wiki_root_unset(monkeypatch, capsys) -> None:
    from benchmarks import export_compare

    monkeypatch.delenv("CKP_PILOT_WIKI_ROOT", raising=False)
    monkeypatch.delenv("CKP_CONFIG_FILE", raising=False)
    exit_code = export_compare.main()
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "pilot corpus binding failed" in captured.err


def test_main_returns_2_when_export_unavailable(tmp_path, monkeypatch, capsys) -> None:
    from benchmarks import export_compare

    wiki_root = tmp_path / "wiki"
    manifest = _write_synthetic_pilot_corpus(wiki_root)
    manifest_file = tmp_path / "pilot-manifest.toml"
    manifest_file.write_text(
        "\n".join(
            f'[[note]]\nrelative_path = "{entry.relative_path}"\n'
            f'content_sha256 = "{entry.content_sha256}"'
            for entry in manifest.entries
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CKP_PILOT_WIKI_ROOT", str(wiki_root))
    monkeypatch.setenv("CKP_PILOT_MANIFEST_PATH", str(manifest_file))
    monkeypatch.setenv(EXPORT_PATH_ENV, str(tmp_path / "does-not-exist.json"))
    monkeypatch.delenv("CKP_CONFIG_FILE", raising=False)

    exit_code = export_compare.main()
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "current export unavailable" in captured.err
    payload = json.loads(captured.out)
    assert payload["compared"] is False
    assert "diffs" not in payload
