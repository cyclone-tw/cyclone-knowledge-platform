"""The real-QMD adapter (issue #24): named index, corpus scoping, fail-loud.

All tests here monkeypatch ``subprocess.run`` -- no real ``qmd`` process, no
real Wiki checkout -- except the gated block at the bottom, which is the
``CKP_REQUIRE_QMD_BASELINE`` counterpart to ``test_index_qdrant.py``'s
``CKP_REQUIRE_QDRANT`` pattern: skip locally when qmd/the wiki checkout are
not available, but let the flag turn that skip into a hard failure so this
path can never silently stop running everywhere. CI never sets the flag or
``CKP_PILOT_WIKI_ROOT`` -- AGENTS.md forbids wiring CI to the real, private
Wiki checkout.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from benchmarks.qmd.adapter import (
    QmdBaselineConfig,
    QmdConfigError,
    QmdUnavailable,
    _normalize_path,
    _qmd_visible_path,
    qmd_binary_available,
    qmd_token_cost,
    run_qmd_query,
)


def _config(tmp_path: Path, **overrides) -> QmdBaselineConfig:
    defaults = dict(
        index_name="cyclone-wiki",
        wiki_root=tmp_path,
        corpus_paths=frozenset({"Core/a.md", "Core/b.md"}),
        binary="qmd",
    )
    defaults.update(overrides)
    return QmdBaselineConfig(**defaults)


def _fake_completed(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["qmd"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# --- empty corpus_paths (Codex Round 1, #24 review) ---------------------


def test_empty_corpus_paths_is_rejected_at_construction(tmp_path: Path) -> None:
    """A caller error, not a legitimate 'compare against zero notes'.

    Left unchecked, this would make every QMD hit fail the scope filter,
    and ``benchmarks.shadow`` would report ``qmd_compared: True,
    qmd_hit_rate: 0.0`` -- indistinguishable from "QMD was compared and
    scored badly," when nothing was ever in scope.
    """
    with pytest.raises(QmdConfigError, match="empty"):
        QmdBaselineConfig(
            index_name="cyclone-wiki",
            wiki_root=tmp_path,
            corpus_paths=frozenset(),
        )


def test_empty_corpus_paths_raises_before_any_subprocess_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Timing is part of the contract: fail at construction, not at query
    time -- before the caller has spent any real QMD subprocess cost.
    """

    def fail_if_invoked(*args, **kwargs):
        raise AssertionError(
            "qmd must never be invoked for a config that fails validation"
        )

    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr("subprocess.run", fail_if_invoked)

    with pytest.raises(QmdConfigError):
        QmdBaselineConfig(
            index_name="cyclone-wiki", wiki_root=tmp_path, corpus_paths=frozenset()
        )
    # If construction had instead succeeded and only ``run_qmd_query``
    # checked, we would need to call it here to prove the check still
    # fires; the ``pytest.raises`` above already proves it never gets that
    # far, and ``fail_if_invoked`` above is a second line of defense in
    # case some future refactor moves the check past construction without
    # this test noticing via the ``raises`` alone.


# --- path normalization ------------------------------------------------


def test_normalize_path_strips_cwd_relative_prefix() -> None:
    assert _normalize_path("./Core/a.md") == "Core/a.md"


def test_normalize_path_extracts_from_qmd_uri() -> None:
    assert (
        _normalize_path("qmd://cyclone-wiki/Core/a.md?index=cyclone-wiki")
        == "Core/a.md"
    )


def test_normalize_path_leaves_unrecognized_shapes_unchanged() -> None:
    assert _normalize_path("/absolute/outside.md") == "/absolute/outside.md"


# --- binary / root preconditions ---------------------------------------


def test_qmd_binary_available_reflects_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda binary: None)
    assert qmd_binary_available("qmd") is False
    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    assert qmd_binary_available("qmd") is True


def test_run_qmd_query_raises_when_binary_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda binary: None)
    with pytest.raises(QmdUnavailable, match="not found on PATH"):
        run_qmd_query("q", config=_config(tmp_path), top_k=5)


def test_run_qmd_query_raises_when_wiki_root_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    missing = Path("/nonexistent-ckp-qmd-root-issue-24")
    with pytest.raises(QmdUnavailable, match="is not a directory"):
        run_qmd_query("q", config=_config(missing, wiki_root=missing), top_k=5)


# --- the happy path, and exactly what command it issues -----------------


def test_run_qmd_query_uses_named_index_and_the_configured_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_run(command, *, cwd, capture_output, text, timeout):
        captured["command"] = command
        captured["cwd"] = cwd
        return _fake_completed(json.dumps([]))

    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr("subprocess.run", fake_run)

    config = _config(tmp_path, index_name="cyclone-wiki")
    run_qmd_query("orbits module", config=config, top_k=5)

    assert captured["cwd"] == tmp_path
    assert "--index" in captured["command"]
    assert (
        captured["command"][captured["command"].index("--index") + 1] == "cyclone-wiki"
    )
    # D3: never the flagless default -- the invocation must always name it.
    assert "search" in captured["command"]
    assert "orbits module" in captured["command"]


def test_run_qmd_query_filters_results_to_the_corpus_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = [
        {"score": 0.9, "file": "./Core/a.md"},
        {"score": 0.8, "file": "./Core/outside-scope.md"},
        {"score": 0.7, "file": "qmd://cyclone-wiki/Core/b.md?index=cyclone-wiki"},
    ]

    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _fake_completed(json.dumps(payload))
    )

    config = _config(tmp_path, corpus_paths=frozenset({"Core/a.md", "Core/b.md"}))
    result = run_qmd_query("q", config=config, top_k=5)

    assert result.raw_paths == (
        "./Core/a.md",
        "./Core/outside-scope.md",
        "qmd://cyclone-wiki/Core/b.md?index=cyclone-wiki",
    )
    assert result.paths == ("Core/a.md", "Core/b.md")
    assert [hit.relative_path for hit in result.hits] == ["Core/a.md", "Core/b.md"]
    assert result.hits[0].score == 0.9


def test_run_qmd_query_deduplicates_repeated_normalized_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = [
        {"score": 0.9, "file": "./Core/a.md"},
        {"score": 0.5, "file": "qmd://cyclone-wiki/Core/a.md?index=cyclone-wiki"},
    ]
    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _fake_completed(json.dumps(payload))
    )
    config = _config(tmp_path, corpus_paths=frozenset({"Core/a.md"}))
    result = run_qmd_query("q", config=config, top_k=5)
    assert result.paths == ("Core/a.md",)
    assert len(result.hits) == 1


# --- qmd's underscore-stripped path spelling (issue #58) ----------------


def test_qmd_visible_path_strips_directory_underscores_only() -> None:
    """Directory stripping is observed qmd behavior; filename stripping is
    not observable (no ``_``-prefixed file exists in the wiki) and must NOT
    be assumed -- an alias wider than observed behavior can credit a
    different file's hit to the corpus (#61 review round 1).
    """
    assert (
        _qmd_visible_path("Core/_inbox/agent-captures/x.md")
        == "Core/inbox/agent-captures/x.md"
    )
    assert _qmd_visible_path("Core/a.md") == "Core/a.md"
    assert _qmd_visible_path("Core/_inbox/_note.md") == "Core/inbox/_note.md"
    assert _qmd_visible_path("_note.md") == "_note.md"


def test_run_qmd_query_never_credits_a_stripped_filename_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-1 blocking finding, regression-pinned through the real entry
    point: with corpus ``Core/_inbox/_note.md``, a hit spelled
    ``Core/inbox/note.md`` is qmd's directory-only rendering of the
    *different* file ``Core/_inbox/note.md`` and must stay filtered out;
    only the directory-stripped spelling ``Core/inbox/_note.md`` is the
    corpus note.
    """
    corpus_path = "Core/_inbox/_note.md"
    payload = [
        {
            "score": 0.9,
            "file": "qmd://cyclone-wiki/Core/inbox/note.md?index=cyclone-wiki",
        },
        {
            "score": 0.7,
            "file": "qmd://cyclone-wiki/Core/inbox/_note.md?index=cyclone-wiki",
        },
    ]
    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _fake_completed(json.dumps(payload))
    )
    config = _config(tmp_path, corpus_paths=frozenset({corpus_path}))
    result = run_qmd_query("q", config=config, top_k=5)
    assert result.paths == (corpus_path,)
    assert result.hits[0].score == 0.7


def test_run_qmd_query_scores_hits_qmd_returns_in_stripped_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #58 false-miss: qmd's own namespace drops the leading underscore
    from directory names, so a hit on ``Core/_inbox/...`` comes back spelled
    ``Core/inbox/...`` (in ``qmd://`` URI form -- ``--full-path`` cannot
    resolve the stripped spelling on disk). Before the mapping, that hit
    failed the ``corpus_paths`` membership test and was scored as a QMD
    miss even though qmd ranked the note first -- a false baseline in the
    platform's favor. Scored paths must come back in the *canonical*
    spelling: ``expected_paths`` and ``qmd_token_cost`` both read it.
    """
    corpus_path = "Core/_inbox/agent-captures/2026-07-04-x.md"
    payload = [
        {
            "score": 0.9,
            "file": (
                "qmd://cyclone-wiki/Core/inbox/agent-captures/2026-07-04-x.md"
                "?index=cyclone-wiki"
            ),
        },
    ]
    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _fake_completed(json.dumps(payload))
    )
    config = _config(tmp_path, corpus_paths=frozenset({corpus_path}))
    result = run_qmd_query("q", config=config, top_k=5)
    assert result.paths == (corpus_path,)
    assert result.hits[0].relative_path == corpus_path
    assert result.hits[0].score == 0.9


def test_run_qmd_query_does_not_credit_stripped_paths_outside_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENTS.md §9 Q2: a compliant-*looking* hit -- same ``Core/inbox/...``
    stripped spelling, but of a path the corpus never contained -- must
    still be filtered out, not waved through by an over-broad reverse map.
    """
    payload = [
        {
            "score": 0.9,
            "file": "qmd://cyclone-wiki/Core/inbox/other.md?index=cyclone-wiki",
        },
    ]
    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _fake_completed(json.dumps(payload))
    )
    config = _config(
        tmp_path, corpus_paths=frozenset({"Core/_inbox/agent-captures/x.md"})
    )
    result = run_qmd_query("q", config=config, top_k=5)
    assert result.paths == ()
    assert result.raw_paths == (payload[0]["file"],)


def test_run_qmd_query_deduplicates_canonical_and_stripped_spellings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = [
        {"score": 0.9, "file": "./Core/_inbox/a.md"},
        {"score": 0.5, "file": "qmd://cyclone-wiki/Core/inbox/a.md?index=cyclone-wiki"},
    ]
    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _fake_completed(json.dumps(payload))
    )
    config = _config(tmp_path, corpus_paths=frozenset({"Core/_inbox/a.md"}))
    result = run_qmd_query("q", config=config, top_k=5)
    assert result.paths == ("Core/_inbox/a.md",)
    assert len(result.hits) == 1
    assert result.hits[0].score == 0.9


def test_corpus_paths_indistinguishable_to_qmd_are_rejected_at_construction(
    tmp_path: Path,
) -> None:
    with pytest.raises(QmdConfigError, match="indistinguishable"):
        _config(
            tmp_path,
            corpus_paths=frozenset({"Core/_inbox/a.md", "Core/inbox/a.md"}),
        )


def test_run_qmd_query_fails_loud_when_stripped_spelling_also_exists_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real, non-corpus file at the stripped spelling makes qmd hits
    unattributable (both files render identically in qmd output), and that
    is a property ``__post_init__`` cannot see -- checked per query, before
    the subprocess ever runs.
    """
    (tmp_path / "Core" / "inbox").mkdir(parents=True)
    (tmp_path / "Core" / "inbox" / "a.md").write_text("decoy\n", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run must not be reached")

    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr("subprocess.run", fail_if_called)
    config = _config(tmp_path, corpus_paths=frozenset({"Core/_inbox/a.md"}))
    with pytest.raises(QmdUnavailable, match="cannot attribute"):
        run_qmd_query("q", config=config, top_k=5)


# --- fail-loud, never a partial/empty stand-in --------------------------


def test_run_qmd_query_raises_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: _fake_completed("", returncode=1, stderr="index not found"),
    )
    with pytest.raises(QmdUnavailable, match="index not found"):
        run_qmd_query("q", config=_config(tmp_path), top_k=5)


def test_run_qmd_query_raises_on_invalid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _fake_completed("not json"))
    with pytest.raises(QmdUnavailable, match="non-JSON"):
        run_qmd_query("q", config=_config(tmp_path), top_k=5)


def test_run_qmd_query_raises_on_non_list_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _fake_completed(json.dumps({"oops": True}))
    )
    with pytest.raises(QmdUnavailable, match="not a list"):
        run_qmd_query("q", config=_config(tmp_path), top_k=5)


def test_run_qmd_query_raises_on_hit_missing_file_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: _fake_completed(json.dumps([{"score": 1.0}])),
    )
    with pytest.raises(QmdUnavailable, match="missing 'file'"):
        run_qmd_query("q", config=_config(tmp_path), top_k=5)


def test_run_qmd_query_raises_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="qmd", timeout=1)

    monkeypatch.setattr("shutil.which", lambda binary: "/usr/local/bin/qmd")
    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(QmdUnavailable, match="qmd invocation failed"):
        run_qmd_query("q", config=_config(tmp_path), top_k=5)


# --- token cost (D1: relative-to-QMD threshold needs this number) -------


def test_qmd_token_cost_counts_body_whitespace_tokens(tmp_path: Path) -> None:
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    body = "alpha beta gamma delta epsilon"
    (note_dir / "note.md").write_text(
        f"---\nprivacy: internal\n---\n\n{body}\n", encoding="utf-8"
    )
    cost = qmd_token_cost(("Core/note.md",), wiki_root=tmp_path)
    assert cost == len(body.split()) == 5


def test_qmd_token_cost_strips_frontmatter_before_counting(tmp_path: Path) -> None:
    """The frontmatter block itself must not inflate the count.

    Pins the same "body tokens only" contract the synthetic corpus's
    ``CORPUS_BODIES`` already uses -- a mutant that counts the whole file
    (frontmatter included) would silently move this number without any
    other test noticing, since the frontmatter here is deliberately long
    enough to change the total if it leaked in.
    """
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    (note_dir / "note.md").write_text(
        "---\ntype: Procedure\nprivacy: internal\nrelated_to: []\n---\n\n"
        "one two three\n",
        encoding="utf-8",
    )
    cost = qmd_token_cost(("Core/note.md",), wiki_root=tmp_path)
    assert cost == 3


def test_qmd_token_cost_sums_across_multiple_paths(tmp_path: Path) -> None:
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    (note_dir / "a.md").write_text("---\nprivacy: internal\n---\n\none two\n")
    (note_dir / "b.md").write_text("---\nprivacy: internal\n---\n\nthree four five\n")
    cost = qmd_token_cost(("Core/a.md", "Core/b.md"), wiki_root=tmp_path)
    assert cost == 5


def test_qmd_token_cost_does_not_raise_on_an_unreadable_path(
    tmp_path: Path,
) -> None:
    """Superseded pin, updated in #40 round 1: the old skip-and-undercount
    contract returned a confident 0 here, which drifted from the platform
    sides' None semantics and fed D1 a fake baseline. Not raising is still
    the contract; the value is now honestly unmeasured."""
    cost = qmd_token_cost(("Core/does-not-exist.md",), wiki_root=tmp_path)
    assert cost is None


def test_qmd_token_cost_never_writes_any_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RP1: reading real note bytes to count them must never persist them.

    Spies on ``Path.write_text``/``Path.write_bytes`` for the duration of
    the call -- the strongest form of this guard, since it does not depend
    on scanning any particular directory afterward.
    """
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    (note_dir / "note.md").write_text(
        "---\nprivacy: internal\n---\n\nfoo bar baz\n", encoding="utf-8"
    )

    write_calls: list[Path] = []
    original_write_text = Path.write_text
    original_write_bytes = Path.write_bytes

    def spy_write_text(self: Path, *args, **kwargs):
        write_calls.append(self)
        return original_write_text(self, *args, **kwargs)

    def spy_write_bytes(self: Path, *args, **kwargs):
        write_calls.append(self)
        return original_write_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy_write_text)
    monkeypatch.setattr(Path, "write_bytes", spy_write_bytes)

    cost = qmd_token_cost(("Core/note.md",), wiki_root=tmp_path)

    assert cost == 3
    assert write_calls == []


# --- gated real-environment smoke (CKP_REQUIRE_QMD_BASELINE) ------------

_QMD_BINARY = os.environ.get("CKP_QMD_BINARY", "qmd")
_PILOT_ROOT = os.environ.get("CKP_PILOT_WIKI_ROOT")
_REQUIRED = os.environ.get("CKP_REQUIRE_QMD_BASELINE") == "1"


def _real_environment_ready() -> bool:
    return qmd_binary_available(_QMD_BINARY) and bool(_PILOT_ROOT)


_ready = _real_environment_ready()
if _REQUIRED and not _ready:
    pytest.fail(
        "CKP_REQUIRE_QMD_BASELINE=1 but qmd is not on PATH or "
        "CKP_PILOT_WIKI_ROOT is unset"
    )

needs_real_qmd = pytest.mark.skipif(
    not _ready, reason="qmd binary or CKP_PILOT_WIKI_ROOT not available locally"
)


@needs_real_qmd
def test_real_qmd_named_index_returns_hits_scoped_to_pilot_manifest() -> None:
    """Manual/local-only smoke check -- never runs in CI (no real wiki there).

    Exercises the actual acceptance scenario: named index, pilot manifest
    scope, real subprocess. Intentionally does not assert on note content,
    only on paths staying inside the pilot allowlist (RP1).
    """
    from ckp.pilot import PILOT_NOTE_PATHS

    config = QmdBaselineConfig(
        index_name="cyclone-wiki",
        wiki_root=Path(_PILOT_ROOT),
        corpus_paths=frozenset(PILOT_NOTE_PATHS),
        binary=_QMD_BINARY,
    )
    result = run_qmd_query("wiki note retrieval procedure", config=config, top_k=5)
    assert all(path in PILOT_NOTE_PATHS for path in result.paths)


def test_qmd_token_cost_is_none_when_any_path_is_unreadable(tmp_path):
    """Codex round 1 on #40: skip-and-undercount turned "could not read
    anything" into a confident 0 -- a fake baseline for D1's relative token
    threshold. One unreadable path makes the whole tuple unmeasured, same
    semantics as measure_token_cost."""
    root = tmp_path
    note = root / "Core" / "ok.md"
    note.parent.mkdir(parents=True)
    note.write_text("---\nprivacy: internal\n---\n\none two three\n", encoding="utf-8")
    assert qmd_token_cost(("Core/ok.md",), wiki_root=root) == 3
    assert qmd_token_cost(("Core/ok.md", "Core/missing.md"), wiki_root=root) is None
    assert qmd_token_cost(("Core/missing.md",), wiki_root=root) is None
