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
    QmdUnavailable,
    _normalize_path,
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


def test_qmd_token_cost_skips_unreadable_paths_instead_of_raising(
    tmp_path: Path,
) -> None:
    cost = qmd_token_cost(("Core/does-not-exist.md",), wiki_root=tmp_path)
    assert cost == 0


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
