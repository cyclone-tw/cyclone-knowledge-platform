"""The shared whitespace-proxy token-cost helper (issue #40).

``benchmarks.token_cost.measure_token_cost`` is the single implementation
both ``benchmarks.shadow`` (lexical/vector) and ``benchmarks.qmd.adapter``
(QMD) now call, so the three sides of the shadow benchmark cannot drift
apart on how a token is counted. These tests exercise the pure function in
isolation; ``tests/test_shadow_benchmark.py`` and ``tests/test_qmd_
adapter.py`` cover how each side wires it into a report.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.questions import CORPUS_BODIES
from benchmarks.token_cost import measure_token_cost, read_real_body_tokens


def test_empty_paths_is_a_real_zero_not_unmeasured() -> None:
    """Nothing returned by an engine is an honest 0, not ``None``."""
    assert measure_token_cost((), wiki_root=None) == 0
    assert measure_token_cost((), wiki_root=Path("/does/not/matter")) == 0


def test_synthetic_path_is_measured_from_corpus_bodies_without_wiki_root() -> None:
    """A synthetic-provenance path never needs ``wiki_root`` at all."""
    path, body = next(iter(CORPUS_BODIES.items()))
    cost = measure_token_cost((path,), wiki_root=None)
    assert cost == len(body.split())


def test_real_path_is_unmeasured_when_wiki_root_is_none() -> None:
    """CI has no wiki checkout -- the honest #26 behavior must survive."""
    assert measure_token_cost(("Core/not-in-corpus-bodies.md",), wiki_root=None) is None


def test_real_path_is_measured_when_wiki_root_is_given(tmp_path: Path) -> None:
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    body = "alpha beta gamma delta epsilon"  # 5 whitespace tokens
    (note_dir / "note.md").write_text(
        f"---\nprivacy: internal\n---\n\n{body}\n", encoding="utf-8"
    )

    cost = measure_token_cost(("Core/note.md",), wiki_root=tmp_path)

    assert cost == len(body.split())


def test_frontmatter_is_stripped_before_counting(tmp_path: Path) -> None:
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    body = "one two three"
    (note_dir / "note.md").write_text(
        "---\n"
        "privacy: internal\n"
        "extra: field with several words that must not be counted\n"
        "---\n\n" + body + "\n",
        encoding="utf-8",
    )

    cost = measure_token_cost(("Core/note.md",), wiki_root=tmp_path)

    assert cost == len(body.split())


def test_unreadable_real_path_makes_the_whole_measurement_unmeasured(
    tmp_path: Path,
) -> None:
    """Unlike ``qmd_token_cost`` (which skips an unreadable path), the
    shared helper's own contract is "any single unresolved path -> None" --
    the same rule ``_token_cost`` always applied on the lexical/vector
    side. A caller that wants QMD's more lenient skip-and-undercount
    behavior gets it via ``read_real_body_tokens`` directly, not through
    this function.
    """
    cost = measure_token_cost(("Core/does-not-exist.md",), wiki_root=tmp_path)
    assert cost is None


def test_sums_across_multiple_real_paths(tmp_path: Path) -> None:
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    (note_dir / "a.md").write_text(
        "---\nprivacy: internal\n---\n\none two\n", encoding="utf-8"
    )
    (note_dir / "b.md").write_text(
        "---\nprivacy: internal\n---\n\nthree four five\n", encoding="utf-8"
    )

    cost = measure_token_cost(("Core/a.md", "Core/b.md"), wiki_root=tmp_path)

    assert cost == 2 + 3


def test_mixes_synthetic_and_real_paths_in_one_call(tmp_path: Path) -> None:
    """A single question can, in principle, return one synthetic-corpus
    path and one real-corpus path (the two sets never overlap by name in
    practice, but the helper must not assume that -- it resolves each path
    independently).
    """
    synthetic_path, synthetic_body = next(iter(CORPUS_BODIES.items()))
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    real_body = "six seven eight nine"
    (note_dir / "real.md").write_text(
        f"---\nprivacy: internal\n---\n\n{real_body}\n", encoding="utf-8"
    )

    cost = measure_token_cost((synthetic_path, "Core/real.md"), wiki_root=tmp_path)

    assert cost == len(synthetic_body.split()) + len(real_body.split())


def test_never_returns_the_body_text_itself(tmp_path: Path) -> None:
    """Only an integer ever leaves this function -- the RP1 read-hash-
    discard contract, proven directly rather than only inferred from the
    return type.
    """
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    sentinel = "SENTINEL-TOKEN-COST-BODY-SHOULD-NEVER-LEAK"
    (note_dir / "note.md").write_text(
        f"---\nprivacy: internal\n---\n\n{sentinel}\n", encoding="utf-8"
    )

    cost = measure_token_cost(("Core/note.md",), wiki_root=tmp_path)

    assert isinstance(cost, int)
    assert sentinel not in repr(cost)


def test_never_writes_any_file(tmp_path: Path) -> None:
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    (note_dir / "note.md").write_text(
        "---\nprivacy: internal\n---\n\nhello world\n", encoding="utf-8"
    )
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    measure_token_cost(("Core/note.md",), wiki_root=tmp_path)

    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert after == before


def test_read_real_body_tokens_returns_none_for_unreadable_path(
    tmp_path: Path,
) -> None:
    """The lower-level primitive ``qmd_token_cost`` also calls -- pinned
    separately so a regression there is attributed correctly rather than
    only surfacing through ``measure_token_cost``'s stricter contract.
    """
    assert read_real_body_tokens("Core/missing.md", wiki_root=tmp_path) is None


def test_read_real_body_tokens_strips_frontmatter(tmp_path: Path) -> None:
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    body = "ten eleven twelve"
    (note_dir / "note.md").write_text(
        f"---\nprivacy: internal\n---\n\n{body}\n", encoding="utf-8"
    )

    assert read_real_body_tokens("Core/note.md", wiki_root=tmp_path) == len(
        body.split()
    )


def test_non_utf8_file_is_unmeasured_not_a_mojibake_count(tmp_path):
    """Codex round 1 on #40: errors="replace" turned a binary file into
    countable mojibake -- a fake measured number. Undecodable is None."""
    root = tmp_path
    target = root / "Core" / "binary.md"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\xff\xfe\x00\x01 not utf-8 \xff")
    assert read_real_body_tokens("Core/binary.md", wiki_root=root) is None
    assert measure_token_cost(("Core/binary.md",), wiki_root=root) is None
