"""Shared whitespace-proxy token-cost measurement (issue #40).

Before this module existed, the three sides of the shadow benchmark
(``benchmarks.shadow``'s ``lexical``/``vector`` engines and
``benchmarks.qmd.adapter``'s QMD baseline) each counted tokens their own
way: the QMD side (#24) could read a real note's body from a live wiki
checkout and count it; the platform's own two sides (#26) could only look
up ``benchmarks.questions.CORPUS_BODIES`` -- an in-memory synthetic-only
map -- and reported ``None`` (unmeasured, never a silent 0) the moment a
returned path was not in it, which today is every real-provenance hit.
D1's "context token 總量不高於 QMD baseline" gate cannot be evaluated while
one side of the comparison is structurally blind on the real corpus.

This module is the single implementation both sides now call, so the
calculation cannot drift between them (e.g. one side quietly forgetting to
strip frontmatter before counting). It resolves a path's body two ways, in
order:

1. ``benchmarks.questions.CORPUS_BODIES`` -- the synthetic corpus, held in
   memory, no I/O. Every synthetic-provenance question is covered this way,
   unchanged from before this module existed.
2. A real Wiki checkout at ``wiki_root``, if the caller supplies one --
   read the note's bytes, strip a leading frontmatter block, count
   whitespace tokens, then discard the text. RP1's line is "content never
   enters this repo", not "content is never read in memory" -- the same
   read-hash-discard pattern ``ckp.pilot.manifest`` already relies on for
   hashing, and ``qmd_token_cost`` (#24) already relied on for its own
   token counting. Nothing here is written to disk, logged, or returned as
   text; only integers leave this module.

``wiki_root=None`` (the default) means "no live wiki checkout is
available" -- CI has none -- and every real-provenance path stays
unmeasured (``None``), the same honest behavior issue #26 established.
This module never reads ``CKP_PILOT_WIKI_ROOT`` or any other environment
variable itself; every caller (``run_shadow_benchmark``, ``qmd_token_cost``)
must be handed the root explicitly, matching the rest of this codebase's
explicit-composition style (AGENTS.md).
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.questions import CORPUS_BODIES


def _strip_frontmatter(text: str) -> str:
    """Drop a leading ``---``-delimited frontmatter block, if present.

    Mirrors the ``lexical``/``vector`` sides counting *body* tokens only
    (``CORPUS_BODIES`` never includes the frontmatter wrapper) -- this is
    the one strip implementation both the QMD side and the platform sides
    now share, closing the "one side quietly stops stripping" drift the
    issue #40 mutation matrix guards against.
    """
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    return text[end + len("\n---") :].lstrip("\n")


def _count_tokens(text: str) -> int:
    return len(text.split())


def read_real_body_tokens(relative_path: str, *, wiki_root: Path) -> int | None:
    """Read one real note's body from ``wiki_root``, count, and discard.

    Returns ``None`` (not raised, not 0) when the file cannot be read as
    UTF-8 text -- missing, permission error, any other ``OSError``, or bytes
    that do not decode. Codex round 1: ``errors="replace"`` turned a non-UTF-8
    file into countable mojibake and a *fake measured number* (a binary read
    704 "tokens"); undecodable is unmeasured, never a count.
    """
    try:
        text = (wiki_root / relative_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return _count_tokens(_strip_frontmatter(text))


def measure_token_cost(paths: tuple[str, ...], *, wiki_root: Path | None) -> int | None:
    """Whitespace-proxy token count across ``paths``, or ``None`` if
    unmeasured.

    The single helper the ``lexical`` and ``vector`` sides of
    ``run_shadow_benchmark`` both call. Each path is resolved via
    ``CORPUS_BODIES`` first (synthetic, in-memory), falling back to a real
    read under ``wiki_root`` when given. ``None`` -- never a silent 0 -- the
    moment a single path resolves through neither route: a real-provenance
    path when ``wiki_root`` is ``None`` (no wiki checkout available, e.g.
    CI), or a path that is in neither ``CORPUS_BODIES`` nor readable on
    disk. An empty ``paths`` tuple (nothing returned by the engine) is a
    real, honest zero and stays ``0``.
    """
    if not paths:
        return 0
    total = 0
    for path in paths:
        body = CORPUS_BODIES.get(path)
        if body is not None:
            total += _count_tokens(body)
            continue
        if wiki_root is None:
            return None
        real_tokens = read_real_body_tokens(path, wiki_root=wiki_root)
        if real_tokens is None:
            return None
        total += real_tokens
    return total


__all__ = [
    "measure_token_cost",
    "read_real_body_tokens",
]
