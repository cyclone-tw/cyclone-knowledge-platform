"""Shell out to the real ``qmd`` CLI and return a corpus-scoped hit list.

Rule sources: Epic #21 D3 (frozen 2026-08-07) and issue #24's own acceptance
list. Three invariants this module exists to enforce, not just document:

1. **Named index only.** The flagless default index is a best-effort mirror
   (``cyclone-wiki`` ``scripts/qmd-refresh.sh``: "agents habitually run
   flagless ``qmd search``/``qmd query``, which read
   ``~/.cache/qmd/index.sqlite``") and can silently lag the named index this
   benchmark is supposed to compare against. ``QmdBaselineConfig.index_name``
   is therefore a required field with no default, and every invocation
   passes ``--index <name>`` explicitly.
2. **Corpus-scoped.** A raw QMD hit list ranges over the whole indexed wiki;
   the caller's corpus (today: the frozen pilot manifest paths from
   ``ckp.pilot``) is a small subset of that. Comparing recall against two
   different denominators is meaningless (D3), so every result this module
   returns is pre-filtered to ``QmdBaselineConfig.corpus_paths`` before it
   ever reaches a caller -- ``QmdQueryResult.paths`` is the *filtered* list;
   ``raw_paths`` is kept only so a caller can report how much got filtered
   out, never to widen the scored set back out. Filtering recognizes each
   corpus path in qmd's own spelling as well (qmd strips the leading
   underscore from directory names -- issue #58, see ``_qmd_visible_path``)
   and always scores the canonical spelling.
3. **Fail loud, never fall back.** Every failure mode below --binary
   missing, wiki root missing, non-zero exit, timeout, malformed JSON, a
   hit missing its ``file`` field-- raises :class:`QmdUnavailable` rather
   than returning an empty or partial result that could be mistaken for "QMD
   ran and found nothing." The caller (``benchmarks.shadow``) is responsible
   for turning that into ``qmd_compared: false`` on the whole report, per
   D3 ("跑不了就整份失敗") -- this module never decides to silently retry
   with a different engine, and it never returns a sentinel value a caller
   could mistake for a real (if empty) answer.

``qmd_token_cost`` (added after the initial #24 review round) reads real note
bytes from ``wiki_root`` purely to count whitespace tokens, then discards
them -- the same "read the live checkout, hash/count it, never retain the
body" pattern ``ckp.pilot.manifest`` already uses to verify pilot notes (RP1:
the red line is content *entering this repo*, not a transient in-memory
read). It never writes anything to disk and never returns the text itself,
only an integer.

**Scope boundary (Epic #21, decided 2026-08-07):** this module delivers the
QMD-calling *capability* only. Wiring ``ckp.pilot.bind_pilot_corpus()`` ->
``QmdBaselineConfig`` -> ``benchmarks.shadow.run_shadow_benchmark()`` into a
single runnable entry point is deliberately left undone here -- it would
also touch the #26 question-set shape and the #30 threshold gate, both
still in flight on parallel issues. That wiring belongs to #30.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from benchmarks.token_cost import read_real_body_tokens

#: The ``qmd`` binary name resolved via ``PATH`` by default. Callers running
#: against a non-PATH install pass an explicit ``binary=`` (an absolute
#: path is fine there -- it is caller-supplied config, not tracked source)
#: to ``QmdBaselineConfig`` instead of relying on this -- this module never
#: hardcodes a host-specific absolute path itself (AGENTS.md §7).
DEFAULT_QMD_BINARY = "qmd"

#: Generous enough for a cold llama.cpp model load on first invocation, but
#: still bounded -- a hung subprocess must not hang the whole benchmark run.
DEFAULT_QMD_TIMEOUT_SECONDS = 60.0


class QmdUnavailable(RuntimeError):
    """QMD could not be queried in a way this module trusts.

    Raised for: binary not on PATH, wiki root missing, non-zero exit,
    timeout, non-JSON stdout, JSON that is not a list, or a hit missing its
    ``file`` key. Every one of these is a *reproducibility* failure, not a
    "found nothing" answer -- callers must not treat catching this the same
    as an empty :class:`QmdQueryResult` (D3: no silent fallback).
    """


class QmdConfigError(ValueError):
    """The caller built an unusable :class:`QmdBaselineConfig`.

    Distinct from :class:`QmdUnavailable` on purpose: this is a *caller*
    mistake in how the comparison was set up, raised at construction time,
    before any subprocess ever runs -- not a runtime failure of a
    correctly-configured comparison. Keeping the two exception types
    separate keeps a `except QmdUnavailable` at a call site from
    accidentally swallowing a config bug too (Codex Round 1 review).
    """


@dataclass(frozen=True)
class QmdBaselineConfig:
    """Everything one shadow-benchmark run needs to call the real QMD.

    ``index_name`` has no default (invariant 1 above). ``corpus_paths`` has
    no default either, and additionally **must be non-empty**: an empty
    scope is a caller configuration error, not a legitimate "compare
    against zero documents" request -- no real scenario needs that. Left
    unchecked, an empty ``corpus_paths`` makes every real QMD hit get
    filtered away, and the report would read as ``qmd_compared: true,
    qmd_hit_rate: 0.0`` -- indistinguishable from "QMD was compared and
    performed badly," when the honest story is "nothing was ever in scope."
    That silently biases every D1 relative threshold in the platform's
    favor (Codex Round 1 finding: the same shape as #26's silent-zero
    token-cost bug, a different entry point). ``__post_init__`` below fails
    loud at construction time -- before the caller has spent any QMD
    subprocess cost -- rather than deferring the check to the first query.
    """

    index_name: str
    wiki_root: Path
    corpus_paths: frozenset[str]
    collection: str | None = None
    binary: str = DEFAULT_QMD_BINARY
    timeout_seconds: float = DEFAULT_QMD_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.corpus_paths:
            raise QmdConfigError(
                "QmdBaselineConfig.corpus_paths is empty: there is nothing "
                "to compare QMD against. This is a caller configuration "
                "error, not a 'QMD found nothing' result -- comparing "
                "against zero documents would make every question read as "
                "qmd_compared=True with a hit rate of 0.0, which looks like "
                "QMD performed badly when in fact nothing was ever in "
                "scope. Pass the actual corpus scope (e.g. "
                "ckp.pilot.PILOT_NOTE_PATHS)."
            )
        # Two corpus paths that qmd renders identically (e.g.
        # ``Core/_inbox/a.md`` and ``Core/inbox/a.md`` -- see
        # ``_qmd_visible_path``) could not be told apart in qmd output, so
        # scoring either would be a guess. No real corpus does this today
        # (the pilot manifest has no such pair); if one ever does, fail at
        # construction rather than silently crediting one path with the
        # other's hits.
        visible_to_canonical: dict[str, str] = {}
        for path in sorted(self.corpus_paths):
            visible = _qmd_visible_path(path)
            other = visible_to_canonical.setdefault(visible, path)
            if other != path:
                raise QmdConfigError(
                    f"corpus_paths {other!r} and {path!r} are "
                    f"indistinguishable in qmd output (both appear as "
                    f"{visible!r} after qmd strips the leading underscore "
                    "from directory names); qmd hits could not be "
                    "attributed to one of them honestly"
                )


@dataclass(frozen=True)
class QmdHit:
    relative_path: str
    score: float | None


@dataclass(frozen=True)
class QmdQueryResult:
    """One query's result, already scoped to ``corpus_paths``.

    ``raw_paths`` is everything qmd returned before filtering, kept only
    for observability (e.g. "qmd found 5, 0 were in scope" is a different,
    more diagnosable story than "qmd found nothing") -- report code must
    score against ``paths``/``hits``, never ``raw_paths``.
    """

    raw_paths: tuple[str, ...]
    paths: tuple[str, ...]
    hits: tuple[QmdHit, ...]


def qmd_binary_available(binary: str = DEFAULT_QMD_BINARY) -> bool:
    return shutil.which(binary) is not None


def qmd_token_cost(paths: tuple[str, ...], *, wiki_root: Path) -> int | None:
    """Whitespace-proxy token count (``TOKEN_COST_METHOD``) for QMD hits.

    Reads each note's body via ``benchmarks.token_cost.read_real_body_tokens``
    -- issue #40's shared strip-frontmatter-then-count implementation, the
    same one the platform's own ``lexical``/``vector`` sides now call, so
    the two sides of the D1 comparison cannot drift apart on how a token is
    counted. Nothing here is written to disk, logged, or returned as text;
    only the integer sum leaves this function -- or ``None``. Codex round 1
    on #40: the old skip-and-undercount turned "could not read anything" into
    a confident ``0``, drifting from the ``None`` semantics the platform
    sides use and handing D1's relative token threshold a fake baseline. One
    unreadable (or undecodable) path now makes the whole tuple unmeasured,
    exactly like ``measure_token_cost``.
    """
    total = 0
    for relative_path in paths:
        tokens = read_real_body_tokens(relative_path, wiki_root=wiki_root)
        if tokens is None:
            return None
        total += tokens
    return total


def _qmd_visible_path(path: str) -> str:
    """``path`` as qmd's own document namespace renders it.

    Observed against the real named ``cyclone-wiki`` index (issue #58,
    2026-08-08, qmd 2.5.3): qmd strips one leading underscore from each
    *directory* name, so ``Core/_inbox/agent-captures/foo.md`` is listed,
    fetched, and returned in search hits as
    ``Core/inbox/agent-captures/foo.md`` -- ``qmd ls`` finds nothing under
    the underscored spelling, and ``qmd get`` canonicalizes either spelling
    to the stripped one. ``--full-path`` cannot undo it: resolving the
    stripped path against the checkout fails (no such file), so those hits
    stay in ``qmd://`` URI form with the stripped spelling inside.

    The *filename* segment is deliberately left untouched. Whether qmd
    also strips a file-level leading underscore is unobservable today (the
    wiki has no ``_``-prefixed ``.md`` file to probe), and transforming it
    anyway would widen the accepted alias space beyond observed behavior:
    a corpus entry ``Core/_inbox/_note.md`` would then also claim hits
    spelled ``Core/inbox/note.md``, which under the observed
    directory-only rule is how qmd renders the *different* file
    ``Core/_inbox/note.md`` -- a false credit (#61 review round 1). If a
    ``_``-prefixed filename ever enters the corpus and qmd does strip it,
    that surfaces as a conspicuous miss to investigate, never as a
    silently wrong baseline.

    Without this mapping, every hit on a note under ``Core/_inbox/`` fails
    the ``corpus_paths`` membership test and gets scored as a QMD miss --
    on the first live #58 probe, QMD ranked the two D2 ``_inbox`` pilot
    notes at rank 1 for their rewritten queries and the adapter still
    reported both as misses. That is a false baseline in the platform's
    favor, the exact shape D3's scope-overlap precondition exists to keep
    out of the comparison.
    """
    segments = path.split("/")
    return "/".join(
        [
            *(
                segment[1:] if segment.startswith("_") else segment
                for segment in segments[:-1]
            ),
            segments[-1],
        ]
    )


def _normalize_path(raw: str) -> str:
    """Map one ``qmd --full-path --format json`` ``file`` value to a
    wiki-root-relative path *as qmd spells it*.

    Observed shapes from a real run against the named ``cyclone-wiki``
    index (2026-08-07, ``qmd --index cyclone-wiki search ... --full-path
    --format json``, invoked with ``cwd`` set to the wiki checkout root):

    * ``"./Core/foo.md"`` -- the common case, cwd-relative.
    * ``"qmd://cyclone-wiki/Core/foo.md?index=cyclone-wiki"`` -- the
      collection-URI form; ``--full-path`` falls back to it whenever the
      indexed path does not resolve on disk, which is every note under an
      underscore-prefixed directory (see ``_qmd_visible_path``).

    Anything else is returned unchanged rather than guessed at: an unmatched
    shape simply fails the ``corpus_paths`` membership test downstream
    instead of being silently mis-mapped into a false match. The returned
    path may still be qmd's underscore-stripped spelling of a real path;
    ``run_qmd_query`` resolves that against ``corpus_paths`` via
    ``_qmd_visible_path``, and this function stays a pure string extractor.
    """
    text = raw.strip()
    if text.startswith("./"):
        return text[2:]
    if text.startswith("qmd://"):
        rest = text[len("qmd://") :].split("?", 1)[0]
        if "/" in rest:
            _, path = rest.split("/", 1)
            return path
    return text


def _build_command(query: str, *, config: QmdBaselineConfig, top_k: int) -> list[str]:
    command = [
        config.binary,
        "--index",
        config.index_name,
        "search",
        query,
        "-n",
        str(top_k),
        "--format",
        "json",
        "--full-path",
    ]
    if config.collection:
        command += ["-c", config.collection]
    return command


def run_qmd_query(
    query: str, *, config: QmdBaselineConfig, top_k: int
) -> QmdQueryResult:
    """Run one lexical (``qmd search``, no LLM rerank) query and scope it.

    ``qmd search`` -- not ``qmd query`` -- is deliberate: this baseline
    replaces the platform's own deterministic lexical engine
    (``benchmarks.shadow``'s ``lexical`` side), so the fair comparison is
    QMD's own deterministic BM25 search, not its LLM-reranked hybrid mode
    (issue #24 non-goal: "不引入 semantic provider"). Raises
    :class:`QmdUnavailable` on every failure mode instead of returning a
    partial or empty result a caller could mistake for a real answer.
    """
    if not qmd_binary_available(config.binary):
        raise QmdUnavailable(f"{config.binary!r} not found on PATH")
    if not config.wiki_root.is_dir():
        raise QmdUnavailable(f"wiki root {config.wiki_root} is not a directory")

    # Membership is tested in both spellings of each corpus path: the
    # canonical one and qmd's underscore-stripped rendering of it
    # (``_qmd_visible_path``); ``__post_init__`` guarantees no two corpus
    # paths share a rendering. One ambiguity ``__post_init__`` cannot see:
    # a *non-corpus* file on disk at the stripped spelling (e.g. a real
    # ``Core/inbox/a.md`` next to corpus ``Core/_inbox/a.md``) would be
    # indistinguishable from the corpus note in qmd output, so a hit could
    # not be attributed honestly -- fail the run rather than guess. Checked
    # here, not at construction: it is a property of the wiki checkout, not
    # of the config, and this function is the first point that may read
    # ``wiki_root``.
    canonical_by_spelling: dict[str, str] = {}
    for corpus_path in sorted(config.corpus_paths):
        canonical_by_spelling[corpus_path] = corpus_path
        visible = _qmd_visible_path(corpus_path)
        if visible == corpus_path:
            continue
        if (config.wiki_root / visible).exists():
            raise QmdUnavailable(
                f"cannot attribute qmd hits for {visible!r}: both "
                f"{corpus_path!r} (corpus) and {visible!r} exist in the "
                "wiki checkout, and qmd renders them identically"
            )
        canonical_by_spelling[visible] = corpus_path

    command = _build_command(query, config=config, top_k=top_k)
    try:
        completed = subprocess.run(
            command,
            cwd=config.wiki_root,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QmdUnavailable(f"qmd invocation failed: {exc}") from exc

    if completed.returncode != 0:
        raise QmdUnavailable(
            f"qmd exited {completed.returncode}: {completed.stderr.strip()[:500]}"
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise QmdUnavailable(f"qmd returned non-JSON stdout: {exc}") from exc

    if not isinstance(payload, list):
        raise QmdUnavailable(
            f"qmd JSON output was not a list of hits (got {type(payload).__name__})"
        )

    raw_paths: list[str] = []
    paths: list[str] = []
    hits: list[QmdHit] = []
    seen: set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict) or "file" not in entry:
            raise QmdUnavailable(f"qmd hit missing 'file' field: {entry!r}")
        raw = str(entry["file"])
        raw_paths.append(raw)
        # Scored paths are always the *canonical* spelling, so downstream
        # consumers (``_hit`` against ``expected_paths``, ``qmd_token_cost``
        # reading ``wiki_root / path``) never see qmd's.
        canonical = canonical_by_spelling.get(_normalize_path(raw))
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        paths.append(canonical)
        score = entry.get("score")
        hits.append(
            QmdHit(
                relative_path=canonical,
                score=score if isinstance(score, (int, float)) else None,
            )
        )

    return QmdQueryResult(
        raw_paths=tuple(raw_paths), paths=tuple(paths), hits=tuple(hits)
    )


__all__ = [
    "DEFAULT_QMD_BINARY",
    "DEFAULT_QMD_TIMEOUT_SECONDS",
    "QmdBaselineConfig",
    "QmdConfigError",
    "QmdHit",
    "QmdQueryResult",
    "QmdUnavailable",
    "qmd_binary_available",
    "qmd_token_cost",
    "run_qmd_query",
]
