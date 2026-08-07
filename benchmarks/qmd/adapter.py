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
   out, never to widen the scored set back out.
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


def _strip_frontmatter(text: str) -> str:
    """Drop a leading ``---``-delimited frontmatter block, if present.

    Mirrors the ``lexical``/``vector`` sides counting *body* tokens only
    (``benchmarks.questions.CORPUS_BODIES`` never includes the frontmatter
    wrapper) -- deliberately a small local reimplementation rather than an
    import of ``ckp.pilot.manifest``'s private ``_frontmatter_body``, which
    returns a different shape (a line list, for a different caller) and is
    not part of that module's public contract.
    """
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    return text[end + len("\n---") :].lstrip("\n")


def qmd_token_cost(paths: tuple[str, ...], *, wiki_root: Path) -> int:
    """Whitespace-proxy token count (``TOKEN_COST_METHOD``) for QMD hits.

    Reads each note's bytes from ``wiki_root``, counts, and discards --
    nothing here is written to disk, logged, or returned as text; only the
    integer sum leaves this function. A path that cannot be read is skipped
    rather than raised: by the time this is called, ``paths`` has already
    passed the corpus-scope filter and the caller (``benchmarks.shadow``)
    has already committed to ``qmd_compared: True`` for this question, so a
    single unreadable file should not retroactively invalidate the whole
    comparison -- it is undercounted instead, same as a lexical/vector
    engine returning fewer results than expected.
    """
    total = 0
    for relative_path in paths:
        try:
            text = (wiki_root / relative_path).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        total += len(_strip_frontmatter(text).split())
    return total


def _normalize_path(raw: str) -> str:
    """Map one ``qmd --full-path --format json`` ``file`` value to a
    wiki-root-relative path.

    Observed shapes from a real run against the named ``cyclone-wiki``
    index (2026-08-07, ``qmd --index cyclone-wiki search ... --full-path
    --format json``, invoked with ``cwd`` set to the wiki checkout root):

    * ``"./Core/foo.md"`` -- the common case, cwd-relative.
    * ``"qmd://cyclone-wiki/Core/foo.md?index=cyclone-wiki"`` -- seen for at
      least one hit in the same response; the collection-URI form.

    Anything else is returned unchanged rather than guessed at: an unmatched
    shape simply fails the ``corpus_paths`` membership test downstream
    instead of being silently mis-mapped into a false match.
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
        normalized = _normalize_path(raw)
        if normalized not in config.corpus_paths or normalized in seen:
            continue
        seen.add(normalized)
        paths.append(normalized)
        score = entry.get("score")
        hits.append(
            QmdHit(
                relative_path=normalized,
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
