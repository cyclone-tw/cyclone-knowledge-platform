"""The QMD shadow benchmark harness (contract §5.3): report only, no cutover.

Both engines answer the same frozen questions over the same privacy-gated
corpus. The report is honest about what it measures:

* the vector side runs the C4 hash embedding, which declares
  ``semantic=false`` -- these numbers are a deterministic baseline, not a
  claim of semantic quality, and Qdrant does not win by being newer;
* ``token_cost`` is a deterministic proxy (whitespace token count of each
  returned note's body), not a model tokenizer count -- ``token_cost_method``
  and ``token_cost_note`` say so explicitly at the top of the report, sourced
  from a single named constant rather than repeated string literals. A
  returned path whose body this module cannot see (see the provenance bullet
  below) makes that question's token cost ``None`` -- **unmeasured, not
  zero**. A missing body must never silently become 0: 0 reads as "measured
  and free", which for a real-provenance question would understate the
  platform's real token cost against a QMD baseline that *can* read the real
  body (#24) -- exactly the false-pass Phase 4 gate issue #26's coordinator
  review caught. Every total this module reports is scoped to measured
  questions only, with an explicit unmeasured count alongside it, so a
  downstream reader can never mistake a partial total for a complete one;
* ``latency_ms`` re-runs each question against each engine ``trials`` times
  (default 5, overridable by the caller) and reports P50/P95 over the
  pooled per-trial samples, not just a single-shot sum;
* stale-exclusion counts how often each engine avoided surfacing a note the
  corpus has marked ``superseded`` -- a superseded note is still public, so
  this is a freshness signal, not a privacy one. **Known limitation**: the
  denominator is "all questions", so a question whose honest answer is
  empty (e.g. ``no-answer``) scores as a free stale-exclusion success for
  any engine that returns nothing at all -- an engine that never returns
  anything gets a 100% stale-exclusion rate. This is not corrected here;
  the metric's denominator is a #30 (P9) decision, not something this
  module should silently redefine;
* privacy is split into two counts that must never be merged: false
  negatives (a non-public path leaked into the report -- zero-tolerance, a
  hard failure) and false positives (a note declared ``public`` that never
  made it into the gated plan -- a quality signal, not a leak). The false
  positive count is computed against the *frozen* corpus's declared privacy
  (``benchmarks.questions.PUBLIC_DECLARED_PATHS``), not against whatever
  ``members`` the caller actually passes in. Every current caller derives
  ``members`` from ``CORPUS`` (synthetic-only), so the two still agree;
  issue #26 (P5) added a real half to the question set
  (``benchmarks.questions.REAL_QUESTIONS``) but not to ``CORPUS`` itself --
  no real note body is ever allowed to live in this repo (RP1) -- so a
  caller that wants privacy-false-positive accounting for real notes must
  still pass its own "declared public" set; this module does not do that;
* the question set (``questions``, a required composition input since #26)
  may mix ``provenance="real"`` and ``provenance="synthetic"`` questions.
  The pooled numbers exist for continuity with pre-#26 reports; they must
  never be read alone once a report mixes real and synthetic questions,
  because a good synthetic score can hide a bad real one (or the reverse)
  -- the pooled hit rate on a mixed set answers "how did the whole set do",
  not "how did the platform do on real Cyclone-Wiki notes", and the two
  questions are not the same one. Token cost for a real-provenance
  question is ``None`` (unmeasured) whenever a returned path's body is not
  in ``benchmarks.questions.CORPUS_BODIES``, which (by RP1) never contains
  a real note's body -- so today, every real question with any hit is
  unmeasured. This module does not fabricate a real-body reader to fix
  that (issue #24 is already building one for the QMD side, via the same
  read-hash-discard pattern ``src/ckp/pilot/manifest.py`` uses;
  duplicating it here would give two implementations that can drift).
  Unifying the two sides -- so a real question's token cost is measured the
  same way on both sides of the shadow comparison -- is follow-up work, not
  this module's job today. See ``benchmarks.questions.KNOWN_COVERAGE_GAPS``
  and the module docstring there for the categories a real note does not
  exist to test at all;
* issue #26 Round 3 (Codex): every *question-level* count and rate in
  ``summary`` is reported twice -- once pooled (the pre-#26 keys, unchanged
  in meaning and computation) and once split under
  ``summary["provenance"]["real"]`` / ``["synthetic"]``. Round 2 split hit
  rate, token cost, and citation correctness only; Round 3 closes the rest
  of the audit: ``lexical_stale_excluded`` / ``vector_stale_excluded`` and
  their ``_rate`` (denominator is this provenance's own question count,
  never the pooled ``total_questions`` -- an unsplit denominator is the
  same shape of bug as an unsplit numerator), ``lexical_no_answer_correct``
  (``None``, not a fake pass, in a provenance that asked no ``no-answer``
  question at all), and ``privacy_false_negatives`` (split as a diagnostic
  convenience -- see below, it was never a self-proof risk pooled). Two
  fields are deliberately **not** split, with the reasoning kept next to
  them rather than left implicit:

  * ``privacy_false_positives`` / ``privacy_false_positive_paths`` are
    corpus-level, not question-level -- computed once, before any question
    is asked, from ``PUBLIC_DECLARED_PATHS - <what the plan indexed>``.
    There is no per-question attribution to split by, and
    ``PUBLIC_DECLARED_PATHS`` only ever names synthetic paths (D2 froze
    every real note at ``internal``, never ``public``), so this metric is
    already synthetic-only in substance, not merely in reporting.
  * ``latency_ms`` stays pooled-only. Unlike a hit rate or a stale-exclusion
    rate, wall-clock latency cannot be inflated by writing an easy
    synthetic question -- the self-proof risk this module exists to guard
    against does not apply the same way to a performance measurement. D1's
    latency threshold is also evaluated against the pooled report, not a
    per-provenance one. Splitting it would need per-provenance trial
    sample pools, which nothing here currently threads through the
    ``trials`` retry loop; left as a documented gap, not a silent one.

  ``rebuild.point_count`` and every top-level metadata field (``schema``,
  ``corpus_version``, ``coverage_gaps``, ``index_provider``,
  ``composed_revision``, ``*_revision``, ``semantic``/``semantic_note``,
  ``token_cost_method``/``token_cost_note``, ``top_k``) describe the run
  itself, not an aggregate over questions -- "split by provenance" does not
  apply to them at all;
* everything except ``latency_ms`` is reproducible byte-for-byte from the
  same commit, and tests pin exactly that.

Privacy is zero-tolerance: a single non-public path in either engine's
output makes ``privacy_false_negatives`` non-zero, and the test suite treats
that as a hard failure (contract §5.3: false-negative exposure is
intolerable).

**QMD baseline (issue #24, Epic #21 D3 frozen)**: an optional third engine,
distinct from ``lexical`` (the platform's own catalog search -- still what
that field name means, unchanged by this addition). When the caller passes
``qmd_config``, every question is also run against the real ``qmd`` CLI,
scoped to ``qmd_config.corpus_paths`` (D3: comparing recall against two
different denominators is meaningless). ``report["qmd_compared"]`` is
``True`` only if every question's QMD call succeeded; the moment any call
raises ``QmdUnavailable``, the *entire* QMD dimension is discarded from the
report -- no partial results, no silent fallback to ``lexical`` standing in
for it (D3: "跑不了就整份失敗"). ``qmd_config=None`` (the default) means "no
QMD comparison was attempted", reported the same way as a mid-run failure:
``qmd_compared: False``.
"""

from __future__ import annotations

import math
import time

from benchmarks.qmd.adapter import (
    QmdBaselineConfig,
    QmdUnavailable,
    qmd_token_cost,
    run_qmd_query,
)
from benchmarks.questions import (
    CORPUS_BODIES,
    CORPUS_VERSION,
    KNOWN_COVERAGE_GAPS,
    PUBLIC_DECLARED_PATHS,
    QUESTION_SET_VERSION,
    SUPERSEDED_PATHS,
    BenchmarkQuestion,
)
from ckp.bundle import (
    BundleMember,
    BundleSnapshot,
    compute_index_revision_from_members,
)
from ckp.catalog import CatalogBuilder
from ckp.embedding import EmbeddingStack
from ckp.gateway.models import QueryRequest
from ckp.gateway.service import query_response
from ckp.index.models import plan_rebuild
from ckp.index.provider import VectorIndexProvider, require_index_provider
from ckp.privacy import PrivacyClass
from ckp.privacy.gate import PrivacyGate

REPORT_SCHEMA = "ckp-shadow-report/3"

#: Named once so the honesty note in the report never drifts into a second,
#: slightly different string somewhere else in the codebase.
TOKEN_COST_METHOD = "whitespace-proxy"
TOKEN_COST_NOTE = (
    "token counts are a whitespace-split proxy, not a model tokenizer count"
)

#: D1's token-cost threshold ("context token 總量 不高於 QMD baseline") needs
#: a QMD-side number to compare against, so unlike stale-exclusion below,
#: this is not optional -- ``qmd_token_cost`` reuses the exact same
#: whitespace-proxy method as the lexical/vector sides (``TOKEN_COST_METHOD``
#: above), just applied to real note bytes read transiently from
#: ``qmd_config.wiki_root`` and immediately discarded (RP1: the rule is
#: "content never enters this repo", not "content is never read in memory" --
#: the same reasoning ``ckp.pilot.manifest`` already relies on for hashing).

#: D2's "known coverage gap": the real pilot corpus (#23's six notes) has no
#: ``superseded`` relationship to measure, so QMD-side stale exclusion is
#: structurally not-applicable, not merely unmeasured. Reported as an
#: explicit note rather than a numeric field so a gate reading this report
#: (#30) cannot mistake an absent key for a zero or a bug.
QMD_STALE_EXCLUSION_NOTE = (
    "QMD side has no stale-exclusion metric: it depends on the corpus's "
    "`superseded` marker, and the real pilot corpus (Epic #21 D2) carries no "
    "superseded relationship to measure -- a known coverage gap recorded in "
    "D2, not a bug in this report."
)

#: Codex Round 1 (#24 review) found a second entry point for the same shape
#: this whole issue exists to close: a non-empty ``corpus_paths`` that
#: happens to share *no* path with anything QMD actually returns (every
#: path mistyped, or pointed at files the named index does not have)
#: produces the identical "qmd_compared: true, qmd_hit_rate: 0.0" report an
#: empty scope would have -- indistinguishable from "QMD was compared and
#: performed badly" unless flagged separately. ``qmd_scope_overlap`` is the
#: flag: ``True`` once any raw QMD hit anywhere in the run fell inside the
#: configured scope, ``False`` if QMD returned real hits but never once one
#: that matched the scope (a config smell, not a quality signal), ``None``
#: when QMD was never compared or never returned a raw hit at all (in which
#: case a scope mismatch cannot be told apart from a genuine "nothing
#: found").
QMD_SCOPE_OVERLAP_NOTE = (
    "true: at least one raw QMD hit somewhere in this run fell inside "
    "corpus_paths. false: QMD returned real hits but none of them, in the "
    "whole run, ever matched corpus_paths -- read this as a likely scope or "
    "path configuration error, not as a retrieval-quality result (Codex "
    "Round 1 finding). null: QMD was not compared, or never returned any "
    "raw hit at all, so a scope mismatch cannot be distinguished from a "
    "genuine 'nothing found' answer."
)

#: Default per-question, per-engine repeat count for the latency sample.
DEFAULT_LATENCY_TRIALS = 5

_PUBLIC_ONLY = frozenset({PrivacyClass.PUBLIC})


def _token_cost(paths: tuple[str, ...]) -> int | None:
    """Whitespace-proxy token count, or ``None`` when any path is unmeasured.

    ``None`` (not ``0``) the moment a single path's body is not in
    ``CORPUS_BODIES`` -- a path this module cannot see the body of is
    unmeasured, never "measured at zero". An empty ``paths`` tuple (nothing
    returned) is a real, honest zero and stays ``0``.
    """
    total = 0
    for path in paths:
        body = CORPUS_BODIES.get(path)
        if body is None:
            return None
        total += len(body.split())
    return total


def _hit(expected: tuple[str, ...], returned: tuple[str, ...]) -> bool | None:
    if not expected:
        return None
    return set(expected).issubset(set(returned))


def _percentile(samples: list[float], pct: float) -> float:
    """Nearest-rank percentile: ``ordered[ceil(pct / 100 * n) - 1]``.

    Chosen over ``statistics.quantiles`` because nearest-rank never
    interpolates between two samples -- the reported value is always one of
    the measured trials, which keeps the number traceable back to an actual
    run even when ``trials`` is small.
    """
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    rank = min(rank, len(ordered))
    return ordered[rank - 1]


def _provenance_summary(tally: dict[str, int]) -> dict:
    """Render one provenance's tally into the same shape as the pooled
    summary's rate/tally fields (issue #26) -- a mutant that folds real and
    synthetic tallies together upstream still produces *some* dict here, but
    a matching test can compare it against the pooled numbers to catch that.

    ``lexical_token_cost_total`` / ``vector_token_cost_total`` sum *measured*
    questions only (issue #26 coordinator review) -- a question whose token
    cost was unmeasured (``_token_cost`` returned ``None``) contributes
    nothing to the total and is counted separately in
    ``lexical_token_cost_unmeasured_questions`` /
    ``vector_token_cost_unmeasured_questions`` instead, so the total's
    coverage is always stated next to it, never silently assumed complete.

    ``*_stale_excluded_rate`` divides by ``question_count`` (this
    provenance's total question count), never by the pooled
    ``total_questions`` -- issue #26 Round 3 (Codex): the pooled
    denominator would silently blend one provenance's question count into
    the other's rate, the exact same shape of bug as an unsplit numerator.

    ``lexical_no_answer_correct`` is ``None`` when this provenance asked no
    ``no-answer`` question at all (today: always the synthetic provenance
    only -- no real D2 note is a no-answer fixture), the same "not asked,
    not a failure" rule ``lexical_hit_rate`` already follows. There is no
    vector equivalent, matching the pooled field: a cosine engine always
    ranks *something*, so scoring its abstention would fake a capability
    (module docstring).
    """
    scored_questions = tally["scored_questions"]
    question_count = tally["question_count"]
    no_answer_total = tally["no_answer_total"]
    return {
        "question_count": question_count,
        "scored_questions": scored_questions,
        "lexical_hit_rate": (
            round(tally["lexical_hits"] / scored_questions, 4)
            if scored_questions
            else None
        ),
        "vector_hit_rate": (
            round(tally["vector_hits"] / scored_questions, 4)
            if scored_questions
            else None
        ),
        "lexical_token_cost_total": tally["lexical_token_cost_total"],
        "lexical_token_cost_unmeasured_questions": tally[
            "lexical_token_cost_unmeasured_questions"
        ],
        "vector_token_cost_total": tally["vector_token_cost_total"],
        "vector_token_cost_unmeasured_questions": tally[
            "vector_token_cost_unmeasured_questions"
        ],
        "citation_correct_questions": tally["citation_correct_questions"],
        # Diagnostic split, not a self-proof fix (issue #26 Round 3 audit):
        # this is zero-tolerance already -- a leak from *either* provenance
        # makes the pooled count non-zero and fails the gate, so pooling it
        # cannot hide a real problem the way an averaged rate can. Split
        # anyway because it is cheap (already tallied per question) and
        # tells a reviewer *which* provenance leaked.
        "privacy_false_negatives": tally["privacy_false_negatives"],
        "lexical_stale_excluded": tally["lexical_stale_excluded"],
        "vector_stale_excluded": tally["vector_stale_excluded"],
        "lexical_stale_excluded_rate": (
            round(tally["lexical_stale_excluded"] / question_count, 4)
            if question_count
            else None
        ),
        "vector_stale_excluded_rate": (
            round(tally["vector_stale_excluded"] / question_count, 4)
            if question_count
            else None
        ),
        "lexical_no_answer_correct": (
            None if no_answer_total == 0 else tally["no_answer_violations"] == 0
        ),
    }


def _latency_block(samples: list[float]) -> dict:
    return {
        "total": round(sum(samples) * 1000.0, 3),
        "p50": round(_percentile(samples, 50) * 1000.0, 3),
        "p95": round(_percentile(samples, 95) * 1000.0, 3),
    }


def run_shadow_benchmark(
    *,
    members: tuple[BundleMember, ...],
    stack: EmbeddingStack,
    index_provider: VectorIndexProvider,
    gate: PrivacyGate,
    top_k: int,
    questions: tuple[BenchmarkQuestion, ...],
    index_admissible: frozenset[PrivacyClass] = _PUBLIC_ONLY,
    trials: int = DEFAULT_LATENCY_TRIALS,
    qmd_config: QmdBaselineConfig | None = None,
) -> dict:
    """Rebuild, query both engines, and assemble the deterministic report.

    ``questions`` is a required composition input (issue #26): the caller
    states exactly which questions to score, same as it already states
    ``members``, ``stack``, and ``gate``. Passing
    ``benchmarks.questions.QUESTIONS`` reproduces pre-#26 behavior
    (synthetic-only); passing ``benchmarks.questions.PILOT_QUESTIONS`` (or
    any other mix) scores real and synthetic questions together, split in
    the report by each question's own ``provenance``.

    ``index_admissible`` (issue #26 Round 2 coordinator review) is forwarded
    verbatim to ``plan_rebuild``'s own ``admissible`` parameter -- it widens
    which privacy classes actually get embedded into the index. Defaults to
    public-only, matching every caller before this parameter existed.
    Passing it wider (e.g. ``frozenset({PrivacyClass.PUBLIC,
    PrivacyClass.INTERNAL})``) is the fix for the bug Codex's Round 1 review
    found: a mixed real+synthetic ``questions`` set produced an honest-
    looking-but-empty real-provenance score, because the six real (``
    internal``) D2 notes never made it into the index at all -- the plan
    admitted public only, no matter what ``gate`` itself was configured to
    admit -- so every real question's retrieval was a guaranteed miss before
    a single query ran.

    Deliberately **not** parametrized: the ``filter_privacy`` argument this
    module passes to each ``index_provider.search()`` call, which stays
    hardcoded to ``_PUBLIC_ONLY`` regardless of ``index_admissible``. Every
    concrete ``VectorIndexProvider`` this repo ships validates that argument
    via ``ckp.index.models.require_public_filter`` (exact equality to
    ``{PrivacyClass.PUBLIC}``, unrelated to and unaware of
    ``index_admissible``) but **never actually uses it to filter which
    points a search can return** -- every provider scores every point in the
    rebuilt plan with no per-point privacy check, because ``IndexedPoint``
    itself carries no privacy field. So passing anything other than
    ``_PUBLIC_ONLY`` there would only ever raise ``IndexRefusal`` without
    changing what comes back; the real fix is entirely on the index-build
    side (``index_admissible`` above), and leaving ``filter_privacy`` as
    ``_PUBLIC_ONLY`` here is intentional, not an oversight left over from
    Round 1.

    ``trials`` controls how many times each question is re-run against each
    engine purely for the latency sample (default
    ``DEFAULT_LATENCY_TRIALS``); it does not change correctness, hit,
    token-cost, citation, or stale-exclusion accounting, which are all
    computed once per question from the first trial's results.

    ``qmd_config`` (issue #24) opts into the real QMD baseline: when given,
    every question also runs once against the ``qmd`` CLI (never repeated
    for ``trials`` -- a slow external subprocess is not worth hammering
    purely for a latency sample), scoped to ``qmd_config.corpus_paths``. Any
    failure discards the whole QMD dimension (``qmd_compared: False``,
    every question's ``"qmd"`` field reset to ``None``) rather than mixing
    partial results with a report that otherwise looks complete.

    Two accounting caveats, documented rather than fixed here (behavior is
    frozen; see the module docstring for the full explanation):

    * ``lexical_stale_excluded`` / ``vector_stale_excluded`` count "no
      superseded path in what was returned", including when nothing was
      returned at all -- a question with an honestly empty answer (or an
      engine that answers nothing) is indistinguishable from one that
      correctly excluded a superseded note.
    * ``privacy_false_positives`` is judged against the frozen corpus's
      declared-public set, not against ``members``; it only stays correct
      because every current caller builds ``members`` from ``CORPUS``.
    * token cost is ``None`` (unmeasured), per question and per engine,
      whenever a returned path's body is not in ``CORPUS_BODIES`` -- today
      that is every real-provenance hit, since no real note body is ever
      allowed to live in this repo (RP1). Unmeasured never contributes 0 to
      a total; see ``_token_cost`` and the ``*_token_cost_unmeasured_
      questions`` fields in ``summary``.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")

    descriptor = require_index_provider(index_provider)
    plan = plan_rebuild(
        members=members, stack=stack, gate=gate, admissible=index_admissible
    )
    rebuild_report = index_provider.rebuild(plan)

    snapshot = BundleSnapshot(
        members=members,
        profile_version=None,
        bundle_commit=None,
        index_revision=compute_index_revision_from_members(members),
        sources={},
        token=(),
    )
    catalog = CatalogBuilder(gate).build(snapshot)

    #: The only paths any engine may surface: what the gated plan actually
    #: indexed under this run's ``index_admissible``. Named ``public_paths``
    #: since #22, kept for #26 (a name change here is not pinned by any
    #: mutation but would be needless diff noise) even though it is no
    #: longer only-ever-public: with a widened ``index_admissible`` this is
    #: the real "approved set" -- the pilot's admitted real notes union
    #: whatever synthetic public notes also made it in -- computed from what
    #: this run's plan actually contains, never from a static allowlist, so
    #: a rogue provider inventing a path nobody has ever seen is still
    #: caught (R1 review) even in a mixed run.
    public_paths = frozenset(point.relative_path for point in plan.points)

    #: Privacy false positives: notes declared ``public`` in the corpus that
    #: never made it into the gated plan. A quality signal (over-blocking),
    #: never merged with the zero-tolerance leak count below (R1 intent).
    fp_paths = sorted(PUBLIC_DECLARED_PATHS - public_paths)

    question_reports = []
    lexical_hits = 0
    vector_hits = 0
    scored = 0
    lexical_tokens_total = 0
    vector_tokens_total = 0
    # Unmeasured counters (issue #26 coordinator review): a question whose
    # token cost could not be measured (``_token_cost`` returned ``None``)
    # must never silently add 0 to the totals above -- it is counted here
    # instead, so the totals' coverage is always stated, never implied.
    lexical_tokens_unmeasured = 0
    vector_tokens_unmeasured = 0
    privacy_false_negatives = 0
    citation_correct_count = 0
    lexical_no_answer_correct = True
    lexical_stale_excluded = 0
    vector_stale_excluded = 0
    lexical_trial_samples: list[float] = []
    vector_trial_samples: list[float] = []

    #: QMD baseline (issue #24). ``qmd_compared`` starts True only if a
    #: config was actually supplied, and flips to False for good the first
    #: time a call fails -- once False, no further qmd calls are attempted
    #: (D3: fail loud, no partial mixing) and every previously collected
    #: question-level qmd block is discarded in the post-loop cleanup below.
    qmd_compared = qmd_config is not None
    qmd_unavailable_reason: str | None = None
    qmd_hits = 0
    qmd_scored = 0
    qmd_tokens_total = 0
    qmd_trial_samples: list[float] = []
    #: Codex Round 1 diagnostic (see ``QMD_SCOPE_OVERLAP_NOTE``): counts
    #: across the whole run, independent of any single question's hit/miss.
    qmd_questions_with_raw_hits = 0
    qmd_questions_with_scoped_hits = 0

    # Per-provenance tallies (issue #26): the same counters as above, kept
    # separately per ``question.provenance`` so a mixed report can never
    # present one pooled number that quietly blends a real and a synthetic
    # score together. Built with a plain dict rather than a dataclass so a
    # third provenance value (there is none today -- see
    # ``benchmarks.questions.PROVENANCES``) would show up as a new key here
    # too, not silently fall into an existing bucket.
    _provenance_zero = {
        "question_count": 0,
        "scored_questions": 0,
        "lexical_hits": 0,
        "vector_hits": 0,
        "lexical_token_cost_total": 0,
        "lexical_token_cost_unmeasured_questions": 0,
        "vector_token_cost_total": 0,
        "vector_token_cost_unmeasured_questions": 0,
        "citation_correct_questions": 0,
        "privacy_false_negatives": 0,
        "lexical_stale_excluded": 0,
        "vector_stale_excluded": 0,
        "no_answer_total": 0,
        "no_answer_violations": 0,
    }
    by_provenance: dict[str, dict[str, int]] = {
        "real": dict(_provenance_zero),
        "synthetic": dict(_provenance_zero),
    }

    for question in questions:
        provenance_tally = by_provenance[question.provenance]
        provenance_tally["question_count"] += 1
        started = time.perf_counter()
        lexical = query_response(
            catalog, QueryRequest(query=question.query, limit=top_k)
        )
        lexical_trial_samples.append(time.perf_counter() - started)
        raw_lexical_paths = tuple(result.citation.path for result in lexical.results)

        started = time.perf_counter()
        query_vector = stack.embedding.embed_query(question.query).values
        vector = index_provider.search(
            query_vector, top_k=top_k, filter_privacy=_PUBLIC_ONLY
        )
        vector_trial_samples.append(time.perf_counter() - started)
        raw_vector_paths = tuple(hit.relative_path for hit in vector.hits)

        # Extra trials measure latency only: their results are discarded,
        # and correctness/hit/token/citation/stale accounting below always
        # uses the first trial's results, so the report stays reproducible
        # byte-for-byte once latency is stripped.
        for _ in range(trials - 1):
            started = time.perf_counter()
            query_response(catalog, QueryRequest(query=question.query, limit=top_k))
            lexical_trial_samples.append(time.perf_counter() - started)

            started = time.perf_counter()
            repeat_vector = stack.embedding.embed_query(question.query).values
            index_provider.search(
                repeat_vector, top_k=top_k, filter_privacy=_PUBLIC_ONLY
            )
            vector_trial_samples.append(time.perf_counter() - started)

        # QMD baseline (issue #24): one call per question, never repeated
        # for ``trials``. A failure here flips ``qmd_compared`` False for
        # the rest of the run and every already-collected qmd block is
        # discarded after the loop (D3: no partial comparison survives).
        qmd_block: dict | None = None
        if qmd_compared:
            try:
                started = time.perf_counter()
                qmd_result = run_qmd_query(
                    question.query, config=qmd_config, top_k=top_k
                )
                qmd_trial_samples.append(time.perf_counter() - started)
            except QmdUnavailable as exc:
                qmd_compared = False
                qmd_unavailable_reason = str(exc)
            else:
                qmd_hit = _hit(question.expected_paths, qmd_result.paths)
                if question.expected_paths:
                    qmd_scored += 1
                    qmd_hits += 1 if qmd_hit else 0
                # Codex Round 1 diagnostic: track whether QMD ever returned
                # anything at all, separately from whether any of it landed
                # inside the configured scope (``QMD_SCOPE_OVERLAP_NOTE``).
                if qmd_result.raw_paths:
                    qmd_questions_with_raw_hits += 1
                if qmd_result.paths:
                    qmd_questions_with_scoped_hits += 1
                # D1's token-cost threshold needs this number; reads real
                # bytes transiently and discards them (see the module-level
                # note above ``QMD_STALE_EXCLUSION_NOTE``).
                qmd_cost = qmd_token_cost(
                    qmd_result.paths, wiki_root=qmd_config.wiki_root
                )
                qmd_tokens_total += qmd_cost
                qmd_block = {
                    "paths": list(qmd_result.paths),
                    "hit": qmd_hit,
                    "result_count": len(qmd_result.paths),
                    "token_cost": qmd_cost,
                    "raw_result_count": len(qmd_result.raw_paths),
                }

        # Leaked paths are counted and then *redacted*: a violating path must
        # never travel onward inside the report it violated (R1 review).
        lexical_paths = tuple(
            path for path in raw_lexical_paths if path in public_paths
        )
        vector_paths = tuple(path for path in raw_vector_paths if path in public_paths)
        privacy_false_negatives += (len(raw_lexical_paths) - len(lexical_paths)) + (
            len(raw_vector_paths) - len(vector_paths)
        )
        # Duplicated rather than factored out of the pooled line above on
        # purpose: scripts/test-c5-mutations.sh M9 pins that exact pooled
        # assignment verbatim (a mutant silences leak counting by replacing
        # its whole right-hand side); refactoring it to share a temporary
        # with this provenance-split accumulation would require updating
        # that pinned string for a purely cosmetic reason, and the
        # coordinator's Round 2 review was explicit that a pinned guard
        # must never be touched except to make it *stronger*.
        provenance_tally["privacy_false_negatives"] += (
            len(raw_lexical_paths) - len(lexical_paths)
        ) + (len(raw_vector_paths) - len(vector_paths))

        # Citation correctness (§5.3): every lexical citation must bind the
        # path it ranks and the exact snapshot revision served; the vector
        # result must be bound to the rebuilt composed revision.
        citation_correct = (
            all(
                result.citation.concept_id == result.concept_id
                and result.citation.index_revision == catalog.index_revision
                for result in lexical.results
            )
            and vector.composed_revision == plan.composed_revision
        )
        if citation_correct:
            citation_correct_count += 1
            provenance_tally["citation_correct_questions"] += 1

        lexical_hit = _hit(question.expected_paths, lexical_paths)
        vector_hit = _hit(question.expected_paths, vector_paths)
        if question.expected_paths:
            scored += 1
            provenance_tally["scored_questions"] += 1
            lexical_hits += 1 if lexical_hit else 0
            vector_hits += 1 if vector_hit else 0
            provenance_tally["lexical_hits"] += 1 if lexical_hit else 0
            provenance_tally["vector_hits"] += 1 if vector_hit else 0
        elif question.category == "no-answer":
            provenance_tally["no_answer_total"] += 1
            if lexical_paths:
                lexical_no_answer_correct = False
                provenance_tally["no_answer_violations"] += 1

        # ``None`` means unmeasured (some returned path's body is not in
        # ``CORPUS_BODIES``, e.g. any real-provenance hit today) -- it must
        # never be added to a total as if it were 0 (issue #26 coordinator
        # review: a silent 0 here would understate the platform's real token
        # cost against a QMD baseline that *can* measure the real body).
        lexical_cost = _token_cost(lexical_paths)
        vector_cost = _token_cost(vector_paths)
        lexical_measured = lexical_cost is not None
        vector_measured = vector_cost is not None
        if lexical_measured:
            lexical_tokens_total += lexical_cost
            provenance_tally["lexical_token_cost_total"] += lexical_cost
        else:
            lexical_tokens_unmeasured += 1
            provenance_tally["lexical_token_cost_unmeasured_questions"] += 1
        if vector_measured:
            vector_tokens_total += vector_cost
            provenance_tally["vector_token_cost_total"] += vector_cost
        else:
            vector_tokens_unmeasured += 1
            provenance_tally["vector_token_cost_unmeasured_questions"] += 1

        lexical_stale_returned = any(path in SUPERSEDED_PATHS for path in lexical_paths)
        vector_stale_returned = any(path in SUPERSEDED_PATHS for path in vector_paths)
        if not lexical_stale_returned:
            lexical_stale_excluded += 1
            provenance_tally["lexical_stale_excluded"] += 1
        if not vector_stale_returned:
            vector_stale_excluded += 1
            provenance_tally["vector_stale_excluded"] += 1

        question_reports.append(
            {
                "question_id": question.question_id,
                "category": question.category,
                "query": question.query,
                "provenance": question.provenance,
                "expected_paths": list(question.expected_paths),
                "citation_correct": citation_correct,
                "lexical": {
                    "paths": list(lexical_paths),
                    "hit": lexical_hit,
                    "result_count": len(lexical_paths),
                    "token_cost": lexical_cost,
                    "token_cost_measured": lexical_measured,
                    "stale_returned": lexical_stale_returned,
                },
                "vector": {
                    "paths": list(vector_paths),
                    "hit": vector_hit,
                    "result_count": len(vector_paths),
                    "top_score": (vector.hits[0].score if vector.hits else None),
                    "token_cost": vector_cost,
                    "token_cost_measured": vector_measured,
                    "stale_returned": vector_stale_returned,
                },
                "qmd": qmd_block,
            }
        )

    total_questions = len(questions)

    # D3 "跑不了就整份失敗": a mid-run QMD failure must not leave earlier
    # questions carrying a qmd block while later ones carry None -- that
    # partial shape would look like "mostly compared" instead of the
    # honest "not compared" the acceptance criteria require. Once
    # ``qmd_compared`` is False for any reason (never configured, or failed
    # partway), every question's qmd block and every qmd summary number is
    # reset together.
    if not qmd_compared:
        for entry in question_reports:
            entry["qmd"] = None
        qmd_hits = 0
        qmd_scored = 0
        qmd_tokens_total = 0
        qmd_trial_samples = []
        qmd_questions_with_raw_hits = 0
        qmd_questions_with_scoped_hits = 0

    # Codex Round 1 diagnostic, computed once the run (or its cleanup
    # above) has settled: see ``QMD_SCOPE_OVERLAP_NOTE`` for the three-way
    # meaning. Deliberately *not* folded into ``qmd_hit_rate`` -- a scope
    # mismatch is a setup problem, not a retrieval-quality number, and the
    # two must stay distinguishable the same way privacy false positives
    # and false negatives are kept apart elsewhere in this report.
    if not qmd_compared or qmd_questions_with_raw_hits == 0:
        qmd_scope_overlap = None
    else:
        qmd_scope_overlap = qmd_questions_with_scoped_hits > 0

    return {
        "schema": REPORT_SCHEMA,
        "corpus_version": CORPUS_VERSION,
        "question_set_version": QUESTION_SET_VERSION,
        # D2 (Epic #21, issue #26): content types with zero real-corpus
        # coverage, always carried on every report -- not conditional on
        # what ``questions`` happens to contain -- so a report can never be
        # read as having verified these on real data by omission.
        "coverage_gaps": list(KNOWN_COVERAGE_GAPS),
        "index_provider": descriptor.provider_id,
        "composed_revision": plan.composed_revision,
        "bundle_index_revision": plan.bundle_index_revision,
        "embedding_revision": plan.embedding_revision,
        # QMD baseline (issue #24, D3): distinct from "lexical" above, which
        # stays what it always was -- the platform's own catalog search.
        # ``qmd_compared`` is the single field callers must check before
        # trusting any "qmd" question block or qmd_hit_rate below.
        "qmd_compared": qmd_compared,
        "qmd_index": qmd_config.index_name if qmd_config is not None else None,
        "qmd_unavailable_reason": qmd_unavailable_reason,
        # D2 known coverage gap (frozen, not a bug): explicit note instead
        # of an absent or zero-valued field, so #30's gate cannot mistake
        # "not applicable" for "measured and zero".
        "qmd_stale_exclusion_note": QMD_STALE_EXCLUSION_NOTE,
        "qmd_scope_overlap": qmd_scope_overlap,
        "qmd_scope_overlap_note": QMD_SCOPE_OVERLAP_NOTE,
        # Read from the descriptor, never hardcoded: a semantic provider
        # must not be reported as a hash baseline, or vice versa (R1 review).
        "semantic": stack.embedding.descriptor.semantic,
        "semantic_note": (
            "vector numbers come from a provider declaring semantic=false: "
            "a reproducible baseline, not semantic quality"
            if not stack.embedding.descriptor.semantic
            else "vector numbers come from a provider declaring semantic=true"
        ),
        "token_cost_method": TOKEN_COST_METHOD,
        "token_cost_note": TOKEN_COST_NOTE,
        "top_k": top_k,
        "rebuild": {
            "point_count": rebuild_report.point_count,
        },
        "questions": question_reports,
        "summary": {
            # Pooled across all of ``questions`` -- unchanged pre-#26 keys,
            # same computation as before. See the module docstring: these
            # must not be read alone once ``questions`` mixes provenances.
            "scored_questions": scored,
            "lexical_hit_rate": round(lexical_hits / scored, 4) if scored else None,
            "vector_hit_rate": round(vector_hits / scored, 4) if scored else None,
            "qmd_hit_rate": (
                round(qmd_hits / qmd_scored, 4) if qmd_compared and qmd_scored else None
            ),
            # Sums *measured* questions only -- a question whose token cost
            # is unmeasured (``None``, see ``_token_cost``) contributes
            # nothing here and is counted in the paired
            # ``*_unmeasured_questions`` field instead, so this total's
            # coverage is always stated, never assumed complete.
            "lexical_token_cost_total": lexical_tokens_total,
            "lexical_token_cost_unmeasured_questions": lexical_tokens_unmeasured,
            "vector_token_cost_total": vector_tokens_total,
            # D1: "context token 總量 不高於 QMD baseline" -- ``None`` (not
            # 0) when qmd_compared is False so a mid-run failure can never
            # leave a valid-looking number behind for #30 to gate against.
            "qmd_token_cost_total": qmd_tokens_total if qmd_compared else None,
            "vector_token_cost_unmeasured_questions": vector_tokens_unmeasured,
            "lexical_no_answer_correct": lexical_no_answer_correct,
            "citation_correct_questions": citation_correct_count,
            "privacy_false_negatives": privacy_false_negatives,
            "privacy_false_positives": len(fp_paths),
            "privacy_false_positive_paths": fp_paths,
            "lexical_stale_excluded": lexical_stale_excluded,
            "vector_stale_excluded": vector_stale_excluded,
            "lexical_stale_excluded_rate": round(
                lexical_stale_excluded / total_questions, 4
            ),
            "vector_stale_excluded_rate": round(
                vector_stale_excluded / total_questions, 4
            ),
            # Issue #26: the same hit-rate/token/citation figures, split by
            # each question's own ``provenance`` so a real note's score can
            # never hide behind a synthetic one's (or the reverse). Always
            # present, both keys, even when one provenance is empty in this
            # particular ``questions`` (its rates report ``None``, not 0 --
            # 0 would read as "answered and failed", not "not asked").
            "provenance": {
                "real": _provenance_summary(by_provenance["real"]),
                "synthetic": _provenance_summary(by_provenance["synthetic"]),
            },
        },
        "latency_ms": {
            "trials_per_question": trials,
            "lexical": _latency_block(lexical_trial_samples),
            "vector": _latency_block(vector_trial_samples),
            # Always 1 trial per question regardless of ``trials`` -- see
            # the qmd_config docstring above. ``None`` (not an empty block)
            # when qmd was never compared, so a caller cannot mistake "no
            # samples" for "measured and it was instant".
            "qmd": (_latency_block(qmd_trial_samples) if qmd_compared else None),
        },
    }


def strip_latency(report: dict) -> dict:
    """The reproducible view of a report: everything except wall-clock."""
    return {key: value for key, value in report.items() if key != "latency_ms"}


__all__ = [
    "DEFAULT_LATENCY_TRIALS",
    "REPORT_SCHEMA",
    "TOKEN_COST_METHOD",
    "TOKEN_COST_NOTE",
    "run_shadow_benchmark",
    "strip_latency",
]
