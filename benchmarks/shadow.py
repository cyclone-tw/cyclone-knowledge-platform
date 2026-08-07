"""The QMD shadow benchmark harness (contract §5.3): report only, no cutover.

Both engines answer the same frozen questions over the same privacy-gated
corpus. The report is honest about what it measures:

* the vector side runs the C4 hash embedding, which declares
  ``semantic=false`` -- these numbers are a deterministic baseline, not a
  claim of semantic quality, and Qdrant does not win by being newer;
* ``token_cost`` is a deterministic proxy (whitespace token count of each
  returned note's body), not a model tokenizer count -- ``token_cost_method``
  and ``token_cost_note`` say so explicitly at the top of the report, sourced
  from a single named constant rather than repeated string literals;
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
  ``members`` from ``CORPUS``, so the two agree today, but this assumption
  breaks the day a caller passes a real/synthetic mixed corpus (tracked as
  #26 / P5) -- at that point false-positive accounting needs to take its
  "declared public" set as an explicit input instead of importing it;
* everything except ``latency_ms`` is reproducible byte-for-byte from the
  same commit, and tests pin exactly that.

Privacy is zero-tolerance: a single non-public path in either engine's
output makes ``privacy_false_negatives`` non-zero, and the test suite treats
that as a hard failure (contract §5.3: false-negative exposure is
intolerable).
"""

from __future__ import annotations

import math
import time

from benchmarks.questions import (
    CORPUS_BODIES,
    CORPUS_VERSION,
    PUBLIC_DECLARED_PATHS,
    QUESTION_SET_VERSION,
    QUESTIONS,
    SUPERSEDED_PATHS,
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

#: Default per-question, per-engine repeat count for the latency sample.
DEFAULT_LATENCY_TRIALS = 5

_PUBLIC_ONLY = frozenset({PrivacyClass.PUBLIC})


def _token_cost(paths: tuple[str, ...]) -> int:
    return sum(len(CORPUS_BODIES.get(path, "").split()) for path in paths)


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
    trials: int = DEFAULT_LATENCY_TRIALS,
) -> dict:
    """Rebuild, query both engines, and assemble the deterministic report.

    ``trials`` controls how many times each question is re-run against each
    engine purely for the latency sample (default
    ``DEFAULT_LATENCY_TRIALS``); it does not change correctness, hit,
    token-cost, citation, or stale-exclusion accounting, which are all
    computed once per question from the first trial's results.

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
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")

    descriptor = require_index_provider(index_provider)
    plan = plan_rebuild(members=members, stack=stack, gate=gate)
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

    #: The only paths any engine may surface: what the gated plan indexed.
    #: Judging leaks against this set (not a fixture list) also catches a
    #: rogue provider inventing paths nobody has ever seen (R1 review).
    public_paths = frozenset(point.relative_path for point in plan.points)

    #: Privacy false positives: notes declared ``public`` in the corpus that
    #: never made it into the gated plan. A quality signal (over-blocking),
    #: never merged with the zero-tolerance leak count below (R1 intent).
    fp_paths = sorted(PUBLIC_DECLARED_PATHS - public_paths)

    questions = []
    lexical_hits = 0
    vector_hits = 0
    scored = 0
    lexical_tokens_total = 0
    vector_tokens_total = 0
    privacy_false_negatives = 0
    citation_correct_count = 0
    lexical_no_answer_correct = True
    lexical_stale_excluded = 0
    vector_stale_excluded = 0
    lexical_trial_samples: list[float] = []
    vector_trial_samples: list[float] = []

    for question in QUESTIONS:
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

        # Leaked paths are counted and then *redacted*: a violating path must
        # never travel onward inside the report it violated (R1 review).
        lexical_paths = tuple(
            path for path in raw_lexical_paths if path in public_paths
        )
        vector_paths = tuple(path for path in raw_vector_paths if path in public_paths)
        privacy_false_negatives += (len(raw_lexical_paths) - len(lexical_paths)) + (
            len(raw_vector_paths) - len(vector_paths)
        )

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

        lexical_hit = _hit(question.expected_paths, lexical_paths)
        vector_hit = _hit(question.expected_paths, vector_paths)
        if question.expected_paths:
            scored += 1
            lexical_hits += 1 if lexical_hit else 0
            vector_hits += 1 if vector_hit else 0
        elif question.category == "no-answer" and lexical_paths:
            lexical_no_answer_correct = False

        lexical_cost = _token_cost(lexical_paths)
        vector_cost = _token_cost(vector_paths)
        lexical_tokens_total += lexical_cost
        vector_tokens_total += vector_cost

        lexical_stale_returned = any(path in SUPERSEDED_PATHS for path in lexical_paths)
        vector_stale_returned = any(path in SUPERSEDED_PATHS for path in vector_paths)
        if not lexical_stale_returned:
            lexical_stale_excluded += 1
        if not vector_stale_returned:
            vector_stale_excluded += 1

        questions.append(
            {
                "question_id": question.question_id,
                "category": question.category,
                "query": question.query,
                "expected_paths": list(question.expected_paths),
                "citation_correct": citation_correct,
                "lexical": {
                    "paths": list(lexical_paths),
                    "hit": lexical_hit,
                    "result_count": len(lexical_paths),
                    "token_cost": lexical_cost,
                    "stale_returned": lexical_stale_returned,
                },
                "vector": {
                    "paths": list(vector_paths),
                    "hit": vector_hit,
                    "result_count": len(vector_paths),
                    "top_score": (vector.hits[0].score if vector.hits else None),
                    "token_cost": vector_cost,
                    "stale_returned": vector_stale_returned,
                },
            }
        )

    total_questions = len(QUESTIONS)

    return {
        "schema": REPORT_SCHEMA,
        "corpus_version": CORPUS_VERSION,
        "question_set_version": QUESTION_SET_VERSION,
        "index_provider": descriptor.provider_id,
        "composed_revision": plan.composed_revision,
        "bundle_index_revision": plan.bundle_index_revision,
        "embedding_revision": plan.embedding_revision,
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
        "questions": questions,
        "summary": {
            "scored_questions": scored,
            "lexical_hit_rate": round(lexical_hits / scored, 4) if scored else None,
            "vector_hit_rate": round(vector_hits / scored, 4) if scored else None,
            "lexical_token_cost_total": lexical_tokens_total,
            "vector_token_cost_total": vector_tokens_total,
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
        },
        "latency_ms": {
            "trials_per_question": trials,
            "lexical": _latency_block(lexical_trial_samples),
            "vector": _latency_block(vector_trial_samples),
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
