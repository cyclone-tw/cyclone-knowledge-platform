"""The P9 threshold gate (issue #30, Epic #21 D1 frozen): pass/fail scoring.

Every test up to the gated block at the bottom builds a synthetic
``benchmarks.shadow``-shaped report dict by hand (never a real
``run_shadow_benchmark`` call) so this whole suite runs in CI with no
Wiki checkout and no ``qmd`` binary -- ``evaluate_gate`` is a pure function
over a dict, and that is exactly what gets exercised here.

The gated block at the bottom mirrors ``tests/test_qmd_adapter.py``'s
``CKP_REQUIRE_QMD_BASELINE`` pattern exactly (itself mirroring
``tests/test_index_qdrant.py``'s ``CKP_REQUIRE_QDRANT``): skip locally when
the real Wiki checkout or the ``qmd`` binary are not available,
``CKP_REQUIRE_GATE_E2E=1`` turns that skip into a hard failure so the
end-to-end runner path can never silently stop running everywhere. CI never
sets the flag or ``CKP_PILOT_WIKI_ROOT`` -- there is no real, private Wiki
checkout in CI.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

import pytest
from benchmarks import gate
from benchmarks.qmd.adapter import qmd_binary_available

# --- A minimal, hand-built "ckp-shadow-report/3"-shaped report -------------


def _provenance_block(
    *,
    question_count: int,
    scored_questions: int,
    lexical_hits: int,
    vector_hits: int,
    lexical_token_cost_total: int,
    lexical_token_cost_unmeasured_questions: int,
    vector_token_cost_total: int,
    vector_token_cost_unmeasured_questions: int,
    citation_correct_questions: int,
    privacy_false_negatives: int,
    lexical_stale_excluded: int,
    vector_stale_excluded: int,
) -> dict[str, Any]:
    return {
        "question_count": question_count,
        "scored_questions": scored_questions,
        "lexical_hit_rate": (
            round(lexical_hits / scored_questions, 4) if scored_questions else None
        ),
        "vector_hit_rate": (
            round(vector_hits / scored_questions, 4) if scored_questions else None
        ),
        "lexical_token_cost_total": lexical_token_cost_total,
        "lexical_token_cost_unmeasured_questions": (
            lexical_token_cost_unmeasured_questions
        ),
        "vector_token_cost_total": vector_token_cost_total,
        "vector_token_cost_unmeasured_questions": (
            vector_token_cost_unmeasured_questions
        ),
        "citation_correct_questions": citation_correct_questions,
        "privacy_false_negatives": privacy_false_negatives,
        "lexical_stale_excluded": lexical_stale_excluded,
        "vector_stale_excluded": vector_stale_excluded,
        "lexical_stale_excluded_rate": (
            round(lexical_stale_excluded / question_count, 4)
            if question_count
            else None
        ),
        "vector_stale_excluded_rate": (
            round(vector_stale_excluded / question_count, 4) if question_count else None
        ),
        "lexical_no_answer_correct": None,
    }


def _real_question(
    question_id: str,
    *,
    hit: bool,
    qmd_hit: bool,
    qmd_cost: int | None,
    lexical_cost: int | None = 35,
) -> dict:
    return {
        "question_id": question_id,
        "category": "precise-note",
        "query": f"query for {question_id}",
        "provenance": "real",
        "expected_paths": [f"Core/{question_id}.md"],
        "citation_correct": True,
        "lexical": {
            "paths": [f"Core/{question_id}.md"] if hit else [],
            "hit": hit,
            "result_count": 1 if hit else 0,
            "token_cost": lexical_cost,
            "token_cost_measured": lexical_cost is not None,
            "stale_returned": False,
        },
        "vector": {
            "paths": [f"Core/{question_id}.md"] if hit else [],
            "hit": hit,
            "result_count": 1 if hit else 0,
            "top_score": 0.9,
            "token_cost": 40,
            "token_cost_measured": True,
            "stale_returned": False,
        },
        "qmd": {
            "paths": [f"Core/{question_id}.md"] if qmd_hit else [],
            "hit": qmd_hit,
            "result_count": 1 if qmd_hit else 0,
            "token_cost": qmd_cost,
            "raw_result_count": 1 if qmd_hit else 0,
        },
    }


def base_report(
    *,
    qmd_compared: bool = True,
    qmd_scope_overlap: bool | None = True,
    real_lexical_hits: int = 2,
    real_vector_hits: int = 2,
    real_qmd_hits: int = 2,
    real_lexical_token_cost_total: int = 70,
    real_lexical_token_cost_unmeasured: int = 0,
    real_qmd_token_cost_total: int = 80,
    real_qmd_token_cost_unmeasured: int = 0,
    real_privacy_false_negatives: int = 0,
    real_citation_correct: int = 2,
    lexical_p95: float = 100.0,
    vector_p95: float = 90.0,
    qmd_p95: float = 60.0,
    semantic: bool = False,
) -> dict[str, Any]:
    """A report that passes every dimension it is possible to pass.

    Two real questions, both scored, both hit by lexical/vector/QMD by
    default, citation-correct, no privacy leak, token totals and latency
    comfortably inside the D1 thresholds. Every test below mutates exactly
    the fields it needs to flip one dimension, starting from this baseline,
    so a failing assertion always points at the one thing the test changed.
    """
    real_questions = [
        _real_question(
            "r01",
            hit=real_lexical_hits >= 1,
            qmd_hit=real_qmd_hits >= 1,
            qmd_cost=(
                None
                if real_qmd_token_cost_unmeasured >= 1
                else real_qmd_token_cost_total // 2
            ),
            lexical_cost=(
                None
                if real_lexical_token_cost_unmeasured >= 1
                else real_lexical_token_cost_total // 2
            ),
        ),
        _real_question(
            "r02",
            hit=real_lexical_hits >= 2,
            qmd_hit=real_qmd_hits >= 2,
            qmd_cost=(
                None
                if real_qmd_token_cost_unmeasured >= 2
                else real_qmd_token_cost_total - real_qmd_token_cost_total // 2
            ),
            lexical_cost=(
                None
                if real_lexical_token_cost_unmeasured >= 2
                else real_lexical_token_cost_total - real_lexical_token_cost_total // 2
            ),
        ),
    ]
    if not qmd_compared:
        for question in real_questions:
            question["qmd"] = None

    real_summary = _provenance_block(
        question_count=2,
        scored_questions=2,
        lexical_hits=real_lexical_hits,
        vector_hits=real_vector_hits,
        lexical_token_cost_total=real_lexical_token_cost_total,
        lexical_token_cost_unmeasured_questions=real_lexical_token_cost_unmeasured,
        vector_token_cost_total=60,
        vector_token_cost_unmeasured_questions=0,
        citation_correct_questions=real_citation_correct,
        privacy_false_negatives=real_privacy_false_negatives,
        lexical_stale_excluded=2,
        vector_stale_excluded=2,
    )
    synthetic_summary = _provenance_block(
        question_count=1,
        scored_questions=1,
        lexical_hits=1,
        vector_hits=1,
        lexical_token_cost_total=10,
        lexical_token_cost_unmeasured_questions=0,
        vector_token_cost_total=10,
        vector_token_cost_unmeasured_questions=0,
        citation_correct_questions=1,
        privacy_false_negatives=0,
        lexical_stale_excluded=1,
        vector_stale_excluded=1,
    )

    return {
        "schema": "ckp-shadow-report/3",
        "corpus_version": "3",
        "question_set_version": "2",
        "coverage_gaps": ["course-module", "coffee-log", "publication"],
        "index_provider": "in-memory",
        "composed_revision": "sha256:composed",
        "bundle_index_revision": "sha256:bundle",
        "embedding_revision": "sha256:embedding",
        "qmd_compared": qmd_compared,
        "qmd_index": "cyclone-wiki" if qmd_compared else None,
        "qmd_unavailable_reason": None if qmd_compared else "qmd exited 1",
        "qmd_stale_exclusion_note": (
            "QMD side has no stale-exclusion metric: it depends on the "
            "corpus's `superseded` marker, and the real pilot corpus (Epic "
            "#21 D2) carries no superseded relationship to measure -- a "
            "known coverage gap recorded in D2, not a bug in this report."
        ),
        "qmd_scope_overlap": qmd_scope_overlap if qmd_compared else None,
        "qmd_scope_overlap_note": "see benchmarks.shadow docstring",
        "semantic": semantic,
        "semantic_note": (
            "vector numbers come from a provider declaring semantic=false: "
            "a reproducible baseline, not semantic quality"
            if not semantic
            else "vector numbers come from a provider declaring semantic=true"
        ),
        "token_cost_method": "whitespace-proxy",
        "token_cost_note": "token counts are a whitespace-split proxy",
        "top_k": 5,
        "rebuild": {"point_count": 3},
        "questions": [
            *real_questions,
            {
                "question_id": "q01",
                "category": "precise-note",
                "query": "synthetic query",
                "provenance": "synthetic",
                "expected_paths": ["espresso-dial-log.md"],
                "citation_correct": True,
                "lexical": {
                    "paths": ["espresso-dial-log.md"],
                    "hit": True,
                    "result_count": 1,
                    "token_cost": 10,
                    "token_cost_measured": True,
                    "stale_returned": False,
                },
                "vector": {
                    "paths": ["espresso-dial-log.md"],
                    "hit": True,
                    "result_count": 1,
                    "top_score": 0.9,
                    "token_cost": 10,
                    "token_cost_measured": True,
                    "stale_returned": False,
                },
                "qmd": None,
            },
        ],
        "summary": {
            "scored_questions": 3,
            "lexical_hit_rate": 1.0,
            "vector_hit_rate": 1.0,
            "qmd_hit_rate": (round(real_qmd_hits / 2, 4) if qmd_compared else None),
            "lexical_token_cost_total": real_lexical_token_cost_total + 10,
            "lexical_token_cost_unmeasured_questions": (
                real_lexical_token_cost_unmeasured
            ),
            "vector_token_cost_total": 70,
            "qmd_token_cost_total": (
                real_qmd_token_cost_total if qmd_compared else None
            ),
            "qmd_token_cost_unmeasured_questions": (
                real_qmd_token_cost_unmeasured if qmd_compared else None
            ),
            "vector_token_cost_unmeasured_questions": 0,
            "lexical_no_answer_correct": True,
            "citation_correct_questions": real_citation_correct + 1,
            "privacy_false_negatives": real_privacy_false_negatives,
            "privacy_false_positives": 0,
            "privacy_false_positive_paths": [],
            "lexical_stale_excluded": 3,
            "vector_stale_excluded": 3,
            "lexical_stale_excluded_rate": 1.0,
            "vector_stale_excluded_rate": 1.0,
            "provenance": {"real": real_summary, "synthetic": synthetic_summary},
        },
        "latency_ms": {
            "trials_per_question": 1,
            "lexical": {
                "total": lexical_p95,
                "p50": lexical_p95,
                "p95": lexical_p95,
            },
            "vector": {"total": vector_p95, "p50": vector_p95, "p95": vector_p95},
            "qmd": (
                {"total": qmd_p95, "p50": qmd_p95, "p95": qmd_p95}
                if qmd_compared
                else None
            ),
        },
    }


def _item(result: dict, dimension: str) -> dict:
    (found,) = (item for item in result["items"] if item["dimension"] == dimension)
    return found


# --- Baseline sanity ---------------------------------------------------


def test_base_report_passes_every_evaluable_dimension() -> None:
    result = gate.evaluate_gate(base_report())
    assert result["verdict"] == "pass"
    for item in result["items"]:
        assert item["status"] in {"pass", "not_evaluable", "recorded"}
    # stale_exclusion and citation-adjacent structural gaps still surface
    # as not_evaluable even on an otherwise perfect report (D2).
    assert result["not_evaluable_dimensions"] == ["stale_exclusion"]


def test_coverage_gaps_are_carried_through_verbatim() -> None:
    report = base_report()
    result = gate.evaluate_gate(report)
    assert result["coverage_gaps"] == report["coverage_gaps"]
    assert result["coverage_gaps"] == ["course-module", "coffee-log", "publication"]


def test_coverage_gaps_removed_from_report_is_visible_in_the_gate_result() -> None:
    """Mutation check: 'remove coverage gaps from the result' must go red."""
    report = base_report()
    report["coverage_gaps"] = []
    result = gate.evaluate_gate(report)
    assert result["coverage_gaps"] == []


# --- RP2: privacy false-negative is zero-tolerance, not a soft threshold --


def test_privacy_false_negative_zero_passes() -> None:
    result = gate.evaluate_gate(base_report(real_privacy_false_negatives=0))
    assert _item(result, "privacy_false_negative")["status"] == "pass"
    assert result["verdict"] == "pass"


def test_privacy_false_negative_of_one_is_a_hard_failure() -> None:
    """Mutation check: '== 0 changed to <= 1' must go red against this."""
    result = gate.evaluate_gate(base_report(real_privacy_false_negatives=1))
    item = _item(result, "privacy_false_negative")
    assert item["status"] == "fail"
    assert result["verdict"] == "fail"
    assert "privacy_false_negative" in result["failing_dimensions"]


def test_privacy_false_negative_gate_reads_the_real_group_not_pooled() -> None:
    report = base_report(real_privacy_false_negatives=1)
    # Pooled number in this fixture only reflects what the caller set on the
    # real provenance block above (there is no separate pooled override) --
    # assert the gate item's own detail names the real-group count, proving
    # it did not fall back to some other source.
    result = gate.evaluate_gate(report)
    item = _item(result, "privacy_false_negative")
    assert item["detail"]["real_false_negatives"] == 1


# --- D3: no QMD baseline means no verdict, not a free pass -----------------


def test_qmd_not_compared_fails_the_whole_gate_even_with_perfect_numbers() -> None:
    """Mutation check: 'qmd_compared: false stops affecting the verdict'
    must go red against this -- every other dimension here is engineered to
    pass, and the gate must still fail overall.
    """
    report = base_report(qmd_compared=False)
    result = gate.evaluate_gate(report)
    assert _item(result, "qmd_compared")["status"] == "fail"
    assert result["verdict"] == "fail"
    assert "qmd_compared" in result["failing_dimensions"]


def test_qmd_not_compared_makes_every_relative_dimension_not_evaluable() -> None:
    report = base_report(qmd_compared=False)
    result = gate.evaluate_gate(report)
    for dimension in (
        "answerable_hit_rate",
        "context_token_total",
        "latency_p95",
    ):
        assert _item(result, dimension)["status"] == "not_evaluable", dimension


def test_qmd_scope_overlap_false_fails_even_with_perfect_numbers() -> None:
    report = base_report(qmd_scope_overlap=False)
    result = gate.evaluate_gate(report)
    assert _item(result, "qmd_scope_overlap")["status"] == "fail"
    assert result["verdict"] == "fail"


def test_qmd_scope_overlap_none_or_true_does_not_fail() -> None:
    for overlap in (None, True):
        result = gate.evaluate_gate(base_report(qmd_scope_overlap=overlap))
        assert _item(result, "qmd_scope_overlap")["status"] == "pass"


# --- D1 thresholds apply to the real group, never the synthetic one --------


def test_real_group_failure_fails_the_gate_even_with_a_perfect_synthetic_group() -> (
    None
):
    """Mutation check: 'apply thresholds to the synthetic group' must go
    red -- this report's *synthetic* numbers (hardcoded in ``base_report``)
    are perfect throughout, and only the real group is made to fail.
    """
    report = base_report(real_lexical_hits=0)  # real lexical hit rate -> 0.0
    result = gate.evaluate_gate(report)
    assert _item(result, "answerable_hit_rate")["status"] == "fail"
    assert result["verdict"] == "fail"


def test_real_group_success_is_not_dragged_down_by_a_bad_synthetic_group() -> None:
    """The converse of the above: a mutant that pools real+synthetic (or
    reads only the synthetic group) could still coincidentally pass the
    first test above while failing this one for the wrong reason, so both
    directions are checked.
    """
    report = base_report()
    report["summary"]["provenance"]["synthetic"]["lexical_hit_rate"] = 0.0
    report["summary"]["provenance"]["synthetic"]["vector_hit_rate"] = 0.0
    result = gate.evaluate_gate(report)
    assert _item(result, "answerable_hit_rate")["status"] == "pass"
    assert result["verdict"] == "pass"


# --- not_evaluable is a real third state, never silently "pass" ------------


def test_no_scored_questions_is_not_evaluable_not_pass() -> None:
    """Round 1 made the absolute dimensions pooled; the empty-run guard moves
    with them -- a run that scored nothing has no citations to be correct."""
    report = base_report()
    report["summary"]["provenance"]["real"]["question_count"] = 0
    report["summary"]["provenance"]["synthetic"]["question_count"] = 0
    report["summary"]["citation_correct_questions"] = 0
    result = gate.evaluate_gate(report)
    item = _item(result, "citation_correctness")
    assert item["status"] == "not_evaluable"
    assert "citation_correctness" in result["not_evaluable_dimensions"]


def test_synthetic_citation_error_fails_the_absolute_gate() -> None:
    """Codex round 1: absolutes are run-wide. A citation bound to the wrong
    commit on a synthetic question is the same engine bug as on a real one;
    real-group scoping governs relative thresholds only."""
    report = base_report()
    report["summary"]["citation_correct_questions"] -= 1
    result = gate.evaluate_gate(report)
    assert _item(result, "citation_correctness")["status"] == "fail"


def test_synthetic_privacy_leak_fails_the_absolute_gate() -> None:
    """RP2 zero tolerance is run-wide: a leak on a synthetic question is
    still a leak by the engine under test."""
    report = base_report()
    report["summary"]["privacy_false_negatives"] = 1
    result = gate.evaluate_gate(report)
    assert _item(result, "privacy_false_negative")["status"] == "fail"


def test_stale_exclusion_is_always_not_evaluable_never_silently_pass() -> None:
    result = gate.evaluate_gate(base_report())
    item = _item(result, "stale_exclusion")
    assert item["status"] == "not_evaluable"
    assert item["reason"]


def test_citation_correctness_absolute_threshold_is_100_percent() -> None:
    report = base_report(real_citation_correct=1)  # 1 of 2, not 100%
    result = gate.evaluate_gate(report)
    assert _item(result, "citation_correctness")["status"] == "fail"
    assert result["verdict"] == "fail"


# --- token totals: a partial total must never be compared ------------------


def test_unmeasured_platform_tokens_makes_the_gate_not_evaluable() -> None:
    report = base_report(real_lexical_token_cost_unmeasured=1)
    result = gate.evaluate_gate(report)
    item = _item(result, "context_token_total")
    assert item["status"] == "not_evaluable"
    assert item["status"] != "pass"


def test_unmeasured_qmd_tokens_makes_the_gate_not_evaluable() -> None:
    report = base_report(real_qmd_token_cost_unmeasured=1)
    result = gate.evaluate_gate(report)
    item = _item(result, "context_token_total")
    assert item["status"] == "not_evaluable"


def test_fully_measured_token_totals_compare_normally() -> None:
    passing = gate.evaluate_gate(
        base_report(real_lexical_token_cost_total=50, real_qmd_token_cost_total=80)
    )
    assert _item(passing, "context_token_total")["status"] == "pass"

    failing = gate.evaluate_gate(
        base_report(real_lexical_token_cost_total=120, real_qmd_token_cost_total=80)
    )
    assert _item(failing, "context_token_total")["status"] == "fail"


def test_token_comparison_reads_only_the_qmd_hit_subset() -> None:
    """Epic #21 D1 clarification (#62): same-work comparison.

    QMD misses r02; the platform answers it and pays 85 there (120 pooled
    vs QMD's pooled 80, so the pre-#62 pooled rule would fail). On the
    same-work subset -- r01 alone -- the platform pays 35 against QMD's
    40 and passes. A mutation that folds QMD-missed questions back into
    either total (the pooled rule) flips this test red; that is the
    mutation this test exists to catch.
    """
    report = base_report(
        real_qmd_hits=1,
        real_lexical_token_cost_total=120,
        real_qmd_token_cost_total=80,
    )
    report["questions"][0]["lexical"]["token_cost"] = 35
    report["questions"][1]["lexical"]["token_cost"] = 85

    result = gate.evaluate_gate(report)
    item = _item(result, "context_token_total")

    assert item["status"] == "pass"
    assert item["detail"]["compared_question_ids"] == ["r01"]
    assert item["detail"]["platform_lexical_token_cost_total_on_subset"] == 35
    assert item["detail"]["qmd_token_cost_total_on_subset"] == 40
    assert (
        item["detail"]["pooled_platform_lexical_token_cost_total_informational"] == 120
    )
    assert item["detail"]["pooled_qmd_token_cost_total_informational"] == 80


def test_token_comparison_still_fails_when_the_subset_itself_is_worse() -> None:
    """Same-work is a clarification, not a loosening: pay more than QMD on
    the questions QMD itself answered and the dimension still fails."""
    report = base_report(
        real_qmd_hits=1,
        real_lexical_token_cost_total=120,
        real_qmd_token_cost_total=40,
    )
    report["questions"][0]["lexical"]["token_cost"] = 90
    report["questions"][1]["lexical"]["token_cost"] = 30

    result = gate.evaluate_gate(report)
    item = _item(result, "context_token_total")

    assert item["status"] == "fail"
    assert "same-work subset" in item["reason"]


def test_unmeasured_tokens_outside_the_subset_do_not_block_evaluation() -> None:
    """The residual-total rule now scopes to the compared subset: an
    unmeasured platform cost on a question QMD missed is reported in the
    pooled informational numbers but must not void the same-work verdict
    (under the pooled rule it did -- that question was never comparable
    work in the first place)."""
    report = base_report(real_qmd_hits=1)
    report["questions"][1]["lexical"]["token_cost"] = None
    report["questions"][1]["lexical"]["token_cost_measured"] = False
    real_summary = report["summary"]["provenance"]["real"]
    real_summary["lexical_token_cost_total"] = 35
    real_summary["lexical_token_cost_unmeasured_questions"] = 1

    result = gate.evaluate_gate(report)
    item = _item(result, "context_token_total")

    assert item["status"] == "pass"
    assert (
        item["detail"][
            "pooled_platform_lexical_token_cost_unmeasured_questions_informational"
        ]
        == 1
    )


def test_token_totals_equal_on_the_subset_pass() -> None:
    """D1 says "not higher than" -- equality passes. Pins the ``<=`` so a
    strictness mutation (``<``) cannot quietly demand the platform beat
    the baseline it is only required to match."""
    report = base_report(real_lexical_token_cost_total=80, real_qmd_token_cost_total=80)
    result = gate.evaluate_gate(report)
    assert _item(result, "context_token_total")["status"] == "pass"


def test_no_qmd_baseline_reports_absent_costs_as_none_not_zero() -> None:
    """Codex round 1 on #65: with ``qmd_compared`` False there is no QMD
    baseline anywhere, and ``_real_qmd_stats``'s zero-initialized totals
    must not surface as "the baseline cost 0" -- absent evidence is
    reported as absent (None), never folded into a number."""
    report = base_report(qmd_compared=False)
    result = gate.evaluate_gate(report)
    item = _item(result, "context_token_total")

    assert item["status"] == "not_evaluable"
    detail = item["detail"]
    assert detail["pooled_qmd_token_cost_total_informational"] is None
    assert detail["pooled_qmd_token_cost_unmeasured_questions_informational"] is None
    assert detail["compared_question_ids"] == []
    assert detail["platform_lexical_token_cost_total_on_subset"] is None
    assert detail["qmd_token_cost_total_on_subset"] is None


def test_empty_same_work_subset_never_passes() -> None:
    """Defense in depth below evaluate_gate: the #58 degenerate-baseline
    rule intercepts a zero-hit QMD before this dimension runs, so the
    empty-subset branch is unreachable through evaluate_gate by
    construction -- but the function must stay honest when called
    directly (an empty subset compares nothing; 0 <= 0 must not pass)."""
    report = base_report(real_qmd_hits=0)
    item = gate._gate_context_token_total(report, gate._real_qmd_stats(report))

    assert item["status"] == "not_evaluable"
    assert "same-work subset is empty" in item["reason"]
    # An empty subset's totals are "nothing to total", never 0 -- while the
    # pooled numbers stay numeric here, because QMD did run and pay.
    assert item["detail"]["platform_lexical_token_cost_total_on_subset"] is None
    assert item["detail"]["qmd_token_cost_total_on_subset"] is None
    assert item["detail"]["pooled_qmd_token_cost_total_informational"] is not None


# --- RP3: semantic=false vector must never decide a gate outcome -----------


def test_semantic_false_vector_advantage_cannot_rescue_a_failing_lexical_rate() -> None:
    """Mutation check: letting ``vector_hit_rate`` participate when
    ``semantic=false`` must go red -- lexical is engineered to miss both
    real questions (0.0 hit rate, below any QMD baseline > 0), vector is
    engineered to hit both (1.0, comfortably above), and the gate must still
    fail because RP3 forbids the hash baseline from deciding anything.
    """
    report = base_report(real_lexical_hits=0, real_vector_hits=2, semantic=False)
    result = gate.evaluate_gate(report)
    item = _item(result, "answerable_hit_rate")
    assert item["status"] == "fail"
    assert item["detail"]["decided_on"] == "lexical"
    assert item["detail"]["platform_vector_hit_rate_informational"] == 1.0


def test_semantic_true_still_does_not_let_vector_decide() -> None:
    """This gate deliberately never promotes vector to decisive even when
    ``semantic=true`` (module docstring: that would be a new, reviewed
    decision, not a default) -- pin that today's behavior does not
    accidentally start honoring ``semantic=true`` without an explicit
    change.
    """
    report = base_report(real_lexical_hits=0, real_vector_hits=2, semantic=True)
    result = gate.evaluate_gate(report)
    assert _item(result, "answerable_hit_rate")["status"] == "fail"


def test_vector_never_appears_as_the_decision_reason_string() -> None:
    report = base_report(real_lexical_hits=0, real_vector_hits=2)
    result = gate.evaluate_gate(report)
    item = _item(result, "answerable_hit_rate")
    assert "vector" not in (item["reason"] or "")


# --- latency: pooled, both engines checked, 2x QMD baseline ----------------


def test_latency_within_2x_qmd_baseline_passes() -> None:
    result = gate.evaluate_gate(
        base_report(lexical_p95=100.0, vector_p95=90.0, qmd_p95=60.0)
    )
    assert _item(result, "latency_p95")["status"] == "pass"


def test_latency_over_2x_qmd_baseline_fails() -> None:
    result = gate.evaluate_gate(
        base_report(lexical_p95=200.0, vector_p95=90.0, qmd_p95=60.0)
    )
    item = _item(result, "latency_p95")
    assert item["status"] == "fail"
    assert "lexical" in item["reason"]


def test_latency_verdict_reads_lexical_only_vector_is_informational() -> None:
    """RP3 (Codex round 1): a slow vector engine must not fail the gate --
    under the hash baseline it decides nothing. The number stays visible in
    detail; the verdict comes from lexical alone."""
    result = gate.evaluate_gate(
        base_report(lexical_p95=100.0, vector_p95=200.0, qmd_p95=60.0)
    )
    item = _item(result, "latency_p95")
    assert item["status"] == "pass"
    assert item["detail"]["platform_vector_p95_ms"] == 200.0

    slow_lexical = gate.evaluate_gate(
        base_report(lexical_p95=200.0, vector_p95=50.0, qmd_p95=60.0)
    )
    assert _item(slow_lexical, "latency_p95")["status"] == "fail"


def test_latency_not_evaluable_when_qmd_not_compared() -> None:
    result = gate.evaluate_gate(base_report(qmd_compared=False))
    assert _item(result, "latency_p95")["status"] == "not_evaluable"


# --- privacy false-positive: recorded, never gated --------------------------


def test_privacy_false_positive_is_recorded_never_pass_or_fail() -> None:
    report = base_report()
    report["summary"]["privacy_false_positives"] = 7
    report["summary"]["privacy_false_positive_paths"] = ["a.md", "b.md"]
    result = gate.evaluate_gate(report)
    item = _item(result, "privacy_false_positive")
    assert item["status"] == "recorded"
    assert item["detail"]["count"] == 7
    # A large false-positive count must never fail the overall verdict.
    assert result["verdict"] == "pass"


# --- Wiring: the runner really connects the gate result to the exit code ---


def test_main_returns_zero_on_pass_and_prints_the_verdict_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_report = base_report()
    fake_result = gate.evaluate_gate(fake_report)
    monkeypatch.setattr(gate, "load_config", lambda: object())
    monkeypatch.setattr(
        gate,
        "run_gate",
        lambda *, config, qmd_binary: (fake_report, fake_result),
    )
    code = gate.main([])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == fake_result


def test_main_returns_one_on_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_report = base_report(qmd_compared=False)
    fake_result = gate.evaluate_gate(fake_report)
    assert fake_result["verdict"] == "fail"
    monkeypatch.setattr(gate, "load_config", lambda: object())
    monkeypatch.setattr(
        gate,
        "run_gate",
        lambda *, config, qmd_binary: (fake_report, fake_result),
    )
    code = gate.main([])
    assert code == 1


def test_main_reads_ckp_qmd_binary_env_var(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    def _fake_run_gate(*, config, qmd_binary):
        seen["qmd_binary"] = qmd_binary
        report = base_report()
        return report, gate.evaluate_gate(report)

    monkeypatch.setenv("CKP_QMD_BINARY", "/opt/custom/qmd")
    monkeypatch.setattr(gate, "load_config", lambda: object())
    monkeypatch.setattr(gate, "run_gate", _fake_run_gate)
    gate.main([])
    assert seen["qmd_binary"] == "/opt/custom/qmd"


def test_main_module_entry_point_is_wired(capsys: pytest.CaptureFixture[str]) -> None:
    """A cheap, independent second signal (AGENTS.md §9 Q3): ``python -m
    benchmarks.gate`` must resolve to this module's own ``main``, not a
    stale or shadowed copy.
    """
    assert gate.__name__ == "benchmarks.gate"
    assert hasattr(gate, "main")


# --- deep-copy sanity: evaluate_gate must not mutate its input -------------


def test_evaluate_gate_does_not_mutate_the_input_report() -> None:
    report = base_report()
    original = deepcopy(report)
    gate.evaluate_gate(report)
    assert report == original


# --- gated real end-to-end runner (CKP_REQUIRE_GATE_E2E) --------------------

_QMD_BINARY = os.environ.get("CKP_QMD_BINARY", "qmd")
_PILOT_ROOT = os.environ.get("CKP_PILOT_WIKI_ROOT")
_REQUIRED = os.environ.get("CKP_REQUIRE_GATE_E2E") == "1"


def _real_environment_ready() -> bool:
    return qmd_binary_available(_QMD_BINARY) and bool(_PILOT_ROOT)


_ready = _real_environment_ready()
if _REQUIRED and not _ready:
    pytest.fail(
        "CKP_REQUIRE_GATE_E2E=1 but qmd is not on PATH or CKP_PILOT_WIKI_ROOT is unset"
    )

needs_real_gate = pytest.mark.skipif(
    not _ready,
    reason="qmd binary or CKP_PILOT_WIKI_ROOT not available locally",
)


@needs_real_gate
def test_real_end_to_end_gate_produces_a_structurally_sound_verdict() -> None:
    """Manual/local-only smoke -- never runs in CI (no real Wiki checkout
    there, AGENTS.md forbids wiring CI to it). Exercises the actual
    acceptance scenario: real pilot corpus, real QMD subprocess, real
    in-memory index rebuild, D1 gate applied to the result.

    Deliberately does not assert a fixed pass/fail verdict or a byte-for-
    byte JSON shape -- the real Wiki checkout's content and QMD's own
    behavior are outside this repo's control, and pinning the verdict here
    would make this test a coin flip on Wiki content drift rather than a
    check that the wiring works. What is pinned: the shape every synthetic
    test above already exercises stays true against a real run too.
    """
    from ckp.config import load_config

    # Explicit env, not ``load_config()``'s real-``os.environ`` default:
    # this test file's own ``CKP_REQUIRE_GATE_E2E`` harness switch (read
    # directly above, never through config layering, same pattern as
    # ``CKP_REQUIRE_QDRANT``/``CKP_REQUIRE_QMD_BASELINE``) is not a known
    # config key, and ``load_config()`` fails closed on any unrecognized
    # ``CKP_*`` name it finds in the real environment.
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": _PILOT_ROOT})
    report, result = gate.run_gate(config=config, qmd_binary=_QMD_BINARY)

    assert result["schema"] == gate.GATE_SCHEMA
    assert result["verdict"] in {"pass", "fail"}
    assert result["not_evaluable_dimensions"]  # stale_exclusion, always (D2)
    assert "stale_exclusion" in result["not_evaluable_dimensions"]
    assert result["coverage_gaps"] == list(report["coverage_gaps"])
    dimensions = {item["dimension"] for item in result["items"]}
    assert dimensions == {
        "citation_correctness",
        "privacy_false_negative",
        "qmd_compared",
        "qmd_scope_overlap",
        "answerable_hit_rate",
        "context_token_total",
        "latency_p95",
        "stale_exclusion",
        "privacy_false_positive",
    }


def test_degenerate_qmd_baseline_fails_and_voids_every_relative_dimension():
    """#58: QMD compared, ran fine, and hit nothing on any scored real
    question. Hit rate would pass against 0.0 exactly as trivially as the
    token total fails against a measured 0 -- meaningless both ways. All
    four relative dimensions go not_evaluable and the gate fails with
    degenerate_qmd_baseline, mirroring D3's absent-baseline rule."""
    report = base_report()
    for question in report["questions"]:
        if question["provenance"] == "real" and question.get("qmd") is not None:
            question["qmd"]["hit"] = False
            question["qmd"]["paths"] = []
            question["qmd"]["token_cost"] = 0
    result = gate.evaluate_gate(report)
    assert result["verdict"] == "fail"
    assert "degenerate_qmd_baseline" in result["failing_dimensions"]
    voided = {
        item["dimension"]
        for item in result["items"]
        if item["status"] == "not_evaluable"
    }
    assert {
        "answerable_hit_rate",
        "context_token_total",
        "latency_p95",
        "stale_exclusion",
    } <= voided
    reasons = [
        item["reason"]
        for item in result["items"]
        if item["status"] == "not_evaluable" and item["dimension"] != "stale_exclusion"
    ]
    assert all("degenerate" in reason for reason in reasons)


def test_gate_env_switches_are_reserved_in_config():
    """Codex round 1 on #30: `main()` calls `load_config()` over the real
    environment before reading `CKP_QMD_BINARY` -- the exact #46 landmine
    reborn in new code. Both gate harness switches must be reserved, or a
    caller exporting them cannot run the runner at all."""
    from ckp.config import RESERVED_ENV, load_config

    assert "CKP_QMD_BINARY" in RESERVED_ENV
    assert "CKP_REQUIRE_GATE_E2E" in RESERVED_ENV
    # The behavioural half: loading config with the switch set must not raise.
    load_config(env={"CKP_QMD_BINARY": "/opt/custom/qmd"})
    load_config(env={"CKP_REQUIRE_GATE_E2E": "1"})


def test_citation_denominator_is_every_question_not_only_scored():
    """The live run that caught this: 16 questions, 13 scored, 16 correct →
    "-3/13 incorrect" under a scored-only denominator. Correctness is tallied
    over every question, so the denominator must be too."""
    report = base_report()
    real_n = report["summary"]["provenance"]["real"]["question_count"]
    syn_n = report["summary"]["provenance"]["synthetic"]["question_count"]
    report["summary"]["scored_questions"] = real_n + syn_n - 1  # one no-answer
    report["summary"]["citation_correct_questions"] = real_n + syn_n
    result = gate.evaluate_gate(report)
    item = _item(result, "citation_correctness")
    assert item["status"] == "pass", item
    assert item["detail"]["total_questions"] == real_n + syn_n
