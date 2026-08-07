"""The §5.3 shadow benchmark: determinism, honesty, and zero privacy leaks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.questions import (
    NON_PUBLIC_PATHS,
    PUBLIC_DECLARED_PATHS,
    QUESTIONS,
    SUPERSEDED_PATHS,
)
from benchmarks.shadow import (
    REPORT_SCHEMA,
    TOKEN_COST_METHOD,
    TOKEN_COST_NOTE,
    run_shadow_benchmark,
    strip_latency,
)

from ckp.index import (
    INDEX_CONTRACT,
    IndexDescriptor,
    InMemoryVectorIndex,
    RebuildReport,
    SearchHit,
    SearchResult,
)
from ckp.index.revision import INDEX_SCHEMA_VERSION
from ckp.privacy import FrontmatterClassifier
from ckp.privacy.gate import PrivacyGate
from index_fixtures import (
    UNUSED_ROOT,
    corpus_members,
    deterministic_stack,
    public_gate,
)


def _run(index_provider=None, top_k: int = 3, trials: int = 1) -> dict:
    return run_shadow_benchmark(
        members=corpus_members(),
        stack=deterministic_stack(),
        index_provider=index_provider or InMemoryVectorIndex(),
        gate=public_gate(),
        top_k=top_k,
        trials=trials,
    )


def test_question_set_covers_the_contract_categories() -> None:
    assert len(QUESTIONS) >= 8
    categories = {question.category for question in QUESTIONS}
    assert {
        "precise-note",
        "cross-note",
        "latest-status",
        "privacy-refusal",
        "no-answer",
    } <= categories
    identifiers = [question.question_id for question in QUESTIONS]
    assert len(identifiers) == len(set(identifiers))


def test_report_shape_and_honesty_fields() -> None:
    report = _run()
    assert report["schema"] == REPORT_SCHEMA
    assert report["semantic"] is False
    assert "not semantic quality" in report["semantic_note"]
    assert report["index_provider"] == "memory-cosine"
    assert report["token_cost_method"] == TOKEN_COST_METHOD
    assert report["token_cost_note"] == TOKEN_COST_NOTE
    # 11 fixture members, 3 non-public: the rebuild indexed exactly the 8
    # public notes, and the report carries no count of what was refused.
    assert report["rebuild"] == {"point_count": 8}
    assert len(report["questions"]) == len(QUESTIONS)
    assert report["summary"]["scored_questions"] == sum(
        1 for question in QUESTIONS if question.expected_paths
    )


def test_both_engines_answer_the_frozen_corpus() -> None:
    summary = _run()["summary"]
    assert summary["lexical_hit_rate"] == 1.0
    assert summary["vector_hit_rate"] == 1.0
    assert summary["lexical_no_answer_correct"] is True
    assert summary["citation_correct_questions"] == len(QUESTIONS)
    assert summary["privacy_false_negatives"] == 0
    assert "privacy_violations" not in summary


def test_report_is_deterministic_apart_from_latency() -> None:
    left = json.dumps(strip_latency(_run()), sort_keys=True, ensure_ascii=False)
    right = json.dumps(strip_latency(_run()), sort_keys=True, ensure_ascii=False)
    assert left == right
    assert "latency_ms" not in left


def test_no_sentinel_or_non_public_path_ever_reaches_the_report() -> None:
    rendered = json.dumps(_run(), sort_keys=True, ensure_ascii=False)
    for forbidden in (
        "INTERNAL-SENTINEL-ALLOCATION",
        "SENSITIVE-SENTINEL-MEETING",
        "STUDENT-SENTINEL-RECORD",
        *sorted(NON_PUBLIC_PATHS),
    ):
        assert forbidden not in rendered


class _LeakyIndex(InMemoryVectorIndex):
    """A rogue provider that surfaces a non-public path in every search.

    ``never-indexed.md`` never existed anywhere -- it proves leak detection
    is judged against the gated plan, not against a fixture list.
    """

    def search(self, query_vector, *, top_k, filter_privacy) -> SearchResult:
        honest = super().search(
            query_vector, top_k=top_k, filter_privacy=filter_privacy
        )
        leaked = (
            SearchHit(relative_path="learner-record.md", score=3.0, rank=0),
            SearchHit(relative_path="never-indexed.md", score=2.0, rank=1),
            *(
                SearchHit(
                    relative_path=hit.relative_path,
                    score=hit.score,
                    rank=hit.rank + 2,
                )
                for hit in honest.hits[: top_k - 2]
            ),
        )
        return SearchResult(composed_revision=honest.composed_revision, hits=leaked)


def test_privacy_violations_are_counted_and_redacted() -> None:
    report = _run(index_provider=_LeakyIndex())
    # Two leaked paths per question: one non-public fixture, one invented.
    assert report["summary"]["privacy_false_negatives"] == 2 * len(QUESTIONS)
    rendered = json.dumps(report, sort_keys=True, ensure_ascii=False)
    assert "learner-record.md" not in rendered
    assert "never-indexed.md" not in rendered


def test_privacy_false_positives_and_false_negatives_stay_separate() -> None:
    """FN (leak, zero-tolerance) and FP (over-blocking) must never merge.

    The frozen corpus has zero false positives (every ``public`` note makes
    it into the gated plan), so this pins the honest-zero shape and the key
    names; a mutant that folds one count into the other is caught by
    ``scripts/test-c5-mutations.sh`` M9b, which forces a false positive to
    exist and asserts it does not inflate ``privacy_false_negatives``.
    """
    summary = _run()["summary"]
    assert summary["privacy_false_positives"] == 0
    assert summary["privacy_false_positive_paths"] == []
    assert summary["privacy_false_negatives"] == 0

    leaky_summary = _run(index_provider=_LeakyIndex())["summary"]
    # The leaky provider only ever manufactures false negatives; it must not
    # move the false-positive count at all.
    assert leaky_summary["privacy_false_positives"] == 0
    assert leaky_summary["privacy_false_negatives"] == 2 * len(QUESTIONS)


def test_privacy_false_positives_do_not_inflate_false_negatives() -> None:
    """Force a real false positive and prove it stays out of the FN count.

    A gate that admits no privacy class at all turns every ``public`` corpus
    note into a false positive (declared public, never indexed) while
    leaking nothing (an empty index returns nothing, so false negatives stay
    zero). This is the scenario that actually distinguishes "FP and FN kept
    separate" from "FP silently folded into FN" -- the frozen corpus alone
    has zero false positives, so a merge bug is invisible without forcing
    one here. ``scripts/test-c5-mutations.sh`` M9b targets exactly this
    test.
    """
    blocking_gate = PrivacyGate(FrontmatterClassifier(UNUSED_ROOT), frozenset())
    report = run_shadow_benchmark(
        members=corpus_members(),
        stack=deterministic_stack(),
        index_provider=InMemoryVectorIndex(),
        gate=blocking_gate,
        top_k=3,
        trials=1,
    )
    summary = report["summary"]
    assert summary["privacy_false_positives"] == len(PUBLIC_DECLARED_PATHS)
    assert summary["privacy_false_positive_paths"] == sorted(PUBLIC_DECLARED_PATHS)
    assert summary["privacy_false_negatives"] == 0


def test_stale_exclusion_is_tracked_per_engine() -> None:
    report = _run()
    summary = report["summary"]
    total = len(QUESTIONS)
    assert 0 <= summary["lexical_stale_excluded"] <= total
    assert 0 <= summary["vector_stale_excluded"] <= total
    assert summary["lexical_stale_excluded_rate"] == round(
        summary["lexical_stale_excluded"] / total, 4
    )
    assert summary["vector_stale_excluded_rate"] == round(
        summary["vector_stale_excluded"] / total, 4
    )
    for question in report["questions"]:
        assert isinstance(question["lexical"]["stale_returned"], bool)
        assert isinstance(question["vector"]["stale_returned"], bool)

    q04 = next(
        q for q in report["questions"] if q["question_id"] == "q04-latest-status"
    )
    outdated = "platform-status-outdated.md"
    assert (outdated in q04["lexical"]["paths"]) == q04["lexical"]["stale_returned"]
    assert (outdated in q04["vector"]["paths"]) == q04["vector"]["stale_returned"]
    assert outdated in SUPERSEDED_PATHS


def test_percentile_is_nearest_rank_not_mean_or_interpolated() -> None:
    """Pin exact values so a mean, or an interpolating quantile, goes red.

    ``[1, 2, 3, 4, 100]`` is chosen so mean (22.0), nearest-rank P50 (3),
    and nearest-rank P95 (100) are all distinct: any implementation that
    silently degrades to the mean, or to an interpolating percentile
    (``statistics.quantiles`` default method gives P50 == 3 too, but a
    different P95 than nearest-rank's 100), is caught by asserting the
    *specific* nearest-rank numbers, not just "some number came back".
    """
    from benchmarks.shadow import _percentile

    samples = [1.0, 2.0, 3.0, 4.0, 100.0]
    mean = sum(samples) / len(samples)
    assert mean == pytest.approx(22.0)

    assert _percentile(samples, 50) == 3.0
    assert _percentile(samples, 95) == 100.0
    assert _percentile(samples, 50) != mean
    assert _percentile(samples, 95) != mean


def test_percentile_of_a_single_sample_is_that_sample() -> None:
    from benchmarks.shadow import _percentile

    assert _percentile([7.5], 50) == 7.5
    assert _percentile([7.5], 95) == 7.5


def test_latency_block_with_a_single_sample_has_p50_and_p95_equal_the_sample() -> None:
    """The per-engine ``latency_ms`` block, with exactly one pooled sample.

    ``run_shadow_benchmark`` with ``trials=1`` still pools one sample *per
    question* (ten questions -> ten samples), so P50 and P95 are not
    generally equal at the report level. This isolates ``_latency_block``
    itself on the single-sample case the coordinator asked to pin: both
    percentiles must equal the one measurement.
    """
    from benchmarks.shadow import _latency_block

    block = _latency_block([0.0421])
    assert block["p50"] == block["p95"] == round(0.0421 * 1000.0, 3)
    assert block["total"] == block["p50"]


def test_token_cost_is_whitespace_split_and_labeled_accordingly() -> None:
    """Bind the ``token_cost_method`` label to the actual computation.

    A mutant that relabels ``TOKEN_COST_METHOD`` (e.g. to
    ``"model-tokenizer"``) without changing how ``token_cost`` is computed
    must fail this test: both assertions live in the same test, against the
    same known note body, so the label and the arithmetic cannot drift
    apart unnoticed.
    """
    from benchmarks.questions import CORPUS_BODIES

    report = _run()
    assert report["token_cost_method"] == "whitespace-proxy"

    espresso_body = CORPUS_BODIES["espresso-dial-log.md"]
    expected_tokens = len(espresso_body.split())
    assert expected_tokens > 0

    q01 = next(q for q in report["questions"] if q["question_id"] == "q01-espresso")
    assert q01["lexical"]["paths"] == ["espresso-dial-log.md"]
    assert q01["lexical"]["token_cost"] == expected_tokens


def test_latency_reports_trials_total_p50_p95() -> None:
    report = _run(trials=4)
    latency = report["latency_ms"]
    assert latency["trials_per_question"] == 4
    for engine in ("lexical", "vector"):
        block = latency[engine]
        assert set(block) == {"total", "p50", "p95"}
        assert block["p50"] <= block["p95"]
        assert block["total"] >= block["p95"]


def test_latency_default_trials_matches_the_named_constant() -> None:
    from benchmarks.shadow import DEFAULT_LATENCY_TRIALS

    report = run_shadow_benchmark(
        members=corpus_members(),
        stack=deterministic_stack(),
        index_provider=InMemoryVectorIndex(),
        gate=public_gate(),
        top_k=3,
    )
    assert report["latency_ms"]["trials_per_question"] == DEFAULT_LATENCY_TRIALS


def test_all_required_composition_no_defaults() -> None:
    import inspect

    signature = inspect.signature(run_shadow_benchmark)
    parameters = signature.parameters
    # ``trials`` is the sole intentional exception (contract §5.3
    # latency-dimension addition): every composition input the harness
    # depends on to build a correct report still has no default, but the
    # repeat count for the latency sample is allowed a documented default
    # so existing callers keep working unchanged.
    assert set(parameters) - {"trials"} == {
        "members",
        "stack",
        "index_provider",
        "gate",
        "top_k",
    }
    for name, parameter in parameters.items():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        if name == "trials":
            assert parameter.default == 5
        else:
            assert parameter.default is inspect.Parameter.empty


def test_report_can_be_written_as_stable_json(tmp_path: Path) -> None:
    report = strip_latency(_run())
    target = tmp_path / "shadow-report.json"
    target.write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assert json.loads(target.read_text(encoding="utf-8")) == report


def test_descriptor_admission_is_enforced_by_the_harness() -> None:
    class NotAProvider:
        pass

    from ckp.index.errors import IndexRefusal

    with pytest.raises(IndexRefusal):
        run_shadow_benchmark(
            members=corpus_members(),
            stack=deterministic_stack(),
            index_provider=NotAProvider(),
            gate=public_gate(),
            top_k=3,
        )


def test_index_descriptor_of_reference_provider() -> None:
    descriptor = InMemoryVectorIndex().descriptor
    assert descriptor == IndexDescriptor(
        contract_version=INDEX_CONTRACT,
        provider_id="memory-cosine",
        provider_version="1",
        schema_version=INDEX_SCHEMA_VERSION,
    )


def test_rebuild_report_is_a_plain_frozen_record() -> None:
    report = InMemoryVectorIndex().rebuild(
        __import__("ckp.index", fromlist=["plan_rebuild"]).plan_rebuild(
            members=corpus_members(),
            stack=deterministic_stack(),
            gate=public_gate(),
        )
    )
    assert isinstance(report, RebuildReport)
    with pytest.raises(AttributeError):
        report.point_count = 0  # type: ignore[misc]
