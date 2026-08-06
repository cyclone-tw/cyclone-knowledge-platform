"""The §5.3 shadow benchmark: determinism, honesty, and zero privacy leaks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.questions import NON_PUBLIC_PATHS, QUESTIONS
from benchmarks.shadow import REPORT_SCHEMA, run_shadow_benchmark, strip_latency

from ckp.index import (
    INDEX_CONTRACT,
    IndexDescriptor,
    InMemoryVectorIndex,
    RebuildReport,
    SearchHit,
    SearchResult,
)
from ckp.index.revision import INDEX_SCHEMA_VERSION
from index_fixtures import corpus_members, deterministic_stack, public_gate


def _run(index_provider=None, top_k: int = 3) -> dict:
    return run_shadow_benchmark(
        members=corpus_members(),
        stack=deterministic_stack(),
        index_provider=index_provider or InMemoryVectorIndex(),
        gate=public_gate(),
        top_k=top_k,
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
    assert summary["privacy_violations"] == 0


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
    assert report["summary"]["privacy_violations"] == 2 * len(QUESTIONS)
    rendered = json.dumps(report, sort_keys=True, ensure_ascii=False)
    assert "learner-record.md" not in rendered
    assert "never-indexed.md" not in rendered


def test_all_required_composition_no_defaults() -> None:
    import inspect

    signature = inspect.signature(run_shadow_benchmark)
    assert all(
        parameter.default is inspect.Parameter.empty
        and parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


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
