"""Qdrant integration: fence, rebuild parity, snapshot, verified restores.

Needs a local Qdrant service. Locally the suite skips when none is
reachable; in CI ``CKP_REQUIRE_QDRANT=1`` turns that skip into a failure so
the integration path can never silently stop running. The fence tests below
the skip line run everywhere -- they refuse before any connection exists.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from benchmarks.questions import QUESTIONS
from benchmarks.shadow import run_shadow_benchmark, strip_latency

from ckp.index import InMemoryVectorIndex, QdrantVectorIndex, plan_rebuild
from ckp.index.errors import IndexErrorCode, IndexRefusal
from index_fixtures import corpus_members, deterministic_stack, public_gate

QDRANT_HOST = os.environ.get("CKP_QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("CKP_QDRANT_PORT", "6333"))
_REQUIRED = os.environ.get("CKP_REQUIRE_QDRANT") == "1"


def _service_reachable() -> bool:
    try:
        from qdrant_client import QdrantClient

        QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3).get_collections()
        return True
    except Exception:
        return False


_reachable = _service_reachable()
if _REQUIRED and not _reachable:
    pytest.fail("CKP_REQUIRE_QDRANT=1 but no Qdrant service is reachable")

needs_qdrant = pytest.mark.skipif(
    not _reachable, reason="no local Qdrant service reachable"
)


def test_remote_hosts_are_refused_before_any_connection() -> None:
    for host in ("qdrant.example.com", "10.0.0.7", "cloud.qdrant.io", ""):
        with pytest.raises(IndexRefusal) as caught:
            QdrantVectorIndex(host=host, port=6333, collection="ckp")
        assert caught.value.code is IndexErrorCode.REMOTE_DENIED
    with pytest.raises(IndexRefusal) as caught:
        QdrantVectorIndex(host="localhost", port=0, collection="ckp")
    assert caught.value.code is IndexErrorCode.REMOTE_DENIED


def test_collection_names_are_validated() -> None:
    for name in ("", "bad name", "x" * 65, "semi;colon"):
        with pytest.raises(IndexRefusal) as caught:
            QdrantVectorIndex(host="localhost", port=6333, collection=name)
        assert caught.value.code is IndexErrorCode.PROVIDER_INVALID


def test_constructor_signature_has_no_defaults() -> None:
    import inspect

    signature = inspect.signature(QdrantVectorIndex.__init__)
    assert all(
        parameter.default is inspect.Parameter.empty
        for name, parameter in signature.parameters.items()
        if name != "self"
    )


def _index() -> QdrantVectorIndex:
    return QdrantVectorIndex(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection=f"ckp-test-{uuid.uuid4().hex[:12]}",
    )


def _plan():
    return plan_rebuild(
        members=corpus_members(), stack=deterministic_stack(), gate=public_gate()
    )


@needs_qdrant
def test_empty_volume_rebuild_reports_the_composed_revision() -> None:
    index = _index()
    try:
        plan = _plan()
        report = index.rebuild(plan)
        assert report.provider_id == "qdrant-local"
        assert report.composed_revision == plan.composed_revision
        assert report.point_count == plan.indexed_count
        second = index.rebuild(plan)
        assert second == report
    finally:
        index.wipe()


@needs_qdrant
def test_search_parity_with_the_reference_index() -> None:
    plan = _plan()
    qdrant = _index()
    memory = InMemoryVectorIndex()
    try:
        qdrant.rebuild(plan)
        memory.rebuild(plan)
        stack = deterministic_stack()
        for query in (
            "espresso grind dial grams",
            "tide caves mapping fieldnotes",
            "platform status storage rebuild",
            "軌道力學 module Kepler",
        ):
            vector = stack.embedding.embed_query(query).values
            qdrant_hits = qdrant.search(vector, top_k=5)
            memory_hits = memory.search(vector, top_k=5)
            assert [hit.relative_path for hit in qdrant_hits.hits] == [
                hit.relative_path for hit in memory_hits.hits
            ]
            for left, right in zip(qdrant_hits.hits, memory_hits.hits, strict=True):
                assert left.score == pytest.approx(right.score, abs=1e-5)
    finally:
        qdrant.wipe()


@needs_qdrant
def test_search_before_rebuild_fails_closed() -> None:
    index = _index()
    with pytest.raises(IndexRefusal) as caught:
        index.search(
            deterministic_stack().embedding.embed_query("x").values,
            top_k=3,
        )
    assert caught.value.code is IndexErrorCode.NOT_BUILT


@needs_qdrant
def test_snapshot_restore_roundtrip_across_providers(tmp_path: Path) -> None:
    plan = _plan()
    qdrant = _index()
    try:
        qdrant.rebuild(plan)
        snapshot_path = qdrant.snapshot(tmp_path)

        memory = InMemoryVectorIndex()
        memory_report = memory.restore(snapshot_path)
        assert memory_report.composed_revision == plan.composed_revision

        second = _index()
        try:
            report = second.restore(snapshot_path)
            assert report.composed_revision == plan.composed_revision
            assert report.payload_digest == plan.payload_digest
        finally:
            second.wipe()
    finally:
        qdrant.wipe()


@needs_qdrant
def test_shadow_benchmark_runs_identically_on_qdrant() -> None:
    members = corpus_members()
    stack = deterministic_stack()
    qdrant = _index()
    try:
        reference = run_shadow_benchmark(
            members=members,
            stack=stack,
            index_provider=InMemoryVectorIndex(),
            gate=public_gate(),
            top_k=3,
            questions=QUESTIONS,
            trials=1,
        )
        against_qdrant = run_shadow_benchmark(
            members=members,
            stack=stack,
            index_provider=qdrant,
            gate=public_gate(),
            top_k=3,
            questions=QUESTIONS,
            trials=1,
        )
    finally:
        qdrant.wipe()

    assert against_qdrant["index_provider"] == "qdrant-local"
    assert against_qdrant["summary"]["privacy_false_negatives"] == 0
    assert (
        against_qdrant["summary"]["vector_hit_rate"]
        == reference["summary"]["vector_hit_rate"]
    )

    def paths_only(report: dict) -> list:
        return [
            (question["question_id"], question["vector"]["paths"])
            for question in strip_latency(report)["questions"]
        ]

    assert paths_only(against_qdrant) == paths_only(reference)


@needs_qdrant
def test_rebuild_wipes_stale_points() -> None:
    from index_fixtures import make_member

    index = _index()
    try:
        index.rebuild(_plan())
        replacement = plan_rebuild(
            members=(
                make_member("only.md", b"---\nprivacy: public\n---\n\nlone survivor\n"),
            ),
            stack=deterministic_stack(),
            gate=public_gate(),
        )
        report = index.rebuild(replacement)
        assert report.point_count == 1
        result = index.search(
            deterministic_stack().embedding.embed_query("espresso").values,
            top_k=100,
        )
        assert [hit.relative_path for hit in result.hits] == ["only.md"]
    finally:
        index.wipe()
