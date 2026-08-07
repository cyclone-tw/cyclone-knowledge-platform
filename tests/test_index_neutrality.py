"""Provider neutrality: a test-defined index runs the same caller unchanged.

The caller under test is ``run_shadow_benchmark`` itself -- the only consumer
of the index interface in this repo. If it needs no edit to accept a
third-party provider defined below, contract §2 line 86 holds: swapping the
index provider is a composition change, not a code change.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from benchmarks.shadow import run_shadow_benchmark, strip_latency

from ckp.index import (
    INDEX_CONTRACT,
    IndexDescriptor,
    InMemoryVectorIndex,
    RebuildPlan,
    RebuildReport,
    SearchHit,
    SearchResult,
    compute_payload_digest,
    read_plan_snapshot,
    require_index_provider,
    require_public_filter,
    require_query_vector,
    require_sealed_plan,
    require_top_k,
    write_plan_snapshot,
)
from ckp.index.errors import IndexErrorCode, IndexRefusal
from ckp.index.revision import INDEX_SCHEMA_VERSION
from index_fixtures import corpus_members, deterministic_stack, public_gate


class ThirdPartyListIndex:
    """An independent implementation: linear scan over a plain list.

    Deliberately shares no code with ``InMemoryVectorIndex`` beyond the
    public shapes and guards -- the point is that a stranger's provider
    satisfies the same contract.
    """

    def __init__(self) -> None:
        self._descriptor = IndexDescriptor(
            contract_version=INDEX_CONTRACT,
            provider_id="third-party-list",
            provider_version="0.1",
            schema_version=INDEX_SCHEMA_VERSION,
        )
        self._rows: list[tuple[str, str, tuple[float, ...]]] | None = None
        self._plan: RebuildPlan | None = None

    @property
    def descriptor(self) -> IndexDescriptor:
        return self._descriptor

    def rebuild(self, plan: RebuildPlan) -> RebuildReport:
        require_sealed_plan(plan)
        self.wipe()
        if compute_payload_digest(plan.points) != plan.payload_digest:
            raise IndexRefusal(IndexErrorCode.PLAN_INVALID)
        self._rows = [(point.relative_path, point.vector) for point in plan.points]
        self._plan = plan
        return RebuildReport(
            provider_id=self._descriptor.provider_id,
            composed_revision=plan.composed_revision,
            point_count=len(self._rows),
            payload_digest=plan.payload_digest,
        )

    def search(self, query_vector, *, top_k, filter_privacy) -> SearchResult:
        if self._rows is None or self._plan is None:
            raise IndexRefusal(IndexErrorCode.NOT_BUILT)
        require_public_filter(filter_privacy)
        limit = require_top_k(top_k)
        vector = require_query_vector(query_vector, dimension=self._plan.dimension)
        scored = sorted(
            (
                (
                    -sum(a * b for a, b in zip(vector, row_vector, strict=True)),
                    path,
                )
                for path, row_vector in self._rows
            ),
        )
        return SearchResult(
            composed_revision=self._plan.composed_revision,
            hits=tuple(
                SearchHit(relative_path=path, score=-negated, rank=rank)
                for rank, (negated, path) in enumerate(scored[:limit])
            ),
        )

    def snapshot(self, target_dir: Path) -> Path:
        if self._plan is None:
            raise IndexRefusal(IndexErrorCode.NOT_BUILT)
        return write_plan_snapshot(
            self._plan,
            provider_id=self._descriptor.provider_id,
            target_dir=target_dir,
        )

    def restore(self, snapshot_path: Path) -> RebuildReport:
        return self.rebuild(read_plan_snapshot(snapshot_path))

    def wipe(self) -> None:
        self._rows = None
        self._plan = None


def test_third_party_provider_passes_admission() -> None:
    descriptor = require_index_provider(ThirdPartyListIndex())
    assert descriptor.provider_id == "third-party-list"


def test_admission_fails_closed_on_wrong_shapes() -> None:
    with pytest.raises(IndexRefusal) as caught:
        require_index_provider(object())
    assert caught.value.code is IndexErrorCode.PROVIDER_INVALID

    class WrongContract(ThirdPartyListIndex):
        def __init__(self) -> None:
            super().__init__()
            self._descriptor = IndexDescriptor(
                contract_version="index/v0",
                provider_id="third-party-list",
                provider_version="0.1",
                schema_version=INDEX_SCHEMA_VERSION,
            )

    with pytest.raises(IndexRefusal) as caught:
        require_index_provider(WrongContract())
    assert caught.value.code is IndexErrorCode.CONTRACT_VERSION_UNKNOWN


def test_the_same_caller_runs_both_providers_unchanged() -> None:
    members = corpus_members()
    stack = deterministic_stack()

    reference = run_shadow_benchmark(
        members=members,
        stack=stack,
        index_provider=InMemoryVectorIndex(),
        gate=public_gate(),
        top_k=3,
        trials=1,
    )
    third_party = run_shadow_benchmark(
        members=members,
        stack=stack,
        index_provider=ThirdPartyListIndex(),
        gate=public_gate(),
        top_k=3,
        trials=1,
    )

    assert third_party["index_provider"] == "third-party-list"
    assert reference["index_provider"] == "memory-cosine"

    def comparable(report: dict) -> dict:
        view = strip_latency(report)
        view.pop("index_provider")
        # Scores are floating point and the two providers legitimately sum in
        # different orders; ranked paths, hits, and counts must still agree.
        view["questions"] = [
            {
                **question,
                "vector": {
                    key: value
                    for key, value in question["vector"].items()
                    if key != "top_score"
                },
            }
            for question in view["questions"]
        ]
        return view

    assert comparable(reference) == comparable(third_party)


def test_caller_source_names_no_concrete_provider() -> None:
    source = inspect.getsource(run_shadow_benchmark)
    for forbidden in ("memory-cosine", "qdrant", "InMemoryVectorIndex"):
        assert forbidden not in source
