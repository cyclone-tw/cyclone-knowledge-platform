"""The in-memory reference index: rebuild, search, snapshot, fail-closed."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from ckp.index import (
    InMemoryVectorIndex,
    RebuildPlan,
    compute_payload_digest,
    plan_rebuild,
    require_index_provider,
)
from ckp.index.errors import IndexErrorCode, IndexRefusal
from index_fixtures import (
    DIMENSION,
    corpus_members,
    deterministic_stack,
    make_member,
    public_gate,
)


def _plan(members=None):
    return plan_rebuild(
        members=members or corpus_members(),
        stack=deterministic_stack(),
        gate=public_gate(),
    )


def _query(text: str) -> tuple[float, ...]:
    return deterministic_stack().embedding.embed_query(text).values


def test_rebuild_reports_the_plan_and_admission_passes() -> None:
    index = InMemoryVectorIndex()
    require_index_provider(index)
    plan = _plan()
    report = index.rebuild(plan)
    assert report.provider_id == "memory-cosine"
    assert report.point_count == plan.indexed_count
    assert report.composed_revision == plan.composed_revision
    assert report.payload_digest == plan.payload_digest


def test_search_before_rebuild_fails_closed() -> None:
    index = InMemoryVectorIndex()
    with pytest.raises(IndexRefusal) as caught:
        index.search(_query("anything"), top_k=3)
    assert caught.value.code is IndexErrorCode.NOT_BUILT


def test_search_orders_truncates_and_guards() -> None:
    index = InMemoryVectorIndex()
    plan = _plan()
    index.rebuild(plan)

    result = index.search(_query("espresso grind dial grams"), top_k=3)
    assert result.composed_revision == plan.composed_revision
    assert len(result.hits) == 3
    assert result.hits[0].relative_path == "espresso-dial-log.md"
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True)

    everything = index.search(_query("synthetic note"), top_k=100)
    assert len(everything.hits) == plan.indexed_count

    with pytest.raises(IndexRefusal) as caught:
        index.search(_query("x"), top_k=0)
    assert caught.value.code is IndexErrorCode.TOP_K_INVALID
    with pytest.raises(IndexRefusal) as caught:
        index.search((1.0,), top_k=3)
    assert caught.value.code is IndexErrorCode.DIMENSION_MISMATCH


def test_equal_scores_tie_break_on_relative_path() -> None:
    # Two identical bodies at different paths embed to the same vector, so
    # their scores are bit-equal and only the frozen tie-break separates them.
    body = b"---\nprivacy: public\n---\n\nidentical twin body words\n"
    members = (
        make_member("twin-b.md", body),
        make_member("twin-a.md", body),
        make_member("other.md", b"---\nprivacy: public\n---\n\ncompletely different\n"),
    )
    index = InMemoryVectorIndex()
    index.rebuild(
        plan_rebuild(members=members, stack=deterministic_stack(), gate=public_gate())
    )
    result = index.search(_query("identical twin body words"), top_k=2)
    assert [hit.relative_path for hit in result.hits] == ["twin-a.md", "twin-b.md"]
    assert result.hits[0].score == result.hits[1].score


def test_full_rebuild_replaces_the_previous_corpus() -> None:
    index = InMemoryVectorIndex()
    index.rebuild(_plan())
    replacement = (
        make_member("only.md", b"---\nprivacy: public\n---\n\nlone survivor\n"),
    )
    index.rebuild(
        plan_rebuild(
            members=replacement, stack=deterministic_stack(), gate=public_gate()
        )
    )
    result = index.search(_query("espresso grind dial grams"), top_k=100)
    assert [hit.relative_path for hit in result.hits] == ["only.md"]


def test_an_unsealed_plan_is_refused() -> None:
    # Hand-built around the privacy gate: structurally the same shape, but
    # not minted by plan_rebuild/read_plan_snapshot -- providers refuse it.
    real = _plan()
    forged = RebuildPlan(
        points=real.points,
        dimension=real.dimension,
        composed_revision=real.composed_revision,
        bundle_index_revision=real.bundle_index_revision,
        embedding_revision=real.embedding_revision,
        indexed_count=real.indexed_count,
        payload_digest=real.payload_digest,
        seal=b"not-the-seal",
    )
    with pytest.raises(IndexRefusal) as caught:
        InMemoryVectorIndex().rebuild(forged)
    assert caught.value.code is IndexErrorCode.PLAN_INVALID


def test_a_replaced_plan_loses_its_seal() -> None:
    # ``dataclasses.replace`` clones the seal, so the seal must bind the
    # contents (R2 review): swapping the point set -- or the digest that
    # vouches for it -- must fail provider admission.
    real = _plan()
    smuggled = dataclasses.replace(real, points=real.points[:1])
    with pytest.raises(IndexRefusal) as caught:
        InMemoryVectorIndex().rebuild(smuggled)
    assert caught.value.code is IndexErrorCode.PLAN_INVALID

    relabeled = dataclasses.replace(
        real,
        points=real.points[:1],
        indexed_count=1,
        payload_digest=f"sha256:{'d' * 64}",
    )
    with pytest.raises(IndexRefusal) as caught:
        InMemoryVectorIndex().rebuild(relabeled)
    assert caught.value.code is IndexErrorCode.PLAN_INVALID

    # Same count, same declared digest, different content: only recomputing
    # the payload digest from the actual points catches this one.
    smuggled_point = dataclasses.replace(real.points[0], relative_path="smuggled.md")
    same_count = dataclasses.replace(real, points=(smuggled_point, *real.points[1:]))
    with pytest.raises(IndexRefusal) as caught:
        InMemoryVectorIndex().rebuild(same_count)
    assert caught.value.code is IndexErrorCode.PLAN_INVALID


def test_snapshot_metadata_tampering_fails_closed(tmp_path: Path) -> None:
    index = InMemoryVectorIndex()
    index.rebuild(_plan())
    snapshot_path = index.snapshot(tmp_path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    for field, value in (
        ("composed_revision", f"sha256:{'e' * 64}"),
        ("embedding_revision", f"sha256:{'e' * 64}"),
        ("bundle_index_revision", f"sha256:{'e' * 64}"),
        ("dimension", 16),
        ("dimension", 0),
        ("indexed_count", 1),
    ):
        edited = dict(payload)
        edited[field] = value
        target = tmp_path / f"tampered-{field}-{value}.json"
        target.write_text(json.dumps(edited, sort_keys=True), encoding="utf-8")
        with pytest.raises(IndexRefusal) as caught:
            InMemoryVectorIndex().restore(target)
        assert caught.value.code is IndexErrorCode.SNAPSHOT_INVALID, field


def test_snapshot_with_non_finite_vectors_fails_closed(tmp_path: Path) -> None:
    index = InMemoryVectorIndex()
    index.rebuild(_plan())
    snapshot_path = index.snapshot(tmp_path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    # float.fromhex would happily parse these (R2 review).
    payload["points"][0]["vector"][0] = "nan"
    target = tmp_path / "tampered-nan.json"
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(IndexRefusal) as caught:
        InMemoryVectorIndex().restore(target)
    assert caught.value.code is IndexErrorCode.SNAPSHOT_INVALID


def test_rebuild_verifies_the_payload_digest() -> None:
    index = InMemoryVectorIndex()
    plan = _plan()
    forged = dataclasses.replace(plan, payload_digest=f"sha256:{'f' * 64}")
    with pytest.raises(IndexRefusal) as caught:
        index.rebuild(forged)
    assert caught.value.code is IndexErrorCode.PLAN_INVALID
    with pytest.raises(IndexRefusal):
        index.search(_query("x"), top_k=1)


def test_snapshot_restore_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    index = InMemoryVectorIndex()
    plan = _plan()
    index.rebuild(plan)
    snapshot_path = index.snapshot(tmp_path)
    assert snapshot_path.name.startswith("ckp-index-memory-cosine-")

    restored = InMemoryVectorIndex()
    report = restored.restore(snapshot_path)
    assert report.composed_revision == plan.composed_revision
    assert report.payload_digest == plan.payload_digest
    query = _query("espresso grind dial grams")
    original = index.search(query, top_k=5)
    recovered = restored.search(query, top_k=5)
    assert original == recovered

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["points"][0]["relative_path"] = "tampered.md"
    tampered = tmp_path / "tampered.snapshot.json"
    tampered.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(IndexRefusal) as caught:
        InMemoryVectorIndex().restore(tampered)
    assert caught.value.code is IndexErrorCode.SNAPSHOT_INVALID

    with pytest.raises(IndexRefusal) as caught:
        InMemoryVectorIndex().restore(tmp_path / "missing.json")
    assert caught.value.code is IndexErrorCode.SNAPSHOT_INVALID


def test_snapshot_preserves_float64_bit_for_bit(tmp_path: Path) -> None:
    index = InMemoryVectorIndex()
    plan = _plan()
    index.rebuild(plan)
    restored = InMemoryVectorIndex()
    restored.restore(index.snapshot(tmp_path))
    assert restored._plan is not None  # noqa: SLF001 - deliberate white-box
    assert compute_payload_digest(restored._plan.points) == plan.payload_digest


def test_wipe_forgets_everything() -> None:
    index = InMemoryVectorIndex()
    index.rebuild(_plan())
    index.wipe()
    with pytest.raises(IndexRefusal) as caught:
        index.search(_query("x"), top_k=1)
    assert caught.value.code is IndexErrorCode.NOT_BUILT


def test_dimension_guard_uses_the_plan(tmp_path: Path) -> None:
    index = InMemoryVectorIndex()
    index.rebuild(_plan())
    wrong = deterministic_stack(dimension=DIMENSION // 2).embedding.embed_query(
        "espresso"
    )
    with pytest.raises(IndexRefusal) as caught:
        index.search(wrong.values, top_k=1)
    assert caught.value.code is IndexErrorCode.DIMENSION_MISMATCH
