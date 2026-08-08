"""``index/v1`` shapes: descriptors, digests, guards, and the rebuild plan."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ckp.index.errors import IndexErrorCode, IndexRefusal
from ckp.index.models import (
    INDEX_CONTRACT,
    IndexDescriptor,
    IndexedPoint,
    SearchHit,
    SearchResult,
    compute_payload_digest,
    compute_point_id,
    plan_rebuild,
    require_query_vector,
    require_top_k,
)
from ckp.index.revision import (
    INDEX_SCHEMA_VERSION,
    compute_composed_index_revision,
)
from ckp.privacy import PrivacyClass
from index_fixtures import corpus_members, deterministic_stack, make_member, public_gate

SHA = "0" * 64


def _point(path: str, vector: tuple[float, ...] = (1.0, 0.0)) -> IndexedPoint:
    return IndexedPoint(
        point_id=compute_point_id(path),
        digest_key=path,
        relative_path=path,
        content_sha256=SHA,
        vector=vector,
    )


def test_descriptor_shape_is_enforced() -> None:
    descriptor = IndexDescriptor(
        contract_version=INDEX_CONTRACT,
        provider_id="synthetic-index",
        provider_version="1",
        schema_version=INDEX_SCHEMA_VERSION,
    )
    assert descriptor.contract_version == "index/v1"
    with pytest.raises(ValidationError):
        IndexDescriptor(
            contract_version=INDEX_CONTRACT,
            provider_id="Bad Provider",
            provider_version="1",
            schema_version="1",
        )
    with pytest.raises(ValidationError):
        IndexDescriptor(
            contract_version=INDEX_CONTRACT,
            provider_id="ok",
            provider_version="1",
            schema_version="1",
            extra_field="nope",  # type: ignore[call-arg]
        )


def test_composed_revision_covers_every_input_and_is_framed() -> None:
    base = compute_composed_index_revision(
        bundle_index_revision=f"sha256:{'a' * 64}",
        embedding_revision=f"sha256:{'b' * 64}",
        index_schema_version="1",
    )
    assert base.startswith("sha256:") and len(base) == 71
    changed_bundle = compute_composed_index_revision(
        bundle_index_revision=f"sha256:{'c' * 64}",
        embedding_revision=f"sha256:{'b' * 64}",
        index_schema_version="1",
    )
    changed_embedding = compute_composed_index_revision(
        bundle_index_revision=f"sha256:{'a' * 64}",
        embedding_revision=f"sha256:{'c' * 64}",
        index_schema_version="1",
    )
    changed_schema = compute_composed_index_revision(
        bundle_index_revision=f"sha256:{'a' * 64}",
        embedding_revision=f"sha256:{'b' * 64}",
        index_schema_version="2",
    )
    assert len({base, changed_bundle, changed_embedding, changed_schema}) == 4
    with pytest.raises(ValueError):
        compute_composed_index_revision(
            bundle_index_revision="not-a-revision",
            embedding_revision=f"sha256:{'b' * 64}",
            index_schema_version="1",
        )


def test_point_id_and_payload_digest_are_deterministic_and_framed() -> None:
    assert compute_point_id("a.md") == compute_point_id("a.md")
    assert compute_point_id("a.md") != compute_point_id("b.md")
    points = (_point("a.md"), _point("b.md"))
    assert compute_payload_digest(points) == compute_payload_digest(points)
    reordered = (points[1], points[0])
    assert compute_payload_digest(points) != compute_payload_digest(reordered)
    moved_vector = (_point("a.md", (0.0, 1.0)), _point("b.md"))
    assert compute_payload_digest(points) != compute_payload_digest(moved_vector)


def test_search_hit_and_result_enforce_the_frozen_total_order() -> None:
    ok = SearchResult(
        composed_revision=f"sha256:{'a' * 64}",
        hits=(
            SearchHit(relative_path="a.md", score=0.9, rank=0),
            SearchHit(relative_path="b.md", score=0.9, rank=1),
            SearchHit(relative_path="c.md", score=0.1, rank=2),
        ),
    )
    assert len(ok.hits) == 3
    with pytest.raises(ValidationError):
        SearchHit(relative_path="a.md", score=float("nan"), rank=0)
    with pytest.raises(ValidationError):
        SearchResult(
            composed_revision=f"sha256:{'a' * 64}",
            hits=(
                SearchHit(relative_path="a.md", score=0.1, rank=0),
                SearchHit(relative_path="b.md", score=0.9, rank=1),
            ),
        )
    with pytest.raises(ValidationError):
        SearchResult(
            composed_revision=f"sha256:{'a' * 64}",
            hits=(
                SearchHit(relative_path="b.md", score=0.9, rank=0),
                SearchHit(relative_path="a.md", score=0.9, rank=1),
            ),
        )
    with pytest.raises(ValidationError):
        SearchResult(
            composed_revision=f"sha256:{'a' * 64}",
            hits=(
                SearchHit(relative_path="a.md", score=0.9, rank=0),
                SearchHit(relative_path="a.md", score=0.5, rank=1),
            ),
        )
    with pytest.raises(ValidationError):
        SearchResult(
            composed_revision=f"sha256:{'a' * 64}",
            hits=(SearchHit(relative_path="a.md", score=0.9, rank=1),),
        )


def test_hit_models_never_carry_note_bodies() -> None:
    assert set(SearchHit.model_fields) == {"relative_path", "score", "rank"}
    assert SearchHit.model_config.get("extra") == "forbid"
    assert set(SearchResult.model_fields) == {"composed_revision", "hits"}


def test_input_guards_fail_closed() -> None:
    with pytest.raises(IndexRefusal) as caught:
        require_query_vector("not a vector", dimension=2)
    assert caught.value.code is IndexErrorCode.QUERY_INVALID
    with pytest.raises(IndexRefusal) as caught:
        require_query_vector((1.0,), dimension=2)
    assert caught.value.code is IndexErrorCode.DIMENSION_MISMATCH
    with pytest.raises(IndexRefusal) as caught:
        require_query_vector((1.0, float("inf")), dimension=2)
    assert caught.value.code is IndexErrorCode.QUERY_INVALID
    with pytest.raises(IndexRefusal) as caught:
        require_query_vector((1.0, True), dimension=2)
    assert caught.value.code is IndexErrorCode.QUERY_INVALID

    for bad in (0, -1, True, 1.5, "3"):
        with pytest.raises(IndexRefusal) as caught:
            require_top_k(bad)
        assert caught.value.code is IndexErrorCode.TOP_K_INVALID


def test_plan_rebuild_orders_gates_and_counts() -> None:
    members = corpus_members()
    stack = deterministic_stack()
    gate = public_gate()
    plan = plan_rebuild(members=members, stack=stack, gate=gate)

    assert plan.indexed_count == len(plan.points)
    # 11 fixture members, 3 non-public: exactly the 8 public notes made it,
    # and the plan deliberately records nothing about what was refused.
    assert plan.indexed_count == len(members) - 3
    assert not hasattr(plan, "excluded_count")
    assert not hasattr(plan, "member_count")
    keys = [point.digest_key for point in plan.points]
    assert keys == sorted(keys)
    assert all(len(point.vector) == plan.dimension for point in plan.points)

    # Input order never matters: a scrambled corpus produces the same plan.
    scrambled = plan_rebuild(members=tuple(reversed(members)), stack=stack, gate=gate)
    assert scrambled.payload_digest == plan.payload_digest
    assert scrambled.composed_revision == plan.composed_revision


def test_plan_rebuild_admissible_defaults_to_public_only() -> None:
    """Issue #26 Round 2 (Codex Finding 2): pin the default explicitly.

    Before this parameter existed, ``plan_rebuild`` hardcoded ``verdict.
    privacy is PrivacyClass.PUBLIC``; every caller predating issue #26 (the
    Gateway; every other test in this file) must see zero behavior change.
    A mutant that hardcodes the check back to literal ``PUBLIC`` regardless
    of what ``admissible`` says would still pass *this* assertion (the
    default *is* public-only) -- ``test_plan_rebuild_admissible_widens_
    which_privacy_classes_are_embedded`` below is the one that catches that,
    by proving the parameter is actually wired, not just present.
    """
    import inspect

    signature = inspect.signature(plan_rebuild)
    assert signature.parameters["admissible"].default == frozenset(
        {PrivacyClass.PUBLIC}
    )


def test_plan_rebuild_admissible_widens_which_privacy_classes_are_embedded() -> None:
    """The parameter must be *wired*, not just accepted (Codex Finding 2).

    ``corpus_members()`` carries one ``internal`` fixture note
    (``budget-internal.md``, 1 of the 3 non-public members the default-gate
    test above excludes). A gate configured to admit
    ``{PUBLIC, INTERNAL}`` alone is not enough to embed it -- issue #26
    Round 1's bug was exactly this: ``plan_rebuild`` ignored what the gate
    admitted and hardcoded ``PUBLIC`` regardless. Passing
    ``admissible={PUBLIC, INTERNAL}`` here must be what actually changes the
    indexed count; a mutant that accepts the parameter but never reads it
    would still index 8 either way and fail the ``== 9`` assertion below.
    """
    from ckp.privacy import FrontmatterClassifier
    from ckp.privacy.gate import PrivacyGate
    from index_fixtures import UNUSED_ROOT

    members = corpus_members()
    stack = deterministic_stack()
    mixed_gate = PrivacyGate(
        FrontmatterClassifier(UNUSED_ROOT),
        frozenset({PrivacyClass.PUBLIC, PrivacyClass.INTERNAL}),
    )

    public_only_plan = plan_rebuild(
        members=members,
        stack=stack,
        gate=mixed_gate,
        admissible=frozenset({PrivacyClass.PUBLIC}),
    )
    assert public_only_plan.indexed_count == 8

    widened_plan = plan_rebuild(
        members=members,
        stack=stack,
        gate=mixed_gate,
        admissible=frozenset({PrivacyClass.PUBLIC, PrivacyClass.INTERNAL}),
    )
    # 8 public + budget-internal.md: the one `internal` fixture member that
    # a matching gate now admits *and* this run's `admissible` allows.
    assert widened_plan.indexed_count == 9
    paths = {point.relative_path for point in widened_plan.points}
    assert "budget-internal.md" in paths

    # Still fails closed: `sensitive`/`student-private` remain excluded even
    # though the *gate* here would admit any class the caller configured it
    # for -- this gate simply was never configured to admit them, matching
    # every other gate in this codebase (Decision §2).
    assert "meeting-sensitive.md" not in paths
    assert "learner-record.md" not in paths


def test_plan_rebuild_admissible_never_admits_student_private() -> None:
    """A second, independent refusal (Decision §2), mirroring
    ``PrivacyGate``'s own unconditional ``student-private`` rule -- a caller
    mistake in ``admissible`` alone must still fail closed even if ``gate``
    itself were somehow misconfigured to admit it.
    """
    members = corpus_members()
    stack = deterministic_stack()
    gate = public_gate()
    with pytest.raises(IndexRefusal) as caught:
        plan_rebuild(
            members=members,
            stack=stack,
            gate=gate,
            admissible=frozenset({PrivacyClass.PUBLIC, PrivacyClass.STUDENT_PRIVATE}),
        )
    assert caught.value.code is IndexErrorCode.PLAN_INVALID


def test_plan_rebuild_fails_closed_on_empty_and_excludes_bad_members() -> None:
    stack = deterministic_stack()
    gate = public_gate()
    with pytest.raises(IndexRefusal) as caught:
        plan_rebuild(members=(), stack=stack, gate=gate)
    assert caught.value.code is IndexErrorCode.PLAN_INVALID

    undecodable = make_member(
        "broken.md", b"---\nprivacy: public\n---\n\n\xff\xfe invalid"
    )
    punctuation_only = make_member(
        "punct.md", b"---\nprivacy: public\n---\n\n...!!!...\n"
    )
    undeclared = make_member("undeclared.md", b"# no frontmatter at all\n")
    ok = make_member("ok.md", b"---\nprivacy: public\n---\n\nreal words here\n")
    plan = plan_rebuild(
        members=(undecodable, punctuation_only, undeclared, ok),
        stack=stack,
        gate=gate,
    )
    assert plan.indexed_count == 1
    assert plan.points[0].relative_path == "ok.md"
