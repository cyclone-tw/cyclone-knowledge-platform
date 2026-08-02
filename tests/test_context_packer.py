"""C6 deterministic item and token-upper-bound context packing."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ckp.gateway.models import CitationResponse, QueryResultResponse, SnapshotRef
from ckp.packer import BoundedContextPacker
from ckp.privacy import FrontmatterClassifier, PrivacyClass, PrivacyGate


def _packer(root: Path) -> BoundedContextPacker:
    return BoundedContextPacker(
        PrivacyGate(
            FrontmatterClassifier(root),
            frozenset({PrivacyClass.PUBLIC}),
        )
    )


def test_packer_requires_a_privacy_gate() -> None:
    parameter = inspect.signature(BoundedContextPacker.__init__).parameters[
        "privacy_gate"
    ]
    assert parameter.default is inspect.Parameter.empty


def _result(
    index: int,
    *,
    snippet: str,
    index_revision: str = "sha256:" + ("a" * 64),
    bundle_commit: str | None = "b" * 40,
) -> QueryResultResponse:
    concept_id = f"synthetic/concept-{index}"
    return QueryResultResponse(
        concept_id=concept_id,
        id=f"00000000-0000-7000-8000-{index:012d}",
        title=f"Synthetic result {index}",
        type="Concept",
        score=10 - index,
        snippet=snippet,
        citation=CitationResponse(
            concept_id=concept_id,
            path=f"synthetic-{index}.md",
            index_revision=index_revision,
            bundle_commit=bundle_commit,
        ),
    )


def test_packer_is_deterministic_cited_and_bounded_by_items_and_utf8_bytes(
    tmp_path: Path,
) -> None:
    results = [
        _result(1, snippet="alpha " * 30),
        _result(2, snippet="茶葉 " * 50),
        _result(3, snippet="gamma " * 30),
    ]
    packer = _packer(tmp_path)
    revision = SnapshotRef(
        index_revision="sha256:" + ("a" * 64),
        bundle_commit="b" * 40,
    )

    first = packer.pack(
        domain="finance",
        revision=revision,
        results=results,
        total_results=3,
        max_items=2,
        max_token_upper_bound=900,
    )
    second = packer.pack(
        domain="finance",
        revision=revision,
        results=results,
        total_results=3,
        max_items=2,
        max_token_upper_bound=900,
    )

    assert first == second
    assert len(first.items) <= 2
    assert first.item_limit == 2
    assert first.token_upper_bound_limit == 900
    assert first.token_upper_bound_used == len(first.context.encode("utf-8"))
    assert first.token_upper_bound_used <= first.token_upper_bound_limit
    assert first.token_upper_bound_method == "utf8-bytes-v1"
    assert first.truncated is True
    assert all(
        item.citation.index_revision == revision.index_revision for item in first.items
    )
    assert all(
        item.citation.bundle_commit == revision.bundle_commit for item in first.items
    )
    assert all(item.citation.path in first.context for item in first.items)
    assert all(item.citation.index_revision in first.context for item in first.items)
    assert all(item.citation.bundle_commit in first.context for item in first.items)


def test_packer_truncates_a_cjk_snippet_without_splitting_utf8(
    tmp_path: Path,
) -> None:
    packer = _packer(tmp_path)
    revision = SnapshotRef(
        index_revision="sha256:" + ("c" * 64),
        bundle_commit="d" * 40,
    )

    packed = packer.pack(
        domain="personal-retrospective",
        revision=revision,
        results=[
            _result(
                1,
                snippet="合成內容" * 80,
                index_revision=revision.index_revision,
                bundle_commit=revision.bundle_commit,
            )
        ],
        total_results=1,
        max_items=1,
        max_token_upper_bound=520,
    )

    assert len(packed.items) == 1
    assert packed.items[0].snippet.endswith("…")
    assert packed.items[0].snippet.encode("utf-8").decode("utf-8")
    assert packed.token_upper_bound_used <= 520
    assert packed.truncated is True


def test_packer_returns_empty_context_when_even_citation_cannot_fit(
    tmp_path: Path,
) -> None:
    packed = _packer(tmp_path).pack(
        domain="finance",
        revision=SnapshotRef(
            index_revision="sha256:" + ("e" * 64),
            bundle_commit="f" * 40,
        ),
        results=[
            _result(
                1,
                snippet="synthetic",
                index_revision="sha256:" + ("e" * 64),
                bundle_commit="f" * 40,
            )
        ],
        total_results=1,
        max_items=1,
        max_token_upper_bound=64,
    )

    assert packed.items == []
    assert packed.context == ""
    assert packed.token_upper_bound_used == 0
    assert packed.truncated is True


def test_packer_rejects_a_result_not_bound_to_the_response_revision(
    tmp_path: Path,
) -> None:
    revision = SnapshotRef(
        index_revision="sha256:" + ("1" * 64),
        bundle_commit="2" * 40,
    )
    mismatched = _result(1, snippet="synthetic").model_copy(
        update={
            "citation": CitationResponse(
                concept_id="synthetic/concept-1",
                path="synthetic-1.md",
                index_revision="sha256:" + ("3" * 64),
                bundle_commit="2" * 40,
            )
        }
    )

    try:
        _packer(tmp_path).pack(
            domain="finance",
            revision=revision,
            results=[mismatched],
            total_results=1,
            max_items=1,
            max_token_upper_bound=512,
        )
    except ValueError as error:
        assert str(error) == "citation-revision-mismatch"
    else:
        raise AssertionError("mismatched citation revision was packed")


def test_packer_rejects_context_without_bundle_commit(tmp_path: Path) -> None:
    revision = SnapshotRef(
        index_revision="sha256:" + ("6" * 64),
        bundle_commit=None,
    )

    with pytest.raises(ValueError, match="missing-bundle-commit"):
        _packer(tmp_path).pack(
            domain="finance",
            revision=revision,
            results=[
                _result(
                    1,
                    snippet="synthetic",
                    index_revision=revision.index_revision,
                    bundle_commit=None,
                )
            ],
            total_results=1,
            max_items=1,
            max_token_upper_bound=512,
        )


def test_item_and_token_guards_each_limit_an_otherwise_small_result_set(
    tmp_path: Path,
) -> None:
    revision = SnapshotRef(
        index_revision="sha256:" + ("4" * 64),
        bundle_commit="5" * 40,
    )
    results = [
        _result(
            index,
            snippet="small synthetic snippet",
            index_revision=revision.index_revision,
            bundle_commit=revision.bundle_commit,
        )
        for index in range(1, 4)
    ]

    item_limited = _packer(tmp_path).pack(
        domain="finance",
        revision=revision,
        results=results,
        total_results=3,
        max_items=1,
        max_token_upper_bound=32_768,
    )
    token_limited = _packer(tmp_path).pack(
        domain="finance",
        revision=revision,
        results=results,
        total_results=3,
        max_items=3,
        max_token_upper_bound=500,
    )

    assert len(item_limited.items) == 1
    assert len(token_limited.items) < 3
    assert token_limited.token_upper_bound_used <= 500
