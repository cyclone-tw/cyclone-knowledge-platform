"""Minimal public-only Catalog listing and deterministic lexical query."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from ckp.bundle import SnapshotCache
from ckp.catalog import CatalogBuilder, CatalogEntry, CatalogSnapshot
from ckp.gateway.models import (
    CatalogFacets,
    CatalogItemResponse,
    CatalogRequest,
    CatalogResponse,
    CitationResponse,
    FacetBucket,
    QueryRequest,
    QueryResponse,
    QueryResultResponse,
    SnapshotRef,
)
from ckp.privacy import PrivacyGate

_MAX_SNIPPET = 320
_WHITESPACE = re.compile(r"\s+")

#: Relevance cutoff (issue #62). Both bars are derived from the measured
#: score distribution of the 16-question benchmark set against the real
#: pilot corpus (per-result scores, coverage, and field decomposition are
#: in #62), not picked by feel:
#:
#: * ``_CUTOFF_FLOOR``: every observed score-1 result was a single generic
#:   token matching in one field (body or metadata) -- noise in all 16
#:   distributions -- and no expected result anywhere scored below 2. The
#:   floor is capped at the query's own token count: every benchmark
#:   question has three or more tokens, so the measured "score 1 is noise"
#:   claim is evidence about matching a small fraction of a longer query,
#:   not about single-token queries, where a score-1 body match is full
#:   coverage of everything the caller asked and must stay servable.
#: * ``_CUTOFF_RATIO_*``: no expected result scored below 2/3 of its
#:   question's top score, and the bound is tight -- the cross-note
#:   question q06's second expected note sits at exactly 2 against a top
#:   of 3, so any stricter ratio starts dropping correct answers. The
#:   comparison is done in integers (``3 * score >= 2 * top``) so no
#:   float rounding can move that boundary.
_CUTOFF_FLOOR = 2
_CUTOFF_RATIO_NUM = 2
_CUTOFF_RATIO_DEN = 3


class GatewayPolicyError(ValueError):
    """The composition root supplied inconsistent privacy dependencies."""


def catalog_response(
    catalog: CatalogSnapshot,
    request: CatalogRequest,
) -> CatalogResponse:
    """Project one already privacy-gated Catalog into the public schema."""
    # Admission already happened in CatalogBuilder. Metadata filtering,
    # count, facets, and only then pagination preserve privacy-before-
    # aggregation from the frozen sink Decision.
    filtered = tuple(
        entry
        for entry in catalog.entries
        if request.type is None or entry.type == request.type
    )
    total = len(filtered)
    page = filtered[request.offset : request.offset + request.limit]
    next_offset = (
        request.offset + request.limit
        if request.offset + request.limit < total
        else None
    )
    return CatalogResponse(
        revision=_snapshot_ref(catalog),
        items=[_catalog_item(entry) for entry in page],
        total=total,
        facets=CatalogFacets(
            types=_facet([entry.type for entry in filtered]),
            content_categories=_facet([entry.content_category for entry in filtered]),
        ),
        next_offset=next_offset,
    )


def query_response(
    catalog: CatalogSnapshot,
    request: QueryRequest,
) -> QueryResponse:
    """Rank one already privacy-gated Catalog and add bound citations.

    Scored entries pass a relevance cutoff before pagination (issue #62):
    an entry is served only when its score clears both the absolute floor
    (``_CUTOFF_FLOOR``) and the top-relative bar (``_CUTOFF_RATIO_NUM /
    _CUTOFF_RATIO_DEN`` of the highest score for this query). Weak
    generic-token matches -- score>0 was the only bar before -- previously
    rode along in every response and dominated its token weight.

    ``total`` counts the results that survive the cutoff, not the raw
    score>0 match count: it is the number of results a caller could
    actually page through with ``limit``/``offset``-style requests, and a
    served relevance policy that hides a result from every page must not
    still advertise it in the count.
    """
    tokens = _tokens(request.query)
    scored = [
        (score, entry)
        for entry in catalog.entries
        if (score := _score(entry, tokens)) > 0
    ]
    scored.sort(key=lambda item: (-item[0], item[1].concept_id))
    if scored:
        top_score = scored[0][0]
        floor = min(_CUTOFF_FLOOR, len(tokens))
        scored = [
            (score, entry)
            for score, entry in scored
            if score >= floor
            and _CUTOFF_RATIO_DEN * score >= _CUTOFF_RATIO_NUM * top_score
        ]
    total = len(scored)
    results = [
        QueryResultResponse(
            concept_id=entry.concept_id,
            id=entry.id,
            title=entry.title,
            type=entry.type,
            score=score,
            snippet=_snippet(entry.body, tokens),
            citation=_citation(entry),
        )
        for score, entry in scored[: request.limit]
    ]
    return QueryResponse(
        revision=_snapshot_ref(catalog),
        query=request.query,
        results=results,
        total=total,
    )


def _normalise(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _tokens(query: str) -> tuple[str, ...]:
    return tuple(token for token in _normalise(query).split() if token)


def _casefold_with_source_offsets(value: str) -> tuple[str, list[int]]:
    folded_parts: list[str] = []
    source_offsets: list[int] = []
    for index, character in enumerate(value):
        folded = character.casefold()
        folded_parts.append(folded)
        source_offsets.extend([index] * len(folded))
    return "".join(folded_parts), source_offsets


def _citation(entry: CatalogEntry) -> CitationResponse:
    return CitationResponse(
        concept_id=entry.citation.concept_id,
        path=entry.citation.path,
        index_revision=entry.citation.index_revision,
        bundle_commit=entry.citation.bundle_commit,
    )


def _snapshot_ref(catalog: CatalogSnapshot) -> SnapshotRef:
    return SnapshotRef(
        index_revision=catalog.index_revision,
        bundle_commit=catalog.bundle_commit,
    )


def _catalog_item(entry: CatalogEntry) -> CatalogItemResponse:
    return CatalogItemResponse(
        concept_id=entry.concept_id,
        id=entry.id,
        title=entry.title,
        type=entry.type,
        description=entry.description,
        status=entry.status,
        content_category=entry.content_category,
        topic_id=entry.topic_id,
        module_id=entry.module_id,
        tags=list(entry.tags) if entry.tags is not None else None,
        citation=_citation(entry),
    )


def _facet(values: list[str | None]) -> list[FacetBucket]:
    counts = Counter(value for value in values if value is not None)
    return [FacetBucket(value=value, count=counts[value]) for value in sorted(counts)]


def _score(entry: CatalogEntry, tokens: tuple[str, ...]) -> int:
    title = _normalise(entry.title or "")
    description = _normalise(entry.description or "")
    metadata = _normalise(
        " ".join(
            value
            for value in (
                entry.type,
                entry.content_category,
                *(entry.tags or ()),
            )
            if value is not None
        )
    )
    body = _normalise(entry.body)
    score = 0
    for token in tokens:
        if token in title:
            score += 4
        if token in description:
            score += 2
        if token in metadata:
            score += 2
        if token in body:
            score += 1
    return score


def _snippet(body: str, tokens: tuple[str, ...]) -> str:
    compact = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", body)).strip()
    if len(compact) <= _MAX_SNIPPET:
        return compact
    normalised, source_offsets = _casefold_with_source_offsets(compact)
    folded_positions = [normalised.find(token) for token in tokens]
    positions = [
        source_offsets[position] for position in folded_positions if position >= 0
    ]
    centre = min(positions, default=0)
    start = max(0, centre - (_MAX_SNIPPET // 3))
    end = min(len(compact), start + _MAX_SNIPPET)
    start = max(0, end - _MAX_SNIPPET)
    snippet = compact[start:end]
    if start:
        snippet = "…" + snippet[1:]
    if end < len(compact):
        snippet = snippet[:-1] + "…"
    return snippet


class KnowledgeGateway:
    """Read-only service whose every result originates in one gated Catalog."""

    def __init__(
        self,
        snapshot_cache: SnapshotCache,
        catalog_builder: CatalogBuilder,
        privacy_gate: PrivacyGate,
    ) -> None:
        if catalog_builder.privacy_gate is not privacy_gate:
            raise GatewayPolicyError(
                "Catalog and Gateway must share the exact PrivacyGate instance"
            )
        self._snapshot_cache = snapshot_cache
        self._catalog_builder = catalog_builder
        self._privacy_gate = privacy_gate

    @property
    def privacy_gate(self) -> PrivacyGate:
        return self._privacy_gate

    def _catalog(self) -> CatalogSnapshot:
        snapshot = self._snapshot_cache.get()
        return self._catalog_builder.build(snapshot)

    def catalog(
        self,
        limit: int | CatalogRequest,
        offset: int | None = None,
        note_type: str | None = None,
    ) -> CatalogResponse:
        request = (
            limit
            if isinstance(limit, CatalogRequest)
            else CatalogRequest(limit=limit, offset=offset or 0, type=note_type)
        )
        catalog = self._catalog()
        return catalog_response(catalog, request)

    def query(
        self, query: str | QueryRequest, limit: int | None = None
    ) -> QueryResponse:
        request = (
            query
            if isinstance(query, QueryRequest)
            else QueryRequest(query=query, limit=limit or 10)
        )
        catalog = self._catalog()
        return query_response(catalog, request)


__all__ = [
    "GatewayPolicyError",
    "KnowledgeGateway",
    "catalog_response",
    "query_response",
]
