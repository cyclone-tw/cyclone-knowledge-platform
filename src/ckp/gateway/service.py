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

#: Lexical scoring (issue #66) and relevance cutoff (issue #62 -> #66).
#: Every constant below is pinned by the measured distribution of the
#: 16-question benchmark set against the real pilot corpus (the per-result
#: scores, coverage, and candidate counts are in #66's probe tables), not
#: picked by feel. #62 proved with the same method that presence-based
#: scoring cannot rank a small on-topic note above a large note that
#: merely mentions every query token (its r05 inversion); this issue
#: replaces the per-token presence points with a BM25F-family weighted
#: term frequency, so that *how often* and *in how small a note* a token
#: matches decides the score, while per-token matching itself stays
#: substring containment (frozen; the real questions were written against
#: that semantic -- r03 depends on a long Chinese token matching as one
#: substring).
#:
#: * ``_K1`` (saturation): a token's weighted term frequency contributes
#:   ``wtf * (k1+1) / (wtf + k1)`` -- repeated mentions help, with
#:   diminishing returns, up to ``k1 + 1`` per token. The probe swept k1
#:   over {0.5, 1.2, 2, 3, 4, 6, 8, 12, 16, 24, 48}: at 3.0 the binding
#:   kill margin is widest (r01's densest sibling falls to 0.698 of the
#:   top score; smaller k1 collapses scores back toward coverage counting
#:   -- #62's proven-impossible family -- and larger k1 lets one dense
#:   token dominate whole-query coverage, re-raising r05's siblings).
#: * length normalization (the BM25 ``b``, folded in at its swept optimum
#:   ``b = 1.0``): each field's term frequency is divided by
#:   ``len(field) / avg len(field over this catalog)``, so a 3931-token
#:   note that mentions everything once no longer outweighs a 904-token
#:   note that is about the query. The averages are computed from the
#:   already privacy-gated catalog of this request, never from a wider
#:   corpus -- a scoring statistic over unadmitted notes would leak their
#:   vocabulary into served rankings (#66 hard constraint).
#: * IDF is deliberately absent, by measurement rather than principle
#:   (#66 named it a candidate): on this corpus the tokens that separate
#:   an expected note from its nearest sibling (r01's "note") are
#:   corpus-common (low IDF) while tokens both share ("retrieval") are
#:   rare, so IDF compresses exactly the gaps the cutoff needs -- every
#:   swept (k1, b, idf=on) configuration had strictly worse binding
#:   margins than its idf=off twin.
#:
#: The cutoff itself (issue #62's serving rule, re-derived on the new
#: distribution as #66 requires):
#:
#: * ``_CUTOFF_FLOOR_COVERAGE``: a result must match at least
#:   ``min(2, len(tokens))`` distinct query tokens. Same evidence as
#:   #62's score floor -- every observed noise result was a single
#:   generic-token match -- restated over coverage because BM25F scores
#:   are no longer integer presence points. The cap at the query's own
#:   token count keeps single-token queries servable (#62's scoped
#:   gateway fixture: matching the only token asked for is full
#:   coverage, not noise).
#: * two-regime top-relative bar: #62 pinned a structural wall for any
#:   single ratio -- q06's second expected answer must survive at 0.648
#:   of the top score while r01's nearest non-answer must fall at 0.698,
#:   and the non-answer is the stronger match on every per-result
#:   relevance signal the probe measured (score ratio, coverage,
#:   density, idf mass, token length). What distinguishes them is the
#:   *candidate field*: q06 has ``2`` floor-passing candidates in the
#:   whole catalog, r01 has ``6``. So the bar adapts to how contested
#:   the query is: with at most ``_SPARSE_CANDIDATE_MAX`` candidates the
#:   catalog itself says the topic is niche, and the runner-up is
#:   served at a third of the top score (``_CUTOFF_RATIO_SPARSE_NUM /
#:   _CUTOFF_RATIO_SPARSE_DEN``); in a crowded field a result must
#:   hold three quarters of the top score (``_CUTOFF_RATIO_CROWDED_NUM
#:   / _CUTOFF_RATIO_CROWDED_DEN``) to justify its context tokens.
#:   The measured distributions pin every constant two-sidedly:
#:   q06 (keep at 2 candidates) and r03 (kill its 0.544 sibling at 3)
#:   force the boundary to exactly 2. 1/3 sits under q06's second
#:   expected answer in *both* measured catalogs -- 0.648 of the top
#:   score against the real pilot corpus, and 0.435 in the
#:   synthetic-only frozen corpus, where nothing dwarfs the average
#:   note length, the tiny notes' term frequencies leave saturation,
#:   and the same pair of notes drops toward its raw frequency ratio
#:   -- while staying far above the strongest sparse non-answer
#:   anywhere (0.025). 3/4 sits between r01's 0.698 and r02's second
#:   expected answer at 0.865. Comparisons are done in
#:   integer-rational form (``den * score >= num * top``) so no
#:   derived float constant sits in the comparison path.
_K1 = 3.0
_CUTOFF_FLOOR_COVERAGE = 2
_SPARSE_CANDIDATE_MAX = 2
_CUTOFF_RATIO_SPARSE_NUM = 1
_CUTOFF_RATIO_SPARSE_DEN = 3
_CUTOFF_RATIO_CROWDED_NUM = 3
_CUTOFF_RATIO_CROWDED_DEN = 4

#: Scored fields and their weights (unchanged from the presence scorer:
#: title 4, description 2, metadata 2, body 1) -- #66 moves the
#: aggregation under them from presence points to weighted term
#: frequency; it does not reweigh the fields.
_FIELD_WEIGHTS = (4.0, 2.0, 2.0, 1.0)


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

    Ranking is the BM25F-family weighted-term-frequency score derived in
    issue #66 (see the scoring constants above): per-token substring
    matching is unchanged, but repeated mentions in a small note now
    outrank single mentions in a large one, which is what lets the
    serving cutoff below separate "the note about the query" from "a
    note that mentions every query word".

    Scored entries pass a relevance cutoff before pagination (#62,
    re-derived in #66): an entry is served only when it matches at least
    ``min(_CUTOFF_FLOOR_COVERAGE, len(tokens))`` distinct query tokens
    and its score clears the top-relative bar for this query's candidate
    field -- a third of the top score (``_CUTOFF_RATIO_SPARSE_NUM /
    _CUTOFF_RATIO_SPARSE_DEN``) when at most ``_SPARSE_CANDIDATE_MAX``
    candidates pass the floor, three quarters
    (``_CUTOFF_RATIO_CROWDED_NUM / _CUTOFF_RATIO_CROWDED_DEN``)
    otherwise. Weak generic-token matches --
    score>0 was the only bar before #62 -- previously rode along in
    every response and dominated its token weight.

    ``total`` counts the results that survive the cutoff, not the raw
    match count: it is the number of results a caller could actually
    page through with ``limit``/``offset``-style requests, and a served
    relevance policy that hides a result from every page must not still
    advertise it in the count (#62).
    """
    tokens = _tokens(request.query)
    entry_fields = [(entry, _entry_fields(entry)) for entry in catalog.entries]
    averages = _field_averages([fields for _, fields in entry_fields])
    floor = min(_CUTOFF_FLOOR_COVERAGE, len(tokens))
    scored = []
    for entry, fields in entry_fields:
        score, coverage = _score(fields, tokens, averages)
        if score > 0 and coverage >= floor:
            scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], item[1].concept_id))
    if scored:
        top_score = scored[0][0]
        if len(scored) <= _SPARSE_CANDIDATE_MAX:
            ratio_num = _CUTOFF_RATIO_SPARSE_NUM
            ratio_den = _CUTOFF_RATIO_SPARSE_DEN
        else:
            ratio_num = _CUTOFF_RATIO_CROWDED_NUM
            ratio_den = _CUTOFF_RATIO_CROWDED_DEN
        scored = [
            (score, entry)
            for score, entry in scored
            if ratio_den * score >= ratio_num * top_score
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
    """Normalised query tokens, deduplicated in first-seen order.

    Duplicates collapse so repeating a word neither doubles its score
    contribution nor counts twice toward the coverage floor -- Codex's
    #68 round-1 review reproduced ``"needle needle missing"`` serving a
    note that matched ``needle`` alone, because each repetition
    incremented coverage past ``min(2, len(tokens))``. After
    deduplication that query *is* the two-token query it asks about,
    and a one-token match stays below its floor.
    """
    return tuple(dict.fromkeys(_normalise(query).split()))


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


def _entry_fields(entry: CatalogEntry) -> tuple[str, str, str, str]:
    """The four scored fields, normalised, in ``_FIELD_WEIGHTS`` order."""
    return (
        _normalise(entry.title or ""),
        _normalise(entry.description or ""),
        _normalise(
            " ".join(
                value
                for value in (
                    entry.type,
                    entry.content_category,
                    *(entry.tags or ()),
                )
                if value is not None
            )
        ),
        _normalise(entry.body),
    )


def _field_averages(
    fields_list: list[tuple[str, str, str, str]],
) -> tuple[float, ...]:
    """Mean normalised length per field over this request's catalog.

    This is the length-normalization yardstick (the scoring constants'
    ``b = 1.0`` note): computed from the already privacy-gated catalog
    only, per request, so no statistic about unadmitted notes can reach
    a served score.
    """
    if not fields_list:
        return (0.0,) * len(_FIELD_WEIGHTS)
    count = len(fields_list)
    return tuple(
        sum(len(fields[index]) for fields in fields_list) / count
        for index in range(len(_FIELD_WEIGHTS))
    )


def _score(
    fields: tuple[str, str, str, str],
    tokens: tuple[str, ...],
    averages: tuple[float, ...],
) -> tuple[float, int]:
    """One entry's BM25F-family score and its distinct-token coverage.

    Per token: each field contributes its substring-occurrence count,
    weighted by the field's weight and divided by the field's relative
    length (``len / avg len``, the ``b = 1.0`` normalization -- a field
    can only reach ``avg > 0`` when some entry, possibly this one, has
    content, so a positive count never divides by zero); the summed
    weighted term frequency then saturates as ``wtf * (k1+1) / (wtf +
    k1)``. Matching stays substring containment: a token counts exactly
    when the old presence scorer would have matched it (``count > 0``
    iff ``token in field``).
    """
    score = 0.0
    coverage = 0
    for token in tokens:
        weighted_tf = 0.0
        for field_text, weight, average in zip(
            fields, _FIELD_WEIGHTS, averages, strict=True
        ):
            occurrences = field_text.count(token)
            if not occurrences:
                continue
            weighted_tf += weight * occurrences * average / len(field_text)
        if weighted_tf > 0:
            coverage += 1
            score += weighted_tf * (_K1 + 1.0) / (weighted_tf + _K1)
    return score, coverage


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
