"""The frozen C3 HTTP request and response schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CatalogRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    type: str | None = None


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=256)
    limit: int = Field(default=10, ge=1, le=20)

    @field_validator("query", mode="before")
    @classmethod
    def strip_and_require_query_text(cls, value):
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must contain non-whitespace text")
        return stripped


class SnapshotRef(BaseModel):
    index_revision: str
    bundle_commit: str | None


class CitationResponse(BaseModel):
    concept_id: str
    path: str
    index_revision: str
    bundle_commit: str | None


class CatalogItemResponse(BaseModel):
    concept_id: str
    id: str | None
    title: str | None
    type: str | None
    description: str | None
    status: str | None
    content_category: str | None
    topic_id: str | None
    module_id: str | None
    tags: list[str] | None
    citation: CitationResponse


class FacetBucket(BaseModel):
    value: str
    count: int


class CatalogFacets(BaseModel):
    types: list[FacetBucket]
    content_categories: list[FacetBucket]


class CatalogResponse(BaseModel):
    revision: SnapshotRef
    items: list[CatalogItemResponse]
    total: int
    facets: CatalogFacets
    next_offset: int | None


class QueryResultResponse(BaseModel):
    concept_id: str
    id: str | None
    title: str | None
    type: str | None
    score: int
    snippet: str = Field(max_length=320)
    citation: CitationResponse


class QueryResponse(BaseModel):
    revision: SnapshotRef
    query: str
    results: list[QueryResultResponse]
    total: int


__all__ = [
    "CatalogFacets",
    "CatalogItemResponse",
    "CatalogRequest",
    "CatalogResponse",
    "CitationResponse",
    "FacetBucket",
    "QueryRequest",
    "QueryResponse",
    "QueryResultResponse",
    "SnapshotRef",
]
