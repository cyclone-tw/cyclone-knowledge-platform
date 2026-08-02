"""C6 bounded context request and response schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ckp.gateway.models import CitationResponse, SnapshotRef


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Scope and bounds are intentionally absent. The server-resolved grant
    # owns domain, privacy, capabilities, item cap, and token upper bound.
    query: str = Field(min_length=1, max_length=256)

    @field_validator("query", mode="before")
    @classmethod
    def strip_and_require_query_text(cls, value):
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must contain non-whitespace text")
        return stripped


class ContextItemResponse(BaseModel):
    concept_id: str
    id: str | None
    title: str | None
    snippet: str = Field(max_length=320)
    citation: CitationResponse


class ContextResponse(BaseModel):
    revision: SnapshotRef
    domain: str
    items: list[ContextItemResponse]
    context: str
    item_limit: int
    token_upper_bound_limit: int
    token_upper_bound_used: int
    token_upper_bound_method: str
    truncated: bool


__all__ = ["ContextItemResponse", "ContextRequest", "ContextResponse"]
