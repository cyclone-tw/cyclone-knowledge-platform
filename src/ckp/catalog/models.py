"""Internal deterministic Catalog projection types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
    concept_id: str
    path: str
    index_revision: str
    bundle_commit: str | None


@dataclass(frozen=True)
class CatalogEntry:
    concept_id: str
    id: str | None
    title: str | None
    type: str | None
    description: str | None
    status: str | None
    content_category: str | None
    topic_id: str | None
    module_id: str | None
    tags: tuple[str, ...] | None
    citation: Citation
    # Gateway needs body text for deterministic lexical matching, but this is
    # never a Catalog API field and never appears in repr/error payloads.
    body: str = field(repr=False)


@dataclass(frozen=True)
class CatalogSnapshot:
    entries: tuple[CatalogEntry, ...]
    index_revision: str
    bundle_commit: str | None


class CatalogUnavailableError(RuntimeError):
    """No complete snapshot can safely back a Catalog response."""


class PrivacyBindingError(RuntimeError):
    """A purported admission does not bind the member being projected."""


__all__ = [
    "CatalogEntry",
    "CatalogSnapshot",
    "CatalogUnavailableError",
    "Citation",
    "PrivacyBindingError",
]
