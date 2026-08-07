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
    # The note's first Markdown "# " heading text (or None if it has none).
    # Same definition ``cyclone-wiki`` ``wiki_dashboard_export.first_heading_body``
    # and ``development_candidates.note_title`` both use for the "title" they
    # export -- a *different* field from ``title`` above, which is the
    # frontmatter ``title:`` scalar. Kept separate rather than merged so a
    # comparison against an export-derived title can use the matching
    # definition instead of silently comparing two different fields that
    # happen to share a name (issue #48).
    heading_title: str | None
    # frontmatter ``candidate_status:`` -- the field
    # ``development_candidates.candidate_record`` actually reads for the
    # ``"status"`` it exports for ``development_candidates`` zone items
    # (``frontmatter.get("candidate_status", "")``), distinct from the plain
    # ``status:`` frontmatter field above that ``projects``/``topics`` read
    # (issue #48).
    candidate_status: str | None
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
