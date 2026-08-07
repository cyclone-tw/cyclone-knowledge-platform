"""Strict YAML parsing for only the metadata the Gateway actually needs."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from ckp.bundle import BundleMember
from ckp.catalog.models import CatalogEntry, Citation


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader plus duplicate-key refusal (never YAML last-key-wins)."""


def _construct_unique_mapping(
    loader: _StrictSafeLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, Hashable):
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            )
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found a duplicate key",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _document_parts(content: bytes) -> tuple[dict[str, Any], str] | None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = [line.rstrip("\r") for line in text.split("\n")]
    if not lines or lines[0] != "---":
        return None
    closing = next(
        (index for index in range(1, len(lines)) if lines[index] == "---"),
        None,
    )
    if closing is None:
        return None
    raw_frontmatter = "\n".join(lines[1:closing])
    try:
        metadata = yaml.load(raw_frontmatter, Loader=_StrictSafeLoader)
    except yaml.YAMLError:
        return None
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) for key in metadata
    ):
        return None
    body = "\n".join(lines[closing + 1 :])
    return metadata, body


def _string(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None


def _tags(metadata: dict[str, Any]) -> tuple[str, ...] | None:
    value = metadata.get("tags")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return tuple(value)


def _heading_title(body: str) -> str | None:
    """The note's first Markdown ``# `` heading text, or ``None``.

    Same definition ``cyclone-wiki`` ``wiki_dashboard_export.first_heading_body``
    and ``development_candidates.note_title`` both use: scan the
    post-frontmatter body for the first line starting with ``"# "`` and take
    the rest of that line, stripped. ``body`` here is already the
    post-frontmatter text (see :func:`_document_parts`), so this does not
    re-strip frontmatter the way the wiki-side helpers do.
    """
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def project_member(
    member: BundleMember,
    *,
    index_revision: str,
    bundle_commit: str | None,
) -> CatalogEntry | None:
    parsed = _document_parts(member.content)
    if parsed is None:
        return None
    metadata, body = parsed
    if not member.relative_path.endswith(".md"):
        return None
    concept_id = member.relative_path[:-3]
    citation = Citation(
        concept_id=concept_id,
        path=member.relative_path,
        index_revision=index_revision,
        bundle_commit=bundle_commit,
    )
    return CatalogEntry(
        concept_id=concept_id,
        id=_string(metadata, "id"),
        title=_string(metadata, "title"),
        type=_string(metadata, "type"),
        description=_string(metadata, "description"),
        status=_string(metadata, "status"),
        content_category=_string(metadata, "content_category"),
        topic_id=_string(metadata, "topic_id"),
        module_id=_string(metadata, "module_id"),
        tags=_tags(metadata),
        citation=citation,
        heading_title=_heading_title(body),
        candidate_status=_string(metadata, "candidate_status"),
        body=body,
    )


__all__ = ["project_member"]
