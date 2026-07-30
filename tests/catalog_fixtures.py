"""Synthetic-only helpers for C3 Catalog and Gateway tests."""

from __future__ import annotations

from pathlib import Path

from ckp.bundle import AnchoredBundleReader, SnapshotCache
from ckp.catalog import CatalogBuilder
from ckp.gateway import KnowledgeGateway
from ckp.privacy import FrontmatterClassifier, PrivacyClass, PrivacyGate


def write_note(
    root: Path,
    relative_path: str,
    *,
    privacy: str | None = "public",
    title: str | None = "Synthetic note",
    note_type: str | None = "Concept",
    description: str | None = None,
    status: str | None = "stable",
    content_category: str | None = "development",
    note_id: str | None = None,
    legacy_uuid: str | None = None,
    topic_id: str | None = None,
    module_id: str | None = None,
    tags: list[str] | None = None,
    extra: list[str] | None = None,
    body: str = "Synthetic public body.",
) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    for key, value in (
        ("privacy", privacy),
        ("title", title),
        ("type", note_type),
        ("description", description),
        ("status", status),
        ("content_category", content_category),
        ("id", note_id),
        ("uuid", legacy_uuid),
        ("topic_id", topic_id),
        ("module_id", module_id),
    ):
        if value is not None:
            lines.append(f"{key}: {value}")
    if tags is not None:
        lines.append("tags:")
        lines.extend(f"  - {tag}" for tag in tags)
    if extra:
        lines.extend(extra)
    lines.extend(["---", "", f"# {title or 'Untitled synthetic'}", "", body, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def services(root: Path):
    reader = AnchoredBundleReader(root)
    cache = SnapshotCache(reader, "**/*.md", expected_profile_version=None)
    gate = PrivacyGate(
        FrontmatterClassifier(reader),
        frozenset({PrivacyClass.PUBLIC}),
    )
    builder = CatalogBuilder(gate)
    gateway = KnowledgeGateway(cache, builder, gate)
    return cache, gate, builder, gateway
