"""Structural tripwires for the C3 read-only result pipeline."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from catalog_fixtures import services, write_note
from ckp.app import PUBLIC_HTTP_ADMISSIBLE
from ckp.catalog import CatalogBuilder
from ckp.gateway import KnowledgeGateway
from ckp.gateway.models import CatalogItemResponse, QueryResultResponse
from ckp.privacy import FrontmatterClassifier, PrivacyClass, PrivacyGate
from conftest import REPO_ROOT

PIPELINE_ROOTS = (
    REPO_ROOT / "src" / "ckp" / "catalog",
    REPO_ROOT / "src" / "ckp" / "gateway",
)
PIPELINE_FILES = (
    REPO_ROOT / "src" / "ckp" / "revision.py",
    REPO_ROOT / "src" / "ckp" / "privacy" / "classifier.py",
)


def _pipeline_sources() -> list[Path]:
    sources = list(PIPELINE_FILES)
    for root in PIPELINE_ROOTS:
        sources.extend(root.rglob("*.py"))
    return sorted(sources)


def test_result_pipeline_has_no_direct_path_reads() -> None:
    """Revision, privacy, Catalog, and Gateway must share the anchored reader.

    This is intentionally an AST guard rather than a text search: prose that
    explains ``Path.read_bytes`` must not trip it, while a dead-code bypass
    must. Actual member bytes may only enter through ``ckp.bundle``.
    """
    offenders: list[str] = []
    for source in _pipeline_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if node.func.attr in {"read_bytes", "read_text"}:
                offenders.append(
                    f"{source.relative_to(REPO_ROOT)}:{node.lineno}:{node.func.attr}"
                )
    assert not offenders, "direct path reads bypass ckp.bundle:\n" + "\n".join(
        offenders
    )


def test_every_policy_dependency_stays_required() -> None:
    constructors = {
        FrontmatterClassifier: ("bundle_root",),
        PrivacyGate: ("classifier", "admissible"),
        CatalogBuilder: ("privacy_gate",),
        KnowledgeGateway: ("snapshot_cache", "catalog_builder", "privacy_gate"),
    }
    for constructor, names in constructors.items():
        parameters = inspect.signature(constructor.__init__).parameters
        for name in names:
            assert parameters[name].default is inspect.Parameter.empty


def test_anonymous_http_policy_is_exactly_public() -> None:
    assert frozenset({PrivacyClass.PUBLIC}) == PUBLIC_HTTP_ADMISSIBLE


def test_result_citations_are_required_fields() -> None:
    assert CatalogItemResponse.model_fields["citation"].is_required()
    assert QueryResultResponse.model_fields["citation"].is_required()


class _OneSnapshotCache:
    """A cache double that makes a second read in one result path explode."""

    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def get(self):
        self.calls += 1
        assert self.calls == 1, "one response mixed more than one snapshot"
        return self.snapshot


def test_each_gateway_result_path_reads_exactly_one_snapshot(tmp_path: Path) -> None:
    write_note(tmp_path, "public.md", title="Kettle")
    cache, gate, builder, _ = services(tmp_path)
    snapshot = cache.get()

    catalog_cache = _OneSnapshotCache(snapshot)
    catalog_gateway = KnowledgeGateway(catalog_cache, builder, gate)
    catalog = catalog_gateway.catalog(limit=20)

    query_cache = _OneSnapshotCache(snapshot)
    query_gateway = KnowledgeGateway(query_cache, builder, gate)
    query = query_gateway.query("kettle")

    assert catalog_cache.calls == query_cache.calls == 1
    assert catalog.revision.index_revision == snapshot.index_revision
    assert query.revision.index_revision == snapshot.index_revision
    assert all(
        item.citation.index_revision == catalog.revision.index_revision
        for item in catalog.items
    )
    assert all(
        item.citation.index_revision == query.revision.index_revision
        for item in query.results
    )
