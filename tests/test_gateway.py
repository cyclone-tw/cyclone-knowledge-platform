"""Public-only HTTP Catalog and bounded lexical query contract."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

from fastapi.testclient import TestClient

from catalog_fixtures import services, write_note
from ckp.app import create_app
from ckp.config import load_config
from ckp.gateway import KnowledgeGateway


def _client(root: Path) -> TestClient:
    config = load_config(env={"CKP_BUNDLE_ROOT": str(root)})
    return TestClient(create_app(config))


def test_gateway_requires_cache_builder_and_gate() -> None:
    parameters = inspect.signature(KnowledgeGateway.__init__).parameters
    for name in ("snapshot_cache", "catalog_builder", "privacy_gate"):
        assert parameters[name].default is inspect.Parameter.empty


def test_catalog_filters_privacy_before_count_facets_and_pagination(
    tmp_path: Path,
) -> None:
    write_note(tmp_path, "a.md", privacy="public", title="Visible", note_type="Concept")
    write_note(
        tmp_path, "b.md", privacy="internal", title="Internal", note_type="Source"
    )
    write_note(
        tmp_path, "c.md", privacy="sensitive", title="Sensitive", note_type="Source"
    )
    write_note(tmp_path, "d.md", privacy="student-private", title="Invented Student")
    write_note(tmp_path, "e.md", privacy=None, title="Undetermined")

    response = _client(tmp_path).get("/catalog")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert [item["title"] for item in body["items"]] == ["Visible"]
    assert body["facets"] == {
        "types": [{"value": "Concept", "count": 1}],
        "content_categories": [{"value": "development", "count": 1}],
    }
    rendered = response.text
    for forbidden in ("Internal", "Sensitive", "Invented Student", "Undetermined"):
        assert forbidden not in rendered


def test_catalog_exact_type_filter_and_bounded_pagination(tmp_path: Path) -> None:
    write_note(tmp_path, "a.md", title="A", note_type="Concept")
    write_note(tmp_path, "b.md", title="B", note_type="Source")
    write_note(tmp_path, "c.md", title="C", note_type="Source")
    client = _client(tmp_path)

    filtered = client.get("/catalog", params={"type": "Source", "limit": 1})
    page_two = client.get(
        "/catalog", params={"type": "Source", "limit": 1, "offset": 1}
    )

    assert filtered.json()["total"] == 2
    assert filtered.json()["next_offset"] == 1
    assert [item["title"] for item in filtered.json()["items"]] == ["B"]
    assert page_two.json()["next_offset"] is None
    assert [item["title"] for item in page_two.json()["items"]] == ["C"]


def test_catalog_rejects_caller_owned_privacy_policy_parameters(
    tmp_path: Path,
) -> None:
    write_note(tmp_path, "public.md", title="Visible")
    client = _client(tmp_path)

    for name in ("privacy", "admissible", "scope"):
        response = client.get("/catalog", params={name: "internal"})
        assert response.status_code == 422, name


def test_query_is_lexical_deterministic_bounded_and_cited(tmp_path: Path) -> None:
    long_body = "before " + ("x" * 400) + " kettle " + ("y" * 400)
    write_note(tmp_path, "b.md", title="Kettle title", body=long_body)
    write_note(tmp_path, "a.md", title="Other", body="A kettle in body.")
    client = _client(tmp_path)

    first = client.post("/query", json={"query": "KETTLE", "limit": 20})
    second = client.post("/query", json={"query": "KETTLE", "limit": 20})

    assert first.status_code == 200
    assert first.json() == second.json()
    body = first.json()
    assert body["query"] == "KETTLE"
    assert body["total"] == 2
    assert [item["concept_id"] for item in body["results"]] == ["b", "a"]
    assert all(0 < len(item["snippet"]) <= 320 for item in body["results"])
    assert all(item["citation"] for item in body["results"])
    assert all(
        item["citation"]["index_revision"] == body["revision"]["index_revision"]
        for item in body["results"]
    )
    assert all(
        item["citation"]["concept_id"] == item["concept_id"]
        and item["citation"]["path"] == f"{item['concept_id']}.md"
        for item in body["results"]
    )


def test_snippet_offset_tracks_casefold_expansion(tmp_path: Path) -> None:
    body = ("ß" * 400) + " needle " + ("z" * 400)
    write_note(tmp_path, "public.md", title="Expansion", body=body)

    response = _client(tmp_path).post("/query", json={"query": "needle"})

    assert response.status_code == 200
    snippet = response.json()["results"][0]["snippet"]
    assert "needle" in snippet.casefold()
    assert len(snippet) <= 320


def test_query_excludes_every_non_public_or_undetermined_match(tmp_path: Path) -> None:
    write_note(tmp_path, "public.md", privacy="public", title="Needle public")
    write_note(tmp_path, "internal.md", privacy="internal", title="Needle internal")
    write_note(tmp_path, "sensitive.md", privacy="sensitive", title="Needle sensitive")
    write_note(
        tmp_path,
        "student.md",
        privacy="student-private",
        title="Needle Invented Student",
    )
    write_note(tmp_path, "unknown.md", privacy=None, title="Needle undetermined")

    response = _client(tmp_path).post("/query", json={"query": "needle"})

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert [item["title"] for item in response.json()["results"]] == ["Needle public"]
    assert "Invented Student" not in response.text


def test_query_schema_rejects_policy_override_and_unbounded_inputs(
    tmp_path: Path,
) -> None:
    write_note(tmp_path, "public.md", title="Visible")
    client = _client(tmp_path)

    assert (
        client.post(
            "/query", json={"query": "visible", "privacy": "internal"}
        ).status_code
        == 422
    )
    assert (
        client.post("/query", json={"query": "visible", "limit": 21}).status_code == 422
    )
    assert client.post("/query", json={"query": " "}).status_code == 422
    assert client.get("/catalog", params={"limit": 101}).status_code == 422

    trimmed = client.post("/query", json={"query": " " + ("x" * 256) + " "})
    assert trimmed.status_code == 200
    assert trimmed.json()["query"] == "x" * 256
    assert client.post("/query", json={"query": "x" * 257}).status_code == 422


def test_gateway_and_revision_use_the_same_snapshot(tmp_path: Path) -> None:
    write_note(tmp_path, "public.md", title="Kettle")
    client = _client(tmp_path)

    gateway = client.post("/query", json={"query": "kettle"}).json()
    revision = client.get("/revision").json()

    assert gateway["revision"]["index_revision"] == revision["index_revision"]
    assert gateway["revision"]["bundle_commit"] == revision["bundle_commit"]
    assert client.app.state.snapshot_cache.rebuild_count == 1


def test_bundle_unavailable_has_a_fixed_non_leaking_error(tmp_path: Path) -> None:
    response = _client(tmp_path / "absent").get("/catalog")
    assert response.status_code == 503
    assert response.json() == {"detail": "bundle-unavailable"}


def test_hardlink_refusal_never_publishes_a_healthy_partial_snapshot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    write_note(root, "plain.md", title="Only safe member")
    outside = write_note(tmp_path, "outside.md", title="Refused hardlink")
    os.link(outside, root / "linked.md")
    client = _client(root)

    health = client.get("/health")
    revision = client.get("/revision")
    catalog = client.get("/catalog")
    query = client.post("/query", json={"query": "safe"})

    assert health.status_code == 503
    assert health.json()["checks"]["bundle_readable"] is False
    assert revision.status_code == 200
    assert revision.json()["index_revision"] is None
    assert revision.json()["sources"]["index_revision"] == "unknown"
    assert catalog.status_code == query.status_code == 503
    assert catalog.json() == query.json() == {"detail": "bundle-unavailable"}
    assert "linked.md" not in catalog.text


def test_service_layer_reuses_one_snapshot_even_if_cache_is_invalidated(
    tmp_path: Path,
) -> None:
    write_note(tmp_path, "public.md", title="Kettle")
    cache, _, _, gateway = services(tmp_path)

    response = gateway.query("kettle", 10)
    cache.invalidate()

    assert all(
        item.citation.index_revision == response.revision.index_revision
        for item in response.results
    )
