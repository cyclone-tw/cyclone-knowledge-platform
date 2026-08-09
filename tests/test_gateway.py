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
    """Two byte-identical runs, bounded snippets, revision-bound citations.

    The expected order pins #66's length normalization: ``a.md`` is a
    compact note whose text is essentially its "kettle" title, while
    ``b.md`` buries the same match in 800 padding characters -- under
    the presence scorer ``b`` won on field points (title + description),
    under weighted term frequency the padding dilutes it below the
    compact note. Both still clear the sparse-field cutoff.
    """
    long_body = "before " + ("x" * 400) + " kettle " + ("y" * 400)
    write_note(
        tmp_path,
        "b.md",
        title="Kettle title",
        description="The kettle reference.",
        body=long_body,
    )
    write_note(tmp_path, "a.md", title="Other kettle", body="Nothing else here.")
    client = _client(tmp_path)

    first = client.post("/query", json={"query": "KETTLE", "limit": 20})
    second = client.post("/query", json={"query": "KETTLE", "limit": 20})

    assert first.status_code == 200
    assert first.json() == second.json()
    body = first.json()
    assert body["query"] == "KETTLE"
    assert body["total"] == 2
    assert [item["concept_id"] for item in body["results"]] == ["a", "b"]
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


def test_query_cutoff_drops_weak_matches_and_counts_only_survivors(
    tmp_path: Path,
) -> None:
    """Issues #62/#66, direction one: weak associations are cut, not served.

    Three candidates pass the coverage floor, so this is a crowded field
    and the strict bar applies: ``partial.md`` and ``second.md`` each
    match only two of three tokens with no repeated mentions and fall
    below three quarters of the top score; ``tagalong.md`` is a single
    generic-token match, cut by the floor before any ratio is read.
    None may be served, and ``total`` must count survivors only -- the
    pre-#62 behavior served all four and reported ``total == 4``.
    """
    write_note(
        tmp_path,
        "target.md",
        title="Umbra logistics manifest",
        body="Umbra logistics manifest, expanded.",
    )
    write_note(tmp_path, "partial.md", title="Logistics", body="Umbra appears once.")
    write_note(tmp_path, "second.md", title="Manifest", body="Umbra appears twice.")
    write_note(tmp_path, "tagalong.md", title="Unrelated", body="Mentions logistics.")

    response = _client(tmp_path).post(
        "/query", json={"query": "umbra logistics manifest"}
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["concept_id"] for item in body["results"]] == ["target"]
    assert body["total"] == 1


def test_query_cutoff_keeps_multi_target_boundary_ties(tmp_path: Path) -> None:
    """Issues #62/#66, direction two: a multi-target answer set is not over-cut.

    The measured distributions' binding case (q06) is a two-candidate
    field whose second expected note holds well under three quarters of
    the top score (0.648 real, 0.435 synthetic-only): only the sparse
    regime's lenient bar serves it. Here ``second.md`` matches two of
    three tokens against a full-coverage top note -- served, because
    after the floor cuts the single-token ``junk.md`` only two
    candidates remain.
    """
    write_note(tmp_path, "first.md", title="Unrelated A", body="tide caves mapping")
    write_note(tmp_path, "second.md", title="Unrelated B", body="tide caves")
    write_note(tmp_path, "junk.md", title="Unrelated C", body="mapping alone")

    response = _client(tmp_path).post("/query", json={"query": "tide caves mapping"})

    assert response.status_code == 200
    body = response.json()
    assert [item["concept_id"] for item in body["results"]] == ["first", "second"]
    assert body["total"] == 2


def test_query_all_weak_matches_serve_nothing(tmp_path: Path) -> None:
    """Issue #62: the floor holds even when every match is weak.

    Both notes match one of the two query tokens, so every candidate
    ties for top and the ratio bar alone would keep everything; only
    the coverage floor says a page of single-token matches is not an
    answer. Empty is the honest response (same doctrine as q10's empty
    no-answer question).
    """
    write_note(tmp_path, "noise1.md", title="Alpha", body="sentinel word")
    write_note(tmp_path, "noise2.md", title="Beta", body="sentinel again")

    response = _client(tmp_path).post("/query", json={"query": "sentinel missingword"})

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["total"] == 0


def test_query_single_token_body_match_stays_servable(tmp_path: Path) -> None:
    """Issue #62: the floor is capped at the query's own token count.

    For a one-token query, a body-only match covers everything the caller
    asked; the multi-token floor evidence (a single-token match is a
    generic tag-along) does not apply and the result must be served.
    """
    write_note(tmp_path, "only.md", title="Unrelated", body="A quokka sighting.")

    response = _client(tmp_path).post("/query", json={"query": "quokka"})

    assert response.status_code == 200
    assert [item["concept_id"] for item in response.json()["results"]] == ["only"]
    assert response.json()["total"] == 1


def test_query_sparse_field_serves_the_runner_up_a_crowd_would_cut(
    tmp_path: Path,
) -> None:
    """Issue #66's two-regime cutoff, pinned from both sides.

    The same runner-up -- three fifths of the top score, two of three
    tokens matched -- is served when the whole catalog yields two
    floor-passing candidates (the sparse bar is one third of the top
    score) and cut when one more floor-passing candidate makes the
    field crowded and raises the bar to three quarters. This pins
    ``_SPARSE_CANDIDATE_MAX`` at exactly 2, the boundary the measured
    distributions fix two-sidedly: q06 must keep its second expected
    answer in a two-candidate field, r03 must kill its strongest
    sibling in a three-candidate one.
    """
    for root in (tmp_path / "sparse", tmp_path / "crowded"):
        write_note(
            root,
            "anchor.md",
            title="Ember cartography atlas",
            body="Ember cartography atlas, expanded.",
        )
        write_note(root, "runner.md", title="Cartography", body="Ember maps.")
    write_note(tmp_path / "crowded", "crowd.md", title="Atlas", body="Ember appears.")
    query = {"query": "ember cartography atlas"}

    sparse = _client(tmp_path / "sparse").post("/query", json=query)
    crowded = _client(tmp_path / "crowded").post("/query", json=query)

    assert sparse.status_code == 200
    assert crowded.status_code == 200
    sparse_body = sparse.json()
    crowded_body = crowded.json()
    assert [item["concept_id"] for item in sparse_body["results"]] == [
        "anchor",
        "runner",
    ]
    assert sparse_body["total"] == 2
    assert [item["concept_id"] for item in crowded_body["results"]] == ["anchor"]
    assert crowded_body["total"] == 1


def test_query_length_normalized_frequency_outranks_a_padded_mention(
    tmp_path: Path,
) -> None:
    """Issue #66's core fix -- the r05 inversion in miniature.

    ``mention.md`` matches every query token and even carries them all
    in its title, but buries them under thirty sentences of padding;
    ``dense.md`` is a compact note whose body actually discusses the
    query (every token, repeated). The presence scorer ranked the
    padded title-holder first -- #62's measured r05 inversion, which no
    cutoff could fix because the wrong score was the higher one --
    while weighted term frequency with length normalization ranks the
    compact note first, and in this three-candidate crowded field the
    strict bar then cuts both the padded mention and the two-token
    aside: the same-work questions' exact-only outcome in miniature.
    """
    write_note(
        tmp_path,
        "mention.md",
        title="Vault ledger inspection",
        body=("Routine unrelated filler. " * 30) + "One vault ledger inspection line.",
    )
    write_note(
        tmp_path,
        "dense.md",
        title="Capture stub",
        body=(
            "Vault ledger inspection: vault ledger inspection findings, vault checks."
        ),
    )
    write_note(
        tmp_path,
        "aside.md",
        title="Ledger",
        body="Vault reference among several other unrelated filler sentences.",
    )

    response = _client(tmp_path).post(
        "/query", json={"query": "vault ledger inspection"}
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["concept_id"] for item in body["results"]] == ["dense"]
    assert body["total"] == 1


def test_query_crowded_field_keeps_a_close_second(tmp_path: Path) -> None:
    """The crowded bar is three quarters, bracketed from both sides.

    Three candidates pass the floor, so the strict bar applies. The
    second candidate holds 0.84 of the top score -- above 3/4, served;
    it stands in for r02's second expected answer at 0.865, the
    real-corpus keep that pins the bar from above. The third holds
    0.60 -- below 3/4, cut. Tightening the bar toward 0.865 flips the
    first assertion; loosening it to 1/2 flips the second.
    """
    write_note(
        tmp_path,
        "first.md",
        title="Ember cartography atlas",
        body="Ember cartography atlas, expanded.",
    )
    write_note(
        tmp_path,
        "second.md",
        title="Ember cartography",
        body="Atlas entries continue.",
    )
    write_note(tmp_path, "third.md", title="Cartography", body="Ember maps.")

    response = _client(tmp_path).post(
        "/query", json={"query": "ember cartography atlas"}
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["concept_id"] for item in body["results"]] == ["first", "second"]
    assert body["total"] == 2


def test_query_sparse_field_still_cuts_a_distant_runner_up(
    tmp_path: Path,
) -> None:
    """The sparse bar is lenient, not absent: a third of the top score.

    Two candidates pass the floor, so the lenient bar applies -- but
    the faint one holds only 0.23 of the top score, below 1/3, and is
    cut: a two-candidate field does not serve every floor-passer.
    Loosening the sparse bar to a fifth serves it.
    """
    write_note(
        tmp_path,
        "anchor.md",
        title="Ember cartography atlas",
        body="Ember cartography atlas, expanded.",
    )
    write_note(
        tmp_path,
        "faint.md",
        title="Unrelated topic",
        body="Ember appears once among cartography filler in this sentence.",
    )

    response = _client(tmp_path).post(
        "/query", json={"query": "ember cartography atlas"}
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["concept_id"] for item in body["results"]] == ["anchor"]
    assert body["total"] == 1


def test_query_duplicate_query_tokens_do_not_inflate_the_floor(
    tmp_path: Path,
) -> None:
    """Codex #68 round 1: repeated tokens must not buy floor coverage.

    ``needle needle missing`` deduplicates to the two-token query it
    actually asks, so a note matching ``needle`` alone covers one of
    two tokens and falls at the floor -- before the fix each repetition
    incremented coverage, so the needle-only match passed ``min(2, 3)``
    and was served. The same dedup keeps single-term semantics:
    ``needle needle`` *is* a one-token query, full coverage, served.
    """
    write_note(tmp_path, "only.md", title="Unrelated", body="A needle sighting.")
    client = _client(tmp_path)

    padded = client.post("/query", json={"query": "needle needle missing"})
    collapsed = client.post("/query", json={"query": "needle needle"})

    assert padded.status_code == 200
    assert padded.json()["results"] == []
    assert padded.json()["total"] == 0
    assert collapsed.status_code == 200
    assert [item["concept_id"] for item in collapsed.json()["results"]] == ["only"]
    assert collapsed.json()["total"] == 1


def test_query_partial_title_match_falls_at_the_coverage_floor(
    tmp_path: Path,
) -> None:
    """Issue #66's floor counts distinct matched tokens, not field points.

    Under #62's score floor a title match was worth four points, so a
    one-token title hit on a three-token query rode over the two-point
    bar. The coverage floor asks how much of the query matched -- one
    token of three is a mention, whichever field it landed in -- so the
    only candidate is cut and empty is the honest answer (same doctrine
    as the all-weak test above).
    """
    write_note(tmp_path, "titleonly.md", title="Granite", body="Nothing more.")

    response = _client(tmp_path).post("/query", json={"query": "granite quarry survey"})

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["total"] == 0


def test_snippet_offset_tracks_casefold_expansion(tmp_path: Path) -> None:
    body = ("ß" * 400) + " needle " + ("z" * 400)
    write_note(
        tmp_path,
        "public.md",
        title="Expansion",
        description="Where the needle hides.",
        body=body,
    )

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
