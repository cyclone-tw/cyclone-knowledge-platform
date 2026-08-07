"""P8 (issue #29): current export vs generated Catalog.

Two layers, matching the AGENTS.md §9 lesson from the first batch ("pure
functions test well, wiring does not test itself"):

* pure-function tests for ``diff_export_against_catalog`` and
  ``load_current_export`` -- no filesystem beyond a JSON file, no privacy
  gate, no pilot binding;
* an end-to-end test for ``compare_export_to_catalog`` against a synthetic
  pilot corpus (a ``tmp_path`` wiki root plus an injected frozen manifest,
  same pattern as ``tests/test_pilot_no_vendoring.py``) -- this is the
  wiring test: it goes through ``bind_pilot_corpus``, the real privacy
  gate, ``AnchoredBundleReader``, and ``CatalogBuilder``, not a mock of any
  of them.

Never touches the real Cyclone-Wiki checkout: every note here is synthetic
text written under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from benchmarks.export_compare import (
    EXPORT_PATH_ENV,
    ExportEntry,
    ExportUnavailable,
    compare_export_to_catalog,
    default_export_path,
    diff_export_against_catalog,
    load_current_export,
    resolve_export_path,
)

from ckp.catalog.models import CatalogEntry, CatalogSnapshot, Citation
from ckp.config import load_config
from ckp.pilot.manifest import PILOT_NOTE_PATHS, PilotManifest, PilotManifestEntry

# --- helpers -----------------------------------------------------------


def _catalog_entry(
    path: str, *, title: str = "Title", status: str = "active"
) -> CatalogEntry:
    concept_id = path[:-3]
    return CatalogEntry(
        concept_id=concept_id,
        id=None,
        title=title,
        type="Procedure",
        description=None,
        status=status,
        content_category="meta",
        topic_id=None,
        module_id=None,
        tags=None,
        citation=Citation(
            concept_id=concept_id,
            path=path,
            index_revision="sha256:test",
            bundle_commit=None,
        ),
        body="body text never leaves this synthetic fixture",
    )


def _full_catalog(**overrides: dict[str, str]) -> CatalogSnapshot:
    entries = tuple(
        _catalog_entry(path, **overrides.get(path, {})) for path in PILOT_NOTE_PATHS
    )
    return CatalogSnapshot(
        entries=entries, index_revision="sha256:test", bundle_commit=None
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _note(
    *, note_type: str = "Procedure", title: str = "Title", status: str = "active"
) -> str:
    return (
        "---\n"
        f"type: {note_type}\n"
        "privacy: internal\n"
        f"title: {title}\n"
        f"status: {status}\n"
        "---\n\n# note\n\nbody\n"
    )


def _write_synthetic_pilot_corpus(
    root, *, title: str = "Title", status: str = "active"
):
    """Write all six pilot notes under ``root`` and return a matching frozen
    manifest, mirroring ``tests/test_pilot_no_vendoring.py``'s pattern."""
    texts = {path: _note(title=title, status=status) for path in PILOT_NOTE_PATHS}
    for path, text in texts.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    manifest = PilotManifest(
        entries=tuple(
            PilotManifestEntry(relative_path=path, content_sha256=_sha256(texts[path]))
            for path in PILOT_NOTE_PATHS
        )
    )
    return manifest


def _write_export(path, zones: dict, *, schema: str = "wiki-export.v1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": schema,
        "exported_at": "2026-08-01T00:00:00Z",
        "commit": "abc1234",
        "zones": zones,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- resolve_export_path -------------------------------------------------


def test_default_export_path_matches_dashboard_wiki_adapter_default(tmp_path) -> None:
    assert default_export_path(tmp_path) == tmp_path / ".cache" / "wiki-export.v1.json"


def test_resolve_export_path_uses_env_override(tmp_path, monkeypatch) -> None:
    override = tmp_path / "elsewhere.json"
    monkeypatch.setenv(EXPORT_PATH_ENV, str(override))
    assert resolve_export_path(tmp_path) == override


def test_resolve_export_path_falls_back_to_default_when_env_unset(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv(EXPORT_PATH_ENV, raising=False)
    assert resolve_export_path(tmp_path) == default_export_path(tmp_path)


# --- load_current_export: never silently "empty" --------------------------


def test_missing_export_file_raises_export_unavailable(tmp_path) -> None:
    with pytest.raises(ExportUnavailable, match="unreadable"):
        load_current_export(tmp_path / "does-not-exist.json")


def test_invalid_json_raises_export_unavailable(tmp_path) -> None:
    bad = tmp_path / "wiki-export.v1.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ExportUnavailable, match="not valid JSON"):
        load_current_export(bad)


def test_wrong_schema_raises_export_unavailable(tmp_path) -> None:
    bad = tmp_path / "wiki-export.v1.json"
    _write_export(bad, zones={}, schema="something-else.v9")
    with pytest.raises(ExportUnavailable, match="schema"):
        load_current_export(bad)


def test_missing_zones_object_raises_export_unavailable(tmp_path) -> None:
    bad = tmp_path / "wiki-export.v1.json"
    bad.write_text(
        json.dumps({"schema": "wiki-export.v1", "exported_at": "x", "commit": "y"}),
        encoding="utf-8",
    )
    with pytest.raises(ExportUnavailable, match="zones"):
        load_current_export(bad)


def test_load_current_export_only_reads_path_addressable_zones(tmp_path) -> None:
    """agent_activity / shared_now carry no per-item path and must be
    ignored, not crash and not silently included as unmatched noise."""
    export_file = tmp_path / "wiki-export.v1.json"
    _write_export(
        export_file,
        zones={
            "shared_now": {"preferences": ["should be ignored"]},
            "agent_activity": [{"agent_id": "X", "active_focus": ["ignored"]}],
            "projects": [
                {
                    "path": "Core/project-cyclone-okf-knowledge-contract.md",
                    "title": "T",
                    "status": "active",
                }
            ],
        },
    )
    result = load_current_export(export_file)
    assert len(result.entries) == 1
    assert result.entries[0].zone == "projects"
    assert result.entries[0].path == "Core/project-cyclone-okf-knowledge-contract.md"


def test_load_current_export_ignores_items_without_a_path(tmp_path) -> None:
    export_file = tmp_path / "wiki-export.v1.json"
    _write_export(export_file, zones={"projects": [{"title": "no path here"}]})
    result = load_current_export(export_file)
    assert result.entries == ()


# --- diff_export_against_catalog: pure classification ----------------------


def test_all_six_match_cleanly_produces_no_diffs() -> None:
    catalog = _full_catalog()
    entries = tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in PILOT_NOTE_PATHS
    )
    diffs, summary = diff_export_against_catalog(entries, catalog)
    assert diffs == []
    assert summary["matched_clean"] == 6
    assert summary["missing_from_export"] == 0
    assert summary["extra_in_export"] == 0
    assert summary["metadata_mismatch"] == 0
    assert summary["path_mismatch"] == 0


def test_path_absent_from_export_is_missing_from_export() -> None:
    catalog = _full_catalog()
    entries = tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in PILOT_NOTE_PATHS[1:]
    )
    diffs, summary = diff_export_against_catalog(entries, catalog)
    assert summary["missing_from_export"] == 1
    missing = [d for d in diffs if d["category"] == "missing_from_export"]
    assert missing == [{"category": "missing_from_export", "path": PILOT_NOTE_PATHS[0]}]


def test_out_of_scope_export_entry_is_silently_excluded_not_reported() -> None:
    """Scope is the pilot manifest -- an export path outside it is out of
    scope, same reasoning D3 applies to the QMD baseline (different scope
    is not a comparison)."""
    catalog = _full_catalog()
    entries = tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in PILOT_NOTE_PATHS
    ) + (
        ExportEntry(
            zone="projects", path="Core/some-other-note.md", title="X", status=None
        ),
    )
    diffs, summary = diff_export_against_catalog(entries, catalog)
    assert diffs == []
    assert summary["matched_clean"] == 6


def test_duplicate_zone_entries_are_extra_in_export_not_merged_into_missing() -> None:
    catalog = _full_catalog()
    dup_path = PILOT_NOTE_PATHS[0]
    entries = tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in PILOT_NOTE_PATHS
    ) + (
        ExportEntry(
            zone="knowledge_feed", path=dup_path, title="Title", status="active"
        ),
    )
    diffs, summary = diff_export_against_catalog(entries, catalog)
    assert summary["extra_in_export"] == 1
    assert summary["missing_from_export"] == 0
    extra = [d for d in diffs if d["category"] == "extra_in_export"]
    assert extra[0]["path"] == dup_path
    assert extra[0]["zones"] == ["knowledge_feed"]


def test_title_mismatch_is_metadata_mismatch_with_field_name_only() -> None:
    catalog = _full_catalog()
    mismatched_path = PILOT_NOTE_PATHS[0]
    entries = tuple(
        ExportEntry(
            zone="projects",
            path=path,
            title=(
                "A Completely Different Title" if path == mismatched_path else "Title"
            ),
            status="active",
        )
        for path in PILOT_NOTE_PATHS
    )
    diffs, summary = diff_export_against_catalog(entries, catalog)
    assert summary["metadata_mismatch"] == 1
    mismatch = next(d for d in diffs if d["category"] == "metadata_mismatch")
    assert mismatch["path"] == mismatched_path
    assert mismatch["fields"] == ["title"]
    # Privacy boundary: only the field *name* is recorded, never the value.
    serialized = json.dumps(diffs)
    assert "A Completely Different Title" not in serialized
    assert serialized.count("title") >= 1  # only the field *name* survives


def test_status_mismatch_is_reported_by_field_name() -> None:
    catalog = _full_catalog()
    mismatched_path = PILOT_NOTE_PATHS[1]
    entries = tuple(
        ExportEntry(
            zone="projects",
            path=path,
            title="Title",
            status=("deprecated" if path == mismatched_path else "active"),
        )
        for path in PILOT_NOTE_PATHS
    )
    diffs, summary = diff_export_against_catalog(entries, catalog)
    mismatch = next(d for d in diffs if d["category"] == "metadata_mismatch")
    assert mismatch["fields"] == ["status"]
    assert "deprecated" not in json.dumps(diffs)


def test_export_field_none_is_not_compared_not_a_mismatch() -> None:
    """An export zone that never carries `status` (e.g. knowledge_feed) must
    not manufacture a mismatch against Catalog's status."""
    catalog = _full_catalog()
    entries = tuple(
        ExportEntry(zone="knowledge_feed", path=path, title="Title", status=None)
        for path in PILOT_NOTE_PATHS
    )
    diffs, summary = diff_export_against_catalog(entries, catalog)
    assert diffs == []
    assert summary["matched_clean"] == 6


def test_path_mismatch_when_raw_path_differs_but_normalizes_the_same() -> None:
    catalog = _full_catalog()
    odd_path = PILOT_NOTE_PATHS[0]
    entries = (
        ExportEntry(
            zone="projects", path="./" + odd_path, title="Title", status="active"
        ),
    ) + tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in PILOT_NOTE_PATHS[1:]
    )
    diffs, summary = diff_export_against_catalog(entries, catalog)
    assert summary["path_mismatch"] == 1
    mismatch = next(d for d in diffs if d["category"] == "path_mismatch")
    assert mismatch["path"] == odd_path


def test_missing_from_catalog_defensive_branch() -> None:
    """Not a normal outcome (bind_pilot_corpus fails loudly first in the
    real wiring) -- exercised directly via a synthetic incomplete Catalog."""
    incomplete = CatalogSnapshot(
        entries=(_catalog_entry(PILOT_NOTE_PATHS[0]),),
        index_revision="sha256:test",
        bundle_commit=None,
    )
    entries = tuple(
        ExportEntry(zone="projects", path=path, title="Title", status="active")
        for path in PILOT_NOTE_PATHS
    )
    diffs, summary = diff_export_against_catalog(entries, incomplete)
    missing_catalog = [d for d in diffs if d["category"] == "missing_from_catalog"]
    assert len(missing_catalog) == len(PILOT_NOTE_PATHS) - 1


# --- summary categories must stay separate (acceptance: 分類, not 一坨) ----


def test_summary_has_all_categories_as_distinct_keys() -> None:
    catalog = _full_catalog()
    diffs, summary = diff_export_against_catalog((), catalog)
    assert set(summary) == {
        "pilot_scope_count",
        "missing_from_export",
        "extra_in_export",
        "metadata_mismatch",
        "path_mismatch",
        "matched_clean",
    }
    assert summary["missing_from_export"] == 6


# --- compare_export_to_catalog: end-to-end wiring --------------------------


def test_compared_false_when_export_unavailable_has_no_diffs_key(tmp_path) -> None:
    """The single most important guard in this module: 'could not compare'
    must be structurally distinguishable from 'compared, found nothing'."""
    wiki_root = tmp_path / "wiki"
    manifest = _write_synthetic_pilot_corpus(wiki_root)
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(wiki_root)})

    report = compare_export_to_catalog(
        config,
        export_path=tmp_path / "no-such-export.json",
        frozen_manifest=manifest,
    )

    assert report["compared"] is False
    assert "diffs" not in report
    assert "summary" not in report
    assert report["export"]["status"] == "unavailable"
    assert "reason" in report["export"]


def test_compared_true_end_to_end_against_synthetic_corpus(tmp_path) -> None:
    wiki_root = tmp_path / "wiki"
    manifest = _write_synthetic_pilot_corpus(wiki_root, title="Title", status="active")
    export_path = tmp_path / "wiki-export.v1.json"
    _write_export(
        export_path,
        zones={
            "projects": [
                {"path": path, "title": "Title", "status": "active"}
                for path in PILOT_NOTE_PATHS
            ]
        },
    )
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(wiki_root)})

    report = compare_export_to_catalog(
        config, export_path=export_path, frozen_manifest=manifest
    )

    assert report["compared"] is True
    assert report["export"]["status"] == "ok"
    assert report["diffs"] == []
    assert report["summary"]["matched_clean"] == 6
    assert report["catalog"]["entry_count"] == 6


def test_compared_true_but_partial_coverage_reports_missing(tmp_path) -> None:
    """The realistic shape of the real wiki-export.v1.json today: it has no
    zone for Procedure/Decision notes at all, so most of the pilot corpus
    is structurally missing_from_export -- this must show up, not be
    hidden by only checking `compared`."""
    wiki_root = tmp_path / "wiki"
    manifest = _write_synthetic_pilot_corpus(wiki_root)
    export_path = tmp_path / "wiki-export.v1.json"
    covered = PILOT_NOTE_PATHS[-1]
    _write_export(
        export_path,
        zones={
            "development_candidates": [
                {"path": covered, "title": "Title", "status": "needs-validation"}
            ]
        },
    )
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(wiki_root)})

    report = compare_export_to_catalog(
        config, export_path=export_path, frozen_manifest=manifest
    )

    assert report["compared"] is True
    missing = [d for d in report["diffs"] if d["category"] == "missing_from_export"]
    assert len(missing) == len(PILOT_NOTE_PATHS) - 1
    assert report["summary"]["missing_from_export"] == len(PILOT_NOTE_PATHS) - 1


def test_result_is_reproducible_run_twice_same_inputs(tmp_path) -> None:
    wiki_root = tmp_path / "wiki"
    manifest = _write_synthetic_pilot_corpus(wiki_root)
    export_path = tmp_path / "wiki-export.v1.json"
    _write_export(
        export_path,
        zones={
            "projects": [
                {"path": path, "title": "Title", "status": "active"}
                for path in PILOT_NOTE_PATHS
            ]
        },
    )
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(wiki_root)})

    first = compare_export_to_catalog(
        config, export_path=export_path, frozen_manifest=manifest
    )
    second = compare_export_to_catalog(
        config, export_path=export_path, frozen_manifest=manifest
    )
    assert first == second


def test_pilot_binding_failure_propagates_not_swallowed(tmp_path) -> None:
    """No CKP_PILOT_WIKI_ROOT -> PilotBindingError, not an empty report."""
    from ckp.pilot.manifest import PilotBindingError

    config = load_config(env={})
    with pytest.raises(PilotBindingError, match="CKP_PILOT_WIKI_ROOT"):
        compare_export_to_catalog(config)


# --- CLI entrypoint ----------------------------------------------------


def test_main_returns_1_when_pilot_wiki_root_unset(monkeypatch, capsys) -> None:
    from benchmarks import export_compare

    monkeypatch.delenv("CKP_PILOT_WIKI_ROOT", raising=False)
    monkeypatch.delenv("CKP_CONFIG_FILE", raising=False)
    exit_code = export_compare.main()
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "pilot corpus binding failed" in captured.err


def test_main_returns_2_when_export_unavailable(tmp_path, monkeypatch, capsys) -> None:
    from benchmarks import export_compare

    wiki_root = tmp_path / "wiki"
    manifest = _write_synthetic_pilot_corpus(wiki_root)
    manifest_file = tmp_path / "pilot-manifest.toml"
    manifest_file.write_text(
        "\n".join(
            f'[[note]]\nrelative_path = "{entry.relative_path}"\n'
            f'content_sha256 = "{entry.content_sha256}"'
            for entry in manifest.entries
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CKP_PILOT_WIKI_ROOT", str(wiki_root))
    monkeypatch.setenv("CKP_PILOT_MANIFEST_PATH", str(manifest_file))
    monkeypatch.setenv(EXPORT_PATH_ENV, str(tmp_path / "does-not-exist.json"))
    monkeypatch.delenv("CKP_CONFIG_FILE", raising=False)

    exit_code = export_compare.main()
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "current export unavailable" in captured.err
    payload = json.loads(captured.out)
    assert payload["compared"] is False
    assert "diffs" not in payload
