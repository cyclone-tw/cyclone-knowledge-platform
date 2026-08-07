"""Deterministic, privacy-gated Catalog projection."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from catalog_fixtures import services, write_note
from ckp.catalog import CatalogBuilder, PrivacyBindingError
from ckp.privacy import Classified, PrivacyClass, PrivacyGate


def test_builder_requires_a_privacy_gate() -> None:
    parameter = inspect.signature(CatalogBuilder.__init__).parameters["privacy_gate"]
    assert parameter.default is inspect.Parameter.empty


class _UnboundPublicClassifier:
    def classify(self, path: Path):
        return Classified(privacy=PrivacyClass.PUBLIC, source="mutation")

    def classify_member(self, member):
        return Classified(privacy=PrivacyClass.PUBLIC, source="mutation")


def test_catalog_rejects_admission_not_bound_to_snapshot_bytes(tmp_path: Path) -> None:
    write_note(tmp_path, "public.md", title="Visible")
    cache, _, _, _ = services(tmp_path)
    gate = PrivacyGate(
        _UnboundPublicClassifier(),
        frozenset({PrivacyClass.PUBLIC}),
    )

    with pytest.raises(PrivacyBindingError):
        CatalogBuilder(gate).build(cache.get())


def test_catalog_is_deterministic_for_the_same_snapshot(tmp_path: Path) -> None:
    write_note(tmp_path, "z.md", title="Zulu")
    write_note(tmp_path, "nested/a.md", title="Alpha")
    cache, _, builder, _ = services(tmp_path)
    snapshot = cache.get()

    first = builder.build(snapshot)
    second = builder.build(snapshot)

    assert first == second
    assert [entry.concept_id for entry in first.entries] == ["nested/a", "z"]
    assert first.index_revision == snapshot.index_revision


def test_only_admitted_public_notes_enter_the_catalog(tmp_path: Path) -> None:
    write_note(tmp_path, "public.md", privacy="public", title="Visible")
    write_note(tmp_path, "internal.md", privacy="internal", title="Hidden internal")
    write_note(tmp_path, "sensitive.md", privacy="sensitive", title="Hidden sensitive")
    write_note(
        tmp_path,
        "student.md",
        privacy="student-private",
        title="Invented Example Student",
    )
    write_note(tmp_path, "missing.md", privacy=None, title="Undetermined")
    cache, _, builder, _ = services(tmp_path)

    catalog = builder.build(cache.get())

    assert [entry.title for entry in catalog.entries] == ["Visible"]
    rendered = repr(catalog)
    for forbidden in (
        "Hidden internal",
        "Hidden sensitive",
        "Invented Example Student",
        "Undetermined",
    ):
        assert forbidden not in rendered


def test_projection_does_not_invent_profile_metadata(tmp_path: Path) -> None:
    write_note(
        tmp_path,
        "legacy.md",
        title=None,
        note_type=None,
        status=None,
        content_category=None,
        legacy_uuid="0192f000-0000-7000-8000-00000000ffff",
        extra=["unknown_future_field: plausible-looking"],
    )
    cache, _, builder, _ = services(tmp_path)

    entry = builder.build(cache.get()).entries[0]

    assert entry.concept_id == "legacy"
    assert entry.id is None  # legacy uuid is not silently promoted to Profile id
    assert entry.title is None
    assert entry.type is None
    assert entry.status is None
    assert entry.content_category is None
    assert not hasattr(entry, "unknown_future_field")


def test_projection_keeps_only_correctly_typed_known_values(tmp_path: Path) -> None:
    path = tmp_path / "wrong-types.md"
    path.write_text(
        "---\n"
        "privacy: public\n"
        "title:\n  nested: value\n"
        "type: [Concept]\n"
        "tags: not-a-list\n"
        "---\n\n# Synthetic\n",
        encoding="utf-8",
    )
    cache, _, builder, _ = services(tmp_path)

    entry = builder.build(cache.get()).entries[0]

    assert entry.title is None
    assert entry.type is None
    assert entry.tags is None


def test_duplicate_yaml_keys_make_the_note_unprojectable(tmp_path: Path) -> None:
    write_note(
        tmp_path,
        "duplicate.md",
        extra=["description: first", "description: second"],
    )
    cache, _, builder, _ = services(tmp_path)

    assert builder.build(cache.get()).entries == ()


def test_every_entry_citation_is_bound_to_its_member_and_snapshot(
    tmp_path: Path,
) -> None:
    write_note(tmp_path, "nested/note.md", title="Cited")
    cache, _, builder, _ = services(tmp_path)
    snapshot = cache.get()

    entry = builder.build(snapshot).entries[0]

    assert entry.citation.concept_id == entry.concept_id == "nested/note"
    assert entry.citation.path == "nested/note.md"
    assert entry.citation.index_revision == snapshot.index_revision
    assert entry.citation.bundle_commit == snapshot.bundle_commit
    host_home_prefix = "/" + "Users/"
    assert host_home_prefix not in repr(entry.citation)


def test_heading_title_and_frontmatter_title_are_independent_fields(
    tmp_path: Path,
) -> None:
    """#48: benchmarks/export_compare.py compares export-derived titles
    (every wiki-export.v1 zone sources "title" from the note's first
    Markdown "# " heading, not frontmatter) against CatalogEntry.
    heading_title, never CatalogEntry.title (the frontmatter title: field).
    A note whose heading text and frontmatter title differ must expose both
    correctly and independently -- this is the exact case that used to
    produce a definitional false-positive metadata_mismatch."""
    path = tmp_path / "diverging-title.md"
    path.write_text(
        "---\n"
        "privacy: public\n"
        "title: Frontmatter Title\n"
        "type: Concept\n"
        "status: stable\n"
        "---\n\n# A Completely Different Heading\n\nbody\n",
        encoding="utf-8",
    )
    cache, _, builder, _ = services(tmp_path)

    entry = builder.build(cache.get()).entries[0]

    assert entry.title == "Frontmatter Title"
    assert entry.heading_title == "A Completely Different Heading"


def test_heading_title_is_none_when_note_has_no_markdown_heading(
    tmp_path: Path,
) -> None:
    path = tmp_path / "no-heading.md"
    path.write_text(
        "---\nprivacy: public\ntitle: Has A Title\n---\n\n"
        "body with no heading at all\n",
        encoding="utf-8",
    )
    cache, _, builder, _ = services(tmp_path)

    entry = builder.build(cache.get()).entries[0]

    assert entry.title == "Has A Title"
    assert entry.heading_title is None


def test_candidate_status_is_read_independently_of_status(tmp_path: Path) -> None:
    """#48's status half: development_candidates zone items export
    "status" from frontmatter candidate_status:, a different field from
    the plain status: frontmatter field projects/topics read. Both must be
    exposed on CatalogEntry, independently."""
    path = tmp_path / "candidate.md"
    path.write_text(
        "---\n"
        "privacy: public\n"
        "title: Candidate\n"
        "status: inbox\n"
        "candidate_status: needs-validation\n"
        "---\n\n# Candidate\n\nbody\n",
        encoding="utf-8",
    )
    cache, _, builder, _ = services(tmp_path)

    entry = builder.build(cache.get()).entries[0]

    assert entry.status == "inbox"
    assert entry.candidate_status == "needs-validation"


def test_public_catalog_omits_non_commit_provenance_from_citations(
    tmp_path: Path,
) -> None:
    write_note(tmp_path, "public.md", title="Cited")
    host_path = "/" + "Users/private/runtime"
    (tmp_path / ".bundle-commit").write_text(host_path + "\n", encoding="utf-8")
    cache, _, builder, _ = services(tmp_path)

    source_snapshot = cache.get()
    catalog = builder.build(source_snapshot)

    # C1 /revision retains its evidence semantics. The public C3 projection
    # must not echo a malformed stamp as host provenance.
    assert source_snapshot.bundle_commit == host_path
    assert catalog.bundle_commit is None
    assert catalog.entries[0].citation.bundle_commit is None
    assert host_path not in repr(catalog)


def test_public_catalog_keeps_commit_shaped_provenance(tmp_path: Path) -> None:
    write_note(tmp_path, "public.md", title="Cited")
    commit = "a" * 40
    (tmp_path / ".bundle-commit").write_text(commit + "\n", encoding="utf-8")
    cache, _, builder, _ = services(tmp_path)

    catalog = builder.build(cache.get())

    assert catalog.bundle_commit == commit
    assert catalog.entries[0].citation.bundle_commit == commit
