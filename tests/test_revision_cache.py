"""Snapshot caching must save work without making stale revision claims."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import ckp.bundle as bundle_module
from ckp.bundle import AnchoredBundleReader, SnapshotCache, SnapshotRaceError


def _cache(root: Path) -> SnapshotCache:
    return SnapshotCache(
        AnchoredBundleReader(root),
        "**/*.md",
        expected_profile_version=None,
    )


def test_same_filesystem_snapshot_reuses_the_exact_snapshot(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_bytes(b"one\n")
    cache = _cache(tmp_path)

    first = cache.get()
    second = cache.get()

    assert second is first
    assert cache.rebuild_count == 1


def test_explicit_invalidation_forces_a_rebuild(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_bytes(b"one\n")
    cache = _cache(tmp_path)
    first = cache.get()

    cache.invalidate()
    second = cache.get()

    assert second is not first
    assert second.index_revision == first.index_revision
    assert cache.rebuild_count == 2


def test_changed_bytes_are_revalidated_without_a_stale_revision(tmp_path: Path) -> None:
    note = tmp_path / "a.md"
    note.write_bytes(b"one\n")
    cache = _cache(tmp_path)
    before = cache.get()

    note.write_bytes(b"two\n")
    after = cache.get()

    assert after is not before
    assert after.index_revision != before.index_revision
    assert cache.rebuild_count == 2


def test_content_hash_not_stat_metadata_decides_a_cache_hit(
    tmp_path: Path, monkeypatch
) -> None:
    """Same-size writes can share timestamps on a portable/bind filesystem."""
    original_signature = bundle_module._file_signature

    def coarse_signature(value):
        return replace(
            original_signature(value),
            modified_ns=0,
            changed_ns=0,
        )

    monkeypatch.setattr(bundle_module, "_file_signature", coarse_signature)
    note = tmp_path / "a.md"
    note.write_bytes(b"one\n")
    cache = _cache(tmp_path)
    before = cache.get()

    note.write_bytes(b"two\n")
    after = cache.get()

    assert after is not before
    assert after.index_revision != before.index_revision


def test_cache_is_owned_per_instance_not_process_global(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_bytes(b"one\n")
    first_cache = _cache(tmp_path)
    second_cache = _cache(tmp_path)

    assert first_cache.get() is not second_cache.get()
    assert first_cache.rebuild_count == second_cache.rebuild_count == 1


def test_empty_bundle_has_an_honest_unavailable_snapshot(tmp_path: Path) -> None:
    snapshot = _cache(tmp_path).get()
    assert snapshot.members == ()
    assert snapshot.index_revision is None


def test_peek_is_none_before_the_first_get(tmp_path: Path) -> None:
    """#39: /revision compares peek() against a fresh recompute. Before any
    request has ever been served, there is nothing to have gone stale from,
    so peek() must be None rather than an invented snapshot."""
    cache = _cache(tmp_path)
    assert cache.peek() is None


def test_peek_returns_the_last_snapshot_get_materialized(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_bytes(b"one\n")
    cache = _cache(tmp_path)
    served = cache.get()
    assert cache.peek() is served


def test_peek_never_reprobes_the_filesystem(tmp_path: Path) -> None:
    """The whole point of peek(): unlike get(), it must report what was
    served *before* this call, even if the bundle changed on disk since."""
    note = tmp_path / "a.md"
    note.write_bytes(b"one\n")
    cache = _cache(tmp_path)
    before = cache.get()

    note.write_bytes(b"two\n")
    peeked = cache.peek()

    assert peeked is before
    assert peeked.index_revision != cache.get().index_revision


def test_cache_does_not_publish_when_before_and_after_probes_disagree(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "a.md").write_bytes(b"stable bytes\n")
    reader = AnchoredBundleReader(tmp_path)
    original_probe = reader.probe
    calls = 0

    def alternating_probe(note_glob: str):
        nonlocal calls
        calls += 1
        probe = original_probe(note_glob)
        if calls % 2 == 0:
            changed_descriptor = replace(
                probe.descriptor,
                refusal="synthetic-probe-change",
            )
            return replace(probe, descriptor=changed_descriptor)
        return probe

    monkeypatch.setattr(reader, "probe", alternating_probe)
    cache = SnapshotCache(reader, "**/*.md", expected_profile_version=None)

    with pytest.raises(SnapshotRaceError):
        cache.get()

    assert calls == 6
    assert cache.rebuild_count == 0
