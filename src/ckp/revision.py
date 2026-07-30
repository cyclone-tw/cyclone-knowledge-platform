"""Revision reporting over the centralized anchored bundle snapshot.

The four Phase 3 fields remain exactly the C1 contract.  C3 changes only how
the evidence is obtained: note bytes, descriptor bytes, and the optional
commit stamp now come from :mod:`ckp.bundle`, so revision, privacy, Catalog,
and Gateway cannot drift into separate containment implementations.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ckp.bundle import (
    BUNDLE_COMMIT_STAMP,
    BUNDLE_DESCRIPTOR,
    INDEX_ALGORITHM,
    AnchoredBundleReader,
    BundleMember,
    BundleSnapshot,
    MemberRefused,
    SnapshotCache,
    SnapshotRaceError,
    compute_index_revision_from_members,
)
from ckp.config import Config

#: The API contract version consumers pin.  C3 adds endpoints without changing
#: the already published C1 response contracts, so the version stays 0.1.
API_VERSION = "0.1"


@dataclass(frozen=True)
class Revision:
    """The four contract fields, plus where each value came from."""

    profile_version: str | None
    api_version: str
    bundle_commit: str | None
    index_revision: str | None
    sources: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "api_version": self.api_version,
            "bundle_commit": self.bundle_commit,
            "index_revision": self.index_revision,
            "sources": dict(self.sources),
        }


def bundle_member_key(path: Path, bundle_root: Path) -> str | None:
    """Return the canonical digest key only after an anchored verified read."""
    result = AnchoredBundleReader(bundle_root).read_path(path)
    if isinstance(result, MemberRefused):
        return None
    assert isinstance(result, BundleMember)
    return result.digest_key


def read_bundle_descriptor(bundle_root: Path) -> dict[str, Any]:
    """Read and parse ``bundle.toml`` through the anchored reader."""
    result = AnchoredBundleReader(bundle_root).read_member(BUNDLE_DESCRIPTOR)
    if isinstance(result, MemberRefused):
        return {}
    try:
        parsed = tomllib.loads(result.content.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return {}
    return parsed


def _capture(bundle_root: Path, note_glob: str):
    reader = AnchoredBundleReader(bundle_root)
    try:
        return reader.capture(note_glob)
    except SnapshotRaceError:
        return None


def iter_note_paths(bundle_root: Path, note_glob: str) -> list[Path]:
    """Verified note paths, sorted by canonical key then actual POSIX path."""
    capture = _capture(bundle_root, note_glob)
    if capture is None:
        return []
    return [bundle_root / member.relative_path for member in capture.members]


def compute_index_revision(bundle_root: Path, note_glob: str) -> str | None:
    """Digest verified note bytes with the unchanged C1 golden algorithm."""
    capture = _capture(bundle_root, note_glob)
    if capture is None or not capture.readable:
        return None
    return compute_index_revision_from_members(capture.members)


def _one_shot_snapshot(
    bundle_root: Path,
    note_glob: str,
    expected_profile_version: str | None,
) -> BundleSnapshot | None:
    cache = SnapshotCache(
        AnchoredBundleReader(bundle_root),
        note_glob,
        expected_profile_version,
    )
    try:
        return cache.get()
    except SnapshotRaceError:
        return None


def resolve_bundle_commit(bundle_root: Path) -> tuple[str | None, str]:
    snapshot = _one_shot_snapshot(bundle_root, "**/*.md", None)
    if snapshot is None:
        return None, "unknown"
    return snapshot.bundle_commit, snapshot.sources["bundle_commit"]


def resolve_profile_version(
    bundle_root: Path,
    config: Config,
) -> tuple[str | None, str]:
    snapshot = _one_shot_snapshot(
        bundle_root,
        config.bundle_note_glob,
        config.profile_expected_version,
    )
    if snapshot is None:
        if config.profile_expected_version:
            return config.profile_expected_version, "config"
        return None, "unknown"
    return snapshot.profile_version, snapshot.sources["profile_version"]


def revision_from_snapshot(snapshot: BundleSnapshot) -> Revision:
    return Revision(
        profile_version=snapshot.profile_version,
        api_version=API_VERSION,
        bundle_commit=snapshot.bundle_commit,
        index_revision=snapshot.index_revision,
        sources={
            "profile_version": snapshot.sources["profile_version"],
            "api_version": "code",
            "bundle_commit": snapshot.sources["bundle_commit"],
            "index_revision": snapshot.sources["index_revision"],
        },
    )


def build_revision(
    config: Config, snapshot_cache: SnapshotCache | None = None
) -> Revision:
    """Build the C1 response, optionally reusing the app-owned C3 cache."""
    cache = snapshot_cache or SnapshotCache(
        AnchoredBundleReader(config.bundle_root),
        config.bundle_note_glob,
        config.profile_expected_version,
    )
    try:
        snapshot = cache.get()
    except SnapshotRaceError:
        expected = config.profile_expected_version
        return Revision(
            profile_version=expected,
            api_version=API_VERSION,
            bundle_commit=None,
            index_revision=None,
            sources={
                "profile_version": "config" if expected else "unknown",
                "api_version": "code",
                "bundle_commit": "unknown",
                "index_revision": "unknown",
            },
        )
    return revision_from_snapshot(snapshot)


def bundle_is_readable(
    bundle_root: Path,
    note_glob: str,
    snapshot_cache: SnapshotCache | None = None,
) -> bool:
    """True only when the same snapshot revision reports can be served."""
    if snapshot_cache is not None:
        try:
            return snapshot_cache.get().index_revision is not None
        except SnapshotRaceError:
            return False
    return compute_index_revision(bundle_root, note_glob) is not None


__all__ = [
    "API_VERSION",
    "BUNDLE_COMMIT_STAMP",
    "BUNDLE_DESCRIPTOR",
    "INDEX_ALGORITHM",
    "Revision",
    "build_revision",
    "bundle_is_readable",
    "bundle_member_key",
    "compute_index_revision",
    "iter_note_paths",
    "read_bundle_descriptor",
    "resolve_bundle_commit",
    "resolve_profile_version",
    "revision_from_snapshot",
]
