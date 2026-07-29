"""The four revision fields the OKF contract requires.

``profile_version``, ``api_version``, ``bundle_commit``, ``index_revision``
(contract §3 Phase 3).

Two design rules shape this module.

**Nothing is fabricated.** When the evidence for a field does not exist the
field is ``None`` and its entry in ``sources`` says so. A plausible-looking
placeholder would make a broken deployment indistinguishable from a healthy
one, which is exactly what revision reporting exists to prevent.

**``index_revision`` is derived, not declared.** Contract §5.5 makes "the
rebuild is not reproducible from the same commit" a rollback trigger, so the
digest is a pure function of the bundle's note bytes and their paths: same
input, same value, on any host.
"""

from __future__ import annotations

import hashlib
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ckp.config import Config

#: The API contract version consumers pin. Single source of truth -- the HTTP
#: app takes its version from here, and a test asserts they never drift.
API_VERSION = "0.1"

#: Bumping the digest algorithm must change every index_revision, so the label
#: is hashed in rather than merely documented.
INDEX_ALGORITHM = "ckp-index-v1"

#: Optional file in the bundle root describing the bundle itself.
BUNDLE_DESCRIPTOR = "bundle.toml"

#: Optional build- or deploy-time provenance stamp, used when git is not
#: available (a container image has no repository to ask).
BUNDLE_COMMIT_STAMP = ".bundle-commit"

_GIT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class Revision:
    """The four contract fields, plus where each one actually came from."""

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


def read_bundle_descriptor(bundle_root: Path) -> dict[str, Any]:
    """Parse ``bundle.toml``. A missing or malformed descriptor is not fatal.

    The descriptor is self-declared evidence: useful when present, never
    required. Callers see an empty mapping and fall back to reporting unknown.
    """
    path = bundle_root / BUNDLE_DESCRIPTOR
    try:
        raw = path.read_bytes()
    except OSError:
        return {}
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return {}


def iter_note_paths(bundle_root: Path, note_glob: str) -> list[Path]:
    """Every note in the bundle, in a deterministic order.

    Sorted by POSIX-normalised relative path so the digest does not depend on
    filesystem walk order, which differs between machines.
    """
    if not bundle_root.is_dir():
        return []
    notes = [p for p in bundle_root.glob(note_glob) if p.is_file()]
    return sorted(notes, key=lambda p: p.relative_to(bundle_root).as_posix())


def compute_index_revision(bundle_root: Path, note_glob: str) -> str | None:
    """Digest the bundle's notes: ``sha256:<hex>``, or None when unreadable.

    Files are read as **bytes**. Reading them as text would let Python's
    universal-newline translation collapse a CRLF note and an LF note onto the
    same digest, so two materially different bundles would report one
    revision. Do not "simplify" this to ``read_text``.
    """
    if not bundle_root.is_dir():
        return None

    digest = hashlib.sha256()
    digest.update(INDEX_ALGORITHM.encode("utf-8"))
    digest.update(b"\n")

    for path in iter_note_paths(bundle_root, note_glob):
        try:
            content = path.read_bytes()
        except OSError:
            return None
        rel = path.relative_to(bundle_root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).hexdigest().encode("ascii"))
        digest.update(b"\n")

    return f"sha256:{digest.hexdigest()}"


def _git_commit(bundle_root: Path) -> str | None:
    """``git rev-parse HEAD`` for the bundle, or None if git cannot answer.

    A bundle vendored inside a repository takes that repository's commit; a
    bundle that is a wiki checkout takes the wiki's. Every failure mode --
    git absent, not a work tree, ownership refusal, hang -- resolves to None
    rather than to a guess.
    """
    if not bundle_root.is_dir():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(bundle_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _stamped_commit(bundle_root: Path) -> str | None:
    try:
        stamp = (bundle_root / BUNDLE_COMMIT_STAMP).read_text(encoding="utf-8")
    except OSError:
        return None
    commit = stamp.strip()
    return commit or None


def resolve_bundle_commit(bundle_root: Path) -> tuple[str | None, str]:
    """Resolve the bundle commit and name the evidence it came from.

    Live git wins over a stamp: a checkout that has moved on since the stamp
    was written should report where it actually is.
    """
    commit = _git_commit(bundle_root)
    if commit:
        return commit, "git"
    commit = _stamped_commit(bundle_root)
    if commit:
        return commit, "stamp"
    return None, "unknown"


def resolve_profile_version(
    bundle_root: Path,
    config: Config,
) -> tuple[str | None, str]:
    """The bundle's own declaration wins; config expresses an expectation."""
    declared = read_bundle_descriptor(bundle_root).get("profile_version")
    if isinstance(declared, str) and declared:
        return declared, "bundle-descriptor"
    expected = config.profile_expected_version
    if expected:
        return expected, "config"
    return None, "unknown"


def build_revision(config: Config) -> Revision:
    bundle_root = config.bundle_root
    profile_version, profile_source = resolve_profile_version(bundle_root, config)
    bundle_commit, commit_source = resolve_bundle_commit(bundle_root)
    index_revision = compute_index_revision(bundle_root, config.bundle_note_glob)

    return Revision(
        profile_version=profile_version,
        api_version=API_VERSION,
        bundle_commit=bundle_commit,
        index_revision=index_revision,
        sources={
            "profile_version": profile_source,
            "api_version": "code",
            "bundle_commit": commit_source,
            "index_revision": "computed" if index_revision else "unknown",
        },
    )


def bundle_is_readable(bundle_root: Path, note_glob: str) -> bool:
    """A bundle is readable when the directory exists and yields at least one note."""
    return bool(iter_note_paths(bundle_root, note_glob))


__all__ = [
    "API_VERSION",
    "INDEX_ALGORITHM",
    "Revision",
    "build_revision",
    "bundle_is_readable",
    "compute_index_revision",
    "iter_note_paths",
    "resolve_bundle_commit",
    "resolve_profile_version",
]
