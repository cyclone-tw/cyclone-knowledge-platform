"""Anchored, immutable reads of an OKF bundle.

Every byte consumed by revision, privacy classification, Catalog, or Gateway
comes through this module.  The bundle root itself may be a symlink (a normal
mount pattern), but after opening that root directory descriptor every child
component is walked with ``dir_fd`` and ``O_NOFOLLOW``.  A member is accepted
only when its pre-open ``lstat``, opened ``fstat``, post-read ``fstat``, and
post-read ``lstat`` all describe the same regular, single-link inode.

The hardlink rule is deliberately conservative and separate from the openat
claim: openat cannot reveal where another hardlink lives.  Rejecting
``st_nlink != 1`` prevents a multi-link file from crossing the boundary, at
the cost of rejecting legitimate in-bundle hardlinks too.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import subprocess
import threading
import tomllib
import unicodedata
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

INDEX_ALGORITHM = "ckp-index-v1"
BUNDLE_DESCRIPTOR = "bundle.toml"
BUNDLE_COMMIT_STAMP = ".bundle-commit"
_GIT_TIMEOUT_SECONDS = 5
_MAX_CAPTURE_ATTEMPTS = 3

REFUSAL_PATH = "invalid-member-path"
REFUSAL_ROOT = "bundle-root-unavailable"
REFUSAL_SYMLINK = "symlink-refused"
REFUSAL_NOT_REGULAR = "non-regular-file"
REFUSAL_HARDLINK = "multiple-hardlinks"
REFUSAL_RACE = "member-changed-during-read"
REFUSAL_UNREADABLE = "member-unreadable"

# C1 deliberately excludes symlink and non-regular note candidates from the
# digest. C3 preserves that compatibility. A hardlink is different: it looks
# like an ordinary regular note on one copy but not another, so silently
# excluding it would let two materially different bundles share a healthy
# revision. Instability and unreadability are likewise snapshot blockers.
_BLOCKING_MEMBER_REFUSALS = frozenset(
    {
        REFUSAL_HARDLINK,
        REFUSAL_PATH,
        REFUSAL_RACE,
        REFUSAL_ROOT,
        REFUSAL_UNREADABLE,
    }
)

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_ROOT_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


@dataclass(frozen=True)
class MemberRefused:
    """A safe refusal: reason code plus a bundle-relative path when known."""

    relative_path: str | None
    reason: str


@dataclass(frozen=True)
class BundleMember:
    """Verified bytes, detached from the mutable filesystem namespace."""

    relative_path: str
    digest_key: str
    content: bytes
    content_sha256: str


@dataclass(frozen=True)
class _FileSignature:
    device: int
    inode: int
    file_type: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _DirectoryIdentity:
    device: int
    inode: int
    file_type: int


@dataclass(frozen=True)
class MemberProbe:
    relative_path: str
    digest_key: str
    signature: _FileSignature | None
    refusal: str | None


@dataclass(frozen=True)
class BundleProbe:
    root_identity: _DirectoryIdentity | None
    members: tuple[MemberProbe, ...]
    descriptor: MemberProbe
    commit_stamp: MemberProbe

    @property
    def token(self) -> tuple[object, ...]:
        return (
            self.root_identity,
            self.members,
            self.descriptor,
            self.commit_stamp,
        )


@dataclass(frozen=True)
class BundleCapture:
    members: tuple[BundleMember, ...]
    descriptor: bytes | None
    commit_stamp: bytes | None
    readable: bool
    token: tuple[object, ...]


@dataclass(frozen=True)
class BundleSnapshot:
    """One immutable source for revision, Catalog, query, and citations."""

    members: tuple[BundleMember, ...]
    profile_version: str | None
    bundle_commit: str | None
    index_revision: str | None
    sources: Mapping[str, str]
    token: tuple[object, ...]


class SnapshotRaceError(RuntimeError):
    """The filesystem changed while a snapshot was being captured."""


def _file_signature(value: os.stat_result) -> _FileSignature:
    return _FileSignature(
        device=value.st_dev,
        inode=value.st_ino,
        file_type=stat.S_IFMT(value.st_mode),
        links=value.st_nlink,
        size=value.st_size,
        modified_ns=value.st_mtime_ns,
        changed_ns=value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> _DirectoryIdentity:
    return _DirectoryIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        file_type=stat.S_IFMT(value.st_mode),
    )


def _digest_key(relative_path: str) -> str:
    return unicodedata.normalize("NFC", relative_path)


def _validate_relative_path(value: str) -> tuple[str, ...] | None:
    if not value or "\x00" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute():
        return None
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    return parts


def _same_directory(left: os.stat_result, right: os.stat_result) -> bool:
    return _directory_identity(left) == _directory_identity(right)


def _reason_for_stat(value: os.stat_result) -> str | None:
    if stat.S_ISLNK(value.st_mode):
        return REFUSAL_SYMLINK
    if not stat.S_ISREG(value.st_mode):
        return REFUSAL_NOT_REGULAR
    if value.st_nlink != 1:
        return REFUSAL_HARDLINK
    return None


def _open_error_reason(exc: OSError, *, checked: bool) -> str:
    if exc.errno == errno.ELOOP:
        return REFUSAL_RACE if checked else REFUSAL_SYMLINK
    if checked and exc.errno in {errno.ENOENT, errno.ENOTDIR, errno.EISDIR}:
        return REFUSAL_RACE
    if exc.errno in {errno.ENOENT, errno.ENOTDIR}:
        return REFUSAL_PATH
    return REFUSAL_UNREADABLE


class AnchoredBundleReader:
    """Read members beneath one root without trusting descendant paths."""

    def __init__(self, bundle_root: Path) -> None:
        self._bundle_root = bundle_root

    @property
    def bundle_root(self) -> Path:
        return self._bundle_root

    def _open_root(self) -> tuple[int, _DirectoryIdentity] | MemberRefused:
        try:
            # Intentionally no O_NOFOLLOW: C1 explicitly permits the root
            # itself to be a symlink.  The returned fd becomes the anchor.
            fd = os.open(self._bundle_root, _ROOT_FLAGS)
        except (OSError, ValueError):
            return MemberRefused(None, REFUSAL_ROOT)
        try:
            root_stat = os.fstat(fd)
            if not stat.S_ISDIR(root_stat.st_mode):
                os.close(fd)
                return MemberRefused(None, REFUSAL_ROOT)
            return fd, _directory_identity(root_stat)
        except OSError:
            os.close(fd)
            return MemberRefused(None, REFUSAL_ROOT)

    def _walk_parent(
        self, root_fd: int, parts: tuple[str, ...]
    ) -> (
        tuple[int, list[int], list[tuple[int, str, _DirectoryIdentity]]] | MemberRefused
    ):
        current = root_fd
        opened: list[int] = []
        lineage: list[tuple[int, str, _DirectoryIdentity]] = []
        relative = "/".join(parts)
        for name in parts[:-1]:
            try:
                before = os.stat(name, dir_fd=current, follow_symlinks=False)
            except OSError:
                self._close_all(opened)
                return MemberRefused(relative, REFUSAL_PATH)
            if stat.S_ISLNK(before.st_mode):
                self._close_all(opened)
                return MemberRefused(relative, REFUSAL_SYMLINK)
            if not stat.S_ISDIR(before.st_mode):
                self._close_all(opened)
                return MemberRefused(relative, REFUSAL_PATH)
            child: int | None = None
            try:
                child = os.open(name, _DIRECTORY_FLAGS, dir_fd=current)
                after = os.fstat(child)
            except OSError as exc:
                if child is not None:
                    with suppress(OSError):
                        os.close(child)
                self._close_all(opened)
                return MemberRefused(relative, _open_error_reason(exc, checked=True))
            if not _same_directory(before, after):
                os.close(child)
                self._close_all(opened)
                return MemberRefused(relative, REFUSAL_RACE)
            identity = _directory_identity(after)
            lineage.append((current, name, identity))
            opened.append(child)
            current = child
        return current, opened, lineage

    @staticmethod
    def _close_all(fds: list[int]) -> None:
        for fd in reversed(fds):
            with suppress(OSError):
                os.close(fd)

    @staticmethod
    def _lineage_is_stable(
        lineage: list[tuple[int, str, _DirectoryIdentity]],
    ) -> bool:
        for parent_fd, name, expected in reversed(lineage):
            try:
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError:
                return False
            if _directory_identity(current) != expected:
                return False
        return True

    def _inspect_at(
        self,
        root_fd: int,
        relative_path: str,
        *,
        read_content: bool,
        expected: _FileSignature | None = None,
    ) -> BundleMember | MemberProbe | MemberRefused:
        parts = _validate_relative_path(relative_path)
        if parts is None:
            return MemberRefused(None, REFUSAL_PATH)
        walked = self._walk_parent(root_fd, parts)
        if isinstance(walked, MemberRefused):
            return walked
        parent_fd, opened, lineage = walked
        name = parts[-1]
        try:
            try:
                before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError:
                return MemberRefused(relative_path, REFUSAL_PATH)
            reason = _reason_for_stat(before)
            if reason is not None:
                return MemberRefused(relative_path, reason)
            before_signature = _file_signature(before)
            if expected is not None and before_signature != expected:
                return MemberRefused(relative_path, REFUSAL_RACE)

            try:
                fd = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
            except OSError as exc:
                return MemberRefused(
                    relative_path, _open_error_reason(exc, checked=True)
                )
            try:
                opened_stat = os.fstat(fd)
                reason = _reason_for_stat(opened_stat)
                if reason is not None:
                    return MemberRefused(relative_path, reason)
                opened_signature = _file_signature(opened_stat)
                if opened_signature != before_signature:
                    return MemberRefused(relative_path, REFUSAL_RACE)

                content = b""
                if read_content:
                    chunks: list[bytes] = []
                    while True:
                        chunk = os.read(fd, 1024 * 1024)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    content = b"".join(chunks)

                after_stat = os.fstat(fd)
                if _reason_for_stat(after_stat) is not None:
                    return MemberRefused(relative_path, REFUSAL_RACE)
                if _file_signature(after_stat) != opened_signature:
                    return MemberRefused(relative_path, REFUSAL_RACE)
            except OSError:
                return MemberRefused(relative_path, REFUSAL_UNREADABLE)
            finally:
                os.close(fd)

            try:
                namespace_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError:
                return MemberRefused(relative_path, REFUSAL_RACE)
            if _file_signature(namespace_stat) != opened_signature:
                return MemberRefused(relative_path, REFUSAL_RACE)
            if not self._lineage_is_stable(lineage):
                return MemberRefused(relative_path, REFUSAL_RACE)

            key = _digest_key(relative_path)
            if not read_content:
                return MemberProbe(relative_path, key, opened_signature, None)
            return BundleMember(
                relative_path=relative_path,
                digest_key=key,
                content=content,
                content_sha256=hashlib.sha256(content).hexdigest(),
            )
        finally:
            self._close_all(opened)

    def read_member(self, relative_path: str) -> BundleMember | MemberRefused:
        root = self._open_root()
        if isinstance(root, MemberRefused):
            return root
        root_fd, _ = root
        try:
            result = self._inspect_at(root_fd, relative_path, read_content=True)
            assert isinstance(result, (BundleMember, MemberRefused))
            return result
        finally:
            os.close(root_fd)

    def read_path(self, path: Path) -> BundleMember | MemberRefused:
        """Read an absolute/lexical child path without resolving symlinks."""
        try:
            root = Path(os.path.abspath(self._bundle_root))
            candidate = Path(os.path.abspath(path))
            relative = candidate.relative_to(root).as_posix()
        except (OSError, ValueError):
            return MemberRefused(None, REFUSAL_PATH)
        return self.read_member(relative)

    def _candidate_paths(self, note_glob: str) -> tuple[str, ...]:
        """Enumerate names only; every actual access is anchored afterwards."""
        try:
            if not self._bundle_root.is_dir():
                return ()
            candidates = self._bundle_root.glob(note_glob)
        except (OSError, ValueError, RuntimeError, NotImplementedError):
            return ()

        root = Path(os.path.abspath(self._bundle_root))
        found: set[str] = set()
        try:
            for candidate in candidates:
                try:
                    absolute = Path(os.path.abspath(candidate))
                    relative = absolute.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                if _validate_relative_path(relative) is not None:
                    found.add(relative)
        except (OSError, ValueError, RuntimeError):
            return ()
        return tuple(sorted(found, key=lambda value: (_digest_key(value), value)))

    def _probe_named(self, root_fd: int, relative_path: str) -> MemberProbe:
        result = self._inspect_at(root_fd, relative_path, read_content=False)
        if isinstance(result, MemberRefused):
            return MemberProbe(
                relative_path=relative_path,
                digest_key=_digest_key(relative_path),
                signature=None,
                refusal=result.reason,
            )
        assert isinstance(result, MemberProbe)
        return result

    def probe(self, note_glob: str) -> BundleProbe:
        candidates = self._candidate_paths(note_glob)
        root = self._open_root()
        if isinstance(root, MemberRefused):
            refused_descriptor = MemberProbe(
                BUNDLE_DESCRIPTOR,
                _digest_key(BUNDLE_DESCRIPTOR),
                None,
                REFUSAL_ROOT,
            )
            refused_stamp = MemberProbe(
                BUNDLE_COMMIT_STAMP,
                _digest_key(BUNDLE_COMMIT_STAMP),
                None,
                REFUSAL_ROOT,
            )
            return BundleProbe(None, (), refused_descriptor, refused_stamp)
        root_fd, root_identity = root
        try:
            members = tuple(
                self._probe_named(root_fd, candidate) for candidate in candidates
            )
            descriptor = self._probe_named(root_fd, BUNDLE_DESCRIPTOR)
            stamp = self._probe_named(root_fd, BUNDLE_COMMIT_STAMP)
            return BundleProbe(root_identity, members, descriptor, stamp)
        finally:
            os.close(root_fd)

    def _read_expected(self, root_fd: int, probe: MemberProbe) -> BundleMember | None:
        if probe.signature is None:
            return None
        result = self._inspect_at(
            root_fd,
            probe.relative_path,
            read_content=True,
            expected=probe.signature,
        )
        if isinstance(result, MemberRefused):
            raise SnapshotRaceError(result.reason)
        assert isinstance(result, BundleMember)
        return result

    def capture(
        self, note_glob: str, probe: BundleProbe | None = None
    ) -> BundleCapture:
        expected = self.probe(note_glob) if probe is None else probe
        root = self._open_root()
        if isinstance(root, MemberRefused):
            if expected.root_identity is not None:
                raise SnapshotRaceError(REFUSAL_RACE)
            return BundleCapture((), None, None, False, expected.token)
        root_fd, root_identity = root
        try:
            if root_identity != expected.root_identity:
                raise SnapshotRaceError(REFUSAL_RACE)
            members = tuple(
                member
                for item in expected.members
                if (member := self._read_expected(root_fd, item)) is not None
            )
            descriptor_member = self._read_expected(root_fd, expected.descriptor)
            stamp_member = self._read_expected(root_fd, expected.commit_stamp)
            return BundleCapture(
                members=members,
                descriptor=(
                    descriptor_member.content if descriptor_member is not None else None
                ),
                commit_stamp=(
                    stamp_member.content if stamp_member is not None else None
                ),
                readable=all(
                    item.refusal not in _BLOCKING_MEMBER_REFUSALS
                    for item in expected.members
                ),
                token=expected.token,
            )
        finally:
            os.close(root_fd)


def compute_index_revision_from_members(
    members: tuple[BundleMember, ...],
) -> str | None:
    if not members:
        return None
    digest = hashlib.sha256()
    digest.update(INDEX_ALGORITHM.encode("utf-8"))
    digest.update(b"\n")
    for member in sorted(
        members, key=lambda item: (item.digest_key, item.relative_path)
    ):
        digest.update(member.digest_key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(member.content_sha256.encode("ascii"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _git_commit(bundle_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(bundle_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _stamp_value(raw: bytes | None) -> str | None:
    if raw is None:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    value = text.split("\n", 1)[0].strip()
    return value or None


def _profile_value(
    raw: bytes | None, expected_profile_version: str | None
) -> tuple[str | None, str]:
    if raw is not None:
        try:
            descriptor = tomllib.loads(raw.decode("utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            descriptor = {}
        declared = descriptor.get("profile_version")
        if isinstance(declared, str) and declared.strip():
            return declared.strip(), "bundle-descriptor"
    if expected_profile_version:
        return expected_profile_version, "config"
    return None, "unknown"


class SnapshotCache:
    """Per-app cache with verified-content validation and invalidation.

    Portable filesystems do not promise that two same-size writes receive
    distinguishable timestamps. Metadata probes therefore protect the
    capture against races, but never prove that cached bytes are current. A
    cache hit requires hashes from a fresh anchored capture to match.
    """

    def __init__(
        self,
        reader: AnchoredBundleReader,
        note_glob: str,
        expected_profile_version: str | None,
    ) -> None:
        self._reader = reader
        self._note_glob = note_glob
        self._expected_profile_version = expected_profile_version
        self._lock = threading.RLock()
        self._snapshot: BundleSnapshot | None = None
        self._cache_token: tuple[object, ...] | None = None
        self._generation = 0
        self._rebuild_count = 0

    @property
    def reader(self) -> AnchoredBundleReader:
        return self._reader

    @property
    def rebuild_count(self) -> int:
        return self._rebuild_count

    def invalidate(self) -> None:
        """Make the next ``get`` rebuild even if metadata did not change."""
        with self._lock:
            self._generation += 1

    def peek(self) -> BundleSnapshot | None:
        """The last snapshot materialized by ``get``, without a fresh probe.

        ``get`` always re-verifies against the filesystem before returning,
        so it can never observe its own staleness -- by the time it answers,
        it has already healed. ``peek`` is the other half of that story: it
        reports what was actually served *before* this call, so a caller
        (``/revision``) can compare "what other requests were just served"
        against a fresh recomputation and flag drift between the two,
        instead of only ever comparing a self-healing value against itself.
        Returns ``None`` before the first ``get`` call on this instance.
        """
        with self._lock:
            return self._snapshot

    @staticmethod
    def _content_digest(raw: bytes | None) -> str | None:
        return hashlib.sha256(raw).hexdigest() if raw is not None else None

    def _snapshot_token(
        self, capture: BundleCapture, live_commit: str | None
    ) -> tuple[object, ...]:
        return (
            tuple(
                (member.digest_key, member.relative_path, member.content_sha256)
                for member in capture.members
            ),
            self._content_digest(capture.descriptor),
            self._content_digest(capture.commit_stamp),
            live_commit,
            capture.readable,
        )

    def _cache_key(self, snapshot_token: tuple[object, ...]) -> tuple[object, ...]:
        return (self._generation, snapshot_token)

    def get(self) -> BundleSnapshot:
        with self._lock:
            for _ in range(_MAX_CAPTURE_ATTEMPTS):
                before = self._reader.probe(self._note_glob)
                live_commit = _git_commit(self._reader.bundle_root)

                try:
                    capture = self._reader.capture(self._note_glob, before)
                except SnapshotRaceError:
                    continue
                after = self._reader.probe(self._note_glob)
                live_commit_after = _git_commit(self._reader.bundle_root)
                if before.token != after.token or live_commit != live_commit_after:
                    continue

                snapshot_token = self._snapshot_token(capture, live_commit_after)
                cache_key = self._cache_key(snapshot_token)
                if self._snapshot is not None and cache_key == self._cache_token:
                    return self._snapshot

                profile_version, profile_source = _profile_value(
                    capture.descriptor, self._expected_profile_version
                )
                stamped_commit = _stamp_value(capture.commit_stamp)
                bundle_commit = live_commit_after or stamped_commit
                commit_source = (
                    "git"
                    if live_commit_after
                    else "stamp"
                    if stamped_commit
                    else "unknown"
                )
                index_revision = (
                    compute_index_revision_from_members(capture.members)
                    if capture.readable
                    else None
                )
                snapshot = BundleSnapshot(
                    members=capture.members,
                    profile_version=profile_version,
                    bundle_commit=bundle_commit,
                    index_revision=index_revision,
                    sources=MappingProxyType(
                        {
                            "profile_version": profile_source,
                            "bundle_commit": commit_source,
                            "index_revision": (
                                "computed" if index_revision else "unknown"
                            ),
                        }
                    ),
                    token=snapshot_token,
                )
                self._snapshot = snapshot
                self._cache_token = cache_key
                self._rebuild_count += 1
                return snapshot
            raise SnapshotRaceError("bundle changed during every capture attempt")


__all__ = [
    "BUNDLE_COMMIT_STAMP",
    "BUNDLE_DESCRIPTOR",
    "INDEX_ALGORITHM",
    "REFUSAL_HARDLINK",
    "REFUSAL_NOT_REGULAR",
    "REFUSAL_PATH",
    "REFUSAL_RACE",
    "REFUSAL_ROOT",
    "REFUSAL_SYMLINK",
    "REFUSAL_UNREADABLE",
    "AnchoredBundleReader",
    "BundleCapture",
    "BundleMember",
    "BundleProbe",
    "BundleSnapshot",
    "MemberProbe",
    "MemberRefused",
    "SnapshotCache",
    "SnapshotRaceError",
    "compute_index_revision_from_members",
]
