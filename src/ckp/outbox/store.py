"""Crash-safe durable record store for the outbox v1.

Every record is one JSON file written atomically (ciphertext-only temp file,
fsync, rename, directory fsync), so a crash leaves either the previous state
or the next one -- never a torn record. The record file is the single commit
point of an enqueue: the derived idempotency index is written *before* the
record and swept when dangling, so no crash ordering can turn a refused
enqueue into a replayable one. Nothing in this module ever holds plaintext
payload bytes; the admission sandbox area is swept, never read.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ckp.outbox.errors import OutboxErrorCode, OutboxRefusal
from ckp.outbox.models import OutboxRecord, record_from_json

_RECORD_SUFFIX = ".json"
#: The only record-name shape this store ever minted. Everything that turns
#: stored text into a path goes through this, so corrupted or malicious
#: index content can never traverse outside the records directory.
_RECORD_NAME = re.compile(r"^[0-9a-f]{64}$")


class OutboxStore:
    """Own one explicit state root: records, derived index, and locks."""

    def __init__(self, *args: object) -> None:
        raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED)

    @classmethod
    def open(
        cls,
        state_root: Path,
        *,
        disjoint_from: tuple[Path, ...],
        lock_timeout_seconds: float,
    ) -> OutboxStore:
        if lock_timeout_seconds <= 0:
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED)
        root = _physical_directory(state_root)
        for other in disjoint_from:
            resolved = _physical_directory(other)
            if _contains(root, resolved) or _contains(resolved, root):
                raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED)
        store = object.__new__(cls)
        store._root = root
        store._records = root / "records"
        store._index = root / "idempotency"
        store._tmp = root / "tmp"
        store._quarantine = root / "quarantine"
        store._admission = root / "admission"
        store._lock_timeout_seconds = lock_timeout_seconds
        for directory in (
            store._records,
            store._index,
            store._tmp,
            store._quarantine,
            store._admission,
        ):
            # A pre-planted symlink here would route durable records outside
            # the disjointness policy checked above; refuse anything that is
            # not a plain physical directory.
            try:
                directory.mkdir(mode=0o700, exist_ok=True)
                value = directory.lstat()
            except OSError as exc:
                raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from exc
            if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
                raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED)
        return store

    @property
    def admission_root(self) -> Path:
        """Where enqueue admission sandboxes live; swept on every recover."""
        return self._admission

    # -- naming -----------------------------------------------------------

    @staticmethod
    def record_name(operation_id: str) -> str:
        return hashlib.sha256(operation_id.encode("utf-8")).hexdigest()

    @staticmethod
    def index_name(idempotency_hash: str) -> str:
        return idempotency_hash.removeprefix("sha256:")

    # -- locks ------------------------------------------------------------

    @contextmanager
    def mutation_lock(self) -> Iterator[None]:
        """Serialize enqueue and state rewrites across processes."""
        with self._flock("store.lock", OutboxErrorCode.STORAGE_FAILED):
            yield

    @contextmanager
    def consumer_leader(self) -> Iterator[None]:
        """At most one replay consumer; contention fails closed, never queues."""
        with self._flock("consumer.lock", OutboxErrorCode.LEASE_UNAVAILABLE):
            yield

    @contextmanager
    def _flock(self, name: str, code: OutboxErrorCode) -> Iterator[None]:
        lock_path = self._root / name
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
                raise OSError("unsafe lock")
        except OSError as exc:
            raise OutboxRefusal(code) from exc
        deadline = time.monotonic() + self._lock_timeout_seconds
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise OutboxRefusal(code) from None
                    time.sleep(0.01)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    # -- reads ------------------------------------------------------------

    def load(self, name: str) -> OutboxRecord | None:
        if _RECORD_NAME.fullmatch(name) is None:
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED)
        path = self._records / (name + _RECORD_SUFFIX)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT) from exc
        return record_from_json(payload)

    def find_for_enqueue(
        self, operation_id: str, idempotency_hash: str
    ) -> OutboxRecord | None:
        """Resolve both indexes; records are truth, the index is derived.

        An index entry pointing at an *existing* other record means the key
        is bound elsewhere and fails closed. An entry pointing at a missing
        record is the residue of a crashed enqueue that never reached its
        commit point; it is deleted here rather than trusted.
        """
        name = self.record_name(operation_id)
        record = self.load(name)
        index_target = self._read_index(idempotency_hash)
        if index_target is not None and index_target != name:
            if self.load(index_target) is not None:
                raise OutboxRefusal(OutboxErrorCode.BINDING_CONFLICT)
            self._drop_index(idempotency_hash)
            index_target = None
        if record is None:
            if index_target is not None:
                # The entry names this operation but its record never
                # committed: a torn enqueue, not an enqueued operation.
                self._drop_index(idempotency_hash)
            # A missing index entry does not unbind the key: some live
            # record may still hold it (the entry can be lost to a crash).
            holder = self._record_holding_key(idempotency_hash)
            if holder is not None:
                self._write_index(idempotency_hash, holder)
                raise OutboxRefusal(OutboxErrorCode.BINDING_CONFLICT)
            return None
        if record.idempotency_hash != idempotency_hash:
            raise OutboxRefusal(OutboxErrorCode.BINDING_CONFLICT)
        return record

    def _record_holding_key(self, idempotency_hash: str) -> str | None:
        """The name of the live record bound to this key, if any exists.

        Consulted only when the derived index has no answer. Records that
        cannot be parsed are skipped here -- recovery quarantines them, and
        their key binding dies with them.
        """
        for path in sorted(self._records.glob("*" + _RECORD_SUFFIX)):
            name = path.name.removesuffix(_RECORD_SUFFIX)
            try:
                record = self.load(name)
            except OutboxRefusal:
                continue
            if record is not None and record.idempotency_hash == idempotency_hash:
                return name
        return None

    def scan(self) -> list[tuple[str, OutboxRecord]]:
        """Every parseable active record in deterministic name order.

        Unparseable or version-unknown files are moved to quarantine as raw
        audit bytes (metadata and ciphertext only, by construction) and are
        never offered for replay.
        """
        results: list[tuple[str, OutboxRecord]] = []
        for path in sorted(self._records.glob("*" + _RECORD_SUFFIX)):
            name = path.name.removesuffix(_RECORD_SUFFIX)
            try:
                record = self.load(name)
            except OutboxRefusal:
                self._quarantine_raw(path)
                continue
            if record is not None:
                results.append((name, record))
        return results

    # -- writes -----------------------------------------------------------

    def persist_new(self, record: OutboxRecord) -> None:
        """Provisional index first, record last: the record is the commit.

        A crash after the index write leaves only a dangling entry that
        recovery and enqueue both discard, so a refused or torn enqueue can
        never resurrect into a replayable record. The reverse order would
        durably enqueue an operation whose caller was told it failed.
        """
        name = self.record_name(record.operation_id)
        path = self._records / (name + _RECORD_SUFFIX)
        if path.exists():
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED)
        self._write_index(record.idempotency_hash, name)
        try:
            self._atomic_write(path, _encode(record))
        except OutboxRefusal:
            # The record may already be visible (rename done, directory
            # fsync failed). The refusal must match the visible state, so
            # roll the create back before reporting it. A crash inside this
            # narrow window is the irreducible filesystem residue.
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            self._drop_index(record.idempotency_hash)
            raise

    def rewrite(self, record: OutboxRecord) -> None:
        name = self.record_name(record.operation_id)
        path = self._records / (name + _RECORD_SUFFIX)
        if not path.exists():
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED)
        self._atomic_write(path, _encode(record))

    def recover(self) -> None:
        """Restore every derived and transient area to a clean state.

        Torn temp files and crashed admission sandboxes are discarded,
        dangling index entries (a crashed enqueue that never reached its
        record commit) are swept, and missing index entries are rebuilt from
        the records that are the source of truth.
        """
        for orphan in self._tmp.iterdir():
            try:
                orphan.unlink()
            except OSError as exc:
                raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from exc
        for leftover in self._admission.iterdir():
            try:
                if leftover.is_dir() and not leftover.is_symlink():
                    shutil.rmtree(leftover)
                else:
                    leftover.unlink()
            except OSError as exc:
                raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from exc
        for entry in self._index.iterdir():
            try:
                target = entry.read_text(encoding="ascii").strip()
            except (OSError, ValueError) as exc:
                raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from exc
            # Shape first: content that is not a minted record name must be
            # swept without ever being turned into a filesystem probe.
            if (
                _RECORD_NAME.fullmatch(target) is None
                or not (self._records / (target + _RECORD_SUFFIX)).exists()
            ):
                try:
                    entry.unlink()
                except OSError as exc:
                    raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from exc
        for name, record in self.scan():
            if self._read_index(record.idempotency_hash) is None:
                self._write_index(record.idempotency_hash, name)

    # -- internals --------------------------------------------------------

    def _read_index(self, idempotency_hash: str) -> str | None:
        path = self._index / self.index_name(idempotency_hash)
        try:
            value = path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from exc
        return value or None

    def _write_index(self, idempotency_hash: str, name: str) -> None:
        path = self._index / self.index_name(idempotency_hash)
        self._atomic_write(path, name.encode("ascii"))

    def _drop_index(self, idempotency_hash: str) -> None:
        path = self._index / self.index_name(idempotency_hash)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from exc

    def _atomic_write(self, destination: Path, content: bytes) -> None:
        try:
            descriptor, temp_name = tempfile.mkstemp(dir=self._tmp)
            try:
                view = memoryview(content)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temp_name, destination)
            self._fsync_directory(destination.parent)
        except OSError as exc:
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from exc

    def _fsync_directory(self, directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _quarantine_raw(self, path: Path) -> None:
        destination = self._quarantine / (path.name + ".bin")
        try:
            os.replace(path, destination)
        except OSError as exc:
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from exc


def _encode(record: OutboxRecord) -> bytes:
    return json.dumps(
        record.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _physical_directory(path: Path) -> Path:
    try:
        absolute = path.expanduser().absolute()
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from exc
    if absolute != resolved or not resolved.is_dir():
        raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED)
    return resolved


def _contains(parent: Path, child: Path) -> bool:
    return parent == child or parent in child.parents


__all__ = ["OutboxStore"]
