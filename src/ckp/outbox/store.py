"""Crash-safe durable record store for the outbox v1.

Every record is one JSON file written atomically (ciphertext-only temp file,
fsync, rename, directory fsync), so a crash leaves either the previous state
or the next one -- never a torn record. The idempotency index is derived data
and is rebuilt from records during recovery; records are the single source of
truth. Nothing in this module ever holds plaintext payload bytes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ckp.outbox.errors import OutboxErrorCode, OutboxRefusal
from ckp.outbox.models import OutboxRecord, record_from_json

_RECORD_SUFFIX = ".json"


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
        store._lock_timeout_seconds = lock_timeout_seconds
        for directory in (
            store._records,
            store._index,
            store._tmp,
            store._quarantine,
        ):
            directory.mkdir(mode=0o700, exist_ok=True)
        return store

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
        """Resolve both indexes; disagreement between them fails closed."""
        name = self.record_name(operation_id)
        record = self.load(name)
        index_target = self._read_index(idempotency_hash)
        if record is None and index_target is None:
            return None
        if record is None:
            # The key is already bound to some other operation's record.
            raise OutboxRefusal(OutboxErrorCode.BINDING_CONFLICT)
        if index_target is not None and index_target != name:
            raise OutboxRefusal(OutboxErrorCode.BINDING_CONFLICT)
        if record.idempotency_hash != idempotency_hash:
            raise OutboxRefusal(OutboxErrorCode.BINDING_CONFLICT)
        return record

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
        name = self.record_name(record.operation_id)
        path = self._records / (name + _RECORD_SUFFIX)
        if path.exists():
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED)
        self._atomic_write(path, _encode(record))
        self._write_index(record.idempotency_hash, name)

    def rewrite(self, record: OutboxRecord) -> None:
        name = self.record_name(record.operation_id)
        path = self._records / (name + _RECORD_SUFFIX)
        if not path.exists():
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED)
        self._atomic_write(path, _encode(record))

    def recover(self) -> None:
        """Discard torn temp files and rebuild missing derived index entries."""
        for orphan in self._tmp.iterdir():
            try:
                orphan.unlink()
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
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from exc

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
