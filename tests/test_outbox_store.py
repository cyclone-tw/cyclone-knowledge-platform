"""Durable store: atomicity, dedup indexes, quarantine, and locks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ckp.outbox.errors import OutboxErrorCode, OutboxRefusal
from ckp.outbox.models import (
    RECORD_VERSION,
    WRITER_CONTRACT,
    EncryptedEnvelope,
    OutboxRecord,
    OutboxState,
    compute_binding_hash,
)
from ckp.outbox.store import OutboxStore

_SHA = "a" * 40
_HASH = "sha256:" + "b" * 64
_TIME = "2026-08-04T10:20:30Z"


def _record(operation_id: str = "synthetic.outbox-1") -> OutboxRecord:
    binding = {
        "record_version": RECORD_VERSION,
        "writer_contract": WRITER_CONTRACT,
        "operation_id": operation_id,
        "idempotency_hash": _HASH,
        "actor": "codex/gpt-5.6",
        "task_hash": _HASH,
        "target_path": "Core/_inbox/c7-synthetic/synthetic-outbox-note.md",
        "request_hash": _HASH,
        "content_hash": _HASH,
        "enqueue_base_commit": _SHA,
        "created_at": _TIME,
        "expires_at": "2026-08-04T11:20:30Z",
    }
    return OutboxRecord.model_validate(
        {
            **binding,
            "binding_hash": compute_binding_hash(**binding),
            "state": OutboxState.QUEUED,
            "attempts": 0,
            "state_changed_at": _TIME,
            "envelope": EncryptedEnvelope(
                cipher="ckp-outbox-cipher-v1",
                key_id="synthetic-outbox-key-1",
                nonce="0" * 32,
                ciphertext="AAAA",
                mac="c" * 64,
            ),
        }
    )


def _store(tmp_path: Path) -> OutboxStore:
    root = tmp_path / "outbox-state"
    root.mkdir(exist_ok=True)
    return OutboxStore.open(root, disjoint_from=(), lock_timeout_seconds=0.2)


def test_store_cannot_be_constructed_directly_or_on_bad_roots(
    tmp_path: Path,
) -> None:
    with pytest.raises(OutboxRefusal):
        OutboxStore()
    with pytest.raises(OutboxRefusal):
        OutboxStore.open(
            tmp_path / "missing", disjoint_from=(), lock_timeout_seconds=1.0
        )
    overlapping = tmp_path / "overlap"
    overlapping.mkdir()
    with pytest.raises(OutboxRefusal):
        OutboxStore.open(
            overlapping,
            disjoint_from=(overlapping,),
            lock_timeout_seconds=1.0,
        )
    nested = overlapping / "inner"
    nested.mkdir()
    with pytest.raises(OutboxRefusal):
        OutboxStore.open(nested, disjoint_from=(overlapping,), lock_timeout_seconds=1.0)
    with pytest.raises(OutboxRefusal):
        OutboxStore.open(overlapping, disjoint_from=(), lock_timeout_seconds=0)


def test_persist_load_roundtrip_and_no_double_create(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _record()
    store.persist_new(record)
    name = store.record_name(record.operation_id)
    loaded = store.load(name)
    assert loaded == record
    with pytest.raises(OutboxRefusal) as refusal:
        store.persist_new(record)
    assert refusal.value.code is OutboxErrorCode.STORAGE_FAILED


def test_find_for_enqueue_resolves_both_indexes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _record()
    assert store.find_for_enqueue(record.operation_id, record.idempotency_hash) is None
    store.persist_new(record)
    found = store.find_for_enqueue(record.operation_id, record.idempotency_hash)
    assert found == record


def test_key_bound_to_another_live_record_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _record()
    store.persist_new(record)
    # The same idempotency key arriving under a different operation id means
    # the key is already bound elsewhere.
    with pytest.raises(OutboxRefusal) as key_reuse:
        store.find_for_enqueue("synthetic.outbox-2", record.idempotency_hash)
    assert key_reuse.value.code is OutboxErrorCode.BINDING_CONFLICT


def test_dangling_index_entries_are_discarded_not_trusted(tmp_path: Path) -> None:
    """A crashed enqueue leaves only a provisional index entry behind.

    Records are the source of truth: an entry naming a record that never
    reached its commit point must be swept, both at enqueue and in recovery,
    and must never turn into a conflict or a replayable operation.
    """
    store = _store(tmp_path)
    record = _record()
    store.persist_new(record)
    index_dir = tmp_path / "outbox-state" / "idempotency"

    # Entry pointing at a record that does not exist: dropped, then healed.
    index_path = index_dir / store.index_name(record.idempotency_hash)
    index_path.write_text("0" * 64, encoding="ascii")
    healed = store.find_for_enqueue(record.operation_id, record.idempotency_hash)
    assert healed == record
    assert not index_path.exists()
    store.recover()
    assert index_path.read_text(encoding="ascii") == store.record_name(
        record.operation_id
    )

    # A pure crashed-enqueue residue: index entry, no record at all.
    other_key = "sha256:" + "f" * 64
    (index_dir / store.index_name(other_key)).write_text("1" * 64, encoding="ascii")
    assert store.find_for_enqueue("synthetic.outbox-9", other_key) is None
    assert not (index_dir / store.index_name(other_key)).exists()


def test_torn_enqueue_is_refused_and_never_resurrects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record write is the commit point of persist_new.

    Failing the second durable write simulates the crash window. With the
    frozen index-first order the surviving artifact is only a dangling index
    entry; a record-first order would durably enqueue an operation whose
    caller was told the enqueue failed.
    """
    store = _store(tmp_path)
    record = _record()
    real_write = store._atomic_write
    calls = {"count": 0}

    def second_write_dies(destination: Path, content: bytes) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED)
        real_write(destination, content)

    monkeypatch.setattr(store, "_atomic_write", second_write_dies)
    with pytest.raises(OutboxRefusal):
        store.persist_new(record)
    monkeypatch.undo()

    restarted = _store(tmp_path)
    restarted.recover()
    assert restarted.scan() == []
    assert not list((tmp_path / "outbox-state" / "idempotency").iterdir())


def test_open_refuses_symlinked_store_subdirectories(tmp_path: Path) -> None:
    """A pre-planted symlink would route records outside the disjoint root."""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "outbox-state"
    root.mkdir()
    (root / "records").symlink_to(outside)
    with pytest.raises(OutboxRefusal) as refusal:
        OutboxStore.open(root, disjoint_from=(), lock_timeout_seconds=1.0)
    assert refusal.value.code is OutboxErrorCode.STORAGE_FAILED


def test_recover_discards_temp_orphans_and_rebuilds_indexes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    record = _record()
    store.persist_new(record)
    orphan = tmp_path / "outbox-state" / "tmp" / "torn-write"
    orphan.write_bytes(b"ciphertext-only-orphan")
    index_path = (
        tmp_path
        / "outbox-state"
        / "idempotency"
        / store.index_name(record.idempotency_hash)
    )
    index_path.unlink()
    store.recover()
    assert not orphan.exists()
    assert index_path.read_text(encoding="ascii") == store.record_name(
        record.operation_id
    )


def test_scan_quarantines_unparseable_and_unknown_version_files(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    good = _record()
    store.persist_new(good)
    records_dir = tmp_path / "outbox-state" / "records"
    (records_dir / ("0" * 64 + ".json")).write_bytes(b"{not json")
    future = good.model_dump(mode="json", exclude_none=True)
    future["record_version"] = "ckp-outbox-record-v9"
    (records_dir / ("1" * 64 + ".json")).write_text(
        json.dumps(future), encoding="utf-8"
    )
    survivors = store.scan()
    assert [record.operation_id for _n, record in survivors] == [good.operation_id]
    quarantine = sorted(
        path.name for path in (tmp_path / "outbox-state" / "quarantine").iterdir()
    )
    assert quarantine == ["0" * 64 + ".json.bin", "1" * 64 + ".json.bin"]


def test_rewrite_requires_an_existing_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(OutboxRefusal) as refusal:
        store.rewrite(_record())
    assert refusal.value.code is OutboxErrorCode.STORAGE_FAILED


def test_locks_are_exclusive_and_fail_closed_on_contention(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    other = OutboxStore.open(
        tmp_path / "outbox-state", disjoint_from=(), lock_timeout_seconds=0.05
    )
    with store.mutation_lock():
        with pytest.raises(OutboxRefusal) as mutation, other.mutation_lock():
            pass
        assert mutation.value.code is OutboxErrorCode.STORAGE_FAILED
    with store.consumer_leader():
        with pytest.raises(OutboxRefusal) as consumer, other.consumer_leader():
            pass
        assert consumer.value.code is OutboxErrorCode.LEASE_UNAVAILABLE
