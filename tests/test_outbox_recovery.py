"""Expiry/recovery: corrupt, wrong-key, and unknown-version records fail
closed with minimal non-sensitive audit state, and restarts stay clean."""

from __future__ import annotations

import json
from pathlib import Path

from ckp.outbox.errors import OutboxErrorCode
from ckp.outbox.models import OutboxRejectReceipt, OutboxState
from outbox_fixtures import (
    SYNTHETIC_KEY_ID,
    assert_no_plaintext_at_rest,
    build_outbox_harness,
    enqueue,
    operation_commits,
)
from writer_fixtures import make_synthetic_repository


def _record_path(harness) -> Path:
    files = list((harness.store_root / "records").iterdir())
    assert len(files) == 1
    return files[0]


def test_restart_after_torn_enqueue_leaves_a_clean_store(tmp_path: Path) -> None:
    harness = build_outbox_harness(tmp_path)
    enqueue(harness.service)
    # A torn write and a lost derived index are exactly what a crash between
    # fsync points leaves behind.
    (harness.store_root / "tmp" / "torn").write_bytes(b"ciphertext-orphan")
    for index in (harness.store_root / "idempotency").iterdir():
        index.unlink()

    receipts = harness.service.replay_pending()
    assert [receipt.state for receipt in receipts] == [OutboxState.COMMITTED]
    assert not list((harness.store_root / "tmp").iterdir())
    assert len(list((harness.store_root / "idempotency").iterdir())) == 1
    assert len(operation_commits(harness.repository)) == 1

    # And the rebuilt index still dedups a duplicate enqueue.
    duplicate = enqueue(harness.service)
    assert not isinstance(duplicate, OutboxRejectReceipt)
    assert duplicate.state is OutboxState.COMMITTED


def test_corrupted_record_bytes_are_quarantined_and_never_replayed(
    tmp_path: Path,
) -> None:
    harness = build_outbox_harness(tmp_path)
    enqueue(harness.service)
    _record_path(harness).write_bytes(b"{torn json")

    receipts = harness.service.replay_pending()
    assert receipts == []
    assert not list((harness.store_root / "records").iterdir())
    quarantined = list((harness.store_root / "quarantine").iterdir())
    assert len(quarantined) == 1
    assert operation_commits(harness.repository) == ()


def test_tampered_record_fields_are_quarantined_by_the_binding(
    tmp_path: Path,
) -> None:
    harness = build_outbox_harness(tmp_path)
    enqueue(harness.service)
    path = _record_path(harness)
    payload = json.loads(path.read_bytes())
    payload["target_path"] = "Core/_inbox/c7-synthetic/redirected-note.md"
    path.write_text(json.dumps(payload), encoding="utf-8")

    receipts = harness.service.replay_pending()
    assert receipts == []
    assert len(list((harness.store_root / "quarantine").iterdir())) == 1
    assert operation_commits(harness.repository) == ()


def test_wrong_key_material_quarantines_without_replay(tmp_path: Path) -> None:
    repository = make_synthetic_repository(tmp_path)
    first = build_outbox_harness(tmp_path, repository=repository)
    enqueue(first.service)

    from ckp.outbox.crypto import SyntheticKeyProvider

    rotated = build_outbox_harness(
        tmp_path,
        repository=repository,
        key_provider=SyntheticKeyProvider(
            {SYNTHETIC_KEY_ID: b"ROTATED-SYNTHETIC-KEY-0123456789abcd"}
        ),
    )
    receipts = rotated.service.replay_pending()
    assert [receipt.state for receipt in receipts] == [OutboxState.QUARANTINED]
    assert receipts[0].code == OutboxErrorCode.RECORD_CORRUPT
    record = rotated.store.scan()[0][1]
    assert record.state is OutboxState.QUARANTINED
    assert record.envelope is None
    assert record.last_code == OutboxErrorCode.RECORD_CORRUPT.value
    assert operation_commits(repository) == ()


def test_unknown_key_id_fails_closed_as_key_denied(tmp_path: Path) -> None:
    repository = make_synthetic_repository(tmp_path)
    first = build_outbox_harness(tmp_path, repository=repository)
    enqueue(first.service)

    from ckp.outbox.crypto import SyntheticKeyProvider

    keyless = build_outbox_harness(
        tmp_path,
        repository=repository,
        key_provider=SyntheticKeyProvider(
            {"a-different-key-id": b"ANOTHER-SYNTHETIC-KEY-0123456789abcd"}
        ),
    )
    receipts = keyless.service.replay_pending()
    assert [receipt.state for receipt in receipts] == [OutboxState.QUARANTINED]
    assert receipts[0].code == OutboxErrorCode.KEY_DENIED
    assert operation_commits(repository) == ()


def test_unknown_record_version_is_never_optimistically_replayed(
    tmp_path: Path,
) -> None:
    harness = build_outbox_harness(tmp_path)
    enqueue(harness.service)
    path = _record_path(harness)
    payload = json.loads(path.read_bytes())
    payload["record_version"] = "ckp-outbox-record-v9"
    path.write_text(json.dumps(payload), encoding="utf-8")

    receipts = harness.service.replay_pending()
    assert receipts == []
    assert len(list((harness.store_root / "quarantine").iterdir())) == 1
    assert operation_commits(harness.repository) == ()


def test_no_plaintext_at_rest_across_the_whole_lifecycle(tmp_path: Path) -> None:
    harness = build_outbox_harness(tmp_path)
    enqueue(harness.service)
    assert_no_plaintext_at_rest(harness.store_root)
    harness.service.replay_pending()
    assert_no_plaintext_at_rest(harness.store_root)


def test_base_movement_policy_v1_table_is_frozen() -> None:
    from ckp.outbox.policy import (
        POLICY_VERSION,
        BaseMovementPolicyV1,
        ReplayDispositionKind,
    )

    policy = BaseMovementPolicyV1(max_transient_attempts=2)
    assert policy.version == POLICY_VERSION == "outbox-base-policy-v1"

    transient = policy.dispose("writer/v1/base-moved", 0)
    assert transient.kind is ReplayDispositionKind.RETRY_TRANSIENT
    exhausted = policy.dispose("writer/v1/base-moved", 2)
    assert exhausted.kind is ReplayDispositionKind.QUARANTINE

    free = policy.dispose("writer/v1/leader-unavailable", 99)
    assert free.kind is ReplayDispositionKind.RETRY_FREE

    for stable in ("writer/v1/path-conflict", "writer/v1/idempotency-conflict"):
        assert policy.dispose(stable, 0).kind is ReplayDispositionKind.CONFLICT

    # Admission-class refusals mean the enqueue-time admission no longer
    # holds; they are never optimistically retried.
    for admission in ("writer/v1/privacy-denied", "writer/v1/target-denied"):
        assert policy.dispose(admission, 0).kind is ReplayDispositionKind.QUARANTINE


def test_unknown_writer_codes_are_never_optimistically_replayed() -> None:
    from ckp.outbox.errors import OutboxRefusal
    from ckp.outbox.policy import BaseMovementPolicyV1, ReplayDispositionKind

    policy = BaseMovementPolicyV1(max_transient_attempts=2)
    for unknown in ("writer/v2/base-moved", "outbox/v1/internal-error", ""):
        assert policy.dispose(unknown, 0).kind is ReplayDispositionKind.QUARANTINE
    import pytest

    for bad_limit in (0, -1, True, "2"):
        with pytest.raises(OutboxRefusal):
            BaseMovementPolicyV1(max_transient_attempts=bad_limit)


def test_crashed_admission_sandboxes_are_swept_on_recover(
    tmp_path: Path,
) -> None:
    """A SIGKILL inside the admission window may strand plaintext on disk;
    the next recovery pass must remove it before any replay work."""
    harness = build_outbox_harness(tmp_path)
    enqueue(harness.service)
    stranded = harness.store.admission_root / "crashed-sandbox"
    stranded.mkdir()
    (stranded / "note.md").write_text(
        "SYNTHETIC-OUTBOX-BODY-NEEDLE stranded by a crash\n", encoding="utf-8"
    )

    harness.service.replay_pending()
    assert list(harness.store.admission_root.iterdir()) == []
    assert_no_plaintext_at_rest(harness.store_root)
