"""Record, binding, and receipt schemas are frozen and payload-free."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ckp.outbox.errors import OutboxErrorCode, OutboxRefusal
from ckp.outbox.models import (
    RECORD_VERSION,
    WRITER_CONTRACT,
    EncryptedEnvelope,
    OutboxEnqueueReceipt,
    OutboxRecord,
    OutboxRejectReceipt,
    OutboxReplayReceipt,
    OutboxState,
    compute_binding_hash,
    record_from_json,
)

_SHA = "a" * 40
_HASH = "sha256:" + "b" * 64
_TIME = "2026-08-04T10:20:30Z"

_BINDING_FIELDS = {
    "record_version": RECORD_VERSION,
    "writer_contract": WRITER_CONTRACT,
    "operation_id": "synthetic.outbox-1",
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


def _envelope() -> EncryptedEnvelope:
    return EncryptedEnvelope(
        cipher="ckp-outbox-cipher-v1",
        key_id="synthetic-outbox-key-1",
        nonce="0" * 32,
        ciphertext="AAAA",
        mac="c" * 64,
    )


def _record(**overrides: object) -> OutboxRecord:
    payload: dict[str, object] = {
        **_BINDING_FIELDS,
        "binding_hash": compute_binding_hash(**_BINDING_FIELDS),
        "state": OutboxState.QUEUED,
        "attempts": 0,
        "state_changed_at": _TIME,
        "envelope": _envelope(),
    }
    payload.update(overrides)
    return OutboxRecord.model_validate(payload)


def test_receipts_have_exact_fields_and_strict_evidence() -> None:
    assert set(OutboxEnqueueReceipt.model_fields) == {
        "operation_id",
        "state",
        "content_hash",
        "enqueue_base_commit",
        "created_at",
        "expires_at",
    }
    assert set(OutboxReplayReceipt.model_fields) == {
        "operation_id",
        "state",
        "code",
        "content_hash",
        "target_path",
        "base_commit",
        "result_commit",
        "actor",
        "timestamp",
    }
    assert set(OutboxRejectReceipt.model_fields) == {"request_id", "code"}
    for model in (OutboxEnqueueReceipt, OutboxReplayReceipt, OutboxRejectReceipt):
        assert model.model_config.get("extra") == "forbid"


def test_committed_replay_receipt_requires_full_evidence() -> None:
    with pytest.raises(ValidationError):
        OutboxReplayReceipt(
            operation_id="synthetic.outbox-1",
            state=OutboxState.COMMITTED,
        )
    receipt = OutboxReplayReceipt(
        operation_id="synthetic.outbox-1",
        state=OutboxState.COMMITTED,
        content_hash=_HASH,
        target_path="Core/_inbox/c7-synthetic/synthetic-outbox-note.md",
        base_commit=_SHA,
        result_commit="d" * 40,
        actor="codex/gpt-5.6",
        timestamp=_TIME,
    )
    assert receipt.state is OutboxState.COMMITTED


def test_non_committed_replay_receipt_must_not_carry_commit_evidence() -> None:
    with pytest.raises(ValidationError):
        OutboxReplayReceipt(
            operation_id="synthetic.outbox-1",
            state=OutboxState.QUEUED,
            result_commit="d" * 40,
        )


def test_reject_receipt_accepts_only_outbox_codes() -> None:
    receipt = OutboxRejectReceipt(
        request_id="server-outbox-request-1",
        code=OutboxErrorCode.PRIVACY_UNCLASSIFIED.value,
    )
    assert receipt.code == "outbox/v1/privacy-unclassified"
    with pytest.raises(ValidationError):
        OutboxRejectReceipt(
            request_id="server-outbox-request-1",
            code="writer/v1/privacy-unclassified",
        )


def test_terminal_records_must_not_retain_an_envelope() -> None:
    with pytest.raises(ValidationError):
        _record(state=OutboxState.COMMITTED)
    purged = _record(state=OutboxState.COMMITTED, envelope=None)
    assert purged.envelope is None


def test_binding_hash_covers_every_frozen_dimension() -> None:
    baseline = compute_binding_hash(**_BINDING_FIELDS)
    replacements = {
        "record_version": "ckp-outbox-record-v2",
        "writer_contract": "writer/v2",
        "operation_id": "synthetic.outbox-2",
        "idempotency_hash": "sha256:" + "0" * 64,
        "actor": "claude-code/claude-fable-5",
        "task_hash": "sha256:" + "1" * 64,
        "target_path": "Core/_inbox/c7-synthetic/other-note.md",
        "request_hash": "sha256:" + "2" * 64,
        "content_hash": "sha256:" + "3" * 64,
        "enqueue_base_commit": "e" * 40,
        "created_at": "2026-08-04T10:20:31Z",
        "expires_at": "2026-08-04T11:20:31Z",
    }
    assert set(replacements) == set(_BINDING_FIELDS)
    for name, replacement in replacements.items():
        mutated = compute_binding_hash(**{**_BINDING_FIELDS, name: replacement})
        assert mutated != baseline, name


def test_record_from_json_fails_closed_on_version_and_binding_drift() -> None:
    good = _record()
    payload = good.model_dump(mode="json", exclude_none=True)
    assert record_from_json(dict(payload)).operation_id == good.operation_id

    with pytest.raises(OutboxRefusal) as unknown_version:
        record_from_json({**payload, "record_version": "ckp-outbox-record-v2"})
    assert unknown_version.value.code is OutboxErrorCode.VERSION_UNKNOWN

    with pytest.raises(OutboxRefusal) as unknown_contract:
        record_from_json({**payload, "writer_contract": "writer/v2"})
    assert unknown_contract.value.code is OutboxErrorCode.VERSION_UNKNOWN

    tampered = dict(payload)
    tampered["target_path"] = "Core/_inbox/c7-synthetic/other-note.md"
    with pytest.raises(OutboxRefusal) as drift:
        record_from_json(tampered)
    assert drift.value.code is OutboxErrorCode.RECORD_CORRUPT

    with pytest.raises(OutboxRefusal) as not_a_record:
        record_from_json({"record_version": RECORD_VERSION})
    assert not_a_record.value.code in {
        OutboxErrorCode.RECORD_CORRUPT,
        OutboxErrorCode.VERSION_UNKNOWN,
    }
