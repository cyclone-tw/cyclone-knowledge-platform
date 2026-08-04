"""Frozen outbox v1 record, states, and receipt schemas.

The record binds the same identity dimensions as the C7 ``OperationIdentity``
plus the durable-queue dimensions (content hash, original base commit,
created/expiry time, contract versions). Receipts never carry note bodies,
prompts, credentials, encryption material, or identifiable student data.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ckp.outbox.errors import OutboxErrorCode

RECORD_VERSION = "ckp-outbox-record-v1"
WRITER_CONTRACT = "writer/v1"
BINDING_DOMAIN = b"ckp-outbox-binding-v1"

_OPERATION_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_REQUEST_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$")
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_FULL_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)


class OutboxState(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    COMMITTED = "committed"
    CONFLICT = "conflict"
    QUARANTINED = "quarantined"
    EXPIRED = "expired"


#: Terminal states purge the encrypted envelope and never transition again.
TERMINAL_STATES = frozenset(
    {
        OutboxState.COMMITTED,
        OutboxState.CONFLICT,
        OutboxState.QUARANTINED,
        OutboxState.EXPIRED,
    }
)


def _require_timestamp(value: str) -> str:
    if _FULL_TIMESTAMP.fullmatch(value) is None:
        raise ValueError("timestamp must be a full timezone-aware RFC 3339 value")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value


class OutboxEnqueueReceipt(BaseModel):
    """The non-content-bearing result of an accepted or deduplicated enqueue."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(
        min_length=1, max_length=64, pattern=_OPERATION_ID.pattern
    )
    state: OutboxState
    content_hash: str = Field(min_length=1)
    enqueue_base_commit: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    expires_at: str = Field(min_length=1)

    @field_validator("content_hash")
    @classmethod
    def require_sha256_content_hash(cls, value: str) -> str:
        if _SHA256_HASH.fullmatch(value) is None:
            raise ValueError("content_hash must be a lowercase sha256 digest")
        return value

    @field_validator("enqueue_base_commit")
    @classmethod
    def require_full_git_sha(cls, value: str) -> str:
        if _FULL_GIT_SHA.fullmatch(value) is None:
            raise ValueError("enqueue_base_commit must be a full lowercase SHA-1")
        return value

    @field_validator("created_at", "expires_at")
    @classmethod
    def require_timezone_aware_timestamp(cls, value: str) -> str:
        return _require_timestamp(value)


_COMMITTED_FIELDS = (
    "target_path",
    "base_commit",
    "result_commit",
    "actor",
    "timestamp",
)


class OutboxReplayReceipt(BaseModel):
    """One replay outcome. Success evidence appears only on a real commit."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(
        min_length=1, max_length=64, pattern=_OPERATION_ID.pattern
    )
    state: OutboxState
    code: str | None = None
    content_hash: str | None = None
    target_path: str | None = None
    base_commit: str | None = None
    result_commit: str | None = None
    actor: str | None = None
    timestamp: str | None = None

    @field_validator("code")
    @classmethod
    def require_stable_outbox_code(cls, value: str | None) -> str | None:
        if value is not None and value not in set(OutboxErrorCode):
            raise ValueError("code is not an outbox v1 code")
        return value

    @field_validator("base_commit", "result_commit")
    @classmethod
    def require_optional_full_git_sha(cls, value: str | None) -> str | None:
        if value is not None and _FULL_GIT_SHA.fullmatch(value) is None:
            raise ValueError("commit must be a full lowercase SHA-1")
        return value

    @field_validator("content_hash")
    @classmethod
    def require_optional_sha256(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_HASH.fullmatch(value) is None:
            raise ValueError("content_hash must be a lowercase sha256 digest")
        return value

    @field_validator("timestamp")
    @classmethod
    def require_optional_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            _require_timestamp(value)
        return value

    @model_validator(mode="after")
    def require_commit_evidence_exactly_on_committed(self) -> OutboxReplayReceipt:
        evidence = (*_COMMITTED_FIELDS, "content_hash")
        if self.state is OutboxState.COMMITTED:
            missing = [name for name in evidence if getattr(self, name) is None]
            if missing:
                raise ValueError("committed replay receipt is missing evidence")
        else:
            present = [name for name in evidence if getattr(self, name) is not None]
            if present:
                raise ValueError(
                    "non-committed replay receipt must not carry commit evidence"
                )
        return self


class OutboxRejectReceipt(BaseModel):
    """A stable refusal without request payload or diagnostic detail."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    code: str = Field(
        json_schema_extra={"enum": [code.value for code in OutboxErrorCode]}
    )

    @field_validator("request_id")
    @classmethod
    def require_server_request_id_shape(cls, value: str) -> str:
        if _REQUEST_ID.fullmatch(value) is None:
            raise ValueError("request_id has invalid shape")
        return value

    @field_validator("code")
    @classmethod
    def require_stable_outbox_code(cls, value: str) -> str:
        if value not in set(OutboxErrorCode):
            raise ValueError("code is not an outbox v1 refusal code")
        return value


class EncryptedEnvelope(BaseModel):
    """The only place ciphertext lives; plaintext and keys never appear here."""

    model_config = ConfigDict(extra="forbid")

    cipher: str = Field(min_length=1)
    key_id: str = Field(min_length=1, max_length=64)
    nonce: str = Field(min_length=32, max_length=64, pattern=r"^[0-9a-f]+$")
    ciphertext: str = Field(min_length=0)
    mac: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class OutboxRecord(BaseModel):
    """One durable operation. The binding covers every immutable dimension."""

    model_config = ConfigDict(extra="forbid")

    record_version: str = Field(min_length=1)
    writer_contract: str = Field(min_length=1)
    operation_id: str = Field(
        min_length=1, max_length=64, pattern=_OPERATION_ID.pattern
    )
    idempotency_hash: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    task_hash: str = Field(min_length=1)
    target_path: str = Field(min_length=1, max_length=240)
    request_hash: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    enqueue_base_commit: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    expires_at: str = Field(min_length=1)
    binding_hash: str = Field(min_length=1)
    state: OutboxState
    attempts: int = Field(ge=0)
    last_code: str | None = None
    state_changed_at: str = Field(min_length=1)
    lease_until: str | None = None
    envelope: EncryptedEnvelope | None = None

    @field_validator(
        "idempotency_hash",
        "task_hash",
        "request_hash",
        "content_hash",
        "binding_hash",
    )
    @classmethod
    def require_sha256(cls, value: str) -> str:
        if _SHA256_HASH.fullmatch(value) is None:
            raise ValueError("hash must be a lowercase sha256 digest")
        return value

    @field_validator("enqueue_base_commit")
    @classmethod
    def require_full_git_sha(cls, value: str) -> str:
        if _FULL_GIT_SHA.fullmatch(value) is None:
            raise ValueError("enqueue_base_commit must be a full lowercase SHA-1")
        return value

    @field_validator("created_at", "expires_at", "state_changed_at")
    @classmethod
    def require_timezone_aware_timestamp(cls, value: str) -> str:
        return _require_timestamp(value)

    @field_validator("lease_until")
    @classmethod
    def require_optional_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            _require_timestamp(value)
        return value

    @model_validator(mode="after")
    def require_terminal_records_purged(self) -> OutboxRecord:
        if self.state in TERMINAL_STATES and self.envelope is not None:
            raise ValueError("terminal records must not retain an envelope")
        return self

    def expected_binding_hash(self) -> str:
        return compute_binding_hash(
            record_version=self.record_version,
            writer_contract=self.writer_contract,
            operation_id=self.operation_id,
            idempotency_hash=self.idempotency_hash,
            actor=self.actor,
            task_hash=self.task_hash,
            target_path=self.target_path,
            request_hash=self.request_hash,
            content_hash=self.content_hash,
            enqueue_base_commit=self.enqueue_base_commit,
            created_at=self.created_at,
            expires_at=self.expires_at,
        )


def compute_binding_hash(
    *,
    record_version: str,
    writer_contract: str,
    operation_id: str,
    idempotency_hash: str,
    actor: str,
    task_hash: str,
    target_path: str,
    request_hash: str,
    content_hash: str,
    enqueue_base_commit: str,
    created_at: str,
    expires_at: str,
) -> str:
    """Length-framed digest over every immutable record dimension."""
    return _framed_hash(
        BINDING_DOMAIN,
        record_version.encode("utf-8"),
        writer_contract.encode("utf-8"),
        operation_id.encode("utf-8"),
        idempotency_hash.encode("ascii"),
        actor.encode("utf-8"),
        task_hash.encode("ascii"),
        target_path.encode("utf-8"),
        request_hash.encode("ascii"),
        content_hash.encode("ascii"),
        enqueue_base_commit.encode("ascii"),
        created_at.encode("ascii"),
        expires_at.encode("ascii"),
    )


def _framed_hash(*values: bytes) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return f"sha256:{digest.hexdigest()}"


def record_from_json(payload: dict[str, Any]) -> OutboxRecord:
    """Parse a stored record, failing closed on version or binding drift."""
    from ckp.outbox.errors import OutboxErrorCode, OutboxRefusal

    if not isinstance(payload, dict):
        raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT)
    if payload.get("record_version") != RECORD_VERSION:
        raise OutboxRefusal(OutboxErrorCode.VERSION_UNKNOWN)
    if payload.get("writer_contract") != WRITER_CONTRACT:
        raise OutboxRefusal(OutboxErrorCode.VERSION_UNKNOWN)
    envelope = payload.get("envelope")
    if envelope is not None and envelope.get("cipher") is not None:
        from ckp.outbox.crypto import CIPHER_VERSION

        if envelope["cipher"] != CIPHER_VERSION:
            raise OutboxRefusal(OutboxErrorCode.VERSION_UNKNOWN)
    try:
        record = OutboxRecord.model_validate(payload)
    except ValueError as exc:
        raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT) from exc
    if record.expected_binding_hash() != record.binding_hash:
        raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT)
    return record


__all__ = [
    "BINDING_DOMAIN",
    "RECORD_VERSION",
    "TERMINAL_STATES",
    "WRITER_CONTRACT",
    "EncryptedEnvelope",
    "OutboxEnqueueReceipt",
    "OutboxRecord",
    "OutboxRejectReceipt",
    "OutboxReplayReceipt",
    "OutboxState",
    "compute_binding_hash",
    "record_from_json",
]
