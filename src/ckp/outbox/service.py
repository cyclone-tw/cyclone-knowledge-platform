"""Enqueue admission and replay for the synthetic-only C8 durable outbox.

Two rules organize this module. First, admission before persistence: every
C7 gate -- identity, provenance, target, privacy, payload scan, and the
worktree validators -- runs before a single byte reaches the store, and a
rejected payload leaves no durable trace. Second, one logical commit: replay
is at-least-once delivery on top of C7's operation/idempotency refs, so a
crash between the Git commit and the record rewrite is healed by the next
replay returning the original receipt instead of minting a second commit.

Rendering, request hashing, and identity hashing are deliberately *imported*
from the C7 service rather than copied: a second renderer would drift, and a
drifted renderer breaks the content-hash binding that replay verifies.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unicodedata
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ckp.bundle import BundleMember
from ckp.outbox.crypto import OutboxCipherV1
from ckp.outbox.errors import (
    OutboxErrorCode,
    OutboxRefusal,
    writer_code_to_outbox_code,
)
from ckp.outbox.models import (
    RECORD_VERSION,
    TERMINAL_STATES,
    WRITER_CONTRACT,
    OutboxEnqueueReceipt,
    OutboxRecord,
    OutboxRejectReceipt,
    OutboxReplayReceipt,
    OutboxState,
)
from ckp.outbox.policy import BaseMovementPolicyV1, ReplayDispositionKind
from ckp.outbox.store import OutboxStore
from ckp.privacy import Admitted, PrivacyGate, Refused
from ckp.writer.errors import WriterRefusal, map_c2_reason
from ckp.writer.identity import ProvenanceMode, WriterActorRegistry
from ckp.writer.models import SuccessReceipt, WriteRequest
from ckp.writer.service import (
    _FORBIDDEN_PROVENANCE,
    WriterService,
    _hash,
    _render,
    _request_hash,
)
from ckp.writer.target import WriterTargetPolicy
from ckp.writer.validation import PayloadScanner, ValidationPipeline

EnqueueReceipt = OutboxEnqueueReceipt | OutboxRejectReceipt

#: Fields the encrypted payload must carry to rebuild one canonical replay.
_PAYLOAD_FIELDS = frozenset(
    {
        "actor_id",
        "task_id",
        "provenance_mode",
        "model_id",
        "operation_id",
        "idempotency_key",
        "target_path",
        "frontmatter",
        "body",
        "created_at",
    }
)


class ReplayClock:
    """Pin the dedicated C7 writer to the enqueue-time timestamp.

    Rendered content embeds the provenance timestamp, so replay must render
    with ``created_at`` for the content-hash binding to hold. The clock is
    set only around one writer call and cleared after; an unset read is a
    composition bug and fails, never silently falls back to wall time.
    """

    def __init__(self) -> None:
        self._current: datetime | None = None

    def pin(self, value: datetime) -> None:
        self._current = value

    def clear(self) -> None:
        self._current = None

    def __call__(self) -> datetime:
        if self._current is None:
            raise OutboxRefusal(OutboxErrorCode.INTERNAL_ERROR)
        return self._current


class OutboxService:
    """Admit, durably bind, and replay synthetic Writer operations."""

    def __init__(
        self,
        *,
        identity_registry: WriterActorRegistry,
        target_policy: WriterTargetPolicy,
        privacy_gate: PrivacyGate,
        payload_scanner: PayloadScanner,
        validation_pipeline: ValidationPipeline,
        cipher: OutboxCipherV1,
        key_id: str,
        store: OutboxStore,
        read_base: Callable[[], str],
        writer: WriterService,
        replay_clock: ReplayClock,
        credential_resolver: Callable[[str], str],
        base_policy: BaseMovementPolicyV1,
        clock: Callable[[], datetime],
        retention_seconds: float,
        lease_seconds: float,
    ) -> None:
        if retention_seconds <= 0 or lease_seconds <= 0:
            raise OutboxRefusal(OutboxErrorCode.INTERNAL_ERROR)
        self._identity_registry = identity_registry
        self._target_policy = target_policy
        self._privacy_gate = privacy_gate
        self._payload_scanner = payload_scanner
        self._validation_pipeline = validation_pipeline
        self._cipher = cipher
        self._key_id = key_id
        self._store = store
        self._read_base = read_base
        self._writer = writer
        self._replay_clock = replay_clock
        self._credential_resolver = credential_resolver
        self._base_policy = base_policy
        self._clock = clock
        self._retention = timedelta(seconds=retention_seconds)
        self._lease = timedelta(seconds=lease_seconds)

    # -- enqueue ----------------------------------------------------------

    def enqueue(
        self,
        request: WriteRequest | Mapping[str, Any],
        *,
        request_id: str,
        actor_credential: str | None,
        task_id: str,
        provenance_mode: ProvenanceMode | str,
        model_id: str | None,
    ) -> EnqueueReceipt:
        """Run every admission gate, then bind and persist -- in that order."""
        try:
            context = self._identity_registry.resolve(
                actor_credential,
                request_id=request_id,
                task_id=task_id,
                provenance_mode=provenance_mode,
                model_id=model_id,
            )
            parsed = (
                request
                if isinstance(request, WriteRequest)
                else WriteRequest.model_validate(request)
            )
            if _FORBIDDEN_PROVENANCE.intersection(parsed.frontmatter):
                raise OutboxRefusal(OutboxErrorCode.PROVENANCE_FORBIDDEN)
            target = self._target_policy.canonicalize(parsed.target_path)
            created = self._clock()
            created_at = _timestamp(created)
            expires_at = _timestamp(created + self._retention)
            content = _render(parsed, context, created_at)
            content_digest = hashlib.sha256(content).hexdigest()
            content_hash = f"sha256:{content_digest}"
            member = BundleMember(
                relative_path=target.path,
                digest_key=unicodedata.normalize("NFC", target.path),
                content=content,
                content_sha256=content_digest,
            )
            verdict = self._privacy_gate.admit_member(member)
            if isinstance(verdict, Refused):
                raise OutboxRefusal(
                    writer_code_to_outbox_code(map_c2_reason(verdict.reason))
                )
            if not isinstance(verdict, Admitted):
                raise OutboxRefusal(OutboxErrorCode.INTERNAL_ERROR)
            if (
                verdict.member_key != target.path
                or verdict.content_sha256 != content_digest
            ):
                raise OutboxRefusal(OutboxErrorCode.INTERNAL_ERROR)
            self._target_policy.authorize(target, verdict.privacy)
            self._payload_scanner.scan(content)

            # The mutation lock covers the sandbox as well as the store
            # mutation: recover() sweeps crashed sandboxes under the same
            # lock, so it can never delete one that is still validating.
            with self._store.mutation_lock():
                self._validate_in_sandbox(target.path, content)
                record = OutboxRecord(
                    record_version=RECORD_VERSION,
                    writer_contract=WRITER_CONTRACT,
                    operation_id=parsed.operation_id,
                    idempotency_hash=_hash(
                        parsed.idempotency_key.get_secret_value().encode("utf-8")
                    ),
                    actor=context.actor,
                    task_hash=_hash(context.task_id.encode("utf-8")),
                    target_path=target.path,
                    request_hash=_request_hash(parsed, context),
                    content_hash=content_hash,
                    enqueue_base_commit=self._read_base(),
                    created_at=created_at,
                    expires_at=expires_at,
                    binding_hash="sha256:" + "0" * 64,
                    state=OutboxState.QUEUED,
                    attempts=0,
                    state_changed_at=created_at,
                    envelope=self._cipher.encrypt(
                        self._key_id,
                        _payload_bytes(parsed, context, created_at),
                    ),
                )
                record = record.model_copy(
                    update={"binding_hash": record.expected_binding_hash()}
                )
                existing = self._store.find_for_enqueue(
                    record.operation_id, record.idempotency_hash
                )
                if existing is not None:
                    if not _identity_matches(existing, record):
                        raise OutboxRefusal(OutboxErrorCode.BINDING_CONFLICT)
                    return _enqueue_receipt(existing)
                self._store.persist_new(record)
            return _enqueue_receipt(record)
        except ValidationError:
            return _reject(request_id, OutboxErrorCode.REQUEST_INVALID)
        except OutboxRefusal as refusal:
            return _reject(request_id, refusal.code)
        except WriterRefusal as refusal:
            return _reject(request_id, writer_code_to_outbox_code(refusal.code))
        except Exception:
            return _reject(request_id, OutboxErrorCode.INTERNAL_ERROR)

    def _validate_in_sandbox(self, target_path: str, content: bytes) -> None:
        """Run the worktree validators before anything durable exists.

        The sandbox is an 0700 directory under the store's admission area
        and is removed before this method returns -- on rejection *and* on
        success. A cleanup failure outranks the validation outcome, because
        it means plaintext is still on disk; and if the process dies inside
        the window, the next ``recover()`` sweeps the whole admission area,
        so crashed sandboxes never outlive one restart.
        """
        sandbox = Path(tempfile.mkdtemp(dir=self._store.admission_root))
        outcome: BaseException | None = None
        try:
            destination = sandbox / target_path
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            destination.write_bytes(content)
            self._validation_pipeline.validate(sandbox, target_path)
        except BaseException as exc:
            outcome = exc
        try:
            shutil.rmtree(sandbox)
        except OSError:
            raise OutboxRefusal(OutboxErrorCode.STORAGE_FAILED) from None
        if outcome is not None:
            raise outcome

    # -- replay -----------------------------------------------------------

    def replay_pending(self) -> list[OutboxReplayReceipt]:
        """Drain due records under a single consumer leader."""
        receipts: list[OutboxReplayReceipt] = []
        with self._store.consumer_leader():
            with self._store.mutation_lock():
                self._store.recover()
            for _name, record in self._store.scan():
                if record.state in TERMINAL_STATES:
                    continue
                now = self._clock()
                if (
                    record.state is OutboxState.LEASED
                    and record.lease_until is not None
                    and _parse(record.lease_until) > now
                ):
                    continue
                if now >= _parse(record.expires_at):
                    expired = self._transition(
                        record,
                        state=OutboxState.EXPIRED,
                        last_code=OutboxErrorCode.RECORD_EXPIRED.value,
                    )
                    receipts.append(
                        OutboxReplayReceipt(
                            operation_id=expired.operation_id,
                            state=OutboxState.EXPIRED,
                            code=OutboxErrorCode.RECORD_EXPIRED,
                        )
                    )
                    continue
                leased = self._transition(
                    record,
                    state=OutboxState.LEASED,
                    last_code=record.last_code,
                    lease_until=_timestamp(now + self._lease),
                )
                receipts.append(self._replay_one(leased))
        return receipts

    def _replay_one(self, record: OutboxRecord) -> OutboxReplayReceipt:
        try:
            parsed, context, created = self._rebind(record)
        except OutboxRefusal as refusal:
            self._transition(
                record,
                state=OutboxState.QUARANTINED,
                last_code=refusal.code.value,
            )
            return OutboxReplayReceipt(
                operation_id=record.operation_id,
                state=OutboxState.QUARANTINED,
                code=refusal.code,
            )
        self._replay_clock.pin(created)
        try:
            receipt = self._writer.execute_authenticated(parsed, context)
        finally:
            self._replay_clock.clear()
        if isinstance(receipt, SuccessReceipt):
            if (
                receipt.operation_id != record.operation_id
                or receipt.content_hash != record.content_hash
                or receipt.target_path != record.target_path
            ):
                self._transition(
                    record,
                    state=OutboxState.QUARANTINED,
                    last_code=OutboxErrorCode.REPLAY_QUARANTINED.value,
                )
                return OutboxReplayReceipt(
                    operation_id=record.operation_id,
                    state=OutboxState.QUARANTINED,
                    code=OutboxErrorCode.REPLAY_QUARANTINED,
                )
            self._transition(record, state=OutboxState.COMMITTED, last_code=None)
            return OutboxReplayReceipt(
                operation_id=record.operation_id,
                state=OutboxState.COMMITTED,
                content_hash=receipt.content_hash,
                target_path=receipt.target_path,
                base_commit=receipt.base_commit,
                result_commit=receipt.result_commit,
                actor=receipt.actor,
                timestamp=receipt.timestamp,
            )
        disposition = self._base_policy.dispose(receipt.code, record.attempts)
        if disposition.kind is ReplayDispositionKind.RETRY_TRANSIENT:
            self._transition(
                record,
                state=OutboxState.QUEUED,
                attempts=record.attempts + 1,
                last_code=receipt.code,
            )
            return OutboxReplayReceipt(
                operation_id=record.operation_id, state=OutboxState.QUEUED
            )
        if disposition.kind is ReplayDispositionKind.RETRY_FREE:
            self._transition(record, state=OutboxState.QUEUED, last_code=receipt.code)
            return OutboxReplayReceipt(
                operation_id=record.operation_id, state=OutboxState.QUEUED
            )
        if disposition.kind is ReplayDispositionKind.CONFLICT:
            self._transition(record, state=OutboxState.CONFLICT, last_code=receipt.code)
            return OutboxReplayReceipt(
                operation_id=record.operation_id,
                state=OutboxState.CONFLICT,
                code=OutboxErrorCode.REPLAY_CONFLICT,
            )
        self._transition(record, state=OutboxState.QUARANTINED, last_code=receipt.code)
        return OutboxReplayReceipt(
            operation_id=record.operation_id,
            state=OutboxState.QUARANTINED,
            code=OutboxErrorCode.REPLAY_QUARANTINED,
        )

    def _rebind(self, record: OutboxRecord) -> tuple[WriteRequest, Any, datetime]:
        """Decrypt and prove the payload still matches every bound dimension."""
        if record.envelope is None:
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT)
        plaintext = self._cipher.decrypt(record.envelope)
        try:
            payload = json.loads(plaintext)
        except ValueError as exc:
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT) from exc
        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_FIELDS:
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT)
        if payload["created_at"] != record.created_at:
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT)
        try:
            parsed = WriteRequest(
                operation_id=payload["operation_id"],
                idempotency_key=payload["idempotency_key"],
                target_path=payload["target_path"],
                frontmatter=payload["frontmatter"],
                body=payload["body"],
            )
        except ValidationError as exc:
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT) from exc
        if (
            parsed.operation_id != record.operation_id
            or parsed.target_path != record.target_path
        ):
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT)
        # The resolver's own exception may embed credential material; sever
        # the chain so no secret can ride along on the refusal.
        try:
            credential = self._credential_resolver(payload["actor_id"])
        except Exception:
            credential = None
        if credential is None or not isinstance(credential, str):
            raise OutboxRefusal(OutboxErrorCode.ACTOR_DENIED)
        name = OutboxStore.record_name(record.operation_id)
        try:
            context = self._identity_registry.resolve(
                credential,
                request_id=f"outbox-replay-{name[:16]}",
                task_id=payload["task_id"],
                provenance_mode=payload["provenance_mode"],
                model_id=payload["model_id"],
            )
        except WriterRefusal as refusal:
            raise OutboxRefusal(writer_code_to_outbox_code(refusal.code)) from refusal
        if context.actor != record.actor:
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT)
        if _hash(context.task_id.encode("utf-8")) != record.task_hash:
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT)
        if (
            _hash(parsed.idempotency_key.get_secret_value().encode("utf-8"))
            != record.idempotency_hash
        ):
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT)
        if _request_hash(parsed, context) != record.request_hash:
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT)
        rendered = _render(parsed, context, record.created_at)
        digest = f"sha256:{hashlib.sha256(rendered).hexdigest()}"
        if digest != record.content_hash:
            # The bound bytes cannot be reproduced; never guess a replay.
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT)
        return parsed, context, _parse(record.created_at)

    def _transition(
        self,
        record: OutboxRecord,
        *,
        state: OutboxState,
        attempts: int | None = None,
        last_code: str | None = None,
        lease_until: str | None = None,
    ) -> OutboxRecord:
        update: dict[str, Any] = {
            "state": state,
            "state_changed_at": _timestamp(self._clock()),
            "lease_until": lease_until,
            "last_code": last_code,
        }
        if attempts is not None:
            update["attempts"] = attempts
        if state in TERMINAL_STATES:
            # The frozen purge policy: terminal records keep their audit
            # metadata and lose their ciphertext immediately.
            update["envelope"] = None
        successor = record.model_copy(update=update)
        with self._store.mutation_lock():
            self._store.rewrite(successor)
        return successor


def _identity_matches(existing: OutboxRecord, candidate: OutboxRecord) -> bool:
    """The C7 ``OperationIdentity`` dimensions decide dedup versus conflict.

    Server-assigned values (created/expiry time, content hash over the
    timestamped render, base commit) are deliberately excluded: a genuine
    duplicate submitted later must dedup, not conflict.
    """
    return (
        existing.operation_id == candidate.operation_id
        and existing.idempotency_hash == candidate.idempotency_hash
        and existing.actor == candidate.actor
        and existing.task_hash == candidate.task_hash
        and existing.target_path == candidate.target_path
        and existing.request_hash == candidate.request_hash
    )


def _payload_bytes(request: WriteRequest, context: Any, created_at: str) -> bytes:
    payload = {
        "actor_id": context.actor_id,
        "task_id": context.task_id,
        "provenance_mode": context.provenance_mode.value,
        "model_id": context.model_id,
        "operation_id": request.operation_id,
        "idempotency_key": request.idempotency_key.get_secret_value(),
        "target_path": request.target_path,
        "frontmatter": request.frontmatter,
        "body": request.body,
        "created_at": created_at,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _enqueue_receipt(record: OutboxRecord) -> OutboxEnqueueReceipt:
    return OutboxEnqueueReceipt(
        operation_id=record.operation_id,
        state=record.state,
        content_hash=record.content_hash,
        enqueue_base_commit=record.enqueue_base_commit,
        created_at=record.created_at,
        expires_at=record.expires_at,
    )


def _reject(request_id: str, code: OutboxErrorCode) -> OutboxRejectReceipt:
    try:
        return OutboxRejectReceipt(request_id=request_id, code=code.value)
    except ValidationError:
        return OutboxRejectReceipt(request_id="server-request-invalid", code=code.value)


def _timestamp(value: datetime) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise OutboxRefusal(OutboxErrorCode.INTERNAL_ERROR)
    return value.isoformat().replace("+00:00", "Z")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


__all__ = ["EnqueueReceipt", "OutboxService", "ReplayClock"]
