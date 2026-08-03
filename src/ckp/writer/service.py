"""Authenticated, synthetic-only C7 Writer transaction coordinator."""

from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

import yaml
from pydantic import ValidationError

from ckp.bundle import BundleMember
from ckp.privacy import Admitted, PrivacyGate, Refused
from ckp.writer.errors import (
    WriterErrorCode,
    WriterRefusal,
    map_c2_reason,
)
from ckp.writer.git import (
    MAX_BASE_ATTEMPTS,
    GitWriterBackend,
    OperationCommit,
    OperationIdentity,
    PublishedOperation,
)
from ckp.writer.identity import (
    AuthenticatedWriterContext,
    ProvenanceMode,
    WriterActorRegistry,
)
from ckp.writer.models import RejectReceipt, SuccessReceipt, WriteRequest
from ckp.writer.target import WriterTargetPolicy
from ckp.writer.validation import PayloadScanner, ValidationPipeline

Receipt = SuccessReceipt | RejectReceipt
_FORBIDDEN_PROVENANCE = frozenset({"generated", "captured", "verified"})


class WriterService:
    """Validate first, then serialize one isolated Git transaction at a time."""

    def __init__(
        self,
        identity_registry: WriterActorRegistry,
        target_policy: WriterTargetPolicy,
        privacy_gate: PrivacyGate,
        payload_scanner: PayloadScanner,
        validation_pipeline: ValidationPipeline,
        git_backend: GitWriterBackend,
        clock: Callable[[], datetime],
    ) -> None:
        self._identity_registry = identity_registry
        self._target_policy = target_policy
        self._privacy_gate = privacy_gate
        self._payload_scanner = payload_scanner
        self._validation_pipeline = validation_pipeline
        self._git = git_backend
        self._clock = clock

    def execute(
        self,
        request: WriteRequest | Mapping[str, Any],
        *,
        request_id: str,
        actor_credential: str | None,
        task_id: str,
        provenance_mode: ProvenanceMode | str,
        model_id: str | None,
    ) -> Receipt:
        """Resolve trusted runtime identity, then run or safely reject."""
        try:
            context = self._identity_registry.resolve(
                actor_credential,
                request_id=request_id,
                task_id=task_id,
                provenance_mode=provenance_mode,
                model_id=model_id,
            )
            return self.execute_authenticated(request, context)
        except WriterRefusal as refusal:
            return _reject(request_id, refusal.code)
        except Exception:
            return _reject(request_id, WriterErrorCode.INTERNAL_ERROR)

    def execute_authenticated(
        self,
        request: WriteRequest | Mapping[str, Any],
        context: AuthenticatedWriterContext,
    ) -> Receipt:
        """Execute with a context sealed by this service's Writer registry."""
        request_id = getattr(context, "request_id", "server-request-invalid")
        worktree = None
        try:
            self._identity_registry.validate(context)
            parsed = (
                request
                if isinstance(request, WriteRequest)
                else WriteRequest.model_validate(request)
            )
            forbidden = _FORBIDDEN_PROVENANCE.intersection(parsed.frontmatter)
            if forbidden:
                raise WriterRefusal(WriterErrorCode.PROVENANCE_FORBIDDEN)
            target = self._target_policy.canonicalize(parsed.target_path)
            timestamp = _timestamp(self._clock())
            content = _render(parsed, context, timestamp)
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
                raise WriterRefusal(map_c2_reason(verdict.reason))
            if not isinstance(verdict, Admitted):
                raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
            if (
                verdict.member_key != target.path
                or verdict.content_sha256 != content_digest
            ):
                raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
            self._target_policy.authorize(target, verdict.privacy)
            self._payload_scanner.scan(content)

            identity = OperationIdentity(
                operation_id=parsed.operation_id,
                idempotency_hash=_hash(
                    parsed.idempotency_key.get_secret_value().encode("utf-8")
                ),
                actor=context.actor,
                task_hash=_hash(context.task_id.encode("utf-8")),
                target_path=target.path,
                request_hash=_request_hash(parsed, context),
            )
            with self._git.leader():
                self._git.repository.assert_shared_pristine()
                existing = self._git.find_existing(identity)
                if existing is not None:
                    _validate_record(existing)
                    return existing.receipt()

                for attempt in range(MAX_BASE_ATTEMPTS):
                    base_commit = self._git.read_base()
                    self._git.inspect_create_target(base_commit, target.path)
                    binding_hash = _binding_hash(
                        identity,
                        content_hash=content_hash,
                        base_commit=base_commit,
                    )
                    operation = OperationCommit(
                        identity=identity,
                        base_commit=base_commit,
                        timestamp=timestamp,
                        content_hash=content_hash,
                        binding_hash=binding_hash,
                    )
                    worktree = self._git.create_worktree(
                        base_commit,
                        parsed.operation_id,
                    )
                    try:
                        self._git.write_new_file(worktree, target.path, content)
                        self._validation_pipeline.validate(worktree, target.path)
                        self._git.assert_diff_allowlist(worktree, target.path)
                        result_commit = self._git.commit(worktree, operation)
                        if self._git.read_base() != base_commit:
                            if attempt + 1 >= MAX_BASE_ATTEMPTS:
                                raise WriterRefusal(WriterErrorCode.BASE_MOVED)
                            continue
                        self._git.publish(operation, result_commit)
                        record = PublishedOperation(
                            operation_id=identity.operation_id,
                            idempotency_hash=identity.idempotency_hash,
                            actor=identity.actor,
                            task_hash=identity.task_hash,
                            target_path=identity.target_path,
                            request_hash=identity.request_hash,
                            base_commit=base_commit,
                            result_commit=result_commit,
                            timestamp=timestamp,
                            content_hash=content_hash,
                            binding_hash=binding_hash,
                        )
                        _validate_record(record)
                        self._git.repository.assert_shared_pristine()
                        return record.receipt()
                    finally:
                        if worktree is not None:
                            self._git.remove_worktree(worktree)
                            worktree = None
                raise WriterRefusal(WriterErrorCode.BASE_MOVED)
        except ValidationError:
            return _reject(request_id, WriterErrorCode.REQUEST_INVALID)
        except WriterRefusal as refusal:
            return _reject(request_id, refusal.code)
        except Exception:
            return _reject(request_id, WriterErrorCode.INTERNAL_ERROR)


def _render(
    request: WriteRequest,
    context: AuthenticatedWriterContext,
    timestamp: str,
) -> bytes:
    # JSON normalization removes shared-object aliases before YAML rendering.
    normalized = json.loads(
        json.dumps(
            copy.deepcopy(request.frontmatter),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    provenance = {"by": context.actor, "at": timestamp}
    normalized[context.provenance_mode.value] = provenance
    try:
        frontmatter = yaml.safe_dump(
            normalized,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
            width=4096,
        ).encode("utf-8")
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise WriterRefusal(WriterErrorCode.REQUEST_INVALID) from exc
    body = request.body.encode("utf-8")
    content = b"---\n" + frontmatter + b"---\n\n" + body
    if not content.endswith(b"\n"):
        content += b"\n"
    return content


def _timestamp(value: datetime) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
    return value.isoformat().replace("+00:00", "Z")


def _request_hash(
    request: WriteRequest,
    context: AuthenticatedWriterContext,
) -> str:
    payload = {
        "operation_id": request.operation_id,
        "target_path": request.target_path,
        "frontmatter": request.frontmatter,
        "body": request.body,
        "provenance_mode": context.provenance_mode.value,
    }
    framed = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _hash(framed)


def _binding_hash(
    identity: OperationIdentity,
    *,
    content_hash: str,
    base_commit: str,
) -> str:
    return _framed_hash(
        b"ckp-writer-binding-v1",
        identity.operation_id.encode("utf-8"),
        identity.idempotency_hash.encode("ascii"),
        identity.actor.encode("utf-8"),
        identity.task_hash.encode("ascii"),
        identity.target_path.encode("utf-8"),
        identity.request_hash.encode("ascii"),
        content_hash.encode("ascii"),
        base_commit.encode("ascii"),
    )


def _validate_record(record: PublishedOperation) -> None:
    identity = OperationIdentity(
        operation_id=record.operation_id,
        idempotency_hash=record.idempotency_hash,
        actor=record.actor,
        task_hash=record.task_hash,
        target_path=record.target_path,
        request_hash=record.request_hash,
    )
    expected = _binding_hash(
        identity,
        content_hash=record.content_hash,
        base_commit=record.base_commit,
    )
    if expected != record.binding_hash:
        raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)


def _framed_hash(*values: bytes) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return f"sha256:{digest.hexdigest()}"


def _hash(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _reject(request_id: str, code: WriterErrorCode) -> RejectReceipt:
    try:
        return RejectReceipt(request_id=request_id, code=code.receipt_code)
    except ValidationError:
        return RejectReceipt(
            request_id="server-request-invalid",
            code=code.receipt_code,
        )


__all__ = ["Receipt", "WriterService"]
