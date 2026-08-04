"""Explicit synthetic-only composition for the C8 durable outbox.

Every dependency is required. The default application never builds an
outbox, no permissive default exists here, and the composed writer is a
dedicated instance whose clock is the outbox replay clock -- pinned per
replay to the enqueue timestamp so the content-hash binding holds.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ckp.outbox.crypto import OutboxCipherV1, OutboxKeyProvider
from ckp.outbox.policy import BaseMovementPolicyV1
from ckp.outbox.service import OutboxService, ReplayClock
from ckp.outbox.store import OutboxStore
from ckp.privacy import PrivacyGate
from ckp.writer.composition import build_c7_synthetic_writer
from ckp.writer.git import GitWriterBackend, SyntheticRepository
from ckp.writer.identity import WriterActorRegistry
from ckp.writer.target import WriterTargetPolicy
from ckp.writer.validation import (
    PayloadScanner,
    ValidationPipeline,
    WorktreeValidator,
)


def build_c8_synthetic_outbox(
    *,
    repository: SyntheticRepository,
    identity_registry: WriterActorRegistry,
    privacy_gate: PrivacyGate,
    payload_scanner: PayloadScanner,
    profile_validator: WorktreeValidator,
    lint_validator: WorktreeValidator,
    privacy_validator: WorktreeValidator,
    key_provider: OutboxKeyProvider,
    key_id: str,
    credential_resolver: Callable[[str], str],
    store_root: Path,
    clock: Callable[[], datetime],
    retention_seconds: float,
    lease_seconds: float,
    max_transient_attempts: int,
    lock_timeout_seconds: float,
) -> OutboxService:
    """Compose every required gate without exposing a production default."""
    replay_clock = ReplayClock()
    writer = build_c7_synthetic_writer(
        repository=repository,
        identity_registry=identity_registry,
        privacy_gate=privacy_gate,
        payload_scanner=payload_scanner,
        profile_validator=profile_validator,
        lint_validator=lint_validator,
        privacy_validator=privacy_validator,
        clock=replay_clock,
        lock_timeout_seconds=lock_timeout_seconds,
    )
    read_backend = GitWriterBackend(
        repository,
        lock_timeout_seconds=lock_timeout_seconds,
    )
    store = OutboxStore.open(
        store_root,
        disjoint_from=(repository.root, repository.state_root),
        lock_timeout_seconds=lock_timeout_seconds,
    )
    return OutboxService(
        identity_registry=identity_registry,
        target_policy=WriterTargetPolicy.c7_synthetic(),
        privacy_gate=privacy_gate,
        payload_scanner=payload_scanner,
        validation_pipeline=ValidationPipeline(
            profile=profile_validator,
            lint=lint_validator,
            privacy=privacy_validator,
        ),
        cipher=OutboxCipherV1(key_provider),
        key_id=key_id,
        store=store,
        read_base=read_backend.read_base,
        writer=writer,
        replay_clock=replay_clock,
        credential_resolver=credential_resolver,
        base_policy=BaseMovementPolicyV1(max_transient_attempts=max_transient_attempts),
        clock=clock,
        retention_seconds=retention_seconds,
        lease_seconds=lease_seconds,
    )


__all__ = ["build_c8_synthetic_outbox"]
