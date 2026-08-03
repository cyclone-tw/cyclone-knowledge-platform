"""Explicit synthetic-only composition for the C7 Writer MVP."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from ckp.privacy import PrivacyGate
from ckp.writer.git import GitWriterBackend, SyntheticRepository
from ckp.writer.identity import WriterActorRegistry
from ckp.writer.service import WriterService
from ckp.writer.target import WriterTargetPolicy
from ckp.writer.validation import (
    PayloadScanner,
    ValidationPipeline,
    WorktreeValidator,
)


def build_c7_synthetic_writer(
    *,
    repository: SyntheticRepository,
    identity_registry: WriterActorRegistry,
    privacy_gate: PrivacyGate,
    payload_scanner: PayloadScanner,
    profile_validator: WorktreeValidator,
    lint_validator: WorktreeValidator,
    privacy_validator: WorktreeValidator,
    clock: Callable[[], datetime],
    lock_timeout_seconds: float,
) -> WriterService:
    """Compose every required gate without exposing a production default."""
    return WriterService(
        identity_registry=identity_registry,
        target_policy=WriterTargetPolicy.c7_synthetic(),
        privacy_gate=privacy_gate,
        payload_scanner=payload_scanner,
        validation_pipeline=ValidationPipeline(
            profile=profile_validator,
            lint=lint_validator,
            privacy=privacy_validator,
        ),
        git_backend=GitWriterBackend(
            repository,
            lock_timeout_seconds=lock_timeout_seconds,
        ),
        clock=clock,
    )


__all__ = ["build_c7_synthetic_writer"]
