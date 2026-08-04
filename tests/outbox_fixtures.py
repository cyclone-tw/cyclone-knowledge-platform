"""Runtime-generated synthetic fixtures for C8 outbox tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ckp.outbox.composition import build_c8_synthetic_outbox
from ckp.outbox.crypto import SyntheticKeyProvider
from ckp.outbox.service import EnqueueReceipt, OutboxService
from ckp.outbox.store import OutboxStore
from ckp.privacy import FrontmatterClassifier, PrivacyClass, PrivacyGate
from ckp.writer.errors import WriterErrorCode
from ckp.writer.git import SyntheticRepository
from ckp.writer.identity import ProvenanceMode, WriterActor, WriterActorRegistry
from ckp.writer.validation import ForbiddenPayloadMarker, LiteralPayloadScanner
from writer_fixtures import (
    SYNTHETIC_CLAUDE_CREDENTIAL,
    SYNTHETIC_CODEX_CREDENTIAL,
    SYNTHETIC_SECRET_MARKER,
    SYNTHETIC_STUDENT_MARKER,
    RecordingValidator,
    make_synthetic_repository,
    run_git,
    valid_request,
)

SYNTHETIC_KEY_ID = "synthetic-outbox-key-1"
SYNTHETIC_KEY_BYTES = b"SYNTHETIC-OUTBOX-KEY-0123456789abcdef"
SYNTHETIC_BODY_NEEDLE = "SYNTHETIC-OUTBOX-BODY-NEEDLE"
OUTBOX_FIXED_TIME = datetime(2026, 8, 4, 10, 20, 30, tzinfo=UTC)

_CREDENTIALS = {
    "codex": SYNTHETIC_CODEX_CREDENTIAL,
    "claude-code": SYNTHETIC_CLAUDE_CREDENTIAL,
}


class MutableClock:
    """A deterministic clock the test advances explicitly."""

    def __init__(self, current: datetime = OUTBOX_FIXED_TIME) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current = self.current + timedelta(seconds=seconds)


@dataclass
class MovingBaseValidator:
    """Move ``main`` during validation to force writer base retries."""

    repository: SyntheticRepository
    times: int
    calls: int = 0

    def validate(self, worktree: Path, target_path: str) -> None:
        del worktree, target_path
        self.calls += 1
        if self.calls <= self.times:
            from writer_fixtures import move_main

            move_main(self.repository)


@dataclass
class OutboxHarness:
    repository: SyntheticRepository
    registry: WriterActorRegistry
    service: OutboxService
    store: OutboxStore
    store_root: Path
    clock: MutableClock
    profile: object
    lint: object
    privacy: object


def build_outbox_harness(
    tmp_path: Path,
    *,
    repository: SyntheticRepository | None = None,
    profile: object | None = None,
    lint: object | None = None,
    privacy: object | None = None,
    clock: MutableClock | None = None,
    key_provider: SyntheticKeyProvider | None = None,
    key_id: str = SYNTHETIC_KEY_ID,
    credential_resolver: object | None = None,
    retention_seconds: float = 3600.0,
    lease_seconds: float = 60.0,
    max_transient_attempts: int = 2,
    lock_timeout_seconds: float = 1.0,
) -> OutboxHarness:
    repository = repository or make_synthetic_repository(tmp_path)
    store_root = tmp_path / "synthetic-outbox-state"
    store_root.mkdir(exist_ok=True)
    clock = clock or MutableClock()
    profile = profile or RecordingValidator("profile")
    lint = lint or RecordingValidator("lint")
    privacy = privacy or RecordingValidator("privacy")
    registry = WriterActorRegistry(
        (
            WriterActor.synthetic("codex", SYNTHETIC_CODEX_CREDENTIAL),
            WriterActor.synthetic("claude-code", SYNTHETIC_CLAUDE_CREDENTIAL),
        )
    )
    privacy_gate = PrivacyGate(
        FrontmatterClassifier(repository.root),
        frozenset(
            {
                PrivacyClass.PUBLIC,
                PrivacyClass.INTERNAL,
                PrivacyClass.SENSITIVE,
            }
        ),
    )
    scanner = LiteralPayloadScanner(
        (
            ForbiddenPayloadMarker(
                SYNTHETIC_SECRET_MARKER,
                WriterErrorCode.SECRET_DETECTED,
            ),
            ForbiddenPayloadMarker(
                SYNTHETIC_STUDENT_MARKER,
                WriterErrorCode.STUDENT_PRIVATE_DENIED,
            ),
        )
    )
    service = build_c8_synthetic_outbox(
        repository=repository,
        identity_registry=registry,
        privacy_gate=privacy_gate,
        payload_scanner=scanner,
        profile_validator=profile,  # type: ignore[arg-type]
        lint_validator=lint,  # type: ignore[arg-type]
        privacy_validator=privacy,  # type: ignore[arg-type]
        key_provider=key_provider
        or SyntheticKeyProvider({key_id: SYNTHETIC_KEY_BYTES}),
        key_id=key_id,
        credential_resolver=credential_resolver or _CREDENTIALS.__getitem__,
        store_root=store_root,
        clock=clock,
        retention_seconds=retention_seconds,
        lease_seconds=lease_seconds,
        max_transient_attempts=max_transient_attempts,
        lock_timeout_seconds=lock_timeout_seconds,
    )
    store = OutboxStore.open(
        store_root,
        disjoint_from=(repository.root, repository.state_root),
        lock_timeout_seconds=lock_timeout_seconds,
    )
    return OutboxHarness(
        repository=repository,
        registry=registry,
        service=service,
        store=store,
        store_root=store_root,
        clock=clock,
        profile=profile,
        lint=lint,
        privacy=privacy,
    )


def outbox_request(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "operation_id": "synthetic.outbox-1",
        "idempotency_key": "synthetic-outbox-key-0001",
        "target_path": "Core/_inbox/c7-synthetic/synthetic-outbox-note.md",
        "body": f"# Synthetic Outbox Note\n\n{SYNTHETIC_BODY_NEEDLE}\n",
    }
    arguments.update(overrides)
    return valid_request(**arguments)  # type: ignore[arg-type]


def enqueue(
    service: OutboxService,
    request: dict[str, object] | None = None,
    *,
    request_id: str = "server-outbox-request-1",
    actor_credential: str | None = SYNTHETIC_CODEX_CREDENTIAL,
    task_id: str = "server-outbox-task-1",
    provenance_mode: ProvenanceMode | str = ProvenanceMode.GENERATED,
    model_id: str | None = "gpt-5.6",
) -> EnqueueReceipt:
    return service.enqueue(
        outbox_request() if request is None else request,
        request_id=request_id,
        actor_credential=actor_credential,
        task_id=task_id,
        provenance_mode=provenance_mode,
        model_id=model_id,
    )


def operation_commits(repository: SyntheticRepository) -> tuple[str, ...]:
    """Every commit on any custom writer ref: the logical-commit oracle."""
    output = str(
        run_git(
            repository.root,
            "for-each-ref",
            "--format=%(objectname)",
            "refs/ckp-writer/operations",
        )
    )
    return tuple(line for line in output.splitlines() if line)


def state_root_files(store_root: Path) -> list[Path]:
    return [path for path in sorted(store_root.rglob("*")) if path.is_file()]


def assert_no_plaintext_at_rest(store_root: Path) -> None:
    """No durable artifact may contain payload plaintext or key material."""
    needles = (
        SYNTHETIC_BODY_NEEDLE.encode("utf-8"),
        b"synthetic-outbox-key-0001",
        SYNTHETIC_KEY_BYTES,
        SYNTHETIC_CODEX_CREDENTIAL.encode("utf-8"),
        SYNTHETIC_CLAUDE_CREDENTIAL.encode("utf-8"),
    )
    for path in state_root_files(store_root):
        content = path.read_bytes()
        for needle in needles:
            assert needle not in content, (path, needle)


__all__ = [
    "OUTBOX_FIXED_TIME",
    "SYNTHETIC_BODY_NEEDLE",
    "SYNTHETIC_KEY_BYTES",
    "SYNTHETIC_KEY_ID",
    "MovingBaseValidator",
    "MutableClock",
    "OutboxHarness",
    "assert_no_plaintext_at_rest",
    "build_outbox_harness",
    "enqueue",
    "operation_commits",
    "outbox_request",
    "state_root_files",
]
