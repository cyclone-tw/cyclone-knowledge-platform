"""Runtime-generated synthetic fixtures for C7 Writer tests."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ckp.privacy import FrontmatterClassifier, PrivacyClass, PrivacyGate
from ckp.writer import WriteRequest
from ckp.writer.composition import build_c7_synthetic_writer
from ckp.writer.errors import WriterErrorCode, WriterRefusal
from ckp.writer.git import (
    SYNTHETIC_MARKER,
    SYNTHETIC_MARKER_CONTENT,
    SyntheticRepository,
)
from ckp.writer.identity import ProvenanceMode, WriterActor, WriterActorRegistry
from ckp.writer.service import Receipt, WriterService
from ckp.writer.validation import ForbiddenPayloadMarker, LiteralPayloadScanner

SYNTHETIC_CODEX_CREDENTIAL = "SYNTHETIC_CREDENTIAL_CODEX_0001"
SYNTHETIC_CLAUDE_CREDENTIAL = "SYNTHETIC_CREDENTIAL_CLAUDE_0001"
SYNTHETIC_SECRET_MARKER = b"SYNTHETIC_SECRET_FIXTURE_DO_NOT_USE"
SYNTHETIC_STUDENT_MARKER = b"SYNTHETIC_STUDENT_IDENTIFIER_DO_NOT_USE"
FIXED_TIME = datetime(2026, 8, 3, 12, 34, 56, tzinfo=UTC)


def run_git(root: Path, *args: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ("git", "-C", str(root), *args),
        capture_output=True,
        check=True,
        text=text,
        timeout=15,
    )
    return completed.stdout


def make_synthetic_repository(
    tmp_path: Path,
    *,
    existing_target: str | None = None,
    symlink_core_route: bool = False,
) -> SyntheticRepository:
    root = tmp_path / "synthetic-git-fixture"
    state_root = tmp_path / "synthetic-writer-state"
    root.mkdir()
    state_root.mkdir()
    run_git(root, "init", "-b", "main")
    run_git(root, "config", "user.name", "Synthetic Fixture")
    run_git(root, "config", "user.email", "fixture@synthetic.invalid")

    (root / SYNTHETIC_MARKER).write_bytes(SYNTHETIC_MARKER_CONTENT)
    private_route = root / "Private/_inbox/c7-synthetic"
    private_route.mkdir(parents=True)
    (private_route / ".keep").write_text("synthetic fixture\n", encoding="utf-8")

    core_parent = root / "Core/_inbox"
    core_parent.mkdir(parents=True)
    core_route = core_parent / "c7-synthetic"
    if symlink_core_route:
        core_route.symlink_to("synthetic-symlink-target")
    else:
        core_route.mkdir()
        (core_route / ".keep").write_text("synthetic fixture\n", encoding="utf-8")

    if existing_target is not None:
        target = root / existing_target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Existing synthetic fixture\n", encoding="utf-8")

    run_git(root, "add", "--all")
    run_git(root, "commit", "-m", "synthetic writer fixture base")
    return SyntheticRepository.open(root, state_root)


@dataclass
class RecordingValidator:
    name: str
    calls: list[tuple[Path, str]] = field(default_factory=list)

    def validate(self, worktree: Path, target_path: str) -> None:
        self.calls.append((worktree, target_path))


@dataclass
class RejectingValidator:
    code: WriterErrorCode
    calls: int = 0

    def validate(self, worktree: Path, target_path: str) -> None:
        del worktree, target_path
        self.calls += 1
        raise WriterRefusal(self.code)


@dataclass
class WriterHarness:
    repository: SyntheticRepository
    registry: WriterActorRegistry
    service: WriterService
    profile: object
    lint: object
    privacy: object


def build_harness(
    tmp_path: Path,
    *,
    repository: SyntheticRepository | None = None,
    profile: object | None = None,
    lint: object | None = None,
    privacy: object | None = None,
    clock: object | None = None,
    lock_timeout_seconds: float = 1.0,
) -> WriterHarness:
    repository = repository or make_synthetic_repository(tmp_path)
    profile = profile or RecordingValidator("profile")
    lint = lint or RecordingValidator("lint")
    privacy = privacy or RecordingValidator("privacy")
    registry = WriterActorRegistry(
        (
            WriterActor.synthetic("codex", SYNTHETIC_CODEX_CREDENTIAL),
            WriterActor.synthetic(
                "claude-code",
                SYNTHETIC_CLAUDE_CREDENTIAL,
            ),
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
    service = build_c7_synthetic_writer(
        repository=repository,
        identity_registry=registry,
        privacy_gate=privacy_gate,
        payload_scanner=scanner,
        profile_validator=profile,  # type: ignore[arg-type]
        lint_validator=lint,  # type: ignore[arg-type]
        privacy_validator=privacy,  # type: ignore[arg-type]
        clock=clock or (lambda: FIXED_TIME),  # type: ignore[arg-type]
        lock_timeout_seconds=lock_timeout_seconds,
    )
    return WriterHarness(
        repository=repository,
        registry=registry,
        service=service,
        profile=profile,
        lint=lint,
        privacy=privacy,
    )


def valid_request(
    *,
    operation_id: str = "synthetic.write-1",
    idempotency_key: str = "synthetic-key-0001",
    target_path: str = "Core/_inbox/c7-synthetic/synthetic-note.md",
    privacy: str = "internal",
    body: str = "# Synthetic Note\n\nSynthetic fixture content only.\n",
    frontmatter: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "type": "Source",
        "title": "Synthetic Writer Note",
        "privacy": privacy,
        "status": "draft",
        "workflow_status": "inbox",
        "content_category": "development",
    }
    if frontmatter:
        metadata.update(frontmatter)
    return {
        "operation_id": operation_id,
        "idempotency_key": idempotency_key,
        "target_path": target_path,
        "frontmatter": metadata,
        "body": body,
    }


def invoke(
    service: WriterService,
    request: WriteRequest | dict[str, object] | None = None,
    *,
    request_id: str = "server-request-1",
    actor_credential: str | None = SYNTHETIC_CODEX_CREDENTIAL,
    task_id: str = "server-task-1",
    provenance_mode: ProvenanceMode | str = ProvenanceMode.GENERATED,
    model_id: str | None = "gpt-5.6",
) -> Receipt:
    return service.execute(
        valid_request() if request is None else request,
        request_id=request_id,
        actor_credential=actor_credential,
        task_id=task_id,
        provenance_mode=provenance_mode,
        model_id=model_id,
    )


def repository_snapshot(
    repository: SyntheticRepository,
) -> tuple[str, str, bytes, bytes]:
    branch = str(run_git(repository.root, "branch", "--show-current")).strip()
    head = str(run_git(repository.root, "rev-parse", "HEAD")).strip()
    status = run_git(
        repository.root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        text=False,
    )
    staged = run_git(
        repository.root,
        "diff",
        "--cached",
        "--binary",
        text=False,
    )
    assert isinstance(status, bytes)
    assert isinstance(staged, bytes)
    return branch, head, status, staged


def custom_refs(repository: SyntheticRepository) -> tuple[str, ...]:
    output = str(
        run_git(
            repository.root,
            "for-each-ref",
            "--format=%(refname):%(objectname)",
            "refs/ckp-writer",
        )
    )
    return tuple(line for line in output.splitlines() if line)


def worktree_paths(repository: SyntheticRepository) -> tuple[str, ...]:
    output = str(run_git(repository.root, "worktree", "list", "--porcelain"))
    return tuple(
        line.removeprefix("worktree ")
        for line in output.splitlines()
        if line.startswith("worktree ")
    )


def move_main(repository: SyntheticRepository) -> str:
    old = str(run_git(repository.root, "rev-parse", "refs/heads/main")).strip()
    tree = str(run_git(repository.root, "rev-parse", f"{old}^{{tree}}")).strip()
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": "Synthetic Base Mover",
            "GIT_AUTHOR_EMAIL": "base-mover@synthetic.invalid",
            "GIT_COMMITTER_NAME": "Synthetic Base Mover",
            "GIT_COMMITTER_EMAIL": "base-mover@synthetic.invalid",
            "GIT_AUTHOR_DATE": FIXED_TIME.isoformat(),
            "GIT_COMMITTER_DATE": FIXED_TIME.isoformat(),
        }
    )
    completed = subprocess.run(
        (
            "git",
            "-C",
            str(repository.root),
            "commit-tree",
            tree,
            "-p",
            old,
            "-m",
            "synthetic base movement",
        ),
        capture_output=True,
        check=True,
        text=True,
        timeout=15,
        env=environment,
    )
    new = completed.stdout.strip()
    run_git(repository.root, "update-ref", "refs/heads/main", new, old)
    return new


__all__ = [
    "FIXED_TIME",
    "SYNTHETIC_CLAUDE_CREDENTIAL",
    "SYNTHETIC_CODEX_CREDENTIAL",
    "SYNTHETIC_SECRET_MARKER",
    "SYNTHETIC_STUDENT_MARKER",
    "RecordingValidator",
    "RejectingValidator",
    "WriterHarness",
    "build_harness",
    "custom_refs",
    "invoke",
    "make_synthetic_repository",
    "move_main",
    "repository_snapshot",
    "run_git",
    "valid_request",
    "worktree_paths",
]
