from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

import ckp.writer.git as writer_git
from ckp.writer.errors import WriterErrorCode, WriterRefusal
from ckp.writer.git import (
    SYNTHETIC_MARKER,
    OperationIdentity,
    PublishedOperation,
    SyntheticRepository,
)
from ckp.writer.models import RejectReceipt, SuccessReceipt
from writer_fixtures import (
    SYNTHETIC_CLAUDE_CREDENTIAL,
    SYNTHETIC_SECRET_MARKER,
    SYNTHETIC_STUDENT_MARKER,
    RecordingValidator,
    RejectingValidator,
    build_harness,
    custom_refs,
    invoke,
    make_synthetic_repository,
    move_main,
    repository_snapshot,
    run_git,
    valid_request,
    worktree_paths,
)


def test_valid_synthetic_core_write_commits_only_to_custom_refs(
    tmp_path: Path,
) -> None:
    harness = build_harness(tmp_path)
    before = repository_snapshot(harness.repository)

    receipt = invoke(harness.service)

    assert isinstance(receipt, SuccessReceipt)
    assert receipt.base_commit == before[1]
    assert repository_snapshot(harness.repository) == before
    assert worktree_paths(harness.repository) == (str(harness.repository.root),)
    refs = custom_refs(harness.repository)
    assert len(refs) == 2
    assert {line.rpartition(":")[2] for line in refs} == {receipt.result_commit}
    parent = str(
        run_git(harness.repository.root, "rev-parse", f"{receipt.result_commit}^")
    ).strip()
    assert parent == receipt.base_commit
    committed = str(
        run_git(
            harness.repository.root,
            "show",
            f"{receipt.result_commit}:{receipt.target_path}",
        )
    )
    assert "generated:" in committed
    assert "by: codex/gpt-5.6" in committed
    assert "privacy: internal" in committed
    assert "Synthetic fixture content only." in committed


@pytest.mark.parametrize(
    "case",
    ["missing", "wrong-content", "symlink", "hardlink", "state-inside", "branch"],
)
def test_repository_capability_requires_physical_committed_marker_and_external_state(
    tmp_path: Path,
    case: str,
) -> None:
    repository = make_synthetic_repository(tmp_path)
    marker = repository.root / SYNTHETIC_MARKER
    state_root = repository.state_root
    if case == "missing":
        marker.unlink()
        run_git(repository.root, "add", "--all")
        run_git(repository.root, "commit", "-m", "synthetic missing marker")
    elif case == "wrong-content":
        marker.write_text("synthetic wrong marker\n", encoding="utf-8")
        run_git(repository.root, "add", "--all")
        run_git(repository.root, "commit", "-m", "synthetic wrong marker")
    elif case == "symlink":
        marker.unlink()
        marker.symlink_to("README.synthetic")
        run_git(repository.root, "add", "--all")
        run_git(repository.root, "commit", "-m", "synthetic marker symlink")
    elif case == "hardlink":
        os.link(marker, repository.root / "SYNTHETIC_MARKER_HARDLINK")
    elif case == "state-inside":
        state_root = repository.root / "synthetic-state"
        state_root.mkdir()
    elif case == "branch":
        run_git(repository.root, "switch", "-c", "synthetic-other")

    with pytest.raises(WriterRefusal) as caught:
        SyntheticRepository.open(repository.root, state_root)
    assert caught.value.code is WriterErrorCode.REPOSITORY_DENIED


def test_valid_synthetic_private_capture_pair_passes(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    request = valid_request(
        operation_id="synthetic.capture-1",
        idempotency_key="synthetic-key-capture-0001",
        target_path="Private/_inbox/c7-synthetic/synthetic-capture.md",
        privacy="sensitive",
    )
    receipt = invoke(
        harness.service,
        request,
        provenance_mode="captured",
        model_id=None,
    )

    assert isinstance(receipt, SuccessReceipt)
    assert receipt.actor == "codex"
    committed = str(
        run_git(
            harness.repository.root,
            "show",
            f"{receipt.result_commit}:{receipt.target_path}",
        )
    )
    assert "captured:" in committed
    assert "by: codex" in committed
    assert "generated:" not in committed


def test_identical_retry_returns_original_receipt_without_new_commit(
    tmp_path: Path,
) -> None:
    harness = build_harness(tmp_path)
    first = invoke(harness.service)
    assert isinstance(first, SuccessReceipt)
    count_after_first = str(
        run_git(harness.repository.root, "rev-list", "--all", "--count")
    ).strip()
    refs_after_first = custom_refs(harness.repository)

    second = invoke(harness.service)

    assert second == first
    assert (
        str(run_git(harness.repository.root, "rev-list", "--all", "--count")).strip()
        == count_after_first
    )
    assert custom_refs(harness.repository) == refs_after_first
    assert len(harness.profile.calls) == 1  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "field_name",
    [
        "operation_id",
        "idempotency_hash",
        "actor",
        "task_hash",
        "target_path",
        "request_hash",
    ],
)
def test_operation_identity_match_binds_every_identity_field(
    field_name: str,
) -> None:
    identity = OperationIdentity(
        operation_id="synthetic.write-1",
        idempotency_hash="sha256:" + "1" * 64,
        actor="codex/gpt-5.6",
        task_hash="sha256:" + "2" * 64,
        target_path="Core/_inbox/c7-synthetic/synthetic-note.md",
        request_hash="sha256:" + "3" * 64,
    )
    record = PublishedOperation(
        operation_id=identity.operation_id,
        idempotency_hash=identity.idempotency_hash,
        actor=identity.actor,
        task_hash=identity.task_hash,
        target_path=identity.target_path,
        request_hash=identity.request_hash,
        base_commit="a" * 40,
        result_commit="b" * 40,
        timestamp="2026-08-03T12:34:56Z",
        content_hash="sha256:" + "4" * 64,
        binding_hash="sha256:" + "5" * 64,
    )
    changed = replace(record, **{field_name: f"changed-{field_name}"})

    assert identity.matches(record)
    assert not identity.matches(changed)


def test_partial_operation_ref_state_fails_closed(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    first = invoke(harness.service)
    assert isinstance(first, SuccessReceipt)
    operation_ref = next(
        line.partition(":")[0]
        for line in custom_refs(harness.repository)
        if "/operations/" in line
    )
    run_git(harness.repository.root, "update-ref", "-d", operation_ref)

    retry = invoke(harness.service)

    assert retry == RejectReceipt(
        request_id="server-request-1",
        code=WriterErrorCode.IDEMPOTENCY_CONFLICT,
    )
    assert len(custom_refs(harness.repository)) == 1
    assert worktree_paths(harness.repository) == (str(harness.repository.root),)


@pytest.mark.parametrize(
    "case",
    ["content", "target", "task", "actor", "operation", "key"],
)
def test_operation_or_key_reuse_with_different_binding_conflicts(
    tmp_path: Path,
    case: str,
) -> None:
    harness = build_harness(tmp_path)
    original = valid_request()
    first = invoke(harness.service, original)
    assert isinstance(first, SuccessReceipt)

    changed = valid_request()
    kwargs: dict[str, object] = {}
    if case == "content":
        changed["body"] = "# Synthetic changed content\n"
    elif case == "target":
        changed["target_path"] = "Core/_inbox/c7-synthetic/synthetic-other.md"
    elif case == "task":
        kwargs["task_id"] = "server-task-2"
    elif case == "actor":
        kwargs["actor_credential"] = SYNTHETIC_CLAUDE_CREDENTIAL
    elif case == "operation":
        changed["operation_id"] = "synthetic.write-2"
    elif case == "key":
        changed["idempotency_key"] = "synthetic-key-0002"

    receipt = invoke(harness.service, changed, **kwargs)  # type: ignore[arg-type]

    assert receipt == RejectReceipt(
        request_id="server-request-1",
        code=WriterErrorCode.IDEMPOTENCY_CONFLICT,
    )
    assert len(custom_refs(harness.repository)) == 2


@pytest.mark.parametrize(
    "case,expected",
    [
        ("missing-actor", WriterErrorCode.IDENTITY_MISSING),
        ("wrong-actor", WriterErrorCode.ACTOR_DENIED),
        ("identity-field", WriterErrorCode.REQUEST_INVALID),
        ("missing-model", WriterErrorCode.MODEL_IDENTITY_MISSING),
        ("generated-spoof", WriterErrorCode.PROVENANCE_FORBIDDEN),
        ("captured-spoof", WriterErrorCode.PROVENANCE_FORBIDDEN),
        ("verified-spoof", WriterErrorCode.PROVENANCE_FORBIDDEN),
        ("privacy-missing", WriterErrorCode.PRIVACY_UNCLASSIFIED),
        ("student-private", WriterErrorCode.STUDENT_PRIVATE_DENIED),
        ("secret-marker", WriterErrorCode.SECRET_DETECTED),
        ("student-marker", WriterErrorCode.STUDENT_PRIVATE_DENIED),
        ("formal-root", WriterErrorCode.TARGET_DENIED),
        ("cross-zone", WriterErrorCode.CROSS_ZONE_DENIED),
        ("traversal", WriterErrorCode.REQUEST_INVALID),
    ],
)
def test_preflight_rejections_happen_before_filesystem_mutation(
    tmp_path: Path,
    case: str,
    expected: WriterErrorCode,
) -> None:
    harness = build_harness(tmp_path)
    request = valid_request()
    kwargs: dict[str, object] = {}
    metadata = request["frontmatter"]
    assert isinstance(metadata, dict)

    if case == "missing-actor":
        kwargs["actor_credential"] = None
    elif case == "wrong-actor":
        kwargs["actor_credential"] = "SYNTHETIC_WRONG_CREDENTIAL"
    elif case == "identity-field":
        request["actor"] = "grok"
    elif case == "missing-model":
        kwargs["model_id"] = None
    elif case.endswith("-spoof"):
        metadata[case.removesuffix("-spoof")] = {"by": "synthetic/spoof"}
    elif case == "privacy-missing":
        metadata.pop("privacy")
    elif case == "student-private":
        metadata["privacy"] = "student-private"
    elif case == "secret-marker":
        request["body"] = SYNTHETIC_SECRET_MARKER.decode()
    elif case == "student-marker":
        request["body"] = SYNTHETIC_STUDENT_MARKER.decode()
    elif case == "formal-root":
        request["target_path"] = "Core/synthetic-formal.md"
    elif case == "cross-zone":
        metadata["privacy"] = "sensitive"
    elif case == "traversal":
        request["target_path"] = "Core/_inbox/c7-synthetic/../escape.md"

    before = repository_snapshot(harness.repository)
    receipt = invoke(harness.service, request, **kwargs)  # type: ignore[arg-type]

    assert receipt == RejectReceipt(
        request_id="server-request-1",
        code=expected,
    )
    assert repository_snapshot(harness.repository) == before
    assert custom_refs(harness.repository) == ()
    assert worktree_paths(harness.repository) == (str(harness.repository.root),)
    assert harness.profile.calls == []  # type: ignore[union-attr]


def test_symlink_ancestor_is_refused_without_worktree_mutation(tmp_path: Path) -> None:
    repository = make_synthetic_repository(tmp_path, symlink_core_route=True)
    harness = build_harness(tmp_path, repository=repository)
    before = repository_snapshot(repository)

    receipt = invoke(harness.service)

    assert receipt == RejectReceipt(
        request_id="server-request-1",
        code=WriterErrorCode.SYMLINK_DENIED,
    )
    assert repository_snapshot(repository) == before
    assert custom_refs(repository) == ()
    assert worktree_paths(repository) == (str(repository.root),)


def test_symlink_target_is_refused_without_following_it(tmp_path: Path) -> None:
    repository = make_synthetic_repository(tmp_path)
    target = repository.root / "Core/_inbox/c7-synthetic/synthetic-note.md"
    target.symlink_to(".keep")
    run_git(repository.root, "add", "--", str(target.relative_to(repository.root)))
    run_git(repository.root, "commit", "-m", "synthetic symlink target")
    harness = build_harness(tmp_path, repository=repository)
    before = repository_snapshot(repository)

    receipt = invoke(harness.service)

    assert receipt == RejectReceipt(
        request_id="server-request-1",
        code=WriterErrorCode.SYMLINK_DENIED,
    )
    assert repository_snapshot(repository) == before
    assert custom_refs(repository) == ()
    assert worktree_paths(repository) == (str(repository.root),)


def test_existing_target_is_a_path_conflict(tmp_path: Path) -> None:
    target = "Core/_inbox/c7-synthetic/synthetic-note.md"
    repository = make_synthetic_repository(tmp_path, existing_target=target)
    harness = build_harness(tmp_path, repository=repository)
    before = repository_snapshot(repository)

    receipt = invoke(harness.service)

    assert receipt == RejectReceipt(
        request_id="server-request-1",
        code=WriterErrorCode.PATH_CONFLICT,
    )
    assert repository_snapshot(repository) == before
    assert custom_refs(repository) == ()


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("profile", WriterErrorCode.PROFILE_VALIDATION_FAILED),
        ("lint", WriterErrorCode.LINT_FAILED),
        ("privacy", WriterErrorCode.PRIVACY_SCAN_FAILED),
    ],
)
def test_validation_failure_cleans_only_invocation_worktree(
    tmp_path: Path,
    stage: str,
    expected: WriterErrorCode,
) -> None:
    repository = make_synthetic_repository(tmp_path)
    validators: dict[str, object] = {
        "profile": RecordingValidator("profile"),
        "lint": RecordingValidator("lint"),
        "privacy": RecordingValidator("privacy"),
    }
    validators[stage] = RejectingValidator(expected)
    harness = build_harness(
        tmp_path,
        repository=repository,
        profile=validators["profile"],
        lint=validators["lint"],
        privacy=validators["privacy"],
    )
    before = repository_snapshot(repository)

    receipt = invoke(harness.service)

    assert receipt == RejectReceipt(
        request_id="server-request-1",
        code=expected,
    )
    assert repository_snapshot(repository) == before
    assert custom_refs(repository) == ()
    assert worktree_paths(repository) == (str(repository.root),)


@dataclass
class _ExtraDiffValidator:
    calls: int = 0

    def validate(self, worktree: Path, target_path: str) -> None:
        del target_path
        self.calls += 1
        (worktree / "SYNTHETIC_EXTRA_DIFF.txt").write_text(
            "synthetic unexpected diff\n",
            encoding="utf-8",
        )


def test_diff_allowlist_rejects_validator_side_effect(tmp_path: Path) -> None:
    repository = make_synthetic_repository(tmp_path)
    harness = build_harness(
        tmp_path,
        repository=repository,
        profile=_ExtraDiffValidator(),
    )
    before = repository_snapshot(repository)

    receipt = invoke(harness.service)

    assert receipt == RejectReceipt(
        request_id="server-request-1",
        code=WriterErrorCode.DIFF_DENIED,
    )
    assert repository_snapshot(repository) == before
    assert custom_refs(repository) == ()
    assert worktree_paths(repository) == (str(repository.root),)


def test_commit_failure_is_stable_and_does_not_stage_shared_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_harness(tmp_path)
    before = repository_snapshot(harness.repository)
    original_git = writer_git._git

    def fail_commit(root: Path, *args: str, **kwargs: object):
        if "commit" in args:
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
        return original_git(root, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(writer_git, "_git", fail_commit)
    receipt = invoke(harness.service)

    assert receipt == RejectReceipt(
        request_id="server-request-1",
        code=WriterErrorCode.COMMIT_FAILED,
    )
    assert repository_snapshot(harness.repository) == before
    assert custom_refs(harness.repository) == ()
    assert worktree_paths(harness.repository) == (str(harness.repository.root),)


@dataclass
class _BaseMovingValidator:
    repository: object
    moves: int
    calls: int = 0
    moved_to: list[str] = field(default_factory=list)

    def validate(self, worktree: Path, target_path: str) -> None:
        del worktree, target_path
        self.calls += 1
        if self.calls <= self.moves:
            self.moved_to.append(move_main(self.repository))  # type: ignore[arg-type]


def test_single_transient_base_move_retries_from_fresh_base(tmp_path: Path) -> None:
    repository = make_synthetic_repository(tmp_path)
    mover = _BaseMovingValidator(repository, moves=1)
    harness = build_harness(tmp_path, repository=repository, profile=mover)

    receipt = invoke(harness.service)

    assert isinstance(receipt, SuccessReceipt)
    assert mover.calls == 2
    assert receipt.base_commit == mover.moved_to[0]
    assert len(custom_refs(repository)) == 2
    assert repository_snapshot(repository)[2:] == (b"", b"")
    assert worktree_paths(repository) == (str(repository.root),)


def test_base_movement_stops_after_two_total_attempts(tmp_path: Path) -> None:
    repository = make_synthetic_repository(tmp_path)
    mover = _BaseMovingValidator(repository, moves=2)
    harness = build_harness(tmp_path, repository=repository, profile=mover)

    receipt = invoke(harness.service)

    assert receipt == RejectReceipt(
        request_id="server-request-1",
        code=WriterErrorCode.BASE_MOVED,
    )
    assert mover.calls == 2
    assert len(mover.moved_to) == 2
    assert custom_refs(repository) == ()
    assert repository_snapshot(repository)[2:] == (b"", b"")
    assert worktree_paths(repository) == (str(repository.root),)


@dataclass
class _BlockingValidator:
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    active: int = 0
    maximum_active: int = 0
    calls: int = 0

    def validate(self, worktree: Path, target_path: str) -> None:
        del worktree, target_path
        with self.state_lock:
            self.active += 1
            self.calls += 1
            self.maximum_active = max(self.maximum_active, self.active)
            first = self.calls == 1
        if first:
            self.entered.set()
            assert self.release.wait(timeout=5)
        with self.state_lock:
            self.active -= 1


def test_concurrent_writers_have_one_leader_and_isolated_worktree(
    tmp_path: Path,
) -> None:
    repository = make_synthetic_repository(tmp_path)
    blocker = _BlockingValidator()
    harness = build_harness(tmp_path, repository=repository, profile=blocker)
    receipts: list[object] = []

    def write(request: dict[str, object]) -> None:
        receipts.append(invoke(harness.service, request))

    first_request = valid_request()
    second_request = valid_request(
        operation_id="synthetic.write-2",
        idempotency_key="synthetic-key-0002",
        target_path="Core/_inbox/c7-synthetic/synthetic-note-2.md",
    )
    first = threading.Thread(target=write, args=(first_request,))
    second = threading.Thread(target=write, args=(second_request,))
    first.start()
    assert blocker.entered.wait(timeout=5)
    second.start()
    time.sleep(0.1)

    assert blocker.calls == 1
    assert blocker.maximum_active == 1
    assert len(worktree_paths(repository)) == 2

    blocker.release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive()
    assert not second.is_alive()
    assert len(receipts) == 2
    assert all(isinstance(receipt, SuccessReceipt) for receipt in receipts)
    assert blocker.maximum_active == 1
    assert len(custom_refs(repository)) == 4
    assert worktree_paths(repository) == (str(repository.root),)


def test_receipts_and_commit_metadata_do_not_echo_sensitive_inputs(
    tmp_path: Path,
) -> None:
    harness = build_harness(tmp_path)
    raw_key = "SYNTHETIC_RAW_IDEMPOTENCY_KEY_0001"
    raw_task = "synthetic-sensitive-task"
    body = "SYNTHETIC_BODY_NEVER_IN_RECEIPT"
    request = valid_request(idempotency_key=raw_key, body=body)
    success = invoke(harness.service, request, task_id=raw_task)
    assert isinstance(success, SuccessReceipt)

    success_wire = success.model_dump_json()
    commit_message = str(
        run_git(
            harness.repository.root,
            "show",
            "-s",
            "--format=%B",
            success.result_commit,
        )
    )
    for forbidden in (raw_key, raw_task, body, "frontmatter", "prompt", "session"):
        assert forbidden not in success_wire
        assert forbidden not in commit_message

    rejected = invoke(
        harness.service,
        valid_request(
            operation_id="synthetic.reject-1",
            idempotency_key="synthetic-reject-key-0001",
            target_path="Core/_inbox/c7-synthetic/synthetic-reject.md",
            body=SYNTHETIC_SECRET_MARKER.decode(),
        ),
        request_id="server-request-reject-1",
    )
    assert isinstance(rejected, RejectReceipt)
    assert set(rejected.model_dump()) == {"request_id", "code"}
    assert SYNTHETIC_SECRET_MARKER.decode() not in rejected.model_dump_json()


def test_dirty_shared_checkout_is_refused_and_never_cleaned(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    user_file = harness.repository.root / "SYNTHETIC_USER_WORK.txt"
    user_file.write_text("preserve synthetic user work\n", encoding="utf-8")

    receipt = invoke(harness.service)

    assert receipt == RejectReceipt(
        request_id="server-request-1",
        code=WriterErrorCode.REPOSITORY_DENIED,
    )
    assert user_file.read_text(encoding="utf-8") == "preserve synthetic user work\n"
    status = run_git(
        harness.repository.root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    assert "SYNTHETIC_USER_WORK.txt" in str(status)
    assert custom_refs(harness.repository) == ()
