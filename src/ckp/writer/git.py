"""Synthetic-only Git transaction primitives for the C7 Writer MVP."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ckp.writer.errors import WriterErrorCode, WriterRefusal
from ckp.writer.models import SuccessReceipt

SYNTHETIC_MARKER = ".ckp-writer-synthetic-fixture"
SYNTHETIC_MARKER_CONTENT = b"ckp-writer-synthetic-v1\n"
MAX_BASE_ATTEMPTS = 2

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_TRAILER_KEYS = (
    "CKP-Writer-Version",
    "CKP-Writer-Operation",
    "CKP-Writer-Idempotency",
    "CKP-Writer-Actor",
    "CKP-Writer-Task",
    "CKP-Writer-Target",
    "CKP-Writer-Request",
    "CKP-Writer-Base",
    "CKP-Writer-Timestamp",
    "CKP-Writer-Content",
    "CKP-Writer-Binding",
)


@dataclass(frozen=True, slots=True)
class _RepositorySeal:
    root: Path
    state_root: Path


@dataclass(frozen=True, slots=True)
class SyntheticRepository:
    """Capability for one clean, explicitly marked synthetic repository."""

    root: Path
    state_root: Path
    _seal: _RepositorySeal = field(repr=False, compare=False, kw_only=True)

    def __post_init__(self) -> None:
        if (
            type(self._seal) is not _RepositorySeal
            or self._seal.root != self.root
            or self._seal.state_root != self.state_root
        ):
            raise WriterRefusal(WriterErrorCode.REPOSITORY_DENIED)

    @classmethod
    def open(cls, root: Path, state_root: Path) -> SyntheticRepository:
        canonical_root = _physical_directory(root)
        canonical_state = _physical_directory(state_root)
        if _contains(canonical_root, canonical_state) or _contains(
            canonical_state, canonical_root
        ):
            raise WriterRefusal(WriterErrorCode.REPOSITORY_DENIED)
        top = _git(canonical_root, "rev-parse", "--show-toplevel").stdout.strip()
        if not top or Path(top).resolve() != canonical_root:
            raise WriterRefusal(WriterErrorCode.REPOSITORY_DENIED)
        branch = _git(canonical_root, "branch", "--show-current").stdout.strip()
        if branch != "main":
            raise WriterRefusal(WriterErrorCode.REPOSITORY_DENIED)
        base = _git(canonical_root, "rev-parse", "refs/heads/main").stdout.strip()
        if _FULL_SHA.fullmatch(base) is None:
            raise WriterRefusal(WriterErrorCode.REPOSITORY_DENIED)
        marker_entry = _git(
            canonical_root,
            "ls-tree",
            base,
            "--",
            SYNTHETIC_MARKER,
        ).stdout.strip()
        if not marker_entry.startswith("100644 blob "):
            raise WriterRefusal(WriterErrorCode.REPOSITORY_DENIED)
        marker = canonical_root / SYNTHETIC_MARKER
        marker_bytes = _read_regular_file_no_follow(marker)
        if marker_bytes != SYNTHETIC_MARKER_CONTENT:
            raise WriterRefusal(WriterErrorCode.REPOSITORY_DENIED)
        repository = cls(
            root=canonical_root,
            state_root=canonical_state,
            _seal=_RepositorySeal(canonical_root, canonical_state),
        )
        repository.assert_shared_pristine()
        return repository

    def assert_shared_pristine(self) -> None:
        branch = _git(self.root, "branch", "--show-current").stdout.strip()
        status = _git(
            self.root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            text=False,
        ).stdout
        if branch != "main" or status:
            raise WriterRefusal(WriterErrorCode.REPOSITORY_DENIED)


@dataclass(frozen=True, slots=True)
class OperationIdentity:
    operation_id: str
    idempotency_hash: str
    actor: str
    task_hash: str
    target_path: str
    request_hash: str

    def matches(self, record: PublishedOperation) -> bool:
        return (
            self.operation_id == record.operation_id
            and self.idempotency_hash == record.idempotency_hash
            and self.actor == record.actor
            and self.task_hash == record.task_hash
            and self.target_path == record.target_path
            and self.request_hash == record.request_hash
        )


@dataclass(frozen=True, slots=True)
class OperationCommit:
    identity: OperationIdentity
    base_commit: str
    timestamp: str
    content_hash: str
    binding_hash: str


@dataclass(frozen=True, slots=True)
class PublishedOperation:
    operation_id: str
    idempotency_hash: str
    actor: str
    task_hash: str
    target_path: str
    request_hash: str
    base_commit: str
    result_commit: str
    timestamp: str
    content_hash: str
    binding_hash: str

    def receipt(self) -> SuccessReceipt:
        return SuccessReceipt(
            operation_id=self.operation_id,
            target_path=self.target_path,
            base_commit=self.base_commit,
            result_commit=self.result_commit,
            actor=self.actor,
            timestamp=self.timestamp,
            content_hash=self.content_hash,
        )


class GitWriterBackend:
    """Own the one synthetic repository's lock, worktrees, commits, and refs."""

    def __init__(
        self,
        repository: SyntheticRepository,
        *,
        lock_timeout_seconds: float,
    ) -> None:
        if lock_timeout_seconds <= 0:
            raise WriterRefusal(WriterErrorCode.LEADER_UNAVAILABLE)
        self.repository = repository
        self._lock_timeout_seconds = lock_timeout_seconds

    @contextmanager
    def leader(self) -> Iterator[None]:
        lock_path = self.repository.state_root / ".ckp-writer.lock"
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
                raise OSError("unsafe lock")
        except OSError as exc:
            raise WriterRefusal(WriterErrorCode.LEADER_UNAVAILABLE) from exc
        deadline = time.monotonic() + self._lock_timeout_seconds
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise WriterRefusal(
                            WriterErrorCode.LEADER_UNAVAILABLE
                        ) from None
                    time.sleep(0.01)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def read_base(self) -> str:
        base = _git(self.repository.root, "rev-parse", "refs/heads/main").stdout.strip()
        if _FULL_SHA.fullmatch(base) is None:
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
        return base

    def inspect_create_target(self, base_commit: str, target_path: str) -> None:
        parts = PurePosixPath(target_path).parts
        for index in range(1, len(parts) + 1):
            candidate = "/".join(parts[:index])
            output = _git(
                self.repository.root,
                "ls-tree",
                base_commit,
                "--",
                candidate,
            ).stdout.strip()
            if not output:
                if index < len(parts):
                    raise WriterRefusal(WriterErrorCode.TARGET_DENIED)
                return
            line = output.splitlines()[0]
            fields = line.split(maxsplit=3)
            if len(fields) != 4:
                raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
            mode, object_type, _object_id, listed_path = fields
            if listed_path != candidate:
                raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
            if mode == "120000":
                raise WriterRefusal(WriterErrorCode.SYMLINK_DENIED)
            if index < len(parts):
                if mode != "040000" or object_type != "tree":
                    raise WriterRefusal(WriterErrorCode.PATH_CONFLICT)
            else:
                raise WriterRefusal(WriterErrorCode.PATH_CONFLICT)

    def find_existing(self, identity: OperationIdentity) -> PublishedOperation | None:
        operation_ref = _operation_ref(identity.operation_id)
        idempotency_ref = _idempotency_ref(identity.idempotency_hash)
        operation_sha = self._resolve_optional_ref(operation_ref)
        idempotency_sha = self._resolve_optional_ref(idempotency_ref)
        if operation_sha is None and idempotency_sha is None:
            return None
        if operation_sha is None or idempotency_sha is None:
            raise WriterRefusal(WriterErrorCode.IDEMPOTENCY_CONFLICT)
        shas = {
            value for value in (operation_sha, idempotency_sha) if value is not None
        }
        if len(shas) != 1:
            raise WriterRefusal(WriterErrorCode.IDEMPOTENCY_CONFLICT)
        result_commit = shas.pop()
        record = self._read_record(result_commit)
        if not identity.matches(record):
            raise WriterRefusal(WriterErrorCode.IDEMPOTENCY_CONFLICT)
        return record

    def create_worktree(self, base_commit: str, operation_id: str) -> Path:
        parent = self.repository.state_root / "worktrees"
        parent.mkdir(mode=0o700, exist_ok=True)
        physical_parent = _physical_directory(parent)
        if physical_parent.parent != self.repository.state_root:
            raise WriterRefusal(WriterErrorCode.REPOSITORY_DENIED)
        prefix = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:12]
        worktree = Path(tempfile.mkdtemp(prefix=f"ckp-{prefix}-", dir=physical_parent))
        worktree.rmdir()
        try:
            _git(
                self.repository.root,
                "worktree",
                "add",
                "--detach",
                str(worktree),
                base_commit,
            )
        except WriterRefusal as exc:
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR) from exc
        return worktree

    def write_new_file(self, worktree: Path, target_path: str, content: bytes) -> None:
        root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        child_flags = root_flags
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            root_flags |= os.O_NOFOLLOW
            child_flags |= os.O_NOFOLLOW
            file_flags |= os.O_NOFOLLOW
        descriptors: list[int] = []
        try:
            current = os.open(worktree, root_flags)
            descriptors.append(current)
            parts = PurePosixPath(target_path).parts
            for part in parts[:-1]:
                current = os.open(part, child_flags, dir_fd=current)
                descriptors.append(current)
            target_fd = os.open(parts[-1], file_flags, 0o600, dir_fd=current)
            descriptors.append(target_fd)
            view = memoryview(content)
            while view:
                written = os.write(target_fd, view)
                view = view[written:]
            os.fsync(target_fd)
            value = os.fstat(target_fd)
            if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
                raise OSError("unsafe target")
        except FileExistsError as exc:
            raise WriterRefusal(WriterErrorCode.PATH_CONFLICT) from exc
        except OSError as exc:
            raise WriterRefusal(WriterErrorCode.SYMLINK_DENIED) from exc
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def assert_diff_allowlist(self, worktree: Path, target_path: str) -> None:
        expected_untracked = b"?? " + target_path.encode("utf-8") + b"\x00"
        status = _git(
            worktree,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            text=False,
        ).stdout
        if status != expected_untracked:
            raise WriterRefusal(WriterErrorCode.DIFF_DENIED)
        cached = _git(
            worktree,
            "diff",
            "--cached",
            "--name-only",
            "-z",
            text=False,
        ).stdout
        if cached:
            raise WriterRefusal(WriterErrorCode.DIFF_DENIED)

    def commit(
        self,
        worktree: Path,
        operation: OperationCommit,
    ) -> str:
        target = operation.identity.target_path
        self.assert_diff_allowlist(worktree, target)
        try:
            _git(worktree, "add", "--", target)
            status = _git(
                worktree,
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                text=False,
            ).stdout
            if status != b"A  " + target.encode("utf-8") + b"\x00":
                raise WriterRefusal(WriterErrorCode.DIFF_DENIED)
            names = _git(
                worktree,
                "diff",
                "--cached",
                "--name-only",
                "-z",
                "--diff-filter=ACMRTUXB",
                text=False,
            ).stdout
            if names != target.encode("utf-8") + b"\x00":
                raise WriterRefusal(WriterErrorCode.DIFF_DENIED)
            environment = {
                "GIT_AUTHOR_NAME": "CKP Synthetic Writer",
                "GIT_AUTHOR_EMAIL": "writer@synthetic.invalid",
                "GIT_COMMITTER_NAME": "CKP Synthetic Writer",
                "GIT_COMMITTER_EMAIL": "writer@synthetic.invalid",
                "GIT_AUTHOR_DATE": operation.timestamp,
                "GIT_COMMITTER_DATE": operation.timestamp,
            }
            _git(
                worktree,
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "--quiet",
                "-m",
                "ckp writer synthetic operation",
                "-m",
                _commit_trailers(operation),
                env=environment,
            )
            result = _git(worktree, "rev-parse", "HEAD").stdout.strip()
            if _FULL_SHA.fullmatch(result) is None:
                raise WriterRefusal(WriterErrorCode.COMMIT_FAILED)
            changed = _git(
                worktree,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                "-z",
                result,
                text=False,
            ).stdout
            if changed != target.encode("utf-8") + b"\x00":
                raise WriterRefusal(WriterErrorCode.DIFF_DENIED)
            return result
        except WriterRefusal as exc:
            if exc.code in {
                WriterErrorCode.DIFF_DENIED,
                WriterErrorCode.COMMIT_FAILED,
            }:
                raise
            raise WriterRefusal(WriterErrorCode.COMMIT_FAILED) from exc
        except Exception as exc:
            raise WriterRefusal(WriterErrorCode.COMMIT_FAILED) from exc

    def publish(self, operation: OperationCommit, result_commit: str) -> None:
        operation_ref = _operation_ref(operation.identity.operation_id)
        idempotency_ref = _idempotency_ref(operation.identity.idempotency_hash)
        transaction = (
            "start\n"
            f"create {operation_ref} {result_commit}\n"
            f"create {idempotency_ref} {result_commit}\n"
            "prepare\n"
            "commit\n"
        )
        try:
            _git(
                self.repository.root,
                "update-ref",
                "--stdin",
                input_bytes=transaction.encode("ascii"),
            )
        except WriterRefusal as exc:
            raise WriterRefusal(WriterErrorCode.IDEMPOTENCY_CONFLICT) from exc

    def remove_worktree(self, worktree: Path) -> None:
        expected_parent = self.repository.state_root / "worktrees"
        try:
            if worktree.parent.resolve() != expected_parent.resolve():
                raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
            _git(
                self.repository.root,
                "worktree",
                "remove",
                "--force",
                str(worktree),
            )
        except WriterRefusal:
            raise
        except OSError as exc:
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR) from exc

    def _resolve_optional_ref(self, ref: str) -> str | None:
        result = _git_optional(
            self.repository.root,
            "rev-parse",
            "--verify",
            "--quiet",
            f"{ref}^{{commit}}",
        )
        if result.returncode == 1:
            return None
        value = result.stdout.strip()
        if result.returncode != 0 or _FULL_SHA.fullmatch(value) is None:
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
        return value

    def _read_record(self, result_commit: str) -> PublishedOperation:
        message = _git(
            self.repository.root,
            "show",
            "-s",
            "--format=%B",
            result_commit,
        ).stdout
        trailers: dict[str, str] = {}
        for line in message.splitlines():
            key, separator, value = line.partition(": ")
            if separator and key in _TRAILER_KEYS:
                if key in trailers:
                    raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
                trailers[key] = value
        if set(trailers) != set(_TRAILER_KEYS) or trailers["CKP-Writer-Version"] != "1":
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
        for key in (
            "CKP-Writer-Idempotency",
            "CKP-Writer-Task",
            "CKP-Writer-Request",
            "CKP-Writer-Content",
            "CKP-Writer-Binding",
        ):
            if _HASH.fullmatch(trailers[key]) is None:
                raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
        if _FULL_SHA.fullmatch(trailers["CKP-Writer-Base"]) is None:
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
        record = PublishedOperation(
            operation_id=trailers["CKP-Writer-Operation"],
            idempotency_hash=trailers["CKP-Writer-Idempotency"],
            actor=trailers["CKP-Writer-Actor"],
            task_hash=trailers["CKP-Writer-Task"],
            target_path=trailers["CKP-Writer-Target"],
            request_hash=trailers["CKP-Writer-Request"],
            base_commit=trailers["CKP-Writer-Base"],
            result_commit=result_commit,
            timestamp=trailers["CKP-Writer-Timestamp"],
            content_hash=trailers["CKP-Writer-Content"],
            binding_hash=trailers["CKP-Writer-Binding"],
        )
        self._validate_result_commit(record)
        return record

    def _validate_result_commit(self, record: PublishedOperation) -> None:
        parents = _git(
            self.repository.root,
            "rev-list",
            "--parents",
            "-n",
            "1",
            record.result_commit,
        ).stdout.split()
        if parents != [record.result_commit, record.base_commit]:
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
        changed = _git(
            self.repository.root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            record.result_commit,
            text=False,
        ).stdout
        if changed != record.target_path.encode("utf-8") + b"\x00":
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
        content = _git(
            self.repository.root,
            "show",
            f"{record.result_commit}:{record.target_path}",
            text=False,
        ).stdout
        assert isinstance(content, bytes)
        actual_hash = f"sha256:{hashlib.sha256(content).hexdigest()}"
        if actual_hash != record.content_hash:
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)


def _physical_directory(path: Path) -> Path:
    try:
        absolute = path.expanduser().absolute()
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise WriterRefusal(WriterErrorCode.REPOSITORY_DENIED) from exc
    if absolute != resolved or not resolved.is_dir():
        raise WriterRefusal(WriterErrorCode.REPOSITORY_DENIED)
    return resolved


def _read_regular_file_no_follow(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
                raise OSError("unsafe marker")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 4096):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise WriterRefusal(WriterErrorCode.REPOSITORY_DENIED) from exc


def _contains(parent: Path, child: Path) -> bool:
    return parent == child or parent in child.parents


def _operation_ref(operation_id: str) -> str:
    digest = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
    return f"refs/ckp-writer/operations/{digest}"


def _idempotency_ref(idempotency_hash: str) -> str:
    if _HASH.fullmatch(idempotency_hash) is None:
        raise WriterRefusal(WriterErrorCode.REQUEST_INVALID)
    return f"refs/ckp-writer/idempotency/{idempotency_hash.removeprefix('sha256:')}"


def _commit_trailers(operation: OperationCommit) -> str:
    identity = operation.identity
    return "\n".join(
        (
            "CKP-Writer-Version: 1",
            f"CKP-Writer-Operation: {identity.operation_id}",
            f"CKP-Writer-Idempotency: {identity.idempotency_hash}",
            f"CKP-Writer-Actor: {identity.actor}",
            f"CKP-Writer-Task: {identity.task_hash}",
            f"CKP-Writer-Target: {identity.target_path}",
            f"CKP-Writer-Request: {identity.request_hash}",
            f"CKP-Writer-Base: {operation.base_commit}",
            f"CKP-Writer-Timestamp: {operation.timestamp}",
            f"CKP-Writer-Content: {operation.content_hash}",
            f"CKP-Writer-Binding: {operation.binding_hash}",
        )
    )


@dataclass(frozen=True, slots=True)
class _GitResult:
    returncode: int
    stdout: str | bytes


def _git(
    root: Path,
    *args: str,
    text: bool = True,
    env: Mapping[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> _GitResult:
    result = _run_git(root, *args, text=text, env=env, input_bytes=input_bytes)
    if result.returncode != 0:
        raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
    return result


def _git_optional(root: Path, *args: str) -> _GitResult:
    return _run_git(root, *args, text=True, env=None, input_bytes=None)


def _run_git(
    root: Path,
    *args: str,
    text: bool,
    env: Mapping[str, str] | None,
    input_bytes: bytes | None,
) -> _GitResult:
    command = ("git", "-C", str(root), *args)
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    try:
        completed = subprocess.run(
            command,
            input=input_bytes,
            capture_output=True,
            check=False,
            text=False,
            env=process_env,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR) from exc
    stdout: str | bytes
    if text:
        try:
            stdout = completed.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR) from exc
    else:
        stdout = completed.stdout
    return _GitResult(returncode=completed.returncode, stdout=stdout)


__all__ = [
    "MAX_BASE_ATTEMPTS",
    "SYNTHETIC_MARKER",
    "SYNTHETIC_MARKER_CONTENT",
    "GitWriterBackend",
    "OperationCommit",
    "OperationIdentity",
    "PublishedOperation",
    "SyntheticRepository",
]
