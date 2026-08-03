"""Payload-safe adapters for C7's preflight and worktree validation gates."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ckp.writer.errors import WriterErrorCode, WriterRefusal


class PayloadScanner(Protocol):
    def scan(self, content: bytes) -> None: ...


class WorktreeValidator(Protocol):
    def validate(self, worktree: Path, target_path: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ForbiddenPayloadMarker:
    marker: bytes
    code: WriterErrorCode

    def __post_init__(self) -> None:
        if not self.marker or self.code not in {
            WriterErrorCode.SECRET_DETECTED,
            WriterErrorCode.STUDENT_PRIVATE_DENIED,
        }:
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)


class LiteralPayloadScanner:
    """Deterministic fixture scanner; production patterns remain injected."""

    def __init__(self, markers: tuple[ForbiddenPayloadMarker, ...]) -> None:
        if not markers:
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
        self._markers = markers

    def scan(self, content: bytes) -> None:
        for rule in self._markers:
            if rule.marker in content:
                raise WriterRefusal(rule.code)


class SubprocessProfileValidator:
    """Invoke one pinned external Profile validator without relaying findings."""

    def __init__(
        self,
        *,
        command_prefix: tuple[str, ...],
        tool_path: Path,
        tool_sha256: str,
        tool_revision: str,
        timeout_seconds: float,
    ) -> None:
        if (
            not command_prefix
            or not tool_revision
            or timeout_seconds <= 0
            or not tool_sha256.startswith("sha256:")
        ):
            raise WriterRefusal(WriterErrorCode.PROFILE_VALIDATION_FAILED)
        self._command_prefix = command_prefix
        self._tool_path = _regular_physical_file(tool_path)
        self._tool_sha256 = tool_sha256
        self._tool_revision = tool_revision
        self._timeout_seconds = timeout_seconds
        self._verify_tool()

    @property
    def tool_revision(self) -> str:
        return self._tool_revision

    def validate(self, worktree: Path, target_path: str) -> None:
        self._verify_tool()
        command = (
            *self._command_prefix,
            str(self._tool_path),
            target_path,
            "--mode",
            "target",
            "--level",
            "draft",
            "--strict",
        )
        _run_sanitized(
            command,
            cwd=worktree,
            timeout_seconds=self._timeout_seconds,
            code=WriterErrorCode.PROFILE_VALIDATION_FAILED,
        )

    def _verify_tool(self) -> None:
        try:
            digest = hashlib.sha256(self._tool_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise WriterRefusal(WriterErrorCode.PROFILE_VALIDATION_FAILED) from exc
        if f"sha256:{digest}" != self._tool_sha256:
            raise WriterRefusal(WriterErrorCode.PROFILE_VALIDATION_FAILED)


class SubprocessWorktreeValidator:
    """Run one trusted argv-only check against the isolated synthetic worktree."""

    def __init__(
        self,
        command: tuple[str, ...],
        *,
        code: WriterErrorCode,
        timeout_seconds: float,
        root_env_var: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if (
            not command
            or timeout_seconds <= 0
            or code
            not in {
                WriterErrorCode.LINT_FAILED,
                WriterErrorCode.PRIVACY_SCAN_FAILED,
            }
        ):
            raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR)
        self._command = command
        self._code = code
        self._timeout_seconds = timeout_seconds
        self._root_env_var = root_env_var
        self._environment = dict(environment or {})

    def validate(self, worktree: Path, target_path: str) -> None:
        del target_path
        environment = dict(self._environment)
        if self._root_env_var is not None:
            environment[self._root_env_var] = str(worktree)
        _run_sanitized(
            self._command,
            cwd=worktree,
            timeout_seconds=self._timeout_seconds,
            code=self._code,
            environment=environment,
        )


class ValidationPipeline:
    """Fixed post-write order: Profile, lint, then privacy scan."""

    def __init__(
        self,
        profile: WorktreeValidator,
        lint: WorktreeValidator,
        privacy: WorktreeValidator,
    ) -> None:
        self._profile = profile
        self._lint = lint
        self._privacy = privacy

    def validate(self, worktree: Path, target_path: str) -> None:
        self._profile.validate(worktree, target_path)
        self._lint.validate(worktree, target_path)
        self._privacy.validate(worktree, target_path)


def _regular_physical_file(path: Path) -> Path:
    try:
        absolute = path.expanduser().absolute()
        resolved = path.expanduser().resolve(strict=True)
        value = resolved.lstat()
    except OSError as exc:
        raise WriterRefusal(WriterErrorCode.PROFILE_VALIDATION_FAILED) from exc
    if absolute != resolved or not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise WriterRefusal(WriterErrorCode.PROFILE_VALIDATION_FAILED)
    return resolved


def _run_sanitized(
    command: tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: float,
    code: WriterErrorCode,
    environment: Mapping[str, str] | None = None,
) -> None:
    process_env = os.environ.copy()
    if environment:
        process_env.update(environment)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            check=False,
            text=False,
            timeout=timeout_seconds,
            env=process_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WriterRefusal(code) from exc
    if completed.returncode != 0:
        raise WriterRefusal(code)


__all__ = [
    "ForbiddenPayloadMarker",
    "LiteralPayloadScanner",
    "PayloadScanner",
    "SubprocessProfileValidator",
    "SubprocessWorktreeValidator",
    "ValidationPipeline",
    "WorktreeValidator",
]
