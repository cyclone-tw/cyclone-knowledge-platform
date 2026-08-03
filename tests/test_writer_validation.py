from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from ckp.writer.errors import WriterErrorCode, WriterRefusal
from ckp.writer.validation import (
    ForbiddenPayloadMarker,
    LiteralPayloadScanner,
    SubprocessProfileValidator,
    SubprocessWorktreeValidator,
    ValidationPipeline,
)
from writer_fixtures import SYNTHETIC_SECRET_MARKER, SYNTHETIC_STUDENT_MARKER


@pytest.mark.parametrize(
    "marker,expected",
    [
        (SYNTHETIC_SECRET_MARKER, WriterErrorCode.SECRET_DETECTED),
        (SYNTHETIC_STUDENT_MARKER, WriterErrorCode.STUDENT_PRIVATE_DENIED),
    ],
)
def test_preflight_scanner_refuses_synthetic_sensitive_markers(
    marker: bytes,
    expected: WriterErrorCode,
) -> None:
    scanner = LiteralPayloadScanner(
        (
            ForbiddenPayloadMarker(marker, expected),
            ForbiddenPayloadMarker(b"SYNTHETIC_UNUSED_MARKER", expected),
        )
    )
    with pytest.raises(WriterRefusal) as caught:
        scanner.scan(b"synthetic prefix\n" + marker + b"\nsynthetic suffix")
    assert caught.value.code is expected
    assert marker.decode() not in str(caught.value)


def _profile_tool(tmp_path: Path, *, exit_code: int = 0) -> tuple[Path, str]:
    tool = tmp_path / "synthetic_profile_validator.py"
    tool.write_text(
        "import sys\n"
        "assert '--report' not in sys.argv\n"
        "assert sys.argv[2:] == "
        "['--mode', 'target', '--level', 'draft', '--strict']\n"
        "print('SYNTHETIC_SENSITIVE_FINDING_MUST_NOT_ESCAPE')\n"
        "print('SYNTHETIC_PRIVATE_METADATA_MUST_NOT_ESCAPE', file=sys.stderr)\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(tool.read_bytes()).hexdigest()
    return tool, f"sha256:{digest}"


def test_profile_adapter_pins_tool_and_uses_single_target_strict_mode(
    tmp_path: Path,
) -> None:
    tool, digest = _profile_tool(tmp_path)
    worktree = tmp_path / "synthetic-worktree"
    worktree.mkdir()
    validator = SubprocessProfileValidator(
        command_prefix=(sys.executable,),
        tool_path=tool,
        tool_sha256=digest,
        tool_revision="synthetic-profile-revision-1",
        timeout_seconds=2,
    )

    validator.validate(
        worktree,
        "Core/_inbox/c7-synthetic/synthetic-note.md",
    )
    assert validator.tool_revision == "synthetic-profile-revision-1"


def test_profile_adapter_sanitizes_findings_and_fails_closed(tmp_path: Path) -> None:
    tool, digest = _profile_tool(tmp_path, exit_code=7)
    validator = SubprocessProfileValidator(
        command_prefix=(sys.executable,),
        tool_path=tool,
        tool_sha256=digest,
        tool_revision="synthetic-profile-revision-2",
        timeout_seconds=2,
    )
    with pytest.raises(WriterRefusal) as caught:
        validator.validate(tmp_path, "synthetic-note.md")
    assert caught.value.code is WriterErrorCode.PROFILE_VALIDATION_FAILED
    assert "SYNTHETIC_SENSITIVE" not in str(caught.value)
    assert "SYNTHETIC_PRIVATE" not in str(caught.value)


def test_profile_adapter_rechecks_digest_before_each_call(tmp_path: Path) -> None:
    tool, digest = _profile_tool(tmp_path)
    validator = SubprocessProfileValidator(
        command_prefix=(sys.executable,),
        tool_path=tool,
        tool_sha256=digest,
        tool_revision="synthetic-profile-revision-3",
        timeout_seconds=2,
    )
    tool.write_text("raise SystemExit(0)\n", encoding="utf-8")

    with pytest.raises(WriterRefusal) as caught:
        validator.validate(tmp_path, "synthetic-note.md")
    assert caught.value.code is WriterErrorCode.PROFILE_VALIDATION_FAILED


def test_profile_adapter_rejects_symlinked_tool(tmp_path: Path) -> None:
    tool, digest = _profile_tool(tmp_path)
    link = tmp_path / "synthetic-profile-link.py"
    link.symlink_to(tool.name)
    with pytest.raises(WriterRefusal) as caught:
        SubprocessProfileValidator(
            command_prefix=(sys.executable,),
            tool_path=link,
            tool_sha256=digest,
            tool_revision="synthetic-profile-revision-4",
            timeout_seconds=2,
        )
    assert caught.value.code is WriterErrorCode.PROFILE_VALIDATION_FAILED


@pytest.mark.parametrize(
    "code",
    [WriterErrorCode.LINT_FAILED, WriterErrorCode.PRIVACY_SCAN_FAILED],
)
def test_worktree_adapter_maps_failure_without_echo(
    tmp_path: Path,
    code: WriterErrorCode,
) -> None:
    tool = tmp_path / "synthetic_worktree_check.py"
    tool.write_text(
        "import os, pathlib, sys\n"
        "assert pathlib.Path(os.environ['SYNTHETIC_ROOT']) == pathlib.Path.cwd()\n"
        "print('SYNTHETIC_BODY_MUST_NOT_ESCAPE')\n"
        "raise SystemExit(9)\n",
        encoding="utf-8",
    )
    validator = SubprocessWorktreeValidator(
        (sys.executable, str(tool)),
        code=code,
        timeout_seconds=2,
        root_env_var="SYNTHETIC_ROOT",
    )
    with pytest.raises(WriterRefusal) as caught:
        validator.validate(tmp_path, "synthetic-note.md")
    assert caught.value.code is code
    assert "SYNTHETIC_BODY" not in str(caught.value)


@dataclass
class _OrderedValidator:
    name: str
    calls: list[str]

    def validate(self, worktree: Path, target_path: str) -> None:
        del worktree, target_path
        self.calls.append(self.name)


def test_validation_pipeline_order_is_profile_lint_privacy(tmp_path: Path) -> None:
    calls: list[str] = []
    pipeline = ValidationPipeline(
        profile=_OrderedValidator("profile", calls),
        lint=_OrderedValidator("lint", calls),
        privacy=_OrderedValidator("privacy", calls),
    )
    pipeline.validate(tmp_path, "synthetic-note.md")
    assert calls == ["profile", "lint", "privacy"]
