"""Local-only bridge from an authenticated C7 actor to Cyclone-Wiki Core.

The Cyclone-Wiki capture wrapper remains authoritative for routing, Git
transactions, validation, push, and QMD refresh.  This adapter deliberately
fixes every capability-bearing wrapper argument and exposes no HTTP surface.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from uuid import RFC_4122, UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from ckp.writer.identity import AuthenticatedWriterContext, WriterActorRegistry

_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
_CAPTURE_PATH = re.compile(
    r"^Core/_inbox/agent-captures/\d{4}-\d{2}-\d{2}-"
    r"(?P<slug>[a-z0-9][a-z0-9-]{0,79})\.md$"
)
_MAX_BODY_BYTES = 1024 * 1024
_CALLER_CREDENTIAL_ENV = frozenset(
    {
        "CKP_WRITER_ACTOR_CREDENTIAL",
        "CKP_WRITER_ACTOR_CREDENTIAL_SHA256",
    }
)
_FIXED_WRAPPER_ARGUMENTS = (
    "--agent",
    "Codex",
    "--kind",
    "agent-discussion",
    "--surface",
    "core",
    "--content-category",
    "development",
    "--git-mode",
    "auto",
)
_RECEIPT_BASE_KEYS = frozenset(
    {
        "wiki_capture_path",
        "wiki_capture_status",
        "wiki_capture_surface",
        "wiki_capture_git_mode",
    }
)
_RECEIPT_V3_KEYS = frozenset(
    {
        "wiki_capture_contract",
        "wiki_capture_id",
        "wiki_capture_concept_id",
        "wiki_capture_request_id",
    }
)
_RECEIPT_V3_COMPLETION_KEYS = frozenset(
    {
        "wiki_capture_path",
        "wiki_capture_status",
        "wiki_capture_surface",
        "wiki_capture_git_mode",
        "wiki_capture_contract",
        "wiki_capture_id",
        "wiki_capture_concept_id",
    }
)
_RECEIPT_V3_PLANNED_KEYS = frozenset(
    {
        "wiki_capture_path",
        "wiki_capture_status",
        "wiki_capture_surface",
        "wiki_capture_git_mode",
        "wiki_capture_contract",
        "wiki_capture_concept_id",
    }
)
_RECEIPT_KEYS = frozenset(_RECEIPT_BASE_KEYS | _RECEIPT_V3_KEYS)
_RECEIPT_GIT_MODES = frozenset({"direct", "isolated"})


class WikiCaptureErrorCode(StrEnum):
    """Stable, content-free failures owned by the local bridge."""

    REQUEST_INVALID = "wiki-capture/v1/request-invalid"
    IDENTITY_DENIED = "wiki-capture/v1/identity-denied"
    WRAPPER_UNAVAILABLE = "wiki-capture/v1/wrapper-unavailable"
    WRITE_FROZEN = "wiki-capture/v1/write-frozen"
    CAPTURE_FAILED = "wiki-capture/v1/capture-failed"
    RECEIPT_INVALID = "wiki-capture/v1/receipt-invalid"


class WikiCaptureRefusal(ValueError):
    """A safe refusal that never relays wrapper output or request content."""

    def __init__(self, code: WikiCaptureErrorCode) -> None:
        self.code = code
        super().__init__(code)


class CoreInboxCaptureRequest(BaseModel):
    """The intentionally narrow request accepted by the local bridge."""

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    event_summary: str = Field(min_length=1, max_length=300)
    body: SecretStr

    @field_validator("slug")
    @classmethod
    def require_kebab_slug(cls, value: str) -> str:
        if _SLUG.fullmatch(value) is None:
            raise ValueError("slug must be lowercase kebab-case")
        return value

    @field_validator("title", "event_summary")
    @classmethod
    def require_single_line_text(cls, value: str) -> str:
        if not value.strip() or any(character in value for character in "\r\n\0"):
            raise ValueError("text must be a nonblank single line")
        return value

    @field_validator("body", mode="before")
    @classmethod
    def require_bounded_text_body(cls, value: object) -> object:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if not isinstance(value, str) or "\0" in value:
            raise ValueError("body must be text without null bytes")
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("body must be UTF-8 encodable") from exc
        if len(encoded) > _MAX_BODY_BYTES:
            raise ValueError("body exceeds 1 MiB")
        return value


class CoreInboxCaptureReceipt(BaseModel):
    """Sanitized evidence returned by the bridge."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: str
    path: str
    surface: str
    git_mode: str
    wiki_capture_contract: int | None = None
    wiki_capture_id: str | None = None
    wiki_capture_concept_id: str | None = None

    def model_dump(self, *args, **kwargs):  # type: ignore[override]
        values = super().model_dump(*args, **kwargs)
        for key in (
            "wiki_capture_contract",
            "wiki_capture_id",
            "wiki_capture_concept_id",
        ):
            if values.get(key) is None:
                values.pop(key, None)
        return values


@dataclass(frozen=True, slots=True)
class CoreInboxCaptureAdapter:
    """Invoke one fixed Cyclone-Wiki Core capture capability."""

    wiki_root: Path
    state_root: Path
    identity_registry: WriterActorRegistry = field(repr=False, compare=False)
    timeout_seconds: float = 300.0
    _wrapper: Path = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise WikiCaptureRefusal(WikiCaptureErrorCode.WRAPPER_UNAVAILABLE)
        wiki_root = _physical_directory(self.wiki_root)
        state_root = _physical_directory(self.state_root)
        if _contains(wiki_root, state_root) or _contains(state_root, wiki_root):
            raise WikiCaptureRefusal(WikiCaptureErrorCode.WRAPPER_UNAVAILABLE)
        wrapper = _regular_executable(wiki_root / "scripts/write_inbox_capture.sh")
        object.__setattr__(self, "wiki_root", wiki_root)
        object.__setattr__(self, "state_root", state_root)
        object.__setattr__(self, "_wrapper", wrapper)

    def capture(
        self,
        request: CoreInboxCaptureRequest,
        context: AuthenticatedWriterContext,
        *,
        dry_run: bool = False,
    ) -> CoreInboxCaptureReceipt:
        """Authenticate first, then delegate one fixed create-only capture."""
        self.identity_registry.validate(context)
        if context.actor_id != "codex":
            raise WikiCaptureRefusal(WikiCaptureErrorCode.IDENTITY_DENIED)

        body_path: Path | None = None
        try:
            body_path = self._write_private_body(request.body.get_secret_value())
            command = (
                str(self._wrapper),
                *_FIXED_WRAPPER_ARGUMENTS,
                "--slug",
                request.slug,
                "--title",
                request.title,
                "--event-summary",
                request.event_summary,
                "--body-file",
                str(body_path),
                *(("--dry-run",) if dry_run else ()),
            )
            environment = os.environ.copy()
            for name in _CALLER_CREDENTIAL_ENV:
                environment.pop(name, None)
            environment["CYCLONE_WIKI_ROOT"] = str(self.wiki_root)
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.wiki_root,
                    env=environment,
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise WikiCaptureRefusal(
                    WikiCaptureErrorCode.WRAPPER_UNAVAILABLE
                ) from exc
            if completed.returncode != 0:
                code = (
                    WikiCaptureErrorCode.WRITE_FROZEN
                    if _has_freeze_receipt(completed.stdout)
                    else WikiCaptureErrorCode.CAPTURE_FAILED
                )
                raise WikiCaptureRefusal(code)
            return _parse_receipt(
                completed.stdout,
                request_id=context.request_id,
                slug=request.slug,
                dry_run=dry_run,
            )
        finally:
            if body_path is not None:
                try:
                    body_path.unlink(missing_ok=True)
                except OSError as exc:
                    raise WikiCaptureRefusal(
                        WikiCaptureErrorCode.CAPTURE_FAILED
                    ) from exc

    def _write_private_body(self, body: str) -> Path:
        descriptor, raw_path = tempfile.mkstemp(
            prefix="ckp-wiki-capture-",
            suffix=".md",
            dir=self.state_root,
        )
        path = Path(raw_path)
        try:
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o600:
                raise OSError("unsafe temporary body")
            payload = body.encode("utf-8")
            view = memoryview(payload)
            while view:
                view = view[os.write(descriptor, view) :]
            os.fsync(descriptor)
            return path
        except OSError as exc:
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise WikiCaptureRefusal(
                    WikiCaptureErrorCode.CAPTURE_FAILED
                ) from cleanup_error
            raise WikiCaptureRefusal(WikiCaptureErrorCode.CAPTURE_FAILED) from exc
        finally:
            os.close(descriptor)


def _parse_receipt(
    output: str,
    *,
    request_id: str,
    slug: str,
    dry_run: bool,
) -> CoreInboxCaptureReceipt:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if not line.startswith("wiki_capture_"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key not in _RECEIPT_KEYS or key in values:
            raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)
        values[key] = value

    if "wiki_capture_request_id" in values:
        raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)

    is_v3_completion = set(values) == _RECEIPT_V3_COMPLETION_KEYS
    is_v3_planned = set(values) == _RECEIPT_V3_PLANNED_KEYS
    if set(values) == _RECEIPT_BASE_KEYS:
        is_v3 = False
    elif is_v3_completion or is_v3_planned:
        is_v3 = True
    else:
        raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)
    if is_v3:
        if values["wiki_capture_contract"] != "3":
            raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)
        if dry_run and is_v3_completion:
            raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)

    status = values["wiki_capture_status"]
    expected_status = {"planned"} if dry_run else {"created"}
    if status not in expected_status or (is_v3_planned and status != "planned"):
        raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)
    if not is_v3 and dry_run and status != "planned":
        raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)

    path = values["wiki_capture_path"]
    concept = values["wiki_capture_concept_id"] if is_v3 else None

    if values["wiki_capture_surface"] != "core":
        raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)
    if values["wiki_capture_git_mode"] not in _RECEIPT_GIT_MODES:
        raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)

    path_match = _CAPTURE_PATH.fullmatch(path)
    if path_match is None or path_match.group("slug") != slug:
        raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)

    if not is_v3 and concept is not None:
        raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)

    if concept is not None:
        if concept.endswith(".md"):
            raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)
        if path[:-3] != concept:
            raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)

    if not is_v3:
        return CoreInboxCaptureReceipt(
            request_id=request_id,
            status=status,
            path=path,
            surface="core",
            git_mode=values["wiki_capture_git_mode"],
        )

    if is_v3_planned and "wiki_capture_id" in values:
        raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)

    if not is_v3_planned:
        if "wiki_capture_id" not in values:
            raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)
        wiki_capture_id = values["wiki_capture_id"]
        try:
            parsed = UUID(wiki_capture_id)
        except ValueError as exc:
            raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID) from exc
        if str(parsed) != wiki_capture_id:
            raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)
        if parsed.version != 7 or parsed.variant != RFC_4122:
            raise WikiCaptureRefusal(WikiCaptureErrorCode.RECEIPT_INVALID)

        return CoreInboxCaptureReceipt(
            request_id=request_id,
            status=status,
            path=path,
            surface="core",
            git_mode=values["wiki_capture_git_mode"],
            wiki_capture_contract=3,
            wiki_capture_id=wiki_capture_id,
            wiki_capture_concept_id=path[:-3],
        )

    return CoreInboxCaptureReceipt(
        request_id=request_id,
        status=status,
        path=path,
        surface="core",
        git_mode=values["wiki_capture_git_mode"],
        wiki_capture_contract=3,
        wiki_capture_id=None,
        wiki_capture_concept_id=path[:-3],
    )


def _has_freeze_receipt(output: str) -> bool:
    lines = set(output.splitlines())
    return {
        "wiki_capture_status=rejected",
        "wiki_capture_error=wiki-write-frozen",
    }.issubset(lines)


def _physical_directory(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise WikiCaptureRefusal(WikiCaptureErrorCode.WRAPPER_UNAVAILABLE) from exc
    if not resolved.is_dir():
        raise WikiCaptureRefusal(WikiCaptureErrorCode.WRAPPER_UNAVAILABLE)
    return resolved


def _regular_executable(path: Path) -> Path:
    try:
        absolute = path.absolute()
        resolved = path.resolve(strict=True)
        value = resolved.stat()
    except OSError as exc:
        raise WikiCaptureRefusal(WikiCaptureErrorCode.WRAPPER_UNAVAILABLE) from exc
    if (
        absolute != resolved
        or not stat.S_ISREG(value.st_mode)
        or not os.access(resolved, os.X_OK)
    ):
        raise WikiCaptureRefusal(WikiCaptureErrorCode.WRAPPER_UNAVAILABLE)
    return resolved


def _contains(parent: Path, child: Path) -> bool:
    return parent == child or parent in child.parents


__all__ = [
    "CoreInboxCaptureAdapter",
    "CoreInboxCaptureReceipt",
    "CoreInboxCaptureRequest",
    "WikiCaptureErrorCode",
    "WikiCaptureRefusal",
]
