"""Frozen Writer v1 input and receipt schemas.

These models deliberately validate only transport-level shape.  Identity,
actor/model semantics, privacy admissibility, and target authorization belong
to the Writer service and policy layers, not to this public request schema.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from ckp.writer.errors import WriterErrorCode

_OPERATION_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_REQUEST_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$")
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_FULL_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)


def _is_json_value(value: object) -> bool:
    """Return whether ``value`` can be represented by JSON without coercion."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False


class WriteRequest(BaseModel):
    """A bounded, non-attributed write request.

    The idempotency key is intentionally secret-redacted in repr and dumps.
    Callers that need the value to perform deduplication must opt in with
    ``get_secret_value()`` at the service boundary.
    """

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=_OPERATION_ID.pattern,
    )
    idempotency_key: SecretStr = Field(min_length=16, max_length=256)
    target_path: str = Field(
        min_length=1,
        max_length=240,
        description="Bundle-relative POSIX Markdown path; policy authorizes it later.",
    )
    frontmatter: dict[str, Any]
    body: str = Field(description="UTF-8 text, no NUL, at most 256 KiB encoded.")

    @field_validator("operation_id")
    @classmethod
    def require_operation_id_shape(cls, value: str) -> str:
        if _OPERATION_ID.fullmatch(value) is None:
            raise ValueError("operation_id has invalid shape")
        return value

    @field_validator("idempotency_key", mode="before")
    @classmethod
    def require_idempotency_key_length(cls, value: object) -> object:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str) and not 16 <= len(value) <= 256:
            raise ValueError("idempotency_key has invalid length")
        return value

    @field_validator("target_path")
    @classmethod
    def require_basic_relative_markdown_path(cls, value: str) -> str:
        # Authorization and zone policy are deliberately outside this model.
        if (
            not value.endswith(".md")
            or value.startswith("/")
            or "\\" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError("target_path is not a relative POSIX Markdown path")
        return value

    @field_validator("frontmatter", mode="before")
    @classmethod
    def require_json_frontmatter_mapping(cls, value: object) -> object:
        if not isinstance(value, Mapping) or not _is_json_value(value):
            raise ValueError("frontmatter must be a JSON-compatible mapping")
        return dict(value)

    @field_validator("body")
    @classmethod
    def require_bounded_utf8_body(cls, value: str) -> str:
        if "\0" in value:
            raise ValueError("body contains a null byte")
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("body is not UTF-8 encodable") from exc
        if len(encoded) > 256 * 1024:
            raise ValueError("body exceeds 256 KiB")
        return value


class SuccessReceipt(BaseModel):
    """The complete, non-content-bearing result of an accepted write."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=_OPERATION_ID.pattern,
    )
    target_path: str = Field(min_length=1, max_length=240)
    base_commit: str = Field(min_length=1)
    result_commit: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)

    @field_validator("operation_id")
    @classmethod
    def require_receipt_operation_id_shape(cls, value: str) -> str:
        if _OPERATION_ID.fullmatch(value) is None:
            raise ValueError("operation_id has invalid shape")
        return value

    @field_validator("target_path")
    @classmethod
    def require_receipt_basic_relative_markdown_path(cls, value: str) -> str:
        if (
            not value.endswith(".md")
            or value.startswith("/")
            or "\\" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError("target_path is not a relative POSIX Markdown path")
        return value

    @field_validator("actor")
    @classmethod
    def require_nonblank_actor(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("actor must not be blank")
        return value

    @field_validator("base_commit", "result_commit")
    @classmethod
    def require_full_git_sha(cls, value: str) -> str:
        if _FULL_GIT_SHA.fullmatch(value) is None:
            raise ValueError("commit must be a full lowercase SHA-1")
        return value

    @field_validator("timestamp")
    @classmethod
    def require_full_timezone_aware_timestamp(cls, value: str) -> str:
        if _FULL_TIMESTAMP.fullmatch(value) is None:
            raise ValueError("timestamp must be a full timezone-aware RFC 3339 value")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be a valid RFC 3339 value") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value

    @field_validator("content_hash")
    @classmethod
    def require_sha256_content_hash(cls, value: str) -> str:
        if _SHA256_CONTENT_HASH.fullmatch(value) is None:
            raise ValueError("content_hash must be a lowercase sha256 digest")
        return value


class RejectReceipt(BaseModel):
    """A stable refusal without request payload or diagnostic detail."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    code: str = Field(
        json_schema_extra={"enum": [code.value for code in WriterErrorCode]}
    )

    @field_validator("request_id")
    @classmethod
    def require_server_request_id_shape(cls, value: str) -> str:
        if _REQUEST_ID.fullmatch(value) is None:
            raise ValueError("request_id has invalid shape")
        return value

    @field_validator("code")
    @classmethod
    def require_stable_writer_code(cls, value: str) -> str:
        if value not in set(WriterErrorCode):
            raise ValueError("code is not a Writer v1 refusal code")
        return value


__all__ = ["RejectReceipt", "SuccessReceipt", "WriteRequest"]
