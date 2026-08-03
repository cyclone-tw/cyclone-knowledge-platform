from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from ckp.writer.errors import WriterErrorCode
from ckp.writer.models import RejectReceipt, SuccessReceipt, WriteRequest


def _request(**changes: object) -> WriteRequest:
    values: dict[str, object] = {
        "operation_id": "write.note-1",
        "idempotency_key": "k" * 16,
        "target_path": "Core/_inbox/example.md",
        "frontmatter": {"title": "Synthetic", "tags": ["test", 2, None]},
        "body": "body",
    }
    values.update(changes)
    return WriteRequest.model_validate(values)


def test_write_request_has_no_attribution_or_content_receipt_fields() -> None:
    request = _request()

    assert set(WriteRequest.model_fields) == {
        "operation_id",
        "idempotency_key",
        "target_path",
        "frontmatter",
        "body",
    }
    assert request.idempotency_key.get_secret_value() == "k" * 16
    assert "k" * 16 not in repr(request)
    assert "k" * 16 not in repr(request.model_dump())


@pytest.mark.parametrize(
    "field,value",
    [
        ("operation_id", "Uppercase"),
        ("operation_id", "a" * 65),
        ("idempotency_key", "short"),
        ("idempotency_key", "k" * 257),
        ("target_path", "/absolute.md"),
        ("target_path", "folder\\windows.md"),
        ("target_path", "folder/../escape.md"),
        ("target_path", "folder/note.txt"),
        ("body", "nul\0body"),
        ("body", "a" * (256 * 1024 + 1)),
    ],
)
def test_write_request_rejects_invalid_bounded_fields(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        _request(**{field: value})


@pytest.mark.parametrize(
    "frontmatter",
    [
        ["not", "a", "mapping"],
        {"nested": {"bad": {1, 2}}},
        {"non-finite": math.nan},
        {1: "non-string-key"},
    ],
)
def test_write_request_requires_json_compatible_frontmatter_mapping(
    frontmatter: object,
) -> None:
    with pytest.raises(ValidationError):
        _request(frontmatter=frontmatter)


def test_receipts_have_exact_fields_and_strict_success_evidence() -> None:
    success = SuccessReceipt(
        operation_id="write.note-1",
        target_path="Core/_inbox/example.md",
        base_commit="a" * 40,
        result_commit="b" * 40,
        actor="codex",
        timestamp="2026-08-03T12:34:56.123456Z",
        content_hash="sha256:" + "c" * 64,
    )
    reject = RejectReceipt(request_id="request-1", code="writer/v1/request-invalid")

    assert set(SuccessReceipt.model_fields) == {
        "operation_id",
        "target_path",
        "base_commit",
        "result_commit",
        "actor",
        "timestamp",
        "content_hash",
    }
    assert set(RejectReceipt.model_fields) == {"request_id", "code"}
    assert success.model_dump()["content_hash"] == "sha256:" + "c" * 64
    with pytest.raises(ValidationError):
        SuccessReceipt.model_validate({**success.model_dump(), "base_commit": "a" * 39})
    with pytest.raises(ValidationError):
        SuccessReceipt.model_validate(
            {**success.model_dump(), "timestamp": "2026-08-03"}
        )
    with pytest.raises(ValidationError):
        SuccessReceipt.model_validate(
            {**success.model_dump(), "content_hash": "c" * 64}
        )
    with pytest.raises(ValidationError):
        RejectReceipt.model_validate({**reject.model_dump(), "detail": "no echo"})


def test_public_json_schemas_publish_frozen_bounds_and_error_values() -> None:
    request_properties = WriteRequest.model_json_schema()["properties"]
    assert request_properties["operation_id"] == {
        "maxLength": 64,
        "minLength": 1,
        "pattern": "^[a-z0-9][a-z0-9._-]*$",
        "title": "Operation Id",
        "type": "string",
    }
    assert request_properties["idempotency_key"] == {
        "format": "password",
        "maxLength": 256,
        "minLength": 16,
        "title": "Idempotency Key",
        "type": "string",
        "writeOnly": True,
    }
    reject_code = RejectReceipt.model_json_schema()["properties"]["code"]
    assert set(reject_code["enum"]) == {code.value for code in WriterErrorCode}
