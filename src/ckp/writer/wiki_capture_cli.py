"""Explicit local CLI for the Cyclone-Wiki Core inbox bridge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pydantic import ValidationError

from ckp.writer.errors import WriterRefusal
from ckp.writer.identity import (
    ProvenanceMode,
    WriterActor,
    WriterActorRegistry,
)
from ckp.writer.wiki_capture import (
    CoreInboxCaptureAdapter,
    CoreInboxCaptureRequest,
    WikiCaptureErrorCode,
    WikiCaptureRefusal,
)

_CREDENTIAL_ENV = "CKP_WRITER_ACTOR_CREDENTIAL"
_CREDENTIAL_HASH_ENV = "CKP_WRITER_ACTOR_CREDENTIAL_SHA256"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ckp-wiki-capture",
        description="Create one authenticated Codex Source note in Core/_inbox.",
    )
    parser.add_argument("--wiki-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--event-summary")
    parser.add_argument("--body-file", required=True, type=Path)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _reject(code: str) -> None:
    print(json.dumps({"status": "rejected", "code": code}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    credential = os.environ.get(_CREDENTIAL_ENV)
    credential_hash = os.environ.get(_CREDENTIAL_HASH_ENV)
    if not credential or not credential_hash:
        _reject(WikiCaptureErrorCode.IDENTITY_DENIED)
        return 2

    try:
        body = arguments.body_file.read_text(encoding="utf-8")
        request = CoreInboxCaptureRequest(
            slug=arguments.slug,
            title=arguments.title,
            event_summary=arguments.event_summary or arguments.title,
            body=body,
        )
        registry = WriterActorRegistry(
            (
                WriterActor(
                    actor_id="codex",
                    credential_sha256=credential_hash,
                ),
            )
        )
        context = registry.resolve(
            credential,
            request_id=arguments.request_id,
            task_id=arguments.task_id,
            provenance_mode=ProvenanceMode.GENERATED,
            model_id=arguments.model_id,
        )
        receipt = CoreInboxCaptureAdapter(
            wiki_root=arguments.wiki_root,
            state_root=arguments.state_root,
            identity_registry=registry,
        ).capture(request, context, dry_run=arguments.dry_run)
    except (OSError, UnicodeError, ValidationError):
        _reject(WikiCaptureErrorCode.REQUEST_INVALID)
        return 2
    except WriterRefusal:
        _reject(WikiCaptureErrorCode.IDENTITY_DENIED)
        return 2
    except WikiCaptureRefusal as refusal:
        _reject(refusal.code)
        return 1

    print(json.dumps(receipt.model_dump(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
