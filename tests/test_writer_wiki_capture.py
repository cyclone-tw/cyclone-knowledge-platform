from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from ckp.writer.errors import WriterErrorCode, WriterRefusal
from ckp.writer.identity import ProvenanceMode, WriterActor, WriterActorRegistry
from ckp.writer.wiki_capture import (
    CoreInboxCaptureAdapter,
    CoreInboxCaptureRequest,
    WikiCaptureErrorCode,
    WikiCaptureRefusal,
)
from ckp.writer.wiki_capture_cli import main as capture_cli

_CREDENTIAL = "local-c7-wiki-credential"
_BODY = "# Synthetic connection smoke\n\nNo private data.\n"

_FAKE_WRAPPER = r"""#!/usr/bin/env python3
import json
import os
import stat
import sys
from pathlib import Path

arguments = sys.argv[1:]
values = {}
flags = set()
index = 0
while index < len(arguments):
    argument = arguments[index]
    if argument == "--dry-run":
        flags.add(argument)
        index += 1
        continue
    values[argument] = arguments[index + 1]
    index += 2

body_path = Path(values["--body-file"])
root = Path(os.environ["CYCLONE_WIKI_ROOT"])
(root / "wrapper-invocation.json").write_text(
    json.dumps(
        {
            "arguments": arguments,
            "body": body_path.read_text(encoding="utf-8"),
            "body_mode": stat.S_IMODE(body_path.stat().st_mode),
            "body_path": str(body_path),
            "credential_visible": "CKP_WRITER_ACTOR_CREDENTIAL" in os.environ,
            "credential_hash_visible": (
                "CKP_WRITER_ACTOR_CREDENTIAL_SHA256" in os.environ
            ),
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)

mode = os.environ.get("FAKE_WRAPPER_MODE", "success")
if mode == "freeze":
    print("wiki_capture_status=rejected")
    print("wiki_capture_error=wiki-write-frozen")
    raise SystemExit(1)
if mode == "failure":
    print("sensitive diagnostic deliberately discarded")
    raise SystemExit(1)

slug = values["--slug"]
path = f"Core/_inbox/agent-captures/2026-08-04-{slug}.md"
surface = "core"
status = "planned" if "--dry-run" in flags else "created"
if mode == "existing":
    status = "existing"
if mode == "private":
    surface = "private"
if mode == "formal":
    path = f"Core/{slug}.md"

print("write_inbox_capture: synthetic fixture log")
print(f"wiki_capture_path={path}")
print(f"wiki_capture_status={status}")
if mode == "duplicate":
    print(f"wiki_capture_status={status}")
print(f"wiki_capture_surface={surface}")
print("wiki_capture_git_mode=isolated")
"""


def _fixture_roots(tmp_path: Path) -> tuple[Path, Path]:
    wiki_root = tmp_path / "wiki"
    state_root = tmp_path / "state"
    (wiki_root / "scripts").mkdir(parents=True)
    state_root.mkdir()
    wrapper = wiki_root / "scripts/write_inbox_capture.sh"
    wrapper.write_text(_FAKE_WRAPPER, encoding="utf-8")
    wrapper.chmod(0o755)
    return wiki_root, state_root


def _registry(*actor_ids: str) -> WriterActorRegistry:
    return WriterActorRegistry(
        tuple(WriterActor.synthetic(actor_id, _CREDENTIAL) for actor_id in actor_ids)
    )


def _context(registry: WriterActorRegistry, model_id: str = "gpt-5.4"):
    return registry.resolve(
        _CREDENTIAL,
        request_id="request-wiki-1",
        task_id="task-wiki-1",
        provenance_mode=ProvenanceMode.GENERATED,
        model_id=model_id,
    )


def _request(slug: str = "c7-wiki-smoke") -> CoreInboxCaptureRequest:
    return CoreInboxCaptureRequest(
        slug=slug,
        title="C7 Wiki smoke",
        event_summary="Connect C7 to the Core inbox",
        body=_BODY,
    )


def test_identity_is_checked_before_wrapper_or_temporary_file(tmp_path: Path) -> None:
    wiki_root, state_root = _fixture_roots(tmp_path)
    registry = _registry("codex")
    foreign_registry = _registry("codex")
    adapter = CoreInboxCaptureAdapter(wiki_root, state_root, registry)

    with pytest.raises(WriterRefusal) as caught:
        adapter.capture(_request(), _context(foreign_registry))

    assert caught.value.code is WriterErrorCode.ACTOR_DENIED
    assert not (wiki_root / "wrapper-invocation.json").exists()
    assert list(state_root.iterdir()) == []


def test_only_codex_actor_can_use_the_fixed_capture_route(tmp_path: Path) -> None:
    wiki_root, state_root = _fixture_roots(tmp_path)
    registry = _registry("claude")
    adapter = CoreInboxCaptureAdapter(wiki_root, state_root, registry)

    with pytest.raises(WikiCaptureRefusal) as caught:
        adapter.capture(_request(), _context(registry))

    assert caught.value.code is WikiCaptureErrorCode.IDENTITY_DENIED
    assert not (wiki_root / "wrapper-invocation.json").exists()
    assert list(state_root.iterdir()) == []


def test_adapter_fixes_route_and_cleans_private_body_file(tmp_path: Path) -> None:
    wiki_root, state_root = _fixture_roots(tmp_path)
    registry = _registry("codex")
    receipt = CoreInboxCaptureAdapter(wiki_root, state_root, registry).capture(
        _request(),
        _context(registry),
    )

    invocation = json.loads(
        (wiki_root / "wrapper-invocation.json").read_text(encoding="utf-8")
    )
    arguments = invocation["arguments"]
    assert arguments[:10] == [
        "--agent",
        "Codex",
        "--kind",
        "agent-discussion",
        "--surface",
        "core",
        "--content-category",
        "engineering",
        "--git-mode",
        "auto",
    ]
    assert "Private" not in arguments
    assert "--append" not in arguments
    assert invocation["body"] == _BODY
    assert invocation["body_mode"] == 0o600
    assert invocation["credential_visible"] is False
    assert invocation["credential_hash_visible"] is False
    assert not Path(invocation["body_path"]).exists()
    assert list(state_root.iterdir()) == []
    assert receipt.model_dump() == {
        "request_id": "request-wiki-1",
        "status": "created",
        "path": "Core/_inbox/agent-captures/2026-08-04-c7-wiki-smoke.md",
        "surface": "core",
        "git_mode": "isolated",
    }
    assert _BODY not in repr(receipt)


def test_dry_run_requires_a_planned_receipt(tmp_path: Path) -> None:
    wiki_root, state_root = _fixture_roots(tmp_path)
    registry = _registry("codex")

    receipt = CoreInboxCaptureAdapter(wiki_root, state_root, registry).capture(
        _request("c7-wiki-dry-run"),
        _context(registry),
        dry_run=True,
    )

    assert receipt.status == "planned"
    invocation = json.loads(
        (wiki_root / "wrapper-invocation.json").read_text(encoding="utf-8")
    )
    assert invocation["arguments"].count("--dry-run") == 1
    assert not Path(invocation["body_path"]).exists()


@pytest.mark.parametrize("mode", ["private", "formal", "duplicate", "existing"])
def test_mismatched_or_duplicate_receipts_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    wiki_root, state_root = _fixture_roots(tmp_path)
    registry = _registry("codex")
    monkeypatch.setenv("FAKE_WRAPPER_MODE", mode)

    with pytest.raises(WikiCaptureRefusal) as caught:
        CoreInboxCaptureAdapter(wiki_root, state_root, registry).capture(
            _request(),
            _context(registry),
        )

    assert caught.value.code is WikiCaptureErrorCode.RECEIPT_INVALID
    invocation = json.loads(
        (wiki_root / "wrapper-invocation.json").read_text(encoding="utf-8")
    )
    assert not Path(invocation["body_path"]).exists()
    assert list(state_root.iterdir()) == []


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("freeze", WikiCaptureErrorCode.WRITE_FROZEN),
        ("failure", WikiCaptureErrorCode.CAPTURE_FAILED),
    ],
)
def test_wrapper_failures_never_return_false_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected: WikiCaptureErrorCode,
) -> None:
    wiki_root, state_root = _fixture_roots(tmp_path)
    registry = _registry("codex")
    monkeypatch.setenv("FAKE_WRAPPER_MODE", mode)

    with pytest.raises(WikiCaptureRefusal) as caught:
        CoreInboxCaptureAdapter(wiki_root, state_root, registry).capture(
            _request(),
            _context(registry),
        )

    assert caught.value.code is expected
    invocation = json.loads(
        (wiki_root / "wrapper-invocation.json").read_text(encoding="utf-8")
    )
    assert not Path(invocation["body_path"]).exists()
    assert list(state_root.iterdir()) == []


def test_cli_uses_separate_credential_hash_and_emits_sanitized_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wiki_root, state_root = _fixture_roots(tmp_path)
    body_file = tmp_path / "body.md"
    body_file.write_text(_BODY, encoding="utf-8")
    digest = hashlib.sha256(_CREDENTIAL.encode()).hexdigest()
    monkeypatch.setenv("CKP_WRITER_ACTOR_CREDENTIAL", _CREDENTIAL)
    monkeypatch.setenv(
        "CKP_WRITER_ACTOR_CREDENTIAL_SHA256",
        f"sha256:{digest}",
    )

    result = capture_cli(
        [
            "--wiki-root",
            str(wiki_root),
            "--state-root",
            str(state_root),
            "--slug",
            "cli-smoke",
            "--title",
            "CLI smoke",
            "--body-file",
            str(body_file),
            "--request-id",
            "request-cli-1",
            "--task-id",
            "task-cli-1",
            "--model-id",
            "gpt-5.4",
            "--dry-run",
        ]
    )

    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert result == 0
    assert payload == {
        "git_mode": "isolated",
        "path": "Core/_inbox/agent-captures/2026-08-04-cli-smoke.md",
        "request_id": "request-cli-1",
        "status": "planned",
        "surface": "core",
    }
    assert output.err == ""
    assert _BODY not in output.out
    assert _CREDENTIAL not in output.out


def test_cli_wrong_credential_rejects_before_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wiki_root, state_root = _fixture_roots(tmp_path)
    body_file = tmp_path / "body.md"
    body_file.write_text(_BODY, encoding="utf-8")
    digest = hashlib.sha256(b"different-credential").hexdigest()
    monkeypatch.setenv("CKP_WRITER_ACTOR_CREDENTIAL", _CREDENTIAL)
    monkeypatch.setenv(
        "CKP_WRITER_ACTOR_CREDENTIAL_SHA256",
        f"sha256:{digest}",
    )

    result = capture_cli(
        [
            "--wiki-root",
            str(wiki_root),
            "--state-root",
            str(state_root),
            "--slug",
            "cli-reject",
            "--title",
            "CLI reject",
            "--body-file",
            str(body_file),
            "--request-id",
            "request-cli-2",
            "--task-id",
            "task-cli-2",
            "--model-id",
            "gpt-5.4",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 2
    assert payload == {
        "status": "rejected",
        "code": WikiCaptureErrorCode.IDENTITY_DENIED,
    }
    assert not (wiki_root / "wrapper-invocation.json").exists()
    assert list(state_root.iterdir()) == []


def test_wrapper_path_must_be_a_physical_executable(
    tmp_path: Path,
) -> None:
    wiki_root, state_root = _fixture_roots(tmp_path)
    wrapper = wiki_root / "scripts/write_inbox_capture.sh"
    wrapper.chmod(stat.S_IRUSR | stat.S_IWUSR)

    with pytest.raises(WikiCaptureRefusal) as caught:
        CoreInboxCaptureAdapter(wiki_root, state_root, _registry("codex"))

    assert caught.value.code is WikiCaptureErrorCode.WRAPPER_UNAVAILABLE
