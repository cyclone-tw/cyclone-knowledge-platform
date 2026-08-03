"""C7 scope and composition tripwires."""

from __future__ import annotations

import inspect
from pathlib import Path

from fastapi.testclient import TestClient

from ckp.app import create_app
from ckp.config import load_config
from ckp.writer.composition import build_c7_synthetic_writer
from ckp.writer.git import SYNTHETIC_MARKER
from ckp.writer.service import WriterService, _binding_hash
from conftest import REPO_ROOT


def test_default_app_keeps_writer_and_direct_formal_surface_closed() -> None:
    bundle = REPO_ROOT / "fixtures/synthetic-bundle"
    app = create_app(load_config(env={"CKP_BUNDLE_ROOT": str(bundle)}))
    client = TestClient(app)
    paths = set(client.get("/openapi.json").json()["paths"])

    assert "writer" not in vars(app.state)
    assert not any(
        token in path.lower()
        for path in paths
        for token in ("write", "outbox", "publication", "promote")
    )


def test_synthetic_composition_has_no_permissive_dependency_defaults() -> None:
    expected = {
        "repository",
        "identity_registry",
        "privacy_gate",
        "payload_scanner",
        "profile_validator",
        "lint_validator",
        "privacy_validator",
        "clock",
        "lock_timeout_seconds",
    }
    signature = inspect.signature(build_c7_synthetic_writer)
    assert set(signature.parameters) == expected
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    service_signature = inspect.signature(WriterService)
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in service_signature.parameters.values()
        if parameter.name != "self"
    )


def test_writer_package_does_not_import_c6_auth_or_out_of_scope_sinks() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "src/ckp/writer").glob("*.py"))
    ).lower()
    for forbidden in (
        "from ckp.auth",
        "import ckp.auth",
        "qdrant",
        "discord",
        "notion",
        "hermes",
        "openab",
        "durable outbox",
    ):
        assert forbidden not in source


def test_c7_synthetic_repo_marker_and_notes_are_runtime_only() -> None:
    assert not (REPO_ROOT / SYNTHETIC_MARKER).exists()
    source_paths = {
        path.relative_to(REPO_ROOT)
        for path in REPO_ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and ".venv" not in path.parts
    }
    assert Path(SYNTHETIC_MARKER) not in source_paths
    assert not any(
        path.suffix == ".md" and "c7-synthetic" in path.as_posix()
        for path in source_paths
    )


def test_c7_expected_writer_modules_are_present() -> None:
    module_names = {path.name for path in (REPO_ROOT / "src/ckp/writer").glob("*.py")}
    assert module_names == {
        "__init__.py",
        "composition.py",
        "errors.py",
        "git.py",
        "identity.py",
        "models.py",
        "service.py",
        "target.py",
        "validation.py",
    }


def test_operation_binding_is_versioned_and_covers_every_frozen_dimension() -> None:
    source = inspect.getsource(_binding_hash)
    for needle in (
        'b"ckp-writer-binding-v1"',
        'identity.operation_id.encode("utf-8")',
        'identity.idempotency_hash.encode("ascii")',
        'identity.actor.encode("utf-8")',
        'identity.task_hash.encode("ascii")',
        'identity.target_path.encode("utf-8")',
        'identity.request_hash.encode("ascii")',
        'content_hash.encode("ascii")',
        'base_commit.encode("ascii")',
    ):
        assert needle in source


def test_git_backend_uses_atomic_refs_and_has_no_shared_cleanup_commands() -> None:
    source = (REPO_ROOT / "src/ckp/writer/git.py").read_text(encoding="utf-8")
    for needle in (
        '"start\\n"',
        '"prepare\\n"',
        '"commit\\n"',
        "refs/ckp-writer/operations/",
        "refs/ckp-writer/idempotency/",
        '"worktree",\n                "add",\n                "--detach"',
    ):
        assert needle in source
    for forbidden in ('"stash"', '"reset"', '"clean"', '"prune"'):
        assert forbidden not in source
