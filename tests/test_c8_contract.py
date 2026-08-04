"""C8 scope and composition tripwires."""

from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from ckp.app import create_app
from ckp.config import load_config
from ckp.outbox.composition import build_c8_synthetic_outbox
from ckp.outbox.models import compute_binding_hash
from ckp.outbox.service import OutboxService
from conftest import REPO_ROOT
from test_privacy_contract import _imports_privacy_package


def test_default_app_keeps_outbox_and_write_surface_closed() -> None:
    bundle = REPO_ROOT / "fixtures/synthetic-bundle"
    app = create_app(load_config(env={"CKP_BUNDLE_ROOT": str(bundle)}))
    client = TestClient(app)
    paths = set(client.get("/openapi.json").json()["paths"])

    assert "outbox" not in vars(app.state)
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
        "key_provider",
        "key_id",
        "credential_resolver",
        "store_root",
        "clock",
        "retention_seconds",
        "lease_seconds",
        "max_transient_attempts",
        "lock_timeout_seconds",
    }
    signature = inspect.signature(build_c8_synthetic_outbox)
    assert set(signature.parameters) == expected
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    service_signature = inspect.signature(OutboxService)
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in service_signature.parameters.values()
        if parameter.name != "self"
    )


def test_outbox_package_wires_the_privacy_package_in() -> None:
    assert _imports_privacy_package(REPO_ROOT / "src/ckp/outbox")


def test_outbox_package_does_not_import_out_of_scope_sinks_or_run_git() -> None:
    """The outbox owns durable state; C7 owns Git, C6 owns read auth.

    ``subprocess`` and ``os.environ`` are pinned out so the outbox can never
    grow its own Git transaction or quietly load a production secret.
    """
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "src/ckp/outbox").glob("*.py"))
    ).lower()
    for forbidden in (
        "from ckp.auth",
        "import ckp.auth",
        "subprocess",
        "os.environ",
        "qdrant",
        "discord",
        "notion",
        "hermes",
        "openab",
    ):
        assert forbidden not in source


def test_c8_expected_outbox_modules_are_present() -> None:
    module_names = {path.name for path in (REPO_ROOT / "src/ckp/outbox").glob("*.py")}
    assert module_names == {
        "__init__.py",
        "composition.py",
        "crypto.py",
        "errors.py",
        "models.py",
        "policy.py",
        "service.py",
        "store.py",
    }


def test_record_binding_is_versioned_and_covers_every_frozen_dimension() -> None:
    source = inspect.getsource(compute_binding_hash)
    for needle in (
        "BINDING_DOMAIN",
        'record_version.encode("utf-8")',
        'writer_contract.encode("utf-8")',
        'operation_id.encode("utf-8")',
        'idempotency_hash.encode("ascii")',
        'actor.encode("utf-8")',
        'task_hash.encode("ascii")',
        'target_path.encode("utf-8")',
        'request_hash.encode("ascii")',
        'content_hash.encode("ascii")',
        'enqueue_base_commit.encode("ascii")',
        'created_at.encode("ascii")',
        'expires_at.encode("ascii")',
    ):
        assert needle in source


def test_replay_composes_the_c7_writer_with_a_pinned_clock() -> None:
    """Replay must reuse the frozen C7 transaction, on the enqueue timestamp."""
    source = inspect.getsource(build_c8_synthetic_outbox)
    for needle in (
        "build_c7_synthetic_writer",
        "replay_clock = ReplayClock()",
        "clock=replay_clock",
    ):
        assert needle in source


def test_terminal_states_purge_and_sandbox_is_always_removed() -> None:
    service_source = (REPO_ROOT / "src/ckp/outbox/service.py").read_text(
        encoding="utf-8"
    )
    assert 'update["envelope"] = None' in service_source
    assert "shutil.rmtree(sandbox, ignore_errors=True)" in service_source
    for forbidden in ('"stash"', '"reset"', '"clean"', '"prune"'):
        assert forbidden not in service_source
