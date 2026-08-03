from __future__ import annotations

import inspect

import pytest

from ckp.auth import CredentialRegistry
from ckp.writer.errors import WriterErrorCode, WriterRefusal
from ckp.writer.identity import ProvenanceMode, WriterActor, WriterActorRegistry
from writer_fixtures import (
    SYNTHETIC_CLAUDE_CREDENTIAL,
    SYNTHETIC_CODEX_CREDENTIAL,
)


def _registry() -> WriterActorRegistry:
    return WriterActorRegistry(
        (WriterActor.synthetic("codex", SYNTHETIC_CODEX_CREDENTIAL),)
    )


def _resolve(
    registry: WriterActorRegistry,
    credential: str | None = SYNTHETIC_CODEX_CREDENTIAL,
    *,
    request_id: str = "server-request-1",
    task_id: str = "server-task-1",
    mode: ProvenanceMode | str = ProvenanceMode.GENERATED,
    model_id: str | None = "gpt-5.6",
):
    return registry.resolve(
        credential,
        request_id=request_id,
        task_id=task_id,
        provenance_mode=mode,
        model_id=model_id,
    )


def test_writer_registry_is_not_the_c6_read_registry() -> None:
    assert WriterActorRegistry is not CredentialRegistry
    assert "CredentialRegistry" not in inspect.getsource(WriterActorRegistry)
    module_source = inspect.getsource(inspect.getmodule(WriterActorRegistry))
    assert "from ckp.auth" not in module_source
    assert "import ckp.auth" not in module_source


@pytest.mark.parametrize("actor_id", ["grok", "asheron", "polylong"])
def test_unregistered_c6_read_actors_cannot_become_writer_actors(
    actor_id: str,
) -> None:
    with pytest.raises(WriterRefusal) as caught:
        WriterActor.synthetic(actor_id, "SYNTHETIC_UNREGISTERED_CREDENTIAL")
    assert caught.value.code is WriterErrorCode.ACTOR_DENIED


@pytest.mark.parametrize(
    "actor_id",
    ["Codex", "codex_alias", "codex/session-1", "process:migration"],
)
def test_writer_registry_does_not_guess_aliases_or_process_identity(
    actor_id: str,
) -> None:
    with pytest.raises(WriterRefusal) as caught:
        WriterActor.synthetic(actor_id, "SYNTHETIC_ALIAS_CREDENTIAL")
    assert caught.value.code is WriterErrorCode.ACTOR_DENIED


@pytest.mark.parametrize("credential", [None, "", "SYNTHETIC_WRONG_CREDENTIAL"])
def test_missing_or_wrong_actor_credential_fails_closed(
    credential: str | None,
) -> None:
    expected = (
        WriterErrorCode.IDENTITY_MISSING
        if not credential
        else WriterErrorCode.ACTOR_DENIED
    )
    with pytest.raises(WriterRefusal) as caught:
        _resolve(_registry(), credential)
    assert caught.value.code is expected


@pytest.mark.parametrize("model_id", [None, "unknown", "UPPER", "bad/model"])
def test_generated_agent_write_requires_runtime_model_identity(
    model_id: str | None,
) -> None:
    with pytest.raises(WriterRefusal) as caught:
        _resolve(_registry(), model_id=model_id)
    assert caught.value.code is WriterErrorCode.MODEL_IDENTITY_MISSING


def test_generated_and_pure_capture_actor_shapes_are_server_minted() -> None:
    registry = _registry()
    generated = _resolve(registry)
    captured = _resolve(
        registry,
        mode=ProvenanceMode.CAPTURED,
        model_id=None,
    )

    assert generated.actor == "codex/gpt-5.6"
    assert generated.provenance_mode is ProvenanceMode.GENERATED
    assert captured.actor == "codex"
    assert captured.provenance_mode is ProvenanceMode.CAPTURED
    registry.validate(generated)
    registry.validate(captured)


@pytest.mark.parametrize(
    "request_id,task_id",
    [
        ("", "server-task-1"),
        ("server request", "server-task-1"),
        ("server-request-1", ""),
        ("server-request-1", "server/task"),
    ],
)
def test_server_request_and_task_bindings_are_bounded(
    request_id: str,
    task_id: str,
) -> None:
    with pytest.raises(WriterRefusal) as caught:
        _resolve(_registry(), request_id=request_id, task_id=task_id)
    assert caught.value.code is WriterErrorCode.IDENTITY_MISSING


def test_context_from_another_registry_cannot_cross_the_writer_boundary() -> None:
    first = _registry()
    second = WriterActorRegistry(
        (WriterActor.synthetic("codex", SYNTHETIC_CLAUDE_CREDENTIAL),)
    )
    context = _resolve(first)

    with pytest.raises(WriterRefusal) as caught:
        second.validate(context)
    assert caught.value.code is WriterErrorCode.ACTOR_DENIED
