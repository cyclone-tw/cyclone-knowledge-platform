"""Server-owned Writer identity, separate from the C6 read principal.

The registry is deliberately injected.  This package does not derive write
authority from ``ckp.auth.AccessActor`` and does not ship a permissive default
actor set.  Identity/Provenance v1 remains authoritative in Cyclone-Wiki.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from ckp.writer.errors import WriterErrorCode, WriterRefusal

_ACTOR_ID = re.compile(r"^[a-z0-9][a-z0-9.-]{0,63}$")
_HUMAN_ACTOR = re.compile(r"^human:[a-z0-9][a-z0-9.-]{0,63}$")
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9.-]{0,63}$")
_RUNTIME_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")

# Identity/Provenance v1 has no Writer registration for these C6 read-side
# principals.  C7 pins that negative capability and does not accept aliases.
_UNREGISTERED_WRITER_ACTORS_V1 = frozenset({"grok", "asheron", "polylong"})


class ProvenanceMode(StrEnum):
    GENERATED = "generated"
    CAPTURED = "captured"


@dataclass(frozen=True, slots=True)
class WriterActor:
    """One explicitly configured Writer actor; no aliases are supported."""

    actor_id: str
    credential_sha256: str
    generated: bool = True
    captured: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.actor_id, str)
            or not isinstance(self.credential_sha256, str)
            or not isinstance(self.generated, bool)
            or not isinstance(self.captured, bool)
        ):
            raise WriterRefusal(WriterErrorCode.ACTOR_DENIED)
        if self.actor_id in _UNREGISTERED_WRITER_ACTORS_V1:
            raise WriterRefusal(WriterErrorCode.ACTOR_DENIED)
        if not (
            _ACTOR_ID.fullmatch(self.actor_id) or _HUMAN_ACTOR.fullmatch(self.actor_id)
        ):
            raise WriterRefusal(WriterErrorCode.ACTOR_DENIED)
        if not _SHA256.fullmatch(self.credential_sha256):
            raise WriterRefusal(WriterErrorCode.IDENTITY_MISSING)
        if not self.generated and not self.captured:
            raise WriterRefusal(WriterErrorCode.ACTOR_DENIED)

    @classmethod
    def synthetic(
        cls,
        actor_id: str,
        credential: str,
        *,
        generated: bool = True,
        captured: bool = True,
    ) -> WriterActor:
        """Build a fixture actor without retaining its plaintext credential."""
        if not isinstance(credential, str) or not credential:
            raise WriterRefusal(WriterErrorCode.IDENTITY_MISSING)
        digest = hashlib.sha256(credential.encode("utf-8")).hexdigest()
        return cls(
            actor_id=actor_id,
            credential_sha256=f"sha256:{digest}",
            generated=generated,
            captured=captured,
        )


@dataclass(frozen=True, slots=True)
class _IdentitySeal:
    registry_token: object
    actor_id: str
    actor: str
    request_id: str
    task_id: str
    provenance_mode: ProvenanceMode
    model_id: str | None


@dataclass(frozen=True, slots=True)
class AuthenticatedWriterContext:
    """Trusted runtime context minted and bound by one Writer registry."""

    actor_id: str
    actor: str
    request_id: str
    task_id: str
    provenance_mode: ProvenanceMode
    model_id: str | None
    _seal: _IdentitySeal = field(repr=False, compare=False, kw_only=True)

    def __post_init__(self) -> None:
        seal = self._seal
        if (
            type(seal) is not _IdentitySeal
            or seal.actor_id != self.actor_id
            or seal.actor != self.actor
            or seal.request_id != self.request_id
            or seal.task_id != self.task_id
            or seal.provenance_mode is not self.provenance_mode
            or seal.model_id != self.model_id
        ):
            raise WriterRefusal(WriterErrorCode.ACTOR_DENIED)


class WriterActorRegistry:
    """Resolve an opaque runtime credential into a sealed write identity."""

    def __init__(self, actors: tuple[WriterActor, ...]) -> None:
        if (
            not isinstance(actors, tuple)
            or not actors
            or any(not isinstance(actor, WriterActor) for actor in actors)
        ):
            raise WriterRefusal(WriterErrorCode.IDENTITY_MISSING)
        by_digest: dict[str, WriterActor] = {}
        actor_ids: set[str] = set()
        for actor in actors:
            if actor.credential_sha256 in by_digest or actor.actor_id in actor_ids:
                raise WriterRefusal(WriterErrorCode.ACTOR_DENIED)
            by_digest[actor.credential_sha256] = actor
            actor_ids.add(actor.actor_id)
        self._actors = MappingProxyType(by_digest)
        self._registry_token = object()

    def resolve(
        self,
        actor_credential: str | None,
        *,
        request_id: str,
        task_id: str,
        provenance_mode: ProvenanceMode | str,
        model_id: str | None,
    ) -> AuthenticatedWriterContext:
        if (
            not isinstance(actor_credential, str)
            or not actor_credential
            or not isinstance(request_id, str)
            or _RUNTIME_ID.fullmatch(request_id) is None
            or not isinstance(task_id, str)
            or _RUNTIME_ID.fullmatch(task_id) is None
        ):
            raise WriterRefusal(WriterErrorCode.IDENTITY_MISSING)
        digest_value = hashlib.sha256(actor_credential.encode("utf-8")).hexdigest()
        digest = f"sha256:{digest_value}"
        actor = self._actors.get(digest)
        if actor is None:
            raise WriterRefusal(WriterErrorCode.ACTOR_DENIED)
        try:
            mode = ProvenanceMode(provenance_mode)
        except (TypeError, ValueError) as exc:
            raise WriterRefusal(WriterErrorCode.IDENTITY_MISSING) from exc

        is_human = actor.actor_id.startswith("human:")
        if mode is ProvenanceMode.GENERATED:
            if not actor.generated:
                raise WriterRefusal(WriterErrorCode.ACTOR_DENIED)
            if is_human:
                if model_id is not None:
                    raise WriterRefusal(WriterErrorCode.ACTOR_DENIED)
                rendered_actor = actor.actor_id
            else:
                if (
                    not isinstance(model_id, str)
                    or model_id == "unknown"
                    or _MODEL_ID.fullmatch(model_id) is None
                ):
                    raise WriterRefusal(WriterErrorCode.MODEL_IDENTITY_MISSING)
                rendered_actor = f"{actor.actor_id}/{model_id}"
        else:
            if not actor.captured:
                raise WriterRefusal(WriterErrorCode.ACTOR_DENIED)
            if model_id is not None:
                raise WriterRefusal(WriterErrorCode.ACTOR_DENIED)
            rendered_actor = actor.actor_id

        seal = _IdentitySeal(
            registry_token=self._registry_token,
            actor_id=actor.actor_id,
            actor=rendered_actor,
            request_id=request_id,
            task_id=task_id,
            provenance_mode=mode,
            model_id=model_id,
        )
        return AuthenticatedWriterContext(
            actor_id=actor.actor_id,
            actor=rendered_actor,
            request_id=request_id,
            task_id=task_id,
            provenance_mode=mode,
            model_id=model_id,
            _seal=seal,
        )

    def validate(self, context: AuthenticatedWriterContext) -> None:
        if (
            not isinstance(context, AuthenticatedWriterContext)
            or context._seal.registry_token is not self._registry_token
        ):
            raise WriterRefusal(WriterErrorCode.ACTOR_DENIED)


__all__ = [
    "AuthenticatedWriterContext",
    "ProvenanceMode",
    "WriterActor",
    "WriterActorRegistry",
]
