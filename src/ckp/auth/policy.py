"""Server-owned C6 access grants and the frozen eligibility matrix.

The matrix is an issuance ceiling, never a standing permission. A request
gets authority only after two opaque credentials resolve server-side: one
identifies the runtime actor and the other selects an expiring task grant.
Neither credential contains caller-editable scope claims.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

from ckp.catalog import CatalogEntry
from ckp.privacy import PrivacyClass


class AccessActor(StrEnum):
    AURELION = "aurelion"
    BAHAMUT = "bahamut"
    OPENCLAW = "openclaw"
    OPENAB = "openab"
    CLAUDE_CODE = "claude-code"
    CODEX = "codex"
    CURSOR = "cursor"
    GROK = "grok"
    ASHERON = "asheron"
    POLYLONG = "polylong"


class KnowledgeDomain(StrEnum):
    FINANCE = "finance"
    CALENDAR_CONTEXT = "calendar-context"
    PRIVATE_WORK = "private-work"
    SOCIAL_CIRCLE = "social-circle"
    PERSONAL_RETROSPECTIVE = "personal-retrospective"
    CORE_ENGINEERING = "core-engineering"


class Capability(StrEnum):
    READ = "read"
    CAPTURE_PRIVATE_INBOX = "capture-private-inbox"
    DIRECT_FORMAL_WRITE = "direct-formal-write"
    REPORT_GENERATION = "report-generation"
    PUBLICATION = "publication"


_READ_REPORT = frozenset({Capability.READ, Capability.REPORT_GENERATION})
_READ_REPORT_CAPTURE = frozenset(
    {
        Capability.READ,
        Capability.REPORT_GENERATION,
        Capability.CAPTURE_PRIVATE_INBOX,
    }
)
_PERSONAL_DOMAINS = frozenset(
    {
        KnowledgeDomain.FINANCE,
        KnowledgeDomain.CALENDAR_CONTEXT,
        KnowledgeDomain.PRIVATE_WORK,
        KnowledgeDomain.SOCIAL_CIRCLE,
        KnowledgeDomain.PERSONAL_RETROSPECTIVE,
    }
)
_C6_SERVED_CAPABILITIES = frozenset({Capability.READ, Capability.REPORT_GENERATION})
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _personal_matrix() -> dict[KnowledgeDomain, frozenset[Capability]]:
    return {domain: _READ_REPORT for domain in KnowledgeDomain}


def _with_capture(
    *domains: KnowledgeDomain,
) -> dict[KnowledgeDomain, frozenset[Capability]]:
    values = _personal_matrix()
    for domain in domains:
        values[domain] = _READ_REPORT_CAPTURE
    return values


def _deny_personal() -> dict[KnowledgeDomain, frozenset[Capability]]:
    return {
        domain: (
            _READ_REPORT if domain is KnowledgeDomain.CORE_ENGINEERING else frozenset()
        )
        for domain in KnowledgeDomain
    }


_MATRIX_SOURCE: dict[AccessActor, dict[KnowledgeDomain, frozenset[Capability]]] = {
    AccessActor.AURELION: _with_capture(*_PERSONAL_DOMAINS),
    AccessActor.BAHAMUT: _with_capture(KnowledgeDomain.PRIVATE_WORK),
    AccessActor.OPENCLAW: _personal_matrix(),
    AccessActor.OPENAB: _with_capture(
        KnowledgeDomain.PRIVATE_WORK,
        KnowledgeDomain.SOCIAL_CIRCLE,
        KnowledgeDomain.PERSONAL_RETROSPECTIVE,
    ),
    AccessActor.CLAUDE_CODE: _with_capture(KnowledgeDomain.PERSONAL_RETROSPECTIVE),
    AccessActor.CODEX: _with_capture(KnowledgeDomain.PERSONAL_RETROSPECTIVE),
    AccessActor.CURSOR: _with_capture(KnowledgeDomain.PERSONAL_RETROSPECTIVE),
    AccessActor.GROK: _with_capture(KnowledgeDomain.PERSONAL_RETROSPECTIVE),
    AccessActor.ASHERON: _deny_personal(),
    AccessActor.POLYLONG: _deny_personal(),
}

AGENT_DOMAIN_CAPABILITIES: Mapping[
    AccessActor, Mapping[KnowledgeDomain, frozenset[Capability]]
] = MappingProxyType(
    {
        actor: MappingProxyType(dict(domains))
        for actor, domains in _MATRIX_SOURCE.items()
    }
)


class GrantPolicyError(ValueError):
    """A server-side grant definition violates the frozen policy."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DomainPolicyError(RuntimeError):
    """The server-owned note/domain registry cannot make a safe decision."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class AccessDenied(PermissionError):
    """A sanitized authorization refusal safe to map into a receipt."""

    def __init__(self, code: str, *, http_status: int = 403) -> None:
        self.code = code
        self.http_status = http_status
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ScopeGrant:
    grant_id: str
    actor: AccessActor
    task_id: str
    domains: frozenset[KnowledgeDomain]
    capabilities: frozenset[Capability]
    privacy_classes: frozenset[PrivacyClass]
    issued_by: str
    issued_at: datetime
    expires_at: datetime
    max_items: int
    max_token_upper_bound: int

    def __post_init__(self) -> None:
        if not isinstance(self.grant_id, str) or not isinstance(self.task_id, str):
            raise GrantPolicyError("missing-task-binding")
        if not isinstance(self.actor, AccessActor):
            raise GrantPolicyError("invalid-access-actor")
        if not isinstance(self.domains, frozenset):
            raise GrantPolicyError("mutable-domain-scope")
        if any(not isinstance(domain, KnowledgeDomain) for domain in self.domains):
            raise GrantPolicyError("invalid-domain")
        if not isinstance(self.capabilities, frozenset):
            raise GrantPolicyError("mutable-capability-scope")
        if any(
            not isinstance(capability, Capability) for capability in self.capabilities
        ):
            raise GrantPolicyError("invalid-capability")
        if not isinstance(self.privacy_classes, frozenset):
            raise GrantPolicyError("mutable-privacy-scope")
        if any(
            not isinstance(privacy, PrivacyClass) for privacy in self.privacy_classes
        ):
            raise GrantPolicyError("invalid-privacy-class")
        if not self.grant_id.strip() or not self.task_id.strip():
            raise GrantPolicyError("missing-task-binding")
        if not self.domains or not self.capabilities or not self.privacy_classes:
            raise GrantPolicyError("empty-scope")
        if not isinstance(self.issued_at, datetime) or not isinstance(
            self.expires_at, datetime
        ):
            raise GrantPolicyError("invalid-grant-time")
        if self.issued_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise GrantPolicyError("naive-grant-time")
        if self.expires_at <= self.issued_at:
            raise GrantPolicyError("invalid-grant-window")
        if (
            isinstance(self.max_items, bool)
            or not isinstance(self.max_items, int)
            or not 1 <= self.max_items <= 20
        ):
            raise GrantPolicyError("invalid-item-limit")
        if (
            isinstance(self.max_token_upper_bound, bool)
            or not isinstance(self.max_token_upper_bound, int)
            or not 64 <= self.max_token_upper_bound <= 32_768
        ):
            raise GrantPolicyError("invalid-token-limit")


@dataclass(frozen=True, slots=True)
class AccessContext:
    actor: AccessActor
    grant_id: str
    task_id: str
    domain: KnowledgeDomain
    capabilities: frozenset[Capability]
    privacy_classes: frozenset[PrivacyClass]
    expires_at: datetime
    max_items: int
    max_token_upper_bound: int


class GrantPolicy:
    """Validate an immutable grant against the issue #8 policy ceiling."""

    def eligibility_snapshot(self) -> dict[str, dict[str, list[str]]]:
        return {
            actor.value: {
                domain.value: sorted(capability.value for capability in capabilities)
                for domain, capabilities in domains.items()
            }
            for actor, domains in AGENT_DOMAIN_CAPABILITIES.items()
        }

    def validate(self, grant: ScopeGrant) -> None:
        if grant.issued_by != "human:cyclone":
            raise GrantPolicyError("owner-trust-required")
        if not grant.capabilities <= _C6_SERVED_CAPABILITIES:
            raise GrantPolicyError("capability-not-served-by-c6")
        if (
            Capability.REPORT_GENERATION in grant.capabilities
            and Capability.READ not in grant.capabilities
        ):
            raise GrantPolicyError("report-requires-read")
        if PrivacyClass.STUDENT_PRIVATE in grant.privacy_classes:
            raise GrantPolicyError("student-private-scope-denied")

        actor_matrix = AGENT_DOMAIN_CAPABILITIES[grant.actor]
        for domain in grant.domains:
            if not grant.capabilities <= actor_matrix[domain]:
                raise GrantPolicyError("agent-domain-capability-denied")
            if (
                domain is KnowledgeDomain.CORE_ENGINEERING
                and PrivacyClass.SENSITIVE in grant.privacy_classes
            ):
                # Private strategy and unreleased meetings route to Private,
                # never into the Core-engineering access domain.
                raise GrantPolicyError("sensitive-core-routing-denied")


def _credential_digest(plaintext: str) -> str:
    if not isinstance(plaintext, str) or not 1 <= len(plaintext) <= 512:
        raise GrantPolicyError("empty-credential")
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ActorCredential:
    credential_sha256: str
    actor: AccessActor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.credential_sha256, str)
            or _SHA256_HEX.fullmatch(self.credential_sha256) is None
        ):
            raise GrantPolicyError("invalid-credential-digest")
        if not isinstance(self.actor, AccessActor):
            raise GrantPolicyError("invalid-access-actor")

    @classmethod
    def from_plaintext(cls, plaintext: str, actor: AccessActor) -> ActorCredential:
        return cls(credential_sha256=_credential_digest(plaintext), actor=actor)


@dataclass(frozen=True, slots=True)
class GrantCredential:
    credential_sha256: str
    grant: ScopeGrant

    def __post_init__(self) -> None:
        if (
            not isinstance(self.credential_sha256, str)
            or _SHA256_HEX.fullmatch(self.credential_sha256) is None
        ):
            raise GrantPolicyError("invalid-credential-digest")
        if not isinstance(self.grant, ScopeGrant):
            raise GrantPolicyError("invalid-grant")

    @classmethod
    def from_plaintext(cls, plaintext: str, grant: ScopeGrant) -> GrantCredential:
        return cls(credential_sha256=_credential_digest(plaintext), grant=grant)


class CredentialRegistry:
    """Resolve actor and task credentials without retaining plaintext."""

    def __init__(
        self,
        policy: GrantPolicy,
        actor_credentials: Iterable[ActorCredential],
        grant_credentials: Iterable[GrantCredential],
        clock: Callable[[], datetime],
    ) -> None:
        self._policy = policy
        self._clock = clock
        self._actors: dict[str, AccessActor] = {}
        self._grants: dict[str, ScopeGrant] = {}
        grant_ids: set[str] = set()
        for credential in actor_credentials:
            if credential.credential_sha256 in self._actors:
                raise GrantPolicyError("duplicate-actor-credential")
            self._actors[credential.credential_sha256] = credential.actor
        for credential in grant_credentials:
            policy.validate(credential.grant)
            if credential.credential_sha256 in self._grants:
                raise GrantPolicyError("duplicate-grant-credential")
            if credential.grant.grant_id in grant_ids:
                raise GrantPolicyError("duplicate-grant-id")
            self._grants[credential.credential_sha256] = credential.grant
            grant_ids.add(credential.grant.grant_id)
        if self._actors.keys() & self._grants.keys():
            raise GrantPolicyError("credential-role-overlap")

    @classmethod
    def deny_all(cls, clock: Callable[[], datetime]) -> CredentialRegistry:
        return cls(GrantPolicy(), (), (), clock)

    def __repr__(self) -> str:
        return (
            "CredentialRegistry("
            f"actor_credentials={len(self._actors)}, "
            f"grant_credentials={len(self._grants)})"
        )

    def resolve(
        self,
        *,
        actor_credential: str | None,
        grant_credential: str | None,
        requested_domain: KnowledgeDomain,
        required_capabilities: frozenset[Capability],
    ) -> AccessContext:
        if not actor_credential or not grant_credential:
            raise AccessDenied("missing-credential", http_status=401)
        try:
            actor_digest = _credential_digest(actor_credential)
            grant_digest = _credential_digest(grant_credential)
        except GrantPolicyError:
            raise AccessDenied("invalid-credential", http_status=401) from None
        actor = self._actors.get(actor_digest)
        grant = self._grants.get(grant_digest)
        if actor is None or grant is None:
            raise AccessDenied("invalid-credential", http_status=401)
        if actor is not grant.actor:
            raise AccessDenied("wrong-actor")

        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise GrantPolicyError("naive-server-clock")
        if now < grant.issued_at:
            raise AccessDenied("scope-not-active")
        if now >= grant.expires_at:
            raise AccessDenied("scope-expired")
        if requested_domain not in grant.domains:
            raise AccessDenied("domain-denied")
        if not required_capabilities <= grant.capabilities:
            raise AccessDenied("capability-denied")

        return AccessContext(
            actor=actor,
            grant_id=grant.grant_id,
            task_id=grant.task_id,
            domain=requested_domain,
            capabilities=grant.capabilities,
            privacy_classes=grant.privacy_classes,
            expires_at=grant.expires_at,
            max_items=grant.max_items,
            max_token_upper_bound=grant.max_token_upper_bound,
        )


@dataclass(frozen=True, slots=True)
class DomainBinding:
    note_id: str
    domain: KnowledgeDomain

    def __post_init__(self) -> None:
        if not isinstance(self.domain, KnowledgeDomain):
            raise GrantPolicyError("invalid-domain")
        try:
            parsed = uuid.UUID(self.note_id)
        except (AttributeError, ValueError):
            raise GrantPolicyError("invalid-domain-binding-id") from None
        if (
            str(parsed) != self.note_id
            or parsed.version != 7
            or parsed.variant != uuid.RFC_4122
        ):
            raise GrantPolicyError("invalid-domain-binding-id")


class DomainRegistry:
    """Server-owned durable note-ID to access-domain bindings."""

    def __init__(self, bindings: Iterable[DomainBinding]) -> None:
        self._domains: dict[str, KnowledgeDomain] = {}
        for binding in bindings:
            if binding.note_id in self._domains:
                raise GrantPolicyError("duplicate-domain-binding")
            self._domains[binding.note_id] = binding.domain

    def select(
        self,
        entries: Iterable[CatalogEntry],
        domain: KnowledgeDomain,
    ) -> tuple[CatalogEntry, ...]:
        # Missing immutable IDs and missing bindings are denied, never guessed
        # from a path, caller label, or body text. A duplicate durable ID makes
        # domain ownership ambiguous, so refuse the whole snapshot.
        selected: list[CatalogEntry] = []
        seen_ids: set[str] = set()
        for entry in entries:
            if entry.id is None:
                continue
            if entry.id in seen_ids:
                raise DomainPolicyError("duplicate-domain-id")
            seen_ids.add(entry.id)
            if self._domains.get(entry.id) is domain:
                selected.append(entry)
        return tuple(selected)


__all__ = [
    "AGENT_DOMAIN_CAPABILITIES",
    "AccessActor",
    "AccessContext",
    "AccessDenied",
    "ActorCredential",
    "Capability",
    "CredentialRegistry",
    "DomainBinding",
    "DomainPolicyError",
    "DomainRegistry",
    "GrantCredential",
    "GrantPolicy",
    "GrantPolicyError",
    "KnowledgeDomain",
    "ScopeGrant",
]
