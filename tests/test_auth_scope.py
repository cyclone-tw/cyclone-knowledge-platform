"""C6 server-owned authorization and frozen eligibility matrix."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ckp.auth import (
    AccessActor,
    AccessDenied,
    ActorCredential,
    Capability,
    CredentialRegistry,
    DomainBinding,
    GrantCredential,
    GrantPolicy,
    GrantPolicyError,
    KnowledgeDomain,
    ScopeGrant,
)
from ckp.packer import ContextRequest
from ckp.privacy import PrivacyClass

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
ACTOR_TOKEN = "synthetic-actor-credential-codex"
OTHER_ACTOR_TOKEN = "synthetic-actor-credential-claude"
GRANT_TOKEN = "synthetic-task-grant-finance"


def _grant(
    *,
    actor: AccessActor = AccessActor.CODEX,
    domains: frozenset[KnowledgeDomain] = frozenset({KnowledgeDomain.FINANCE}),
    capabilities: frozenset[Capability] = frozenset(
        {Capability.READ, Capability.REPORT_GENERATION}
    ),
    privacy_classes: frozenset[PrivacyClass] = frozenset(
        {
            PrivacyClass.PUBLIC,
            PrivacyClass.INTERNAL,
            PrivacyClass.SENSITIVE,
        }
    ),
    issued_at: datetime = NOW - timedelta(minutes=5),
    expires_at: datetime = NOW + timedelta(hours=1),
) -> ScopeGrant:
    return ScopeGrant(
        grant_id="grant-synthetic-finance-001",
        actor=actor,
        task_id="task-synthetic-personal-read-001",
        domains=domains,
        capabilities=capabilities,
        privacy_classes=privacy_classes,
        issued_by="human:cyclone",
        issued_at=issued_at,
        expires_at=expires_at,
        max_items=3,
        max_token_upper_bound=1_024,
    )


def _registry(grant: ScopeGrant | None = None) -> CredentialRegistry:
    grant = grant or _grant()
    return CredentialRegistry(
        policy=GrantPolicy(),
        actor_credentials=(
            ActorCredential.from_plaintext(ACTOR_TOKEN, AccessActor.CODEX),
            ActorCredential.from_plaintext(OTHER_ACTOR_TOKEN, AccessActor.CLAUDE_CODE),
        ),
        grant_credentials=(GrantCredential.from_plaintext(GRANT_TOKEN, grant),),
        clock=lambda: NOW,
    )


def test_frozen_agent_domain_capability_matrix_matches_issue_8() -> None:
    snapshot = GrantPolicy().eligibility_snapshot()
    all_personal = {
        "finance": ["read", "report-generation"],
        "calendar-context": ["read", "report-generation"],
        "private-work": ["read", "report-generation"],
        "social-circle": ["read", "report-generation"],
        "personal-retrospective": ["read", "report-generation"],
        "core-engineering": ["read", "report-generation"],
    }
    assert snapshot == {
        "aurelion": {
            **all_personal,
            "finance": ["capture-private-inbox", "read", "report-generation"],
            "calendar-context": [
                "capture-private-inbox",
                "read",
                "report-generation",
            ],
            "private-work": [
                "capture-private-inbox",
                "read",
                "report-generation",
            ],
            "social-circle": [
                "capture-private-inbox",
                "read",
                "report-generation",
            ],
            "personal-retrospective": [
                "capture-private-inbox",
                "read",
                "report-generation",
            ],
        },
        "bahamut": {
            **all_personal,
            "private-work": [
                "capture-private-inbox",
                "read",
                "report-generation",
            ],
        },
        "openclaw": all_personal,
        "openab": {
            **all_personal,
            "private-work": [
                "capture-private-inbox",
                "read",
                "report-generation",
            ],
            "social-circle": [
                "capture-private-inbox",
                "read",
                "report-generation",
            ],
            "personal-retrospective": [
                "capture-private-inbox",
                "read",
                "report-generation",
            ],
        },
        "claude-code": {
            **all_personal,
            "personal-retrospective": [
                "capture-private-inbox",
                "read",
                "report-generation",
            ],
        },
        "codex": {
            **all_personal,
            "personal-retrospective": [
                "capture-private-inbox",
                "read",
                "report-generation",
            ],
        },
        "cursor": {
            **all_personal,
            "personal-retrospective": [
                "capture-private-inbox",
                "read",
                "report-generation",
            ],
        },
        "grok": {
            **all_personal,
            "personal-retrospective": [
                "capture-private-inbox",
                "read",
                "report-generation",
            ],
        },
        "asheron": {
            "finance": [],
            "calendar-context": [],
            "private-work": [],
            "social-circle": [],
            "personal-retrospective": [],
            "core-engineering": ["read", "report-generation"],
        },
        "polylong": {
            "finance": [],
            "calendar-context": [],
            "private-work": [],
            "social-circle": [],
            "personal-retrospective": [],
            "core-engineering": ["read", "report-generation"],
        },
    }
    assert all(
        "direct-formal-write" not in capabilities and "publication" not in capabilities
        for domains in snapshot.values()
        for capabilities in domains.values()
    )


def test_domains_are_frozen_and_context_request_has_no_scope_fields() -> None:
    assert [domain.value for domain in KnowledgeDomain] == [
        "finance",
        "calendar-context",
        "private-work",
        "social-circle",
        "personal-retrospective",
        "core-engineering",
    ]
    assert set(ContextRequest.model_fields) == {"query"}

    with pytest.raises(GrantPolicyError, match="invalid-domain-binding-id"):
        DomainBinding("synthetic-not-a-uuid", KnowledgeDomain.FINANCE)


def test_valid_task_grant_resolves_from_two_server_owned_credentials() -> None:
    registry = _registry()

    context = registry.resolve(
        actor_credential=ACTOR_TOKEN,
        grant_credential=GRANT_TOKEN,
        requested_domain=KnowledgeDomain.FINANCE,
        required_capabilities=frozenset(
            {Capability.READ, Capability.REPORT_GENERATION}
        ),
    )

    assert context.actor is AccessActor.CODEX
    assert context.domain is KnowledgeDomain.FINANCE
    assert context.task_id == "task-synthetic-personal-read-001"
    assert context.max_items == 3
    assert context.max_token_upper_bound == 1_024
    assert "synthetic-actor-credential" not in repr(registry)
    assert "synthetic-task-grant" not in repr(registry)


@pytest.mark.parametrize(
    ("actor_credential", "grant_credential", "domain", "required", "code"),
    [
        (
            None,
            GRANT_TOKEN,
            KnowledgeDomain.FINANCE,
            {Capability.READ},
            "missing-credential",
        ),
        (
            ACTOR_TOKEN,
            None,
            KnowledgeDomain.FINANCE,
            {Capability.READ},
            "missing-credential",
        ),
        (
            "synthetic-invalid",
            GRANT_TOKEN,
            KnowledgeDomain.FINANCE,
            {Capability.READ},
            "invalid-credential",
        ),
        (
            OTHER_ACTOR_TOKEN,
            GRANT_TOKEN,
            KnowledgeDomain.FINANCE,
            {Capability.READ},
            "wrong-actor",
        ),
        (
            ACTOR_TOKEN,
            GRANT_TOKEN,
            KnowledgeDomain.CALENDAR_CONTEXT,
            {Capability.READ},
            "domain-denied",
        ),
        (
            ACTOR_TOKEN,
            GRANT_TOKEN,
            KnowledgeDomain.FINANCE,
            {Capability.DIRECT_FORMAL_WRITE},
            "capability-denied",
        ),
    ],
)
def test_resolution_failures_are_stable_and_do_not_echo_inputs(
    actor_credential: str | None,
    grant_credential: str | None,
    domain: KnowledgeDomain,
    required: set[Capability],
    code: str,
) -> None:
    with pytest.raises(AccessDenied) as caught:
        _registry().resolve(
            actor_credential=actor_credential,
            grant_credential=grant_credential,
            requested_domain=domain,
            required_capabilities=frozenset(required),
        )

    assert caught.value.code == code
    rendered = str(caught.value)
    assert rendered == code
    assert "synthetic" not in rendered


def test_expired_and_not_yet_valid_grants_fail_closed() -> None:
    for grant, expected in (
        (
            _grant(expires_at=NOW),
            "scope-expired",
        ),
        (
            _grant(
                issued_at=NOW + timedelta(seconds=1),
                expires_at=NOW + timedelta(hours=1),
            ),
            "scope-not-active",
        ),
    ):
        with pytest.raises(AccessDenied) as caught:
            _registry(grant).resolve(
                actor_credential=ACTOR_TOKEN,
                grant_credential=GRANT_TOKEN,
                requested_domain=KnowledgeDomain.FINANCE,
                required_capabilities=frozenset({Capability.READ}),
            )
        assert caught.value.code == expected


def test_credential_roles_cannot_share_one_token_and_oversize_input_is_denied() -> None:
    with pytest.raises(GrantPolicyError, match="credential-role-overlap"):
        CredentialRegistry(
            policy=GrantPolicy(),
            actor_credentials=(
                ActorCredential.from_plaintext(ACTOR_TOKEN, AccessActor.CODEX),
            ),
            grant_credentials=(GrantCredential.from_plaintext(ACTOR_TOKEN, _grant()),),
            clock=lambda: NOW,
        )

    with pytest.raises(AccessDenied) as caught:
        _registry().resolve(
            actor_credential="x" * 513,
            grant_credential=GRANT_TOKEN,
            requested_domain=KnowledgeDomain.FINANCE,
            required_capabilities=frozenset({Capability.READ}),
        )
    assert caught.value.code == "invalid-credential"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("domains", {KnowledgeDomain.FINANCE}, "mutable-domain-scope"),
        ("capabilities", {Capability.READ}, "mutable-capability-scope"),
        ("privacy_classes", {PrivacyClass.INTERNAL}, "mutable-privacy-scope"),
        ("max_items", True, "invalid-item-limit"),
        ("max_items", 21, "invalid-item-limit"),
        ("max_items", "3", "invalid-item-limit"),
        ("max_token_upper_bound", True, "invalid-token-limit"),
        ("max_token_upper_bound", 63, "invalid-token-limit"),
        ("max_token_upper_bound", "1024", "invalid-token-limit"),
    ],
)
def test_grant_scope_and_bounds_are_immutable_and_structurally_valid(
    field: str,
    value: object,
    code: str,
) -> None:
    with pytest.raises(GrantPolicyError, match=code):
        replace(_grant(), **{field: value})


def test_credential_digests_and_grant_ids_are_unambiguous() -> None:
    with pytest.raises(GrantPolicyError, match="invalid-credential-digest"):
        ActorCredential("not-a-sha256", AccessActor.CODEX)

    with pytest.raises(GrantPolicyError, match="duplicate-grant-id"):
        CredentialRegistry(
            policy=GrantPolicy(),
            actor_credentials=(
                ActorCredential.from_plaintext(ACTOR_TOKEN, AccessActor.CODEX),
            ),
            grant_credentials=(
                GrantCredential.from_plaintext(GRANT_TOKEN, _grant()),
                GrantCredential.from_plaintext(
                    "synthetic-second-task-grant",
                    replace(_grant(), task_id="task-synthetic-second"),
                ),
            ),
            clock=lambda: NOW,
        )


def test_policy_rejects_non_owner_personal_scope_sensitive_core_and_c6_writes() -> None:
    policy = GrantPolicy()
    invalid = (
        replace(_grant(), issued_by="process:dashboard-writer"),
        _grant(
            domains=frozenset({KnowledgeDomain.CORE_ENGINEERING}),
            privacy_classes=frozenset({PrivacyClass.SENSITIVE}),
        ),
        _grant(capabilities=frozenset({Capability.READ, Capability.PUBLICATION})),
        _grant(
            actor=AccessActor.ASHERON,
            domains=frozenset({KnowledgeDomain.FINANCE}),
        ),
    )

    for grant in invalid:
        with pytest.raises(GrantPolicyError):
            policy.validate(grant)


def test_report_capability_cannot_exist_without_read() -> None:
    with pytest.raises(GrantPolicyError, match="report-requires-read"):
        GrantPolicy().validate(
            _grant(capabilities=frozenset({Capability.REPORT_GENERATION}))
        )
