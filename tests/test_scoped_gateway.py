"""C6 protected Gateway, privacy-first aggregation, and QMD parity."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from catalog_fixtures import write_note
from ckp.app import create_app
from ckp.auth import (
    AccessActor,
    AccessDenied,
    ActorCredential,
    Capability,
    CredentialRegistry,
    DomainBinding,
    DomainRegistry,
    GrantCredential,
    GrantPolicy,
    KnowledgeDomain,
    ScopeGrant,
)
from ckp.config import load_config
from ckp.privacy import PrivacyClass

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
ACTOR_TOKEN = "synthetic-http-actor-codex"
CLAUDE_TOKEN = "synthetic-http-actor-claude"
GRANT_TOKEN = "synthetic-http-grant-finance"
BUNDLE_COMMIT = "a" * 40

FINANCE_PUBLIC_ID = "00000000-0000-7000-8000-000000000001"
FINANCE_INTERNAL_ID = "00000000-0000-7000-8000-000000000002"
FINANCE_SENSITIVE_ID = "00000000-0000-7000-8000-000000000003"
CALENDAR_SENSITIVE_ID = "00000000-0000-7000-8000-000000000004"
STUDENT_ID = "00000000-0000-7000-8000-000000000005"


def _write_bundle(root: Path) -> DomainRegistry:
    root.mkdir(parents=True)
    (root / ".bundle-commit").write_text(BUNDLE_COMMIT + "\n", encoding="utf-8")
    write_note(
        root,
        "finance-public.md",
        privacy="public",
        title="Synthetic Finance Public",
        note_type="Concept",
        note_id=FINANCE_PUBLIC_ID,
        body="scope-needle amount TWD 1200 category synthetic utilities.",
    )
    write_note(
        root,
        "finance-internal.md",
        privacy="internal",
        title="Synthetic Finance Internal",
        note_type="Source",
        note_id=FINANCE_INTERNAL_ID,
        body="scope-needle synthetic monthly summary without account identifiers.",
    )
    write_note(
        root,
        "finance-sensitive.md",
        privacy="sensitive",
        title="Synthetic Finance Sensitive",
        note_type="Decision",
        note_id=FINANCE_SENSITIVE_ID,
        body="scope-needle synthetic private allocation summary.",
    )
    write_note(
        root,
        "calendar-sensitive.md",
        privacy="sensitive",
        title="Synthetic Calendar Hidden",
        note_type="Procedure",
        note_id=CALENDAR_SENSITIVE_ID,
        body="scope-needle CALENDAR-CROSS-DOMAIN-SENTINEL event-id synthetic-42.",
    )
    write_note(
        root,
        "student.md",
        privacy="student-private",
        title="Invented Student Hidden",
        note_type="Student Reference",
        note_id=STUDENT_ID,
        body="scope-needle STUDENT-PRIVATE-SENTINEL entirely invented fixture.",
    )
    write_note(
        root,
        "unmapped-sensitive.md",
        privacy="sensitive",
        title="Synthetic Unmapped Hidden",
        note_type="Concept",
        note_id="00000000-0000-7000-8000-000000000006",
        body="scope-needle UNMAPPED-SENTINEL.",
    )
    return DomainRegistry(
        (
            DomainBinding(FINANCE_PUBLIC_ID, KnowledgeDomain.FINANCE),
            DomainBinding(FINANCE_INTERNAL_ID, KnowledgeDomain.FINANCE),
            DomainBinding(FINANCE_SENSITIVE_ID, KnowledgeDomain.FINANCE),
            DomainBinding(CALENDAR_SENSITIVE_ID, KnowledgeDomain.CALENDAR_CONTEXT),
            DomainBinding(STUDENT_ID, KnowledgeDomain.FINANCE),
        )
    )


def _grant(
    *,
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
    expires_at: datetime = NOW + timedelta(hours=1),
) -> ScopeGrant:
    return ScopeGrant(
        grant_id="grant-synthetic-http-finance",
        actor=AccessActor.CODEX,
        task_id="task-synthetic-http-finance",
        domains=frozenset({KnowledgeDomain.FINANCE}),
        capabilities=capabilities,
        privacy_classes=privacy_classes,
        issued_by="human:cyclone",
        issued_at=NOW - timedelta(minutes=1),
        expires_at=expires_at,
        max_items=2,
        max_token_upper_bound=1_024,
    )


def _client(
    root: Path,
    *,
    grant: ScopeGrant | None = None,
) -> tuple[TestClient, CredentialRegistry]:
    domains = _write_bundle(root)
    registry = CredentialRegistry(
        policy=GrantPolicy(),
        actor_credentials=(
            ActorCredential.from_plaintext(ACTOR_TOKEN, AccessActor.CODEX),
            ActorCredential.from_plaintext(CLAUDE_TOKEN, AccessActor.CLAUDE_CODE),
        ),
        grant_credentials=(
            GrantCredential.from_plaintext(GRANT_TOKEN, grant or _grant()),
        ),
        clock=lambda: NOW,
    )
    config = load_config(env={"CKP_BUNDLE_ROOT": str(root)})
    return (
        TestClient(
            create_app(
                config,
                credential_registry=registry,
                domain_registry=domains,
            )
        ),
        registry,
    )


def _headers(*, actor: str = ACTOR_TOKEN, grant: str = GRANT_TOKEN) -> dict[str, str]:
    return {
        "X-CKP-Actor-Credential": actor,
        "X-CKP-Grant-Credential": grant,
    }


def _assert_sanitized_reject(response, code: str) -> None:
    assert response.json().keys() == {"request_id", "status", "code"}
    assert response.json()["status"] == "rejected"
    assert response.json()["code"] == code
    assert re.fullmatch(r"[0-9a-f]{32}", response.json()["request_id"])


def test_scoped_read_filters_privacy_and_domain_before_every_projection(
    tmp_path: Path,
) -> None:
    client, _ = _client(tmp_path / "bundle")

    catalog = client.get(
        "/scoped/catalog/finance", params={"limit": 100}, headers=_headers()
    )
    query = client.post(
        "/scoped/query/finance",
        json={"query": "scope-needle", "limit": 20},
        headers=_headers(),
    )
    context = client.post(
        "/context/finance",
        json={"query": "scope-needle"},
        headers=_headers(),
    )

    assert catalog.status_code == query.status_code == context.status_code == 200
    assert catalog.json()["total"] == 3
    assert query.json()["total"] == 3
    assert catalog.json()["facets"] == {
        "types": [
            {"value": "Concept", "count": 1},
            {"value": "Decision", "count": 1},
            {"value": "Source", "count": 1},
        ],
        "content_categories": [{"value": "development", "count": 3}],
    }
    rendered = catalog.text + query.text + context.text
    for forbidden in (
        "CALENDAR-CROSS-DOMAIN-SENTINEL",
        "Synthetic Calendar Hidden",
        "STUDENT-PRIVATE-SENTINEL",
        "Invented Student Hidden",
        "UNMAPPED-SENTINEL",
        "Synthetic Unmapped Hidden",
    ):
        assert forbidden not in rendered

    packed = context.json()
    assert len(packed["items"]) <= 2
    assert packed["token_upper_bound_used"] <= packed["token_upper_bound_limit"]
    assert packed["token_upper_bound_method"] == "utf8-bytes-v1"
    assert packed["revision"] == query.json()["revision"]
    for item in packed["items"]:
        assert item["citation"]["bundle_commit"] == BUNDLE_COMMIT
        assert (
            item["citation"]["index_revision"] == packed["revision"]["index_revision"]
        )


def test_lower_privacy_grant_cannot_leak_sensitive_counts_facets_or_citations(
    tmp_path: Path,
) -> None:
    client, _ = _client(
        tmp_path / "bundle",
        grant=_grant(
            privacy_classes=frozenset({PrivacyClass.PUBLIC, PrivacyClass.INTERNAL})
        ),
    )

    catalog = client.get(
        "/scoped/catalog/finance", params={"limit": 100}, headers=_headers()
    )
    query = client.post(
        "/scoped/query/finance",
        json={"query": "scope-needle", "limit": 20},
        headers=_headers(),
    )

    assert catalog.json()["total"] == query.json()["total"] == 2
    assert catalog.json()["facets"]["types"] == [
        {"value": "Concept", "count": 1},
        {"value": "Source", "count": 1},
    ]
    assert "Synthetic Finance Sensitive" not in catalog.text + query.text
    assert "finance-sensitive.md" not in catalog.text + query.text


def test_wrong_actor_expired_missing_capability_and_cross_domain_fail_closed(
    tmp_path: Path,
) -> None:
    client, _ = _client(tmp_path / "wrong-actor")
    wrong = client.post(
        "/context/finance",
        json={"query": "REJECT-QUERY-SENTINEL"},
        headers=_headers(actor=CLAUDE_TOKEN),
    )
    _assert_sanitized_reject(wrong, "wrong-actor")
    assert "REJECT-QUERY-SENTINEL" not in wrong.text
    assert CLAUDE_TOKEN not in wrong.text

    cross = client.get("/scoped/catalog/calendar-context", headers=_headers())
    _assert_sanitized_reject(cross, "domain-denied")

    expired_client, _ = _client(tmp_path / "expired", grant=_grant(expires_at=NOW))
    expired = expired_client.get("/scoped/catalog/finance", headers=_headers())
    _assert_sanitized_reject(expired, "scope-expired")

    read_only_client, _ = _client(
        tmp_path / "read-only",
        grant=_grant(capabilities=frozenset({Capability.READ})),
    )
    missing_capability = read_only_client.post(
        "/context/finance",
        json={"query": "scope-needle"},
        headers=_headers(),
    )
    _assert_sanitized_reject(missing_capability, "capability-denied")


def test_request_cannot_add_scope_and_validation_receipt_does_not_echo_query(
    tmp_path: Path,
) -> None:
    client, _ = _client(tmp_path / "bundle")

    response = client.post(
        "/context/finance",
        json={
            "query": "VALIDATION-QUERY-SENTINEL",
            "privacy": "sensitive",
            "domains": ["finance", "calendar-context"],
            "max_items": 1000,
        },
        headers=_headers(),
    )

    assert response.status_code == 422
    _assert_sanitized_reject(response, "invalid-request")
    assert "VALIDATION-QUERY-SENTINEL" not in response.text
    assert "sensitive" not in response.text


def test_gateway_and_offline_qmd_use_the_same_scoped_catalog(tmp_path: Path) -> None:
    client, _registry = _client(tmp_path / "bundle")
    offline = client.app.state.offline_qmd_scope.plan(
        actor_credential=ACTOR_TOKEN,
        grant_credential=GRANT_TOKEN,
        domain=KnowledgeDomain.FINANCE,
    )

    catalog = client.get(
        "/scoped/catalog/finance", params={"limit": 100}, headers=_headers()
    ).json()

    assert (
        client.app.state.offline_qmd_scope.selector
        is client.app.state.scoped_gateway.selector
    )
    assert set(offline.paths) == {item["citation"]["path"] for item in catalog["items"]}
    assert offline.index_revision == catalog["revision"]["index_revision"]
    assert offline.bundle_commit == catalog["revision"]["bundle_commit"]
    assert offline.domain is KnowledgeDomain.FINANCE
    assert offline.actor == "codex"
    assert offline.grant_id == "grant-synthetic-http-finance"
    assert offline.task_id == "task-synthetic-http-finance"
    assert offline.is_valid_at(NOW)
    assert not offline.is_valid_at(NOW + timedelta(hours=2))


def test_offline_qmd_revalidates_actor_domain_and_expiry(tmp_path: Path) -> None:
    client, _ = _client(tmp_path / "valid")
    for actor, domain, code in (
        (CLAUDE_TOKEN, KnowledgeDomain.FINANCE, "wrong-actor"),
        (ACTOR_TOKEN, KnowledgeDomain.CALENDAR_CONTEXT, "domain-denied"),
    ):
        with pytest.raises(AccessDenied, match=code):
            client.app.state.offline_qmd_scope.plan(
                actor_credential=actor,
                grant_credential=GRANT_TOKEN,
                domain=domain,
            )

    expired_client, _ = _client(
        tmp_path / "expired-offline", grant=_grant(expires_at=NOW)
    )
    with pytest.raises(AccessDenied, match="scope-expired"):
        expired_client.app.state.offline_qmd_scope.plan(
            actor_credential=ACTOR_TOKEN,
            grant_credential=GRANT_TOKEN,
            domain=KnowledgeDomain.FINANCE,
        )


def test_scoped_read_requires_bundle_commit_but_public_c3_remains_available(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    client, _ = _client(root)
    (root / ".bundle-commit").unlink()

    scoped = client.get("/scoped/catalog/finance", headers=_headers())
    public = client.get("/catalog")

    assert scoped.status_code == 503
    assert scoped.json() == {"detail": "bundle-unavailable"}
    assert public.status_code == 200
    assert public.json()["revision"]["bundle_commit"] is None


def test_duplicate_durable_id_makes_scoped_snapshot_unavailable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    client, _ = _client(root)
    write_note(
        root,
        "duplicate-id.md",
        privacy="internal",
        title="Synthetic Duplicate ID",
        note_id=FINANCE_PUBLIC_ID,
        body="DUPLICATE-ID-SENTINEL",
    )

    response = client.get("/scoped/catalog/finance", headers=_headers())

    assert response.status_code == 503
    assert response.json() == {"detail": "bundle-unavailable"}
    assert "DUPLICATE-ID-SENTINEL" not in response.text


def test_anonymous_c3_routes_remain_public_only_even_with_scope_headers(
    tmp_path: Path,
) -> None:
    client, _ = _client(tmp_path / "bundle")

    response = client.post(
        "/query",
        json={"query": "scope-needle", "limit": 20},
        headers=_headers(),
    )

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert [item["title"] for item in response.json()["results"]] == [
        "Synthetic Finance Public"
    ]


def test_default_app_has_no_standing_personal_grants(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    config = load_config(env={"CKP_BUNDLE_ROOT": str(root)})
    client = TestClient(create_app(config))

    response = client.post("/context/finance", json={"query": "scope-needle"})

    assert response.status_code == 401
    _assert_sanitized_reject(response, "missing-credential")
