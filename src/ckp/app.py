"""HTTP surface: health/revision plus the read-only C3 Gateway.

The walking skeleton deliberately uses the framework the Gateway will use, so
later children extend this app rather than replace it. The API schema this
repo owns (contract §4) is what FastAPI publishes from these models.

``/health`` answers "is this process able to serve"; ``/revision`` answers
"what exactly is it serving". They are separate because a running process
with an unreadable bundle is a real state, and conflating the two would let it
report healthy.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ckp.auth import (
    AccessContext,
    AccessDenied,
    Capability,
    CredentialRegistry,
    DomainPolicyError,
    DomainRegistry,
    KnowledgeDomain,
)
from ckp.bundle import AnchoredBundleReader, SnapshotCache, SnapshotRaceError
from ckp.catalog import (
    CatalogBuilder,
    CatalogUnavailableError,
    PrivacyBindingError,
)
from ckp.config import Config, load_config
from ckp.gateway import (
    CatalogRequest,
    CatalogResponse,
    KnowledgeGateway,
    QueryRequest,
    QueryResponse,
)
from ckp.gateway.scoped import (
    OfflineQmdScope,
    ScopedCatalogSelector,
    ScopedKnowledgeGateway,
)
from ckp.packer import BoundedContextPacker, ContextRequest, ContextResponse
from ckp.privacy import FrontmatterClassifier, PrivacyClass, PrivacyGate
from ckp.revision import API_VERSION, build_revision

# C6 adds separate protected routes. The anonymous C3 surface stays
# public-only, and no caller field can widen that fixed policy.
PUBLIC_HTTP_ADMISSIBLE = frozenset({PrivacyClass.PUBLIC})
_READ = frozenset({Capability.READ})
_READ_AND_REPORT = frozenset({Capability.READ, Capability.REPORT_GENERATION})


class HealthChecks(BaseModel):
    config_loaded: bool = Field(description="Config layers resolved without error")
    bundle_readable: bool = Field(
        description="Bundle directory yields at least one note"
    )


class HealthResponse(BaseModel):
    status: str = Field(description="ok when every check passes, else degraded")
    checks: HealthChecks
    config_layers: list[str] = Field(
        description="Config layers that contributed, most general first"
    )


class RevisionResponse(BaseModel):
    """The four fields required by OKF contract §3 Phase 3.

    Any field may be ``null``: absent evidence is reported as absent rather
    than filled with a placeholder.
    """

    profile_version: str | None
    api_version: str
    bundle_commit: str | None
    index_revision: str | None
    sources: dict[str, str] = Field(
        description="Where each field came from, so self-declared and derived "
        "evidence stay distinguishable"
    )


def create_app(
    config: Config | None = None,
    *,
    credential_registry: CredentialRegistry | None = None,
    domain_registry: DomainRegistry | None = None,
) -> FastAPI:
    resolved = load_config() if config is None else config

    app = FastAPI(
        title="Cyclone Knowledge Platform",
        version=API_VERSION,
        summary="Phase 3 public and task-scoped read-only Knowledge Gateway.",
    )
    app.state.config = resolved
    reader = AnchoredBundleReader(resolved.bundle_root)
    snapshot_cache = SnapshotCache(
        reader,
        resolved.bundle_note_glob,
        resolved.profile_expected_version,
    )
    classifier = FrontmatterClassifier(reader)
    privacy_gate = PrivacyGate(classifier, PUBLIC_HTTP_ADMISSIBLE)
    catalog_builder = CatalogBuilder(privacy_gate)
    gateway = KnowledgeGateway(snapshot_cache, catalog_builder, privacy_gate)
    resolved_credentials = credential_registry or CredentialRegistry.deny_all(
        lambda: datetime.now(UTC)
    )
    resolved_domains = domain_registry or DomainRegistry(())
    scoped_selector = ScopedCatalogSelector(
        snapshot_cache,
        classifier,
        resolved_domains,
    )
    scoped_gateway = ScopedKnowledgeGateway(
        scoped_selector,
        BoundedContextPacker,
    )
    offline_qmd_scope = OfflineQmdScope(scoped_selector, resolved_credentials)
    app.state.snapshot_cache = snapshot_cache
    app.state.privacy_gate = privacy_gate
    app.state.catalog_builder = catalog_builder
    app.state.gateway = gateway
    app.state.credential_registry = resolved_credentials
    app.state.domain_registry = resolved_domains
    app.state.scoped_selector = scoped_selector
    app.state.scoped_gateway = scoped_gateway
    app.state.offline_qmd_scope = offline_qmd_scope

    def rejection(code: str, status_code: int) -> JSONResponse:
        # Server-generated ID plus a stable code is the whole reject receipt.
        # Never serialize exceptions, request bodies, headers, or note metadata.
        return JSONResponse(
            status_code=status_code,
            content={
                "request_id": secrets.token_hex(16),
                "status": "rejected",
                "code": code,
            },
        )

    @app.exception_handler(AccessDenied)
    async def access_denied_handler(
        _request: Request, error: AccessDenied
    ) -> JSONResponse:
        return rejection(error.code, error.http_status)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        # FastAPI's default detail includes rejected input values. A query can
        # itself be sensitive, so C6 returns a body-free machine receipt.
        return rejection("invalid-request", 422)

    def resolve_access(
        domain: KnowledgeDomain,
        actor_credential: str | None,
        grant_credential: str | None,
        required_capabilities: frozenset[Capability],
    ) -> AccessContext:
        return resolved_credentials.resolve(
            actor_credential=actor_credential,
            grant_credential=grant_credential,
            requested_domain=domain,
            required_capabilities=required_capabilities,
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> JSONResponse:
        cfg: Config = app.state.config
        try:
            readable = app.state.snapshot_cache.get().index_revision is not None
        except SnapshotRaceError:
            readable = False
        payload = HealthResponse(
            status="ok" if readable else "degraded",
            checks=HealthChecks(config_loaded=True, bundle_readable=readable),
            config_layers=list(cfg.layers),
        )
        # A process that cannot read its bundle is up but not serving. Say so
        # with a status code a load balancer already understands.
        return JSONResponse(
            status_code=200 if readable else 503,
            content=payload.model_dump(),
        )

    @app.get("/revision", response_model=RevisionResponse)
    def revision() -> RevisionResponse:
        return RevisionResponse(
            **build_revision(app.state.config, app.state.snapshot_cache).as_dict()
        )

    def unavailable() -> HTTPException:
        return HTTPException(status_code=503, detail="bundle-unavailable")

    def protected_unavailable() -> JSONResponse:
        return rejection("bundle-unavailable", 503)

    @app.get("/catalog", response_model=CatalogResponse)
    def catalog(
        request: Annotated[CatalogRequest, Query()],
    ) -> CatalogResponse:
        try:
            return app.state.gateway.catalog(request)
        except (CatalogUnavailableError, PrivacyBindingError, SnapshotRaceError):
            raise unavailable() from None

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        try:
            return app.state.gateway.query(request)
        except (CatalogUnavailableError, PrivacyBindingError, SnapshotRaceError):
            raise unavailable() from None

    @app.get("/scoped/catalog/{domain}", response_model=CatalogResponse)
    def scoped_catalog(
        domain: KnowledgeDomain,
        request: Annotated[CatalogRequest, Query()],
        actor_credential: Annotated[
            str | None, Header(alias="X-CKP-Actor-Credential")
        ] = None,
        grant_credential: Annotated[
            str | None, Header(alias="X-CKP-Grant-Credential")
        ] = None,
    ) -> CatalogResponse:
        context = resolve_access(
            domain,
            actor_credential,
            grant_credential,
            _READ,
        )
        try:
            return app.state.scoped_gateway.catalog(context, request)
        except (
            CatalogUnavailableError,
            DomainPolicyError,
            PrivacyBindingError,
            SnapshotRaceError,
        ):
            return protected_unavailable()

    @app.post("/scoped/query/{domain}", response_model=QueryResponse)
    def scoped_query(
        domain: KnowledgeDomain,
        request: QueryRequest,
        actor_credential: Annotated[
            str | None, Header(alias="X-CKP-Actor-Credential")
        ] = None,
        grant_credential: Annotated[
            str | None, Header(alias="X-CKP-Grant-Credential")
        ] = None,
    ) -> QueryResponse:
        context = resolve_access(
            domain,
            actor_credential,
            grant_credential,
            _READ,
        )
        try:
            return app.state.scoped_gateway.query(context, request)
        except (
            CatalogUnavailableError,
            DomainPolicyError,
            PrivacyBindingError,
            SnapshotRaceError,
        ):
            return protected_unavailable()

    @app.post("/context/{domain}", response_model=ContextResponse)
    def context(
        domain: KnowledgeDomain,
        request: ContextRequest,
        actor_credential: Annotated[
            str | None, Header(alias="X-CKP-Actor-Credential")
        ] = None,
        grant_credential: Annotated[
            str | None, Header(alias="X-CKP-Grant-Credential")
        ] = None,
    ) -> ContextResponse:
        access = resolve_access(
            domain,
            actor_credential,
            grant_credential,
            _READ_AND_REPORT,
        )
        try:
            return app.state.scoped_gateway.context(access, request)
        except (
            CatalogUnavailableError,
            DomainPolicyError,
            PrivacyBindingError,
            SnapshotRaceError,
        ):
            return protected_unavailable()

    return app
