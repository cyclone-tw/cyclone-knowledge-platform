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

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
from ckp.privacy import FrontmatterClassifier, PrivacyClass, PrivacyGate
from ckp.revision import API_VERSION, build_revision

# C6 authentication/scope does not exist yet.  The anonymous HTTP surface is
# therefore public-only and the caller gets no request field that can widen it.
PUBLIC_HTTP_ADMISSIBLE = frozenset({PrivacyClass.PUBLIC})


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


def create_app(config: Config | None = None) -> FastAPI:
    resolved = load_config() if config is None else config

    app = FastAPI(
        title="Cyclone Knowledge Platform",
        version=API_VERSION,
        summary="Phase 3 read-only Catalog and Knowledge Gateway.",
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
    app.state.snapshot_cache = snapshot_cache
    app.state.privacy_gate = privacy_gate
    app.state.catalog_builder = catalog_builder
    app.state.gateway = gateway

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

    return app
