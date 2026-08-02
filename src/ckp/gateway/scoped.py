"""C6 authenticated scoped reads and offline QMD policy projection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ckp.auth import (
    AccessContext,
    AccessDenied,
    Capability,
    CredentialRegistry,
    DomainPolicyError,
    DomainRegistry,
    KnowledgeDomain,
)
from ckp.bundle import SnapshotCache
from ckp.catalog import CatalogBuilder, CatalogSnapshot
from ckp.gateway.models import (
    CatalogRequest,
    CatalogResponse,
    QueryRequest,
    QueryResponse,
)
from ckp.gateway.service import catalog_response, query_response
from ckp.packer import (
    BoundedContextPacker,
    ContextRequest,
    ContextResponse,
)
from ckp.privacy import Classifier, PrivacyGate

_READ = frozenset({Capability.READ})
_READ_AND_REPORT = frozenset({Capability.READ, Capability.REPORT_GENERATION})


def _require(context: AccessContext, required: frozenset[Capability]) -> None:
    if not required <= context.capabilities:
        raise AccessDenied("capability-denied")


class ScopedCatalogSelector:
    """Privacy first, then server-owned domain binding, on one snapshot."""

    def __init__(
        self,
        snapshot_cache: SnapshotCache,
        classifier: Classifier,
        domain_registry: DomainRegistry,
    ) -> None:
        self._snapshot_cache = snapshot_cache
        self._classifier = classifier
        self._domain_registry = domain_registry

    def select_with_gate(
        self, context: AccessContext
    ) -> tuple[CatalogSnapshot, PrivacyGate]:
        _require(context, _READ)
        snapshot = self._snapshot_cache.get()
        # Build the Catalog through a per-grant PrivacyGate before the domain
        # registry, search, counts, facets, snippets, or citations see notes.
        gate = PrivacyGate(self._classifier, context.privacy_classes)
        admitted = CatalogBuilder(gate).build(snapshot)
        # Protected reads must be attributable to one durable bundle commit.
        # The anonymous C3 surface may honestly report null evidence, but a
        # personal context result without commit provenance is unusable.
        if admitted.bundle_commit is None:
            raise DomainPolicyError("missing-bundle-commit")
        scoped_entries = self._domain_registry.select(admitted.entries, context.domain)
        return (
            CatalogSnapshot(
                entries=scoped_entries,
                index_revision=admitted.index_revision,
                bundle_commit=admitted.bundle_commit,
            ),
            gate,
        )

    def select(self, context: AccessContext) -> CatalogSnapshot:
        catalog, _gate = self.select_with_gate(context)
        return catalog


class ScopedKnowledgeGateway:
    def __init__(
        self,
        selector: ScopedCatalogSelector,
        packer_type: type[BoundedContextPacker],
    ) -> None:
        self._selector = selector
        self._packer_type = packer_type

    @property
    def selector(self) -> ScopedCatalogSelector:
        return self._selector

    def catalog(
        self, context: AccessContext, request: CatalogRequest
    ) -> CatalogResponse:
        _require(context, _READ)
        return catalog_response(self._selector.select(context), request)

    def query(self, context: AccessContext, request: QueryRequest) -> QueryResponse:
        _require(context, _READ)
        return query_response(self._selector.select(context), request)

    def context(
        self, context: AccessContext, request: ContextRequest
    ) -> ContextResponse:
        _require(context, _READ_AND_REPORT)
        # The caller supplies no limit. The server-owned grant chooses the
        # upper bounds; QueryRequest's fixed limit is only an internal ranking
        # window and can never widen the final context.
        catalog, privacy_gate = self._selector.select_with_gate(context)
        ranked = query_response(
            catalog,
            QueryRequest(query=request.query, limit=20),
        )
        return self._packer_type(privacy_gate).pack(
            domain=context.domain.value,
            revision=ranked.revision,
            results=ranked.results,
            total_results=ranked.total,
            max_items=context.max_items,
            max_token_upper_bound=context.max_token_upper_bound,
        )


@dataclass(frozen=True, slots=True)
class QmdScopePlan:
    actor: str
    grant_id: str
    task_id: str
    domain: KnowledgeDomain
    paths: tuple[str, ...]
    index_revision: str
    bundle_commit: str | None
    expires_at: datetime

    def is_valid_at(self, now: datetime) -> bool:
        if now.tzinfo is None:
            raise ValueError("QMD scope checks require a timezone-aware time")
        return now < self.expires_at


class OfflineQmdScope:
    """Emit an allowlist from the exact selector used by the Gateway."""

    def __init__(
        self,
        selector: ScopedCatalogSelector,
        credential_registry: CredentialRegistry,
    ) -> None:
        self._selector = selector
        self._credential_registry = credential_registry

    @property
    def selector(self) -> ScopedCatalogSelector:
        return self._selector

    def plan(
        self,
        *,
        actor_credential: str | None,
        grant_credential: str | None,
        domain: KnowledgeDomain,
    ) -> QmdScopePlan:
        # Do not accept a cached or caller-built AccessContext. Every offline
        # invocation resolves the same opaque credentials as the Gateway, so
        # revocation-by-registry, actor binding, domain and expiry stay equal.
        context = self._credential_registry.resolve(
            actor_credential=actor_credential,
            grant_credential=grant_credential,
            requested_domain=domain,
            required_capabilities=_READ,
        )
        catalog = self._selector.select(context)
        return QmdScopePlan(
            actor=context.actor.value,
            grant_id=context.grant_id,
            task_id=context.task_id,
            domain=context.domain,
            paths=tuple(entry.citation.path for entry in catalog.entries),
            index_revision=catalog.index_revision,
            bundle_commit=catalog.bundle_commit,
            expires_at=context.expires_at,
        )


__all__ = [
    "OfflineQmdScope",
    "QmdScopePlan",
    "ScopedCatalogSelector",
    "ScopedKnowledgeGateway",
]
