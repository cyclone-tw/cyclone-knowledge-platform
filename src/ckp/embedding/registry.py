"""Explicit provider registry. Empty by default, fail closed on everything.

A registry that ships with builtins already registered, or that falls back to
"the default provider" when a name is unknown, is how a deployment ends up
running something nobody chose. Registration is always an explicit act, and
every rejection has a stable code.
"""

from __future__ import annotations

from ckp.embedding.errors import EmbeddingErrorCode, EmbeddingRefusal
from ckp.embedding.models import EMBEDDING_CONTRACT, ProviderDescriptor, ProviderKind
from ckp.embedding.provider import EmbeddingProvider, RerankerProvider

_NAME_MAX = 64


def _require_registrable(descriptor: ProviderDescriptor, kind: ProviderKind) -> None:
    """The Phase 3 offline boundary, checked at registration, not at query."""
    if descriptor.contract_version != EMBEDDING_CONTRACT:
        raise EmbeddingRefusal(EmbeddingErrorCode.CONTRACT_VERSION_UNKNOWN)
    if descriptor.kind is not kind:
        raise EmbeddingRefusal(EmbeddingErrorCode.PROVIDER_KIND_MISMATCH)
    if descriptor.requires_network:
        # Phase 3 non-goal: no cloud provider, no model download. This is the
        # single place that decision is enforced, so C5 cannot reach around it.
        raise EmbeddingRefusal(EmbeddingErrorCode.NETWORK_PROVIDER_DENIED)
    if not descriptor.deterministic:
        # A non-deterministic provider cannot back a recomputable
        # ``index_revision`` (contract §5.5).
        raise EmbeddingRefusal(EmbeddingErrorCode.NONDETERMINISTIC_PROVIDER_DENIED)


class ProviderRegistry:
    """Named providers, resolved only by exact name."""

    def __init__(self) -> None:
        self._embedding: dict[str, EmbeddingProvider] = {}
        self._reranker: dict[str, RerankerProvider] = {}

    def register_embedding(self, name: str, provider: EmbeddingProvider) -> None:
        self._register(self._embedding, name, provider, ProviderKind.EMBEDDING)

    def register_reranker(self, name: str, provider: RerankerProvider) -> None:
        self._register(self._reranker, name, provider, ProviderKind.RERANKER)

    def resolve_embedding(self, name: str) -> EmbeddingProvider:
        return self._resolve(self._embedding, name)

    def resolve_reranker(self, name: str) -> RerankerProvider:
        return self._resolve(self._reranker, name)

    def embedding_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._embedding))

    def reranker_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._reranker))

    def _register(
        self,
        slot: dict[str, object],
        name: str,
        provider: object,
        kind: ProviderKind,
    ) -> None:
        if not isinstance(name, str) or not name or len(name) > _NAME_MAX:
            raise EmbeddingRefusal(EmbeddingErrorCode.PROVIDER_UNKNOWN)
        if kind is ProviderKind.EMBEDDING:
            expected: type = EmbeddingProvider
        else:
            expected = RerankerProvider
        if not isinstance(provider, expected):
            raise EmbeddingRefusal(EmbeddingErrorCode.PROVIDER_KIND_MISMATCH)
        descriptor = getattr(provider, "descriptor", None)
        if not isinstance(descriptor, ProviderDescriptor):
            raise EmbeddingRefusal(EmbeddingErrorCode.DESCRIPTOR_INVALID)
        _require_registrable(descriptor, kind)
        if name in slot:
            # Silent override would make the winner depend on import order.
            raise EmbeddingRefusal(EmbeddingErrorCode.PROVIDER_DUPLICATE)
        slot[name] = provider

    @staticmethod
    def _resolve(slot: dict[str, object], name: str):
        if not isinstance(name, str) or name not in slot:
            # No default, no nearest match, no fallback.
            raise EmbeddingRefusal(EmbeddingErrorCode.PROVIDER_UNKNOWN)
        return slot[name]


__all__ = [
    "ProviderRegistry",
]
