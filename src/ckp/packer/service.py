"""Provider-neutral, deterministic bounded context packing.

`utf8-bytes-v1` counts the exact UTF-8 bytes of the context string. Byte-level
LLM tokenizers cannot require more tokens than input bytes, so this is a
portable conservative upper bound without coupling C6 to one model vendor.
"""

from __future__ import annotations

import re

from ckp.gateway.models import QueryResultResponse, SnapshotRef
from ckp.packer.models import ContextItemResponse, ContextResponse
from ckp.privacy import PrivacyGate

TOKEN_UPPER_BOUND_METHOD = "utf8-bytes-v1"
_WHITESPACE = re.compile(r"\s+")


def _compact(value: str | None) -> str:
    return _WHITESPACE.sub(" ", value or "").strip()


def _render(index: int, item: ContextItemResponse) -> str:
    citation = item.citation
    bundle_commit = citation.bundle_commit
    if bundle_commit is None:
        raise ValueError("missing-bundle-commit")
    title = _compact(item.title) or item.concept_id
    snippet = _compact(item.snippet)
    return (
        f"[{index}] {title}\n"
        f"{snippet}\n"
        "citation: "
        f"concept_id={citation.concept_id}; "
        f"path={citation.path}; "
        f"bundle_commit={bundle_commit}; "
        f"index_revision={citation.index_revision}"
    )


def _as_context_item(result: QueryResultResponse) -> ContextItemResponse:
    return ContextItemResponse(
        concept_id=result.concept_id,
        id=result.id,
        title=result.title,
        snippet=_compact(result.snippet),
        citation=result.citation,
    )


def _fit_item(
    item: ContextItemResponse,
    *,
    index: int,
    byte_budget: int,
) -> tuple[ContextItemResponse, str, bool] | None:
    rendered = _render(index, item)
    if len(rendered.encode("utf-8")) <= byte_budget:
        return item, rendered, False

    # Keep citation completeness non-negotiable. If the citation-only block
    # cannot fit, omit the whole item instead of emitting an uncited fragment.
    minimum = item.model_copy(update={"snippet": ""})
    if len(_render(index, minimum).encode("utf-8")) > byte_budget:
        return None

    source = item.snippet
    low = 0
    high = len(source)
    best = minimum
    best_rendered = _render(index, minimum)
    while low <= high:
        midpoint = (low + high) // 2
        prefix = source[:midpoint].rstrip()
        snippet = prefix + ("…" if midpoint < len(source) else "")
        candidate = item.model_copy(update={"snippet": snippet})
        candidate_rendered = _render(index, candidate)
        if len(candidate_rendered.encode("utf-8")) <= byte_budget:
            best = candidate
            best_rendered = candidate_rendered
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best, best_rendered, True


class BoundedContextPacker:
    """Pack ranked results under server-supplied item and byte-token caps."""

    def __init__(self, privacy_gate: PrivacyGate) -> None:
        # The packer is itself a read projection. Requiring the exact gate
        # chosen by ScopedCatalogSelector keeps red line R1 explicit at this
        # boundary instead of treating "upstream probably filtered" as proof.
        if not isinstance(privacy_gate, PrivacyGate):
            raise TypeError("privacy_gate must be a PrivacyGate")
        self._privacy_gate = privacy_gate

    @property
    def privacy_gate(self) -> PrivacyGate:
        return self._privacy_gate

    def pack(
        self,
        *,
        domain: str,
        revision: SnapshotRef,
        results: list[QueryResultResponse],
        total_results: int,
        max_items: int,
        max_token_upper_bound: int,
    ) -> ContextResponse:
        if max_items < 1 or max_token_upper_bound < 1:
            raise ValueError("context bounds must be positive")
        if total_results < len(results):
            raise ValueError("total_results cannot be smaller than supplied results")
        if revision.bundle_commit is None:
            raise ValueError("missing-bundle-commit")
        for result in results:
            citation = result.citation
            if (
                citation.concept_id != result.concept_id
                or citation.index_revision != revision.index_revision
                or citation.bundle_commit != revision.bundle_commit
            ):
                raise ValueError("citation-revision-mismatch")

        items: list[ContextItemResponse] = []
        blocks: list[str] = []
        snippet_truncated = False
        for result in results[:max_items]:
            separator_bytes = 2 if blocks else 0
            used = len("\n\n".join(blocks).encode("utf-8"))
            remaining = max_token_upper_bound - used - separator_bytes
            fitted = _fit_item(
                _as_context_item(result),
                index=len(items) + 1,
                byte_budget=remaining,
            )
            if fitted is None:
                break
            item, rendered, was_truncated = fitted
            items.append(item)
            blocks.append(rendered)
            snippet_truncated = snippet_truncated or was_truncated

        context = "\n\n".join(blocks)
        used = len(context.encode("utf-8"))
        return ContextResponse(
            revision=revision,
            domain=domain,
            items=items,
            context=context,
            item_limit=max_items,
            token_upper_bound_limit=max_token_upper_bound,
            token_upper_bound_used=used,
            token_upper_bound_method=TOKEN_UPPER_BOUND_METHOD,
            truncated=snippet_truncated or total_results > len(items),
        )


__all__ = ["BoundedContextPacker", "TOKEN_UPPER_BOUND_METHOD"]
