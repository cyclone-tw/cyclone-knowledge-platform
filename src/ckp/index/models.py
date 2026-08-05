"""Frozen ``index/v1`` shapes and the one rebuild planner.

Privacy gating, embedding, ordering, and revision composition happen exactly
once, in :func:`plan_rebuild` -- a provider receives finished points and can
neither widen the privacy sink nor reorder the corpus (C8 lesson: reuse, do
not duplicate). Nothing here knows the name of any concrete index provider.

A search result is a *total* order (score descending, then ``relative_path``
ascending), enforced by the model rather than trusted per provider, mirroring
the C4 rerank contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ckp.bundle import BundleMember, compute_index_revision_from_members
from ckp.embedding import EmbeddingRefusal, EmbeddingStack
from ckp.index.errors import IndexErrorCode, IndexRefusal
from ckp.index.revision import (
    INDEX_SCHEMA_VERSION,
    compute_composed_index_revision,
)
from ckp.privacy import PrivacyClass
from ckp.privacy.gate import Admitted, PrivacyGate

#: Bumped only by a breaking change to the shapes in this module.
INDEX_CONTRACT = "index/v1"

_POINT_DOMAIN = b"ckp-index-point-v1"
_PAYLOAD_DOMAIN = b"ckp-index-payload-v1"


class IndexDescriptor(BaseModel):
    """Everything a consumer needs to know about an index provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: str = Field(min_length=1)
    provider_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9._-]*$"
    )
    provider_version: str = Field(
        min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9._-]*$"
    )
    schema_version: str = Field(min_length=1, max_length=16)


@dataclass(frozen=True)
class IndexedPoint:
    """One prepared point: identity, evidence binding, vector. No note body."""

    point_id: str
    digest_key: str
    relative_path: str
    content_sha256: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class RebuildPlan:
    """Everything a provider needs to rebuild from an empty volume.

    ``payload_digest`` is the determinism oracle: two plans built from the
    same commit with the same stack must produce byte-identical digests, and
    a provider's post-rebuild verification recomputes it from what was
    actually stored.
    """

    points: tuple[IndexedPoint, ...]
    dimension: int
    composed_revision: str
    bundle_index_revision: str
    embedding_revision: str
    member_count: int
    indexed_count: int
    excluded_count: int
    payload_digest: str


class SearchHit(BaseModel):
    """One placed result: identifiers, score, rank. Deliberately no body."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    content_sha256: str = Field(min_length=64, max_length=64)
    score: float
    rank: int = Field(ge=0)

    @model_validator(mode="after")
    def require_finite_score(self) -> SearchHit:
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        return self


class SearchResult(BaseModel):
    """A total order over hits, bound to the revision that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    composed_revision: str = Field(min_length=1)
    hits: tuple[SearchHit, ...]

    @model_validator(mode="after")
    def require_frozen_total_order(self) -> SearchResult:
        seen: set[str] = set()
        for position, hit in enumerate(self.hits):
            if hit.rank != position:
                raise ValueError("rank must be the dense 0-based position")
            if hit.relative_path in seen:
                raise ValueError("relative_path must not repeat in a result")
            seen.add(hit.relative_path)
        for previous, hit in zip(self.hits, self.hits[1:], strict=False):
            previous_key = (-previous.score, previous.relative_path)
            if previous_key >= (-hit.score, hit.relative_path):
                raise ValueError(
                    "hit order must be score descending, then relative_path ascending"
                )
        return self


@dataclass(frozen=True)
class RebuildReport:
    """What a provider proved after a rebuild or verified restore."""

    provider_id: str
    composed_revision: str
    point_count: int
    indexed_count: int
    excluded_count: int
    payload_digest: str


def compute_point_id(digest_key: str) -> str:
    """Derived, never random: the same note always lands on the same point."""
    payload = digest_key.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(_POINT_DOMAIN).to_bytes(8, "big"))
    digest.update(_POINT_DOMAIN)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def compute_payload_digest(points: tuple[IndexedPoint, ...]) -> str:
    """Bit-exact digest over every stored field of every point, in order."""
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    for point in points:
        for text in (
            point.point_id,
            point.digest_key,
            point.relative_path,
            point.content_sha256,
        ):
            raw = text.encode("utf-8")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        vector_bytes = struct.pack(f">{len(point.vector)}d", *point.vector)
        digest.update(len(vector_bytes).to_bytes(8, "big"))
        digest.update(vector_bytes)
    return f"sha256:{digest.hexdigest()}"


def require_query_vector(value: object, *, dimension: int) -> tuple[float, ...]:
    """Accept one finite vector of the declared dimension, failing closed."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise IndexRefusal(IndexErrorCode.QUERY_INVALID)
    values = tuple(value)
    if len(values) != dimension:
        raise IndexRefusal(IndexErrorCode.DIMENSION_MISMATCH)
    for component in values:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise IndexRefusal(IndexErrorCode.QUERY_INVALID)
        if not math.isfinite(component):
            raise IndexRefusal(IndexErrorCode.QUERY_INVALID)
    return tuple(float(component) for component in values)


def require_top_k(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise IndexRefusal(IndexErrorCode.TOP_K_INVALID)
    return value


def require_public_filter(value: object) -> frozenset[PrivacyClass]:
    """The only admissible search filter is exactly ``{public}``.

    The index never stores anything else, so any wider or different filter is
    a caller expecting data this index must not have -- refuse rather than
    silently narrow (AGENTS.md §8: do not pretend a capability exists).
    """
    if not isinstance(value, frozenset) or value != frozenset({PrivacyClass.PUBLIC}):
        raise IndexRefusal(IndexErrorCode.PRIVACY_FILTER_INVALID)
    return value


def _note_text(decoded: str) -> str:
    """The embeddable text of a note: its body, without the frontmatter.

    Frontmatter is metadata, not content -- embedding it would give every
    note the same ``privacy: public`` tokens and quietly flatten similarity.
    The split mirrors the classifier's framing: a leading ``---`` line, then
    everything up to the next ``---`` line is metadata.
    """
    if not decoded.startswith("---\n"):
        return decoded
    closing = decoded.find("\n---\n", len("---\n") - 1)
    if closing < 0:
        # An unterminated frontmatter block is all metadata, no body.
        return ""
    return decoded[closing + len("\n---\n") :]


def plan_rebuild(
    *,
    members: tuple[BundleMember, ...],
    stack: EmbeddingStack,
    gate: PrivacyGate,
) -> RebuildPlan:
    """Gate, embed, and order the corpus into a provider-ready plan.

    Exclusion is silent and counted, never detailed: a refused member's
    path or content appearing in any output would defeat the point of
    refusing it. Only members the gate admits as ``public`` are embedded;
    an undetermined class, a non-public class, undecodable bytes, or text
    the embedder refuses (empty after tokenization) all fail closed into
    ``excluded_count``.
    """
    if not isinstance(members, tuple) or not members:
        raise IndexRefusal(IndexErrorCode.PLAN_INVALID)
    bundle_revision = compute_index_revision_from_members(members)
    if bundle_revision is None:
        raise IndexRefusal(IndexErrorCode.PLAN_INVALID)

    ordered = sorted(members, key=lambda item: (item.digest_key, item.relative_path))
    admitted: list[tuple[BundleMember, str]] = []
    excluded = 0
    for member in ordered:
        verdict = gate.admit_member(member)
        admitted_public = (
            isinstance(verdict, Admitted) and verdict.privacy is PrivacyClass.PUBLIC
        )
        if not admitted_public:
            excluded += 1
            continue
        try:
            text = _note_text(member.content.decode("utf-8"))
        except UnicodeDecodeError:
            excluded += 1
            continue
        admitted.append((member, text))

    points: list[IndexedPoint] = []
    for member, text in admitted:
        # ``embed_query`` is bit-identical to the matching ``embed_documents``
        # row (C4 frozen invariant 2), and refusals stay attributable to one
        # member instead of poisoning a whole batch.
        try:
            vector = stack.embedding.embed_query(text).values
        except EmbeddingRefusal:
            excluded += 1
            continue
        points.append(
            IndexedPoint(
                point_id=compute_point_id(member.digest_key),
                digest_key=member.digest_key,
                relative_path=member.relative_path,
                content_sha256=member.content_sha256,
                vector=vector,
            )
        )

    dimension = stack.embedding.descriptor.dimension
    if dimension is None:  # pragma: no cover - stack admission already forbids
        raise IndexRefusal(IndexErrorCode.PLAN_INVALID)
    frozen_points = tuple(points)
    return RebuildPlan(
        points=frozen_points,
        dimension=dimension,
        composed_revision=compute_composed_index_revision(
            bundle_index_revision=bundle_revision,
            embedding_revision=stack.embedding_revision,
            index_schema_version=INDEX_SCHEMA_VERSION,
        ),
        bundle_index_revision=bundle_revision,
        embedding_revision=stack.embedding_revision,
        member_count=len(members),
        indexed_count=len(frozen_points),
        excluded_count=excluded,
        payload_digest=compute_payload_digest(frozen_points),
    )


SNAPSHOT_SCHEMA = "ckp-index-snapshot/1"


def write_plan_snapshot(
    plan: RebuildPlan, *, provider_id: str, target_dir: Path
) -> Path:
    """Provider-neutral snapshot: the canonical float64 plan, as JSON.

    The snapshot is the *recoverable point set*, not a storage engine dump --
    which is what lets a snapshot taken against one provider be restored into
    another, and lets restore verification recompute the exact payload digest.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (
        f"ckp-index-{provider_id}-"
        f"{plan.composed_revision.removeprefix('sha256:')[:12]}.snapshot.json"
    )
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "provider_id": provider_id,
        "dimension": plan.dimension,
        "composed_revision": plan.composed_revision,
        "bundle_index_revision": plan.bundle_index_revision,
        "embedding_revision": plan.embedding_revision,
        "member_count": plan.member_count,
        "indexed_count": plan.indexed_count,
        "excluded_count": plan.excluded_count,
        "payload_digest": plan.payload_digest,
        "points": [
            {
                "point_id": point.point_id,
                "digest_key": point.digest_key,
                "relative_path": point.relative_path,
                "content_sha256": point.content_sha256,
                "vector": [value.hex() for value in point.vector],
            }
            for point in plan.points
        ],
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return target


def read_plan_snapshot(snapshot_path: Path) -> RebuildPlan:
    """Parse and verify a snapshot, failing closed on any drift."""
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise IndexRefusal(IndexErrorCode.SNAPSHOT_INVALID) from error
    if not isinstance(payload, dict) or payload.get("schema") != SNAPSHOT_SCHEMA:
        raise IndexRefusal(IndexErrorCode.SNAPSHOT_INVALID)
    try:
        points = tuple(
            IndexedPoint(
                point_id=raw["point_id"],
                digest_key=raw["digest_key"],
                relative_path=raw["relative_path"],
                content_sha256=raw["content_sha256"],
                vector=tuple(float.fromhex(value) for value in raw["vector"]),
            )
            for raw in payload["points"]
        )
        plan = RebuildPlan(
            points=points,
            dimension=int(payload["dimension"]),
            composed_revision=payload["composed_revision"],
            bundle_index_revision=payload["bundle_index_revision"],
            embedding_revision=payload["embedding_revision"],
            member_count=int(payload["member_count"]),
            indexed_count=int(payload["indexed_count"]),
            excluded_count=int(payload["excluded_count"]),
            payload_digest=payload["payload_digest"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise IndexRefusal(IndexErrorCode.SNAPSHOT_INVALID) from error
    # A snapshot edited on disk -- or truncated in transit -- must fail
    # closed here, not surface later as a quietly different index.
    if compute_payload_digest(points) != plan.payload_digest:
        raise IndexRefusal(IndexErrorCode.SNAPSHOT_INVALID)
    if len(points) != plan.indexed_count:
        raise IndexRefusal(IndexErrorCode.SNAPSHOT_INVALID)
    return plan


__all__ = [
    "INDEX_CONTRACT",
    "SNAPSHOT_SCHEMA",
    "IndexDescriptor",
    "IndexedPoint",
    "RebuildPlan",
    "RebuildReport",
    "SearchHit",
    "SearchResult",
    "compute_payload_digest",
    "compute_point_id",
    "plan_rebuild",
    "read_plan_snapshot",
    "require_public_filter",
    "require_query_vector",
    "require_top_k",
    "write_plan_snapshot",
]
