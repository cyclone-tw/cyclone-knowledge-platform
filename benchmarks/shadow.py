"""The QMD shadow benchmark harness (contract §5.3): report only, no cutover.

Both engines answer the same frozen questions over the same privacy-gated
corpus. The report is honest about what it measures:

* the vector side runs the C4 hash embedding, which declares
  ``semantic=false`` -- these numbers are a deterministic baseline, not a
  claim of semantic quality, and Qdrant does not win by being newer;
* ``token_cost`` is a deterministic proxy (whitespace token count of each
  returned note's body), not a model tokenizer count;
* everything except ``latency_ms`` is reproducible byte-for-byte from the
  same commit, and tests pin exactly that.

Privacy is zero-tolerance: a single non-public path in either engine's
output makes ``privacy_violations`` non-zero, and the test suite treats that
as a hard failure (contract §5.3: false-negative exposure is intolerable).
"""

from __future__ import annotations

import time

from benchmarks.questions import (
    CORPUS_BODIES,
    CORPUS_VERSION,
    QUESTION_SET_VERSION,
    QUESTIONS,
)
from ckp.bundle import (
    BundleMember,
    BundleSnapshot,
    compute_index_revision_from_members,
)
from ckp.catalog import CatalogBuilder
from ckp.embedding import EmbeddingStack
from ckp.gateway.models import QueryRequest
from ckp.gateway.service import query_response
from ckp.index.models import plan_rebuild
from ckp.index.provider import VectorIndexProvider, require_index_provider
from ckp.privacy import PrivacyClass
from ckp.privacy.gate import PrivacyGate

REPORT_SCHEMA = "ckp-shadow-report/2"

_PUBLIC_ONLY = frozenset({PrivacyClass.PUBLIC})


def _token_cost(paths: tuple[str, ...]) -> int:
    return sum(len(CORPUS_BODIES.get(path, "").split()) for path in paths)


def _hit(expected: tuple[str, ...], returned: tuple[str, ...]) -> bool | None:
    if not expected:
        return None
    return set(expected).issubset(set(returned))


def run_shadow_benchmark(
    *,
    members: tuple[BundleMember, ...],
    stack: EmbeddingStack,
    index_provider: VectorIndexProvider,
    gate: PrivacyGate,
    top_k: int,
) -> dict:
    """Rebuild, query both engines, and assemble the deterministic report."""
    descriptor = require_index_provider(index_provider)
    plan = plan_rebuild(members=members, stack=stack, gate=gate)
    rebuild_report = index_provider.rebuild(plan)

    snapshot = BundleSnapshot(
        members=members,
        profile_version=None,
        bundle_commit=None,
        index_revision=compute_index_revision_from_members(members),
        sources={},
        token=(),
    )
    catalog = CatalogBuilder(gate).build(snapshot)

    #: The only paths any engine may surface: what the gated plan indexed.
    #: Judging leaks against this set (not a fixture list) also catches a
    #: rogue provider inventing paths nobody has ever seen (R1 review).
    public_paths = frozenset(point.relative_path for point in plan.points)

    questions = []
    lexical_hits = 0
    vector_hits = 0
    scored = 0
    lexical_tokens_total = 0
    vector_tokens_total = 0
    privacy_violations = 0
    citation_correct_count = 0
    lexical_no_answer_correct = True
    lexical_elapsed = 0.0
    vector_elapsed = 0.0

    for question in QUESTIONS:
        started = time.perf_counter()
        lexical = query_response(
            catalog, QueryRequest(query=question.query, limit=top_k)
        )
        lexical_elapsed += time.perf_counter() - started
        raw_lexical_paths = tuple(result.citation.path for result in lexical.results)

        started = time.perf_counter()
        query_vector = stack.embedding.embed_query(question.query).values
        vector = index_provider.search(
            query_vector, top_k=top_k, filter_privacy=_PUBLIC_ONLY
        )
        vector_elapsed += time.perf_counter() - started
        raw_vector_paths = tuple(hit.relative_path for hit in vector.hits)

        # Leaked paths are counted and then *redacted*: a violating path must
        # never travel onward inside the report it violated (R1 review).
        lexical_paths = tuple(
            path for path in raw_lexical_paths if path in public_paths
        )
        vector_paths = tuple(path for path in raw_vector_paths if path in public_paths)
        privacy_violations += (len(raw_lexical_paths) - len(lexical_paths)) + (
            len(raw_vector_paths) - len(vector_paths)
        )

        # Citation correctness (§5.3): every lexical citation must bind the
        # path it ranks and the exact snapshot revision served; the vector
        # result must be bound to the rebuilt composed revision.
        citation_correct = (
            all(
                result.citation.concept_id == result.concept_id
                and result.citation.index_revision == catalog.index_revision
                for result in lexical.results
            )
            and vector.composed_revision == plan.composed_revision
        )
        if citation_correct:
            citation_correct_count += 1

        lexical_hit = _hit(question.expected_paths, lexical_paths)
        vector_hit = _hit(question.expected_paths, vector_paths)
        if question.expected_paths:
            scored += 1
            lexical_hits += 1 if lexical_hit else 0
            vector_hits += 1 if vector_hit else 0
        elif question.category == "no-answer" and lexical_paths:
            lexical_no_answer_correct = False

        lexical_cost = _token_cost(lexical_paths)
        vector_cost = _token_cost(vector_paths)
        lexical_tokens_total += lexical_cost
        vector_tokens_total += vector_cost

        questions.append(
            {
                "question_id": question.question_id,
                "category": question.category,
                "query": question.query,
                "expected_paths": list(question.expected_paths),
                "citation_correct": citation_correct,
                "lexical": {
                    "paths": list(lexical_paths),
                    "hit": lexical_hit,
                    "result_count": len(lexical_paths),
                    "token_cost": lexical_cost,
                },
                "vector": {
                    "paths": list(vector_paths),
                    "hit": vector_hit,
                    "result_count": len(vector_paths),
                    "top_score": (vector.hits[0].score if vector.hits else None),
                    "token_cost": vector_cost,
                },
            }
        )

    return {
        "schema": REPORT_SCHEMA,
        "corpus_version": CORPUS_VERSION,
        "question_set_version": QUESTION_SET_VERSION,
        "index_provider": descriptor.provider_id,
        "composed_revision": plan.composed_revision,
        "bundle_index_revision": plan.bundle_index_revision,
        "embedding_revision": plan.embedding_revision,
        # Read from the descriptor, never hardcoded: a semantic provider
        # must not be reported as a hash baseline, or vice versa (R1 review).
        "semantic": stack.embedding.descriptor.semantic,
        "semantic_note": (
            "vector numbers come from a provider declaring semantic=false: "
            "a reproducible baseline, not semantic quality"
            if not stack.embedding.descriptor.semantic
            else "vector numbers come from a provider declaring semantic=true"
        ),
        "top_k": top_k,
        "rebuild": {
            "point_count": rebuild_report.point_count,
        },
        "questions": questions,
        "summary": {
            "scored_questions": scored,
            "lexical_hit_rate": round(lexical_hits / scored, 4) if scored else None,
            "vector_hit_rate": round(vector_hits / scored, 4) if scored else None,
            "lexical_token_cost_total": lexical_tokens_total,
            "vector_token_cost_total": vector_tokens_total,
            "lexical_no_answer_correct": lexical_no_answer_correct,
            "citation_correct_questions": citation_correct_count,
            "privacy_violations": privacy_violations,
        },
        "latency_ms": {
            "lexical": round(lexical_elapsed * 1000.0, 3),
            "vector": round(vector_elapsed * 1000.0, 3),
        },
    }


def strip_latency(report: dict) -> dict:
    """The reproducible view of a report: everything except wall-clock."""
    return {key: value for key, value in report.items() if key != "latency_ms"}


__all__ = [
    "REPORT_SCHEMA",
    "run_shadow_benchmark",
    "strip_latency",
]
