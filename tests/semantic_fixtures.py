"""Shared constants for the issue #25 semantic-relationship checks.

Kept separate from both ``tests/test_semantic_contract.py`` (weight-free)
and ``tests/test_semantic_embedding_provider.py`` (weight-gated: needs the
pinned model cache) so the threshold *values themselves* can be sanity
checked without needing the pinned weights on disk --
``tests/test_semantic_contract.py::test_semantic_relationship_thresholds_have_real_margin``
imports this module directly, the same way ``embedding_fixtures.py`` is
shared between the C4 test files.

Round 3 (Codex Finding 1 confirmed in CI): these thresholds replace a
``GOLDEN`` dict of frozen output digests that turned out to be host-specific
-- see ``tests/test_semantic_embedding_provider.py``'s module docstring for
the full story. Cosine similarity between two freshly computed vectors is a
macroscopic relationship, not a bitwise one: the LSB-scale floating-point
drift a different CPU's SIMD kernel can introduce moves a cosine value by a
negligible amount next to these thresholds' margins, which is what makes
this stable across hosts in a way exact vectors are not.
"""

from __future__ import annotations

#: Both thresholds were set with real headroom, not tuned to the boundary --
#: measured on the development machine (Apple Silicon, arm64) before these
#: numbers were chosen:
#:   zh related pair    -> cosine 0.8400   (margin to RELATED_MIN: +0.14)
#:   en related pair    -> cosine 0.8734   (margin to RELATED_MIN: +0.17)
#:   zh unrelated pair  -> cosine 0.3031   (margin to UNRELATED_MAX: +0.20)
#:   en unrelated pair  -> cosine 0.3666   (margin to UNRELATED_MAX: +0.13)
#: A loosely-topic-associated (rather than near-paraphrase) English pair was
#: tried first and scored only 0.4645 -- this Chinese-specialized checkpoint
#: has a visibly weaker semantic signal for English-only pairs than for
#: Chinese ones, which is exactly why the English "related" pair used in
#: ``tests/test_semantic_embedding_provider.py`` is a near-paraphrase (same
#: claim, reworded) rather than a loosely associated one: it is the pairing
#: this model can actually be expected to score high on, not the easiest
#: number to pick.
RELATED_MIN = 0.7
UNRELATED_MAX = 0.5


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


__all__ = ["RELATED_MIN", "UNRELATED_MAX", "cosine"]
