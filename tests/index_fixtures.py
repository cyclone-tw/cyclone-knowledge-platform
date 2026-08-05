"""Shared fixtures for the C5 index and shadow-benchmark tests.

Everything is synthetic and built in-process from the frozen benchmark
corpus; no fixture Markdown on disk gains a non-public class
(``tests/test_no_wiki_content.py`` stays at full strength).
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

from benchmarks.questions import CORPUS, CorpusNote

from ckp.bundle import BundleMember
from ckp.embedding import EmbeddingStack, build_c4_deterministic_stack
from ckp.privacy import FrontmatterClassifier, PrivacyClass
from ckp.privacy.gate import PrivacyGate

DIMENSION = 256

#: Never materialized; the classifier only ever sees in-memory members.
UNUSED_ROOT = Path("/nonexistent-ckp-c5-root")


def make_member(relative_path: str, content: bytes) -> BundleMember:
    return BundleMember(
        relative_path=relative_path,
        digest_key=unicodedata.normalize("NFC", relative_path),
        content=content,
        content_sha256=hashlib.sha256(content).hexdigest(),
    )


def corpus_members(notes: tuple[CorpusNote, ...] = CORPUS) -> tuple[BundleMember, ...]:
    return tuple(make_member(note.relative_path, note.content) for note in notes)


def public_gate() -> PrivacyGate:
    return PrivacyGate(
        FrontmatterClassifier(UNUSED_ROOT),
        frozenset({PrivacyClass.PUBLIC}),
    )


def deterministic_stack(dimension: int = DIMENSION) -> EmbeddingStack:
    return build_c4_deterministic_stack(
        dimension=dimension,
        embedding_name="hash",
        reranker_name="cosine",
    )
