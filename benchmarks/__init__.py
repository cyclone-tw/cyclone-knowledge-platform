"""C5 shadow benchmark package: frozen corpus, questions, and harness."""

from benchmarks.questions import (
    CORPUS,
    CORPUS_VERSION,
    NON_PUBLIC_PATHS,
    QUESTION_SET_VERSION,
    QUESTIONS,
    BenchmarkQuestion,
    CorpusNote,
)
from benchmarks.shadow import REPORT_SCHEMA, run_shadow_benchmark, strip_latency

__all__ = [
    "CORPUS",
    "CORPUS_VERSION",
    "NON_PUBLIC_PATHS",
    "QUESTIONS",
    "QUESTION_SET_VERSION",
    "REPORT_SCHEMA",
    "BenchmarkQuestion",
    "CorpusNote",
    "run_shadow_benchmark",
    "strip_latency",
]
