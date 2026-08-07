"""C5 shadow benchmark package: frozen corpus, questions, and harness."""

from benchmarks.questions import (
    CORPUS,
    CORPUS_VERSION,
    KNOWN_COVERAGE_GAPS,
    NON_PUBLIC_PATHS,
    PILOT_QUESTIONS,
    QUESTION_SET_VERSION,
    QUESTIONS,
    REAL_QUESTIONS,
    BenchmarkQuestion,
    CorpusNote,
)
from benchmarks.shadow import REPORT_SCHEMA, run_shadow_benchmark, strip_latency

__all__ = [
    "CORPUS",
    "CORPUS_VERSION",
    "KNOWN_COVERAGE_GAPS",
    "NON_PUBLIC_PATHS",
    "PILOT_QUESTIONS",
    "QUESTIONS",
    "QUESTION_SET_VERSION",
    "REAL_QUESTIONS",
    "REPORT_SCHEMA",
    "BenchmarkQuestion",
    "CorpusNote",
    "run_shadow_benchmark",
    "strip_latency",
]
