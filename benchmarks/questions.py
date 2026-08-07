"""The frozen shadow-benchmark corpus and question set (contract §5.3).

Everything here is invented. No Cyclone-Wiki note, person, student, or log
line appears; the corpus exists so the benchmark is reproducible from a bare
checkout with no fixture mutation. Changing either the corpus or the question
set changes what the benchmark measures, so both are versioned together.
"""

from __future__ import annotations

from dataclasses import dataclass

CORPUS_VERSION = "2"
QUESTION_SET_VERSION = "1"


@dataclass(frozen=True)
class CorpusNote:
    """One synthetic note: path, declared privacy, body.

    ``superseded`` marks a note that a newer note has replaced (contract
    §5.3 stale-exclusion metric). It carries no privacy meaning -- a
    superseded note is still ``public``; it is just not the answer anymore.
    """

    relative_path: str
    privacy: str
    body: str
    superseded: bool = False

    @property
    def content(self) -> bytes:
        return f"---\nprivacy: {self.privacy}\n---\n\n{self.body}\n".encode()


#: Paths are flat on purpose: the corpus tests retrieval, not tree walking.
CORPUS: tuple[CorpusNote, ...] = (
    CorpusNote(
        "espresso-dial-log.md",
        "public",
        "# Espresso dial log\n\nSynthetic espresso grind dial: 18 grams in, "
        "36 grams out, 27 seconds, medium roast zephyr blend tastes balanced.",
    ),
    CorpusNote(
        "kettlebell-program.md",
        "public",
        "# Kettlebell program\n\nSynthetic strength block: swings, presses, "
        "and goblet squats, three sessions per week, progressive overload.",
    ),
    CorpusNote(
        "course-module-orbits.md",
        "public",
        "# Course module: orbits\n\nSynthetic astronomy module covering "
        "Kepler laws, transfer windows, and 軌道力學 basics for beginners.",
    ),
    CorpusNote(
        "platform-status-latest.md",
        "public",
        "# Platform status\n\nSynthetic status note: the knowledge gateway "
        "returned to normal operation after the storage rebuild completed.",
    ),
    CorpusNote(
        "platform-status-outdated.md",
        "public",
        "# Platform status (superseded)\n\nSynthetic older status note: the "
        "storage rebuild is still in progress and reads are degraded.",
        superseded=True,
    ),
    CorpusNote(
        "person-profile-vex.md",
        "public",
        "# Profile: Vex Meridian\n\nSynthetic person profile: Vex Meridian "
        "is an invented cartographer who maps imaginary tide caves.",
    ),
    CorpusNote(
        "external-source-summary.md",
        "public",
        "# External source summary\n\nSynthetic digest of an invented paper "
        "on deterministic feature hashing for offline retrieval baselines.",
    ),
    CorpusNote(
        "tide-cave-fieldnotes.md",
        "public",
        "# Tide cave fieldnotes\n\nSynthetic fieldnotes: bioluminescent "
        "moss density readings across three imaginary tide caves at low tide.",
    ),
    CorpusNote(
        "budget-internal.md",
        "internal",
        "# Internal budget\n\nSynthetic internal allocation note: "
        "INTERNAL-SENTINEL-ALLOCATION should never surface in any output.",
    ),
    CorpusNote(
        "meeting-sensitive.md",
        "sensitive",
        "# Sensitive meeting\n\nSynthetic sensitive note: "
        "SENSITIVE-SENTINEL-MEETING should never surface in any output.",
    ),
    CorpusNote(
        "learner-record.md",
        "student-private",
        "# Invented learner record\n\nSynthetic student-private fixture: "
        "STUDENT-SENTINEL-RECORD, entirely invented, must never surface.",
    ),
)

#: Bodies by path, for deterministic token-cost accounting in the harness.
CORPUS_BODIES: dict[str, str] = {note.relative_path: note.body for note in CORPUS}

#: Paths that must never appear in any engine's output, ever.
NON_PUBLIC_PATHS: frozenset[str] = frozenset(
    note.relative_path for note in CORPUS if note.privacy != "public"
)

#: Paths superseded by a newer note (contract §5.3 stale-exclusion metric).
SUPERSEDED_PATHS: frozenset[str] = frozenset(
    note.relative_path for note in CORPUS if note.superseded
)

#: Paths declared ``public`` in the corpus, independent of what any gated
#: rebuild actually indexed. Used to measure privacy false positives: a
#: public note the gate should have indexed but did not.
PUBLIC_DECLARED_PATHS: frozenset[str] = frozenset(
    note.relative_path for note in CORPUS if note.privacy == "public"
)


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One fixed question: id, §5.3 category, query, expected public paths.

    ``expected_paths`` empty means the honest answer is "nothing": the
    lexical engine must return no results, and the vector engine's returned
    similarities are recorded but not scored (a cosine engine always ranks
    *something*; pretending it can abstain would fake a capability).
    """

    question_id: str
    category: str
    query: str
    expected_paths: tuple[str, ...]


QUESTIONS: tuple[BenchmarkQuestion, ...] = (
    BenchmarkQuestion(
        "q01-espresso",
        "precise-note",
        "espresso grind dial grams",
        ("espresso-dial-log.md",),
    ),
    BenchmarkQuestion(
        "q02-kettlebell",
        "precise-note",
        "kettlebell swings progressive overload",
        ("kettlebell-program.md",),
    ),
    BenchmarkQuestion(
        "q03-orbits-zh",
        "course-module",
        "軌道力學 module Kepler",
        ("course-module-orbits.md",),
    ),
    BenchmarkQuestion(
        "q04-latest-status",
        "latest-status",
        "platform status storage rebuild",
        ("platform-status-latest.md", "platform-status-outdated.md"),
    ),
    BenchmarkQuestion(
        "q05-person",
        "person",
        "Vex Meridian cartographer",
        ("person-profile-vex.md",),
    ),
    BenchmarkQuestion(
        "q06-cross-note",
        "cross-note",
        "tide caves mapping fieldnotes",
        ("person-profile-vex.md", "tide-cave-fieldnotes.md"),
    ),
    BenchmarkQuestion(
        "q07-external",
        "external-source",
        "deterministic feature hashing paper",
        ("external-source-summary.md",),
    ),
    BenchmarkQuestion(
        "q08-privacy-internal",
        "privacy-refusal",
        "internal budget allocation sentinel",
        (),
    ),
    BenchmarkQuestion(
        "q09-privacy-student",
        "privacy-refusal",
        "invented learner record sentinel",
        (),
    ),
    BenchmarkQuestion(
        "q10-no-answer",
        "no-answer",
        "quantum sourdough referendum",
        (),
    ),
)


__all__ = [
    "CORPUS",
    "CORPUS_BODIES",
    "CORPUS_VERSION",
    "NON_PUBLIC_PATHS",
    "PUBLIC_DECLARED_PATHS",
    "QUESTIONS",
    "QUESTION_SET_VERSION",
    "SUPERSEDED_PATHS",
    "BenchmarkQuestion",
    "CorpusNote",
]
