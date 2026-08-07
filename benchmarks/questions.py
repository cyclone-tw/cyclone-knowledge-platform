"""The frozen shadow-benchmark corpus and question set (contract §5.3).

The ``CORPUS`` below (and every question that only targets it) is entirely
invented -- no Cyclone-Wiki note, person, student, or log line appears; it
exists so the synthetic half of the benchmark is reproducible from a bare
checkout with no fixture mutation.

Issue #26 (P5, Epic #21 D2) adds a second, real half: ``REAL_QUESTIONS``
targets six real Cyclone-Wiki notes frozen by D2 and read live through
``ckp.pilot`` (issue #23) -- this module carries only their *paths* (via
``ckp.pilot.PILOT_NOTE_PATHS``) and hand-written queries about what their
filenames say they are about, never their bodies. ``tests/test_no_wiki_content.py``
enforces that no real note content ever lands in this repo; nothing here
weakens that guard.

Every ``BenchmarkQuestion`` carries a ``provenance`` of ``"real"`` or
``"synthetic"``. This exists because a mixed corpus that does not label its
own questions makes it impossible to tell how much of a good score came from
notes the implementer wrote to be found (self-proof) versus notes that
existed independently in the wiki. Consumers of the question set (the shadow
harness, and eventually the P9 report/gate) must keep real and synthetic
statistics separate rather than average them into one number.

Changing the corpus or either question tuple changes what the benchmark
measures, so ``CORPUS_VERSION`` and ``QUESTION_SET_VERSION`` are bumped
together whenever either changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ckp.pilot import PILOT_NOTE_PATHS

CORPUS_VERSION = "3"
QUESTION_SET_VERSION = "2"


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


#: Values ``BenchmarkQuestion.provenance`` may take. Kept as a frozenset
#: (not just documented) so validation and any future iteration over "the
#: known provenances" read from one place.
PROVENANCES: frozenset[str] = frozenset({"real", "synthetic"})


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One fixed question: id, §5.3 category, query, expected paths, origin.

    ``expected_paths`` empty means the honest answer is "nothing": the
    lexical engine must return no results, and the vector engine's returned
    similarities are recorded but not scored (a cosine engine always ranks
    *something*; pretending it can abstain would fake a capability).

    ``provenance`` is ``"synthetic"`` for a question scored against
    ``CORPUS`` (this module's invented fixture) and ``"real"`` for a
    question scored against a note frozen by D2 and read live through
    ``ckp.pilot`` -- never both. Required, not defaulted (AGENTS.md §8:
    computed/declared values should not silently default to a
    plausible-looking value), so every question here states its own origin
    explicitly rather than inheriting one by omission.
    """

    question_id: str
    category: str
    query: str
    expected_paths: tuple[str, ...]
    provenance: str

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCES:
            raise ValueError(
                f"{self.question_id}: provenance must be one of "
                f"{sorted(PROVENANCES)!r}, got {self.provenance!r}"
            )


#: Synthetic questions, scored against ``CORPUS``. Unchanged in substance
#: from the pre-#26 set (issue #26 only adds the ``provenance`` label here;
#: see ``REAL_QUESTIONS`` below for the six D2 real-note questions this
#: issue adds).
#:
#: ``q01-espresso`` and ``q02-kettlebell`` are this set's answer to the
#: contract's "咖啡／重訓 log" question type -- their *category* value is
#: ``"precise-note"`` (the retrieval capability under test: one query, one
#: exact note) while their *content* is the coffee/strength-training topic
#: the contract names; ``COFFEE_LOG_QUESTION_IDS`` below records that
#: mapping explicitly so it is a checked fact, not an implicit reading of
#: two note titles.
QUESTIONS: tuple[BenchmarkQuestion, ...] = (
    BenchmarkQuestion(
        "q01-espresso",
        "precise-note",
        "espresso grind dial grams",
        ("espresso-dial-log.md",),
        "synthetic",
    ),
    BenchmarkQuestion(
        "q02-kettlebell",
        "precise-note",
        "kettlebell swings progressive overload",
        ("kettlebell-program.md",),
        "synthetic",
    ),
    BenchmarkQuestion(
        "q03-orbits-zh",
        "course-module",
        "軌道力學 module Kepler",
        ("course-module-orbits.md",),
        "synthetic",
    ),
    BenchmarkQuestion(
        "q04-latest-status",
        "latest-status",
        "platform status storage rebuild",
        ("platform-status-latest.md", "platform-status-outdated.md"),
        "synthetic",
    ),
    BenchmarkQuestion(
        "q05-person",
        "person",
        "Vex Meridian cartographer",
        ("person-profile-vex.md",),
        "synthetic",
    ),
    BenchmarkQuestion(
        "q06-cross-note",
        "cross-note",
        "tide caves mapping fieldnotes",
        ("person-profile-vex.md", "tide-cave-fieldnotes.md"),
        "synthetic",
    ),
    BenchmarkQuestion(
        "q07-external",
        "external-source",
        "deterministic feature hashing paper",
        ("external-source-summary.md",),
        "synthetic",
    ),
    BenchmarkQuestion(
        "q08-privacy-internal",
        "privacy-refusal",
        "internal budget allocation sentinel",
        (),
        "synthetic",
    ),
    BenchmarkQuestion(
        "q09-privacy-student",
        "privacy-refusal",
        "invented learner record sentinel",
        (),
        "synthetic",
    ),
    BenchmarkQuestion(
        "q10-no-answer",
        "no-answer",
        "quantum sourdough referendum",
        (),
        "synthetic",
    ),
)

#: The subset of ``QUESTIONS`` that stands in for the contract's "咖啡／重訓
#: log" question type (D2: no real coffee/strength-training note exists in
#: the wiki, so this type is synthetic-only -- see ``KNOWN_COVERAGE_GAPS``).
COFFEE_LOG_QUESTION_IDS: tuple[str, ...] = ("q01-espresso", "q02-kettlebell")

#: Real questions (issue #26, Epic #21 D2): six notes D2 froze, read live
#: through ``ckp.pilot.bind_pilot_corpus`` -- never vendored here. Queries
#: below are hand-written from what each note's *filename* says it is about;
#: none is copied from, or paraphrases, the note's actual body (this repo
#: never reads those bodies -- see the module docstring and
#: ``tests/test_no_wiki_content.py``).
#:
#: Mapping to ``PILOT_NOTE_PATHS`` (Core procedure ×2, Decision, Project,
#: external Source ×2 -- D2's six):
#:
#: * ``PILOT_NOTE_PATHS[0]`` (procedure) -> precise-note
#: * ``PILOT_NOTE_PATHS[0]`` + ``PILOT_NOTE_PATHS[1]`` (both procedures)
#:   -> cross-note (D2's own rationale for freezing a second procedure note
#:   was explicitly "測跨頁整合")
#: * ``PILOT_NOTE_PATHS[2]`` (decision) -> latest-status (the currently
#:   authoritative decision, not a superseded/current pair like the
#:   synthetic ``q04`` -- real corpus has no superseded pair, see
#:   ``KNOWN_COVERAGE_GAPS`` note on stale-exclusion below)
#: * ``PILOT_NOTE_PATHS[3]`` (project) -> precise-note (a second example,
#:   over a different note type)
#: * ``PILOT_NOTE_PATHS[4]``, ``PILOT_NOTE_PATHS[5]`` (external captures)
#:   -> external-source
REAL_QUESTIONS: tuple[BenchmarkQuestion, ...] = (
    BenchmarkQuestion(
        "r01-wiki-note-retrieval",
        "precise-note",
        "agent 讀取 wiki note 的標準程序 retrieval procedure",
        (PILOT_NOTE_PATHS[0],),
        "real",
    ),
    BenchmarkQuestion(
        "r02-retrieval-and-memory-scopes",
        "cross-note",
        "agent wiki note retrieval 與 memory read scopes 兩份程序的關聯",
        (PILOT_NOTE_PATHS[0], PILOT_NOTE_PATHS[1]),
        "real",
    ),
    BenchmarkQuestion(
        "r03-openwiki-role-boundary",
        "latest-status",
        "Cyclone-Wiki 與 OpenWiki 目前的角色邊界決策",
        (PILOT_NOTE_PATHS[2],),
        "real",
    ),
    BenchmarkQuestion(
        "r04-okf-knowledge-contract",
        "precise-note",
        "Cyclone OKF knowledge contract 專案範圍",
        (PILOT_NOTE_PATHS[3],),
        "real",
    ),
    BenchmarkQuestion(
        "r05-relayapi-analysis",
        "external-source",
        "RelayAPI repo 分析摘要",
        (PILOT_NOTE_PATHS[4],),
        "real",
    ),
    BenchmarkQuestion(
        "r06-hermes-os-loops",
        "external-source",
        "Hermes OS 2 100 stars loops 摘要",
        (PILOT_NOTE_PATHS[5],),
        "real",
    ),
)

#: The full §5.3 pilot question set: synthetic plus real, in that order.
#: This is what covers all eight contract question types when taken as a
#: whole -- neither half covers all eight alone (D2; see
#: ``KNOWN_COVERAGE_GAPS``).
PILOT_QUESTIONS: tuple[BenchmarkQuestion, ...] = QUESTIONS + REAL_QUESTIONS

#: D2 (Epic #21 comment, frozen 2026-08-07): a wiki-wide survey at freeze
#: time found zero notes of these three §Phase4 content types anywhere in
#: Cyclone-Wiki, so the real half of the pilot question set has -- and can
#: have -- no coverage of them. They stay synthetic-only. This is a known
#: gap, not a defect: Phase 5 must not read a synthetic-only pass on these
#: as evidence the platform works on *real* notes of these types, because
#: no such real note currently exists to test against.
#:
#: * ``"course-module"`` -- no course Module note exists (contract category
#:   and ``QUESTIONS`` category value both named ``"course-module"``).
#: * ``"coffee-log"`` -- no coffee/strength-training log note exists; see
#:   ``COFFEE_LOG_QUESTION_IDS`` for the synthetic stand-in.
#: * ``"publication"`` -- no ``type: Publication`` note exists; this is one
#:   of the six §Phase4 sample content types but is not exercised as its own
#:   ``category`` value at all in either question tuple (there is no real or
#:   synthetic Publication-type note to query).
KNOWN_COVERAGE_GAPS: tuple[str, ...] = ("course-module", "coffee-log", "publication")


__all__ = [
    "COFFEE_LOG_QUESTION_IDS",
    "CORPUS",
    "CORPUS_BODIES",
    "CORPUS_VERSION",
    "KNOWN_COVERAGE_GAPS",
    "NON_PUBLIC_PATHS",
    "PILOT_QUESTIONS",
    "PROVENANCES",
    "PUBLIC_DECLARED_PATHS",
    "QUESTIONS",
    "QUESTION_SET_VERSION",
    "REAL_QUESTIONS",
    "SUPERSEDED_PATHS",
    "BenchmarkQuestion",
    "CorpusNote",
]
