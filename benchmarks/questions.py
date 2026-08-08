"""The frozen shadow-benchmark corpus and question set (contract §5.3).

The ``CORPUS`` below (and every question that only targets it) is entirely
invented -- no Cyclone-Wiki note, person, student, or log line appears; it
exists so the synthetic half of the benchmark is reproducible from a bare
checkout with no fixture mutation.

Issue #26 (P5, Epic #21 D2) adds a second, real half: ``REAL_QUESTIONS``
targets six real Cyclone-Wiki notes frozen by D2 and read live through
``ckp.pilot`` (issue #23) -- this module carries only their *paths* (a local
mirror of ``ckp.pilot.PILOT_NOTE_PATHS``, see ``_PILOT_NOTE_PATHS`` below)
and hand-written queries. ``tests/test_no_wiki_content.py`` enforces that no
real note content ever lands in this repo; nothing here weakens that guard.

**Real-question writing rule (issue #58, question set version 3).** Every
term in a real question's query must come from one of the two sources RP1
permits the question writer to look at:

* the note's *filename* (its path segments, split on hyphens), and
* the note's *H1 heading* -- the same first ``# `` line the Catalog already
  exposes as ``heading_title`` (issue #48 established that surfacing it is
  privacy-safe; #58 extends that ruling to question writing).

The note *body* is never read and never paraphrased (RP1). Version 2's
queries added natural-language connective phrases that came from neither
source (``"...的標準程序"``, ``"...專案範圍"``, ``"分析摘要"``); live probing
in #58 showed ``qmd search`` (BM25) is *conjunctive* -- every query term
must appear in a document for it to match at all -- so each invented term
made the frozen query retrieve literally nothing, wiki-wide, and the whole
QMD baseline degenerated to zero hits. Version 3's queries therefore stay
inside the filename/H1 vocabulary: near-verbatim H1 wording where that
wording retrieves under BM25 (r03--r05), naturalized filename tokens where
the H1's punctuation defeats tokenization (r01, r06 -- its H1 spells
``2.0：`` with a fullwidth colon), and the union of both filenames' tokens
for the cross-note question (r02; see its entry for why QMD structurally
cannot answer it).

Every ``BenchmarkQuestion`` carries a ``provenance`` of ``"real"`` or
``"synthetic"``. This exists because a mixed corpus that does not label its
own questions makes it impossible to tell how much of a good score came from
notes the implementer wrote to be found (self-proof) versus notes that
existed independently in the wiki. Consumers of the question set (the shadow
harness, and eventually the P9 report/gate) must keep real and synthetic
statistics separate rather than average them into one number.

Changing the corpus or either question tuple changes what the benchmark
measures. Each change bumps its own version -- ``CORPUS_VERSION`` for
``CORPUS``, ``QUESTION_SET_VERSION`` for ``QUESTIONS``/``REAL_QUESTIONS``
-- and ``tests/test_benchmark_versions.py`` pins each version to a
fingerprint of exactly the content it names, independently (its
``test_corpus_and_question_set_versions_are_independent`` makes not
bumping the untouched side a checked guarantee, not an accident: #58
bumped only the question side).
"""

from __future__ import annotations

from dataclasses import dataclass

#: D2's six frozen real-note paths, mirrored from ``ckp.pilot.
#: PILOT_NOTE_PATHS`` rather than imported from it (issue #26 Round 2,
#: Codex Finding 1: the runtime image deliberately excludes ``ckp.pilot``
#: -- issue #23 -- but the C5 container smoke test imports ``benchmarks``
#: at module load time; a top-level ``from ckp.pilot import
#: PILOT_NOTE_PATHS`` here made that image fail with
#: ``ModuleNotFoundError: No module named 'ckp.pilot'``). ``ckp.pilot``
#: itself must **not** go back into the runtime image to fix this -- that
#: would undo #23's design. ``tests/test_shadow_benchmark.py::
#: test_real_questions_target_only_the_frozen_pilot_note_paths`` asserts
#: this mirror stays byte-identical to ``ckp.pilot.PILOT_NOTE_PATHS`` in
#: every environment that *does* have ``ckp.pilot`` installed (every
#: environment except the runtime image), so the two copies can never
#: silently drift apart unnoticed.
_PILOT_NOTE_PATHS: tuple[str, ...] = (
    "Core/procedure-agent-wiki-note-retrieval.md",
    "Core/procedure-agent-memory-read-scopes.md",
    "Core/decision-cyclone-wiki-openwiki-role-boundary.md",
    "Core/project-cyclone-okf-knowledge-contract.md",
    "Core/_inbox/agent-captures/2026-07-04-relayapi-repo-analysis.md",
    "Core/_inbox/agent-captures/2026-06-23-hermes-os-2-100-stars-loops.md",
)

CORPUS_VERSION = "3"
QUESTION_SET_VERSION = "3"


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
#: below are hand-written from each note's filename tokens and H1 heading
#: only (the #58 rule in the module docstring); none is copied from, or
#: paraphrases, the note's actual body (this repo never reads those bodies
#: -- see the module docstring and ``tests/test_no_wiki_content.py``).
#:
#: Each query below was probed live against the named ``cyclone-wiki`` QMD
#: index (issue #58, 2026-08-08, ``qmd search -n 5``): r01 and r03--r06
#: rank their target note top-5 wiki-wide (r01, r03, r05, r06 at rank 1;
#: r04 at rank 2). r02 is the one exception, and the miss is structural,
#: not a wording accident: ``qmd search`` is conjunctive, and probing shows
#: its first target never matches any query containing ``scopes`` (the
#: second target's discriminating vocabulary), while the vocabulary the
#: two targets share (``procedure``/``agent``/``wiki``) is too generic to
#: rank either one top-5 wiki-wide. No plain-keyword query drawn from the
#: sanctioned sources can put both notes in one top-5, so r02 measures a
#: real capability boundary of the baseline (single conjunctive query vs.
#: cross-note integration -- the very capability D2 froze a second
#: procedure note to test), and QMD's expected score on it is a recorded
#: miss, not a rigged one. OR-syntax was probed too and is not honored by
#: ``qmd search``; the platform's own lexical engine is disjunctive
#: (per-token additive scoring), so the identical query string remains
#: answerable there.
#:
#: Mapping to ``_PILOT_NOTE_PATHS`` (Core procedure ×2, Decision, Project,
#: external Source ×2 -- D2's six):
#:
#: * ``_PILOT_NOTE_PATHS[0]`` (procedure) -> precise-note
#: * ``_PILOT_NOTE_PATHS[0]`` + ``_PILOT_NOTE_PATHS[1]`` (both procedures)
#:   -> cross-note (D2's own rationale for freezing a second procedure note
#:   was explicitly "測跨頁整合")
#: * ``_PILOT_NOTE_PATHS[2]`` (decision) -> latest-status (the currently
#:   authoritative decision, not a superseded/current pair like the
#:   synthetic ``q04`` -- real corpus has no superseded pair, see
#:   ``KNOWN_COVERAGE_GAPS`` note on stale-exclusion below)
#: * ``_PILOT_NOTE_PATHS[3]`` (project) -> precise-note (a second example,
#:   over a different note type)
#: * ``_PILOT_NOTE_PATHS[4]``, ``_PILOT_NOTE_PATHS[5]`` (external captures)
#:   -> external-source
REAL_QUESTIONS: tuple[BenchmarkQuestion, ...] = (
    # Filename tokens naturalized ("procedure-agent-wiki-note-retrieval").
    BenchmarkQuestion(
        "r01-wiki-note-retrieval",
        "precise-note",
        "agent wiki note retrieval procedure",
        (_PILOT_NOTE_PATHS[0],),
        "real",
    ),
    # Union of both filenames' tokens -- the structural QMD miss documented
    # in the tuple comment above.
    BenchmarkQuestion(
        "r02-retrieval-and-memory-scopes",
        "cross-note",
        "agent wiki note retrieval memory read scopes procedure",
        (_PILOT_NOTE_PATHS[0], _PILOT_NOTE_PATHS[1]),
        "real",
    ),
    # H1 wording verbatim, minus the "Decision: " taxonomy prefix.
    BenchmarkQuestion(
        "r03-openwiki-role-boundary",
        "latest-status",
        "Cyclone-Wiki 與 OpenWiki 的角色分工與整合方式",
        (_PILOT_NOTE_PATHS[2],),
        "real",
    ),
    # H1 verbatim (the note's H1 carries no taxonomy prefix).
    BenchmarkQuestion(
        "r04-okf-knowledge-contract",
        "precise-note",
        "Cyclone OKF Knowledge Contract",
        (_PILOT_NOTE_PATHS[3],),
        "real",
    ),
    # H1 verbatim.
    BenchmarkQuestion(
        "r05-relayapi-analysis",
        "external-source",
        "relayAPI repo analysis and Cyclone application notes",
        (_PILOT_NOTE_PATHS[4],),
        "real",
    ),
    # Filename tokens: the H1's fullwidth-punctuated "2.0：100 Stars 之後的
    # 三條 Loops" wording retrieves nothing under BM25 tokenization.
    BenchmarkQuestion(
        "r06-hermes-os-loops",
        "external-source",
        "Hermes OS 2 100 stars loops",
        (_PILOT_NOTE_PATHS[5],),
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
