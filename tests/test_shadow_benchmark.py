"""The §5.3 shadow benchmark: determinism, honesty, and zero privacy leaks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.qmd.adapter import QmdBaselineConfig, QmdQueryResult, QmdUnavailable
from benchmarks.questions import (
    CORPUS_BODIES,
    KNOWN_COVERAGE_GAPS,
    NON_PUBLIC_PATHS,
    PILOT_QUESTIONS,
    PUBLIC_DECLARED_PATHS,
    QUESTIONS,
    REAL_QUESTIONS,
    SUPERSEDED_PATHS,
    BenchmarkQuestion,
)
from benchmarks.shadow import (
    REPORT_SCHEMA,
    TOKEN_COST_METHOD,
    TOKEN_COST_NOTE,
    run_shadow_benchmark,
    strip_latency,
)

from ckp.index import (
    INDEX_CONTRACT,
    IndexDescriptor,
    InMemoryVectorIndex,
    RebuildReport,
    SearchHit,
    SearchResult,
)
from ckp.index.revision import INDEX_SCHEMA_VERSION
from ckp.privacy import FrontmatterClassifier, PrivacyClass
from ckp.privacy.gate import PrivacyGate
from index_fixtures import (
    UNUSED_ROOT,
    corpus_members,
    deterministic_stack,
    make_member,
    public_gate,
)


def _run(
    index_provider=None,
    top_k: int = 3,
    trials: int = 1,
    questions: tuple[BenchmarkQuestion, ...] = QUESTIONS,
    qmd_config: QmdBaselineConfig | None = None,
) -> dict:
    return run_shadow_benchmark(
        members=corpus_members(),
        stack=deterministic_stack(),
        index_provider=index_provider or InMemoryVectorIndex(),
        gate=public_gate(),
        top_k=top_k,
        questions=questions,
        trials=trials,
        qmd_config=qmd_config,
    )


def test_question_set_covers_the_contract_categories() -> None:
    """The pre-#26 synthetic-only set: unchanged, still a >=5-category subset."""
    assert len(QUESTIONS) >= 8
    categories = {question.category for question in QUESTIONS}
    assert {
        "precise-note",
        "cross-note",
        "latest-status",
        "privacy-refusal",
        "no-answer",
    } <= categories
    identifiers = [question.question_id for question in QUESTIONS]
    assert len(identifiers) == len(set(identifiers))


def test_pilot_question_set_covers_all_eight_contract_categories() -> None:
    """Issue #26 acceptance #1: §5.3's eight question types, all present.

    Taken as the *combined* real+synthetic set -- D2 froze real notes for
    only some of the eight (see ``KNOWN_COVERAGE_GAPS``), so neither half
    alone covers all eight; ``PILOT_QUESTIONS`` (synthetic + real) must.
    "咖啡／重訓 log" has no literal ``category`` value in either question
    tuple (the synthetic stand-ins are tagged ``"precise-note"`` -- the
    retrieval capability under test -- while their *content* is the coffee/
    strength-training topic the contract names); ``COFFEE_LOG_QUESTION_IDS``
    is the checked record of that mapping, asserted separately below.
    """
    from benchmarks.questions import COFFEE_LOG_QUESTION_IDS

    categories = {question.category for question in PILOT_QUESTIONS}
    assert {
        "precise-note",
        "cross-note",
        "latest-status",
        "privacy-refusal",
        "course-module",
        "person",
        "external-source",
    } <= categories

    ids_by_provenance = {q.question_id: q for q in PILOT_QUESTIONS}
    assert COFFEE_LOG_QUESTION_IDS
    for question_id in COFFEE_LOG_QUESTION_IDS:
        assert ids_by_provenance[question_id].provenance == "synthetic"


def test_pilot_questions_is_synthetic_then_real_with_no_overlap() -> None:
    assert PILOT_QUESTIONS == QUESTIONS + REAL_QUESTIONS
    synthetic_ids = {q.question_id for q in QUESTIONS}
    real_ids = {q.question_id for q in REAL_QUESTIONS}
    assert not synthetic_ids & real_ids
    identifiers = [q.question_id for q in PILOT_QUESTIONS]
    assert len(identifiers) == len(set(identifiers))


def test_every_question_provenance_matches_its_id_prefix() -> None:
    """A cheap, independent cross-check (AGENTS.md §9 Q3): every ``q*`` id is
    ``synthetic`` and every ``r*`` id is ``real``. This does not replace the
    ``provenance`` field -- it is a second signal that would catch a single
    mislabeled question even if nobody hand-checks the field against the
    corpus that question actually targets.
    """
    for question in PILOT_QUESTIONS:
        if question.question_id.startswith("q"):
            assert question.provenance == "synthetic", question.question_id
        elif question.question_id.startswith("r"):
            assert question.provenance == "real", question.question_id
        else:
            raise AssertionError(f"unexpected question id {question.question_id!r}")


def test_real_questions_target_only_the_frozen_pilot_note_paths() -> None:
    """Every real question's ``expected_paths`` must be drawn from D2's six.

    Guards against a real question accidentally naming a path that is not
    (or no longer) part of the frozen allowlist -- which would make the
    question set claim real-corpus coverage for a note nothing actually
    freezes or verifies.

    Deliberately checked against ``benchmarks.questions._PILOT_NOTE_PATHS``
    (the local mirror), never against ``ckp.pilot.PILOT_NOTE_PATHS``
    directly -- issue #26 Round 3 (Codex): the C5 container smoke actually
    *runs* this file's tests inside the image, which excludes ``ckp.pilot``
    by design (#23), and a runtime (not just module-level) ``from ckp.pilot
    import ...`` here made that smoke fail with ``ModuleNotFoundError`` even
    though the module-level import in ``benchmarks.questions`` had already
    been fixed in Round 2 -- the same class of bug, one hop further down
    the same file (AGENTS.md §9 Q3). The drift check against the *real*
    ``ckp.pilot.PILOT_NOTE_PATHS`` lives in
    ``test_pilot_note_paths_mirror_matches_ckp_pilot`` below, which skips
    (not fails) where ``ckp.pilot`` is not installed -- so this test keeps
    running everywhere, including inside the container image.
    """
    from benchmarks.questions import _PILOT_NOTE_PATHS

    allowed = set(_PILOT_NOTE_PATHS)
    for question in REAL_QUESTIONS:
        assert question.provenance == "real"
        assert question.expected_paths, question.question_id
        assert set(question.expected_paths) <= allowed, question.question_id
    # Every one of the six frozen paths is exercised by at least one
    # question -- D2 froze six notes; none of them should be dead weight
    # the question set never actually asks about.
    exercised = {
        path for question in REAL_QUESTIONS for path in question.expected_paths
    }
    assert exercised == allowed


def test_pilot_note_paths_mirror_matches_ckp_pilot() -> None:
    """Drift guard: ``benchmarks.questions._PILOT_NOTE_PATHS`` (the local
    mirror, see its definition for why it is not an import) must stay
    byte-identical to the real ``ckp.pilot.PILOT_NOTE_PATHS``.

    Skips via ``pytest.importorskip`` rather than failing when ``ckp.pilot``
    is not installed -- true inside the runtime/C5-smoke image by design
    (#23), where there is nothing to drift-check against and the skip is
    not hiding a real problem, unlike a D5-style "silent skip" over a
    *reachable* live checkout. Everywhere ``ckp.pilot`` *is* installed (dev,
    CI's ``lint-and-test`` job, any environment except the runtime image)
    this still runs and still fails loudly on drift.
    """
    ckp_pilot = pytest.importorskip("ckp.pilot")
    from benchmarks.questions import _PILOT_NOTE_PATHS

    assert _PILOT_NOTE_PATHS == ckp_pilot.PILOT_NOTE_PATHS


def test_known_coverage_gaps_are_recorded_and_synthetic_only() -> None:
    """Issue #26 acceptance: the known real-data coverage gap must be
    recorded, non-empty, and every category it names must have zero
    representation in ``REAL_QUESTIONS`` (otherwise the "gap" is a lie).
    """
    assert set(KNOWN_COVERAGE_GAPS) == {"course-module", "coffee-log", "publication"}
    real_categories = {question.category for question in REAL_QUESTIONS}
    for gap_category in KNOWN_COVERAGE_GAPS:
        assert gap_category not in real_categories


def _synthetic_question_labeled_real(question_id: str) -> BenchmarkQuestion:
    """A question over ``CORPUS`` content, deliberately mislabeled
    ``provenance="real"`` for tests only.

    This never touches the live Wiki -- it targets an ordinary synthetic
    fixture path (``tide-cave-fieldnotes.md``) so the split-accounting
    machinery in ``run_shadow_benchmark`` can be exercised end to end
    without ``CKP_PILOT_WIKI_ROOT`` (matching the established pattern in
    ``tests/test_pilot_no_vendoring.py``: real-checkout binding is a manual
    smoke check, never a pytest dependency). The mismatch between the label
    and the content is the point -- it proves the harness sorts by the
    *label* on the question, not by anything it infers about the path.
    """
    return BenchmarkQuestion(
        question_id,
        "external-source",
        "tide caves mapping fieldnotes",
        ("tide-cave-fieldnotes.md",),
        "real",
    )


def test_summary_splits_real_and_synthetic_hit_rates_separately() -> None:
    """Issue #26's central acceptance: real and synthetic stats never merge
    into one number that could look like self-proof.

    Mixes one "real"-labeled question (which the synthetic-only fixture
    corpus can actually answer, so its hit rate is real signal, not a
    guaranteed miss) into the synthetic set, and checks that:

    * the two provenance blocks report different, independently correct
      numbers, not the same pooled figure copied twice;
    * the pooled top-level numbers still reflect all questions combined
      (continuity with the pre-#26 shape);
    * a provenance with a question present always reports numeric rates,
      never ``None`` (``None`` is reserved for "no scored question exists
      in this provenance at all").
    """
    mixed = QUESTIONS + (_synthetic_question_labeled_real("r-test-mixed"),)
    report = _run(questions=mixed)
    summary = report["summary"]

    provenance = summary["provenance"]
    assert provenance["synthetic"]["scored_questions"] == sum(
        1 for q in QUESTIONS if q.expected_paths
    )
    assert provenance["real"]["scored_questions"] == 1
    assert provenance["real"]["lexical_hit_rate"] == 1.0
    assert provenance["real"]["vector_hit_rate"] == 1.0
    assert provenance["synthetic"]["lexical_hit_rate"] == 1.0
    assert provenance["synthetic"]["vector_hit_rate"] == 1.0

    # Pooled continues to reflect everything, unchanged in meaning.
    assert summary["scored_questions"] == (
        provenance["synthetic"]["scored_questions"]
        + provenance["real"]["scored_questions"]
    )
    assert summary["citation_correct_questions"] == (
        provenance["synthetic"]["citation_correct_questions"]
        + provenance["real"]["citation_correct_questions"]
    )


def test_provenance_split_reflects_a_genuine_real_miss_against_synthetic_hits() -> None:
    """Issue #26 Round 2, Codex Finding 3: an asymmetric fixture, not just
    field presence.

    Codex injected "``_hit`` always returns ``True``" and
    ``test_summary_splits_real_and_synthetic_hit_rates_separately`` stayed
    green, because that test's fixture makes *both* provenances hit --
    there was no case where the two provenances' actual scores differ, so a
    mutant that always returns the same answer regardless of input was
    invisible to it. This uses an *unmodified* ``REAL_QUESTIONS`` entry
    mixed into the synthetic-only fixture corpus (``corpus_members()``,
    unchanged from every other test in this file): its expected path is a
    real D2 path that cannot possibly appear in a synthetic-only corpus's
    gated plan, so it is a genuine, guaranteed miss -- not a contrived
    always-false stub -- while the synthetic questions in the same run
    genuinely hit. A ``_hit`` that always returns ``True`` turns the real
    block's rate from ``0.0`` into ``1.0``; one that always returns
    ``False`` turns the synthetic block's rate from ``1.0`` into ``0.0``.
    Either mutation must fail at least one assertion here.
    """
    real_question = REAL_QUESTIONS[0]
    mixed = QUESTIONS + (real_question,)
    report = _run(questions=mixed)
    provenance = report["summary"]["provenance"]

    assert provenance["real"]["scored_questions"] == 1
    assert provenance["real"]["lexical_hit_rate"] == 0.0
    assert provenance["real"]["vector_hit_rate"] == 0.0
    assert provenance["synthetic"]["lexical_hit_rate"] == 1.0
    assert provenance["synthetic"]["vector_hit_rate"] == 1.0

    by_id = {q["question_id"]: q for q in report["questions"]}
    real_entry = by_id[real_question.question_id]
    assert real_entry["lexical"]["hit"] is False
    assert real_entry["vector"]["hit"] is False
    for question in QUESTIONS:
        if not question.expected_paths:
            continue
        entry = by_id[question.question_id]
        assert entry["lexical"]["hit"] is True, question.question_id
        assert entry["vector"]["hit"] is True, question.question_id


def test_provenance_split_covers_stale_exclusion_with_correct_denominators() -> None:
    """Issue #26 Round 3 (Codex Finding): stale exclusion was still pooled-
    only after Round 2 split hit rate/token cost/citation.

    The vector engine always returns *something* up to ``top_k`` even for
    an off-topic query -- documented behavior (module docstring: "a cosine
    engine always ranks *something*") -- so ``REAL_QUESTIONS[0]`` mixed
    into the default synthetic-only fixture corpus reliably gets back
    top-3 spillover that happens to include the superseded note: a real
    question can never be a *correct* hit here (D2's real notes are never
    in this fixture corpus at all), but it can absolutely be an *honest*
    stale-exclusion failure. The lexical side used to spill the same way,
    but the #62 relevance cutoff now cuts every weak cross-corpus match
    for this query (its returns are empty), which counts as excluded
    under the documented "no superseded path in what was returned"
    accounting -- the module docstring's own caveat that an empty answer
    is indistinguishable from a correct exclusion. Pinned empirically
    (see the values asserted below) as a real, deterministic asymmetry on
    both axes: synthetic vs real (two different causes -- ``q04-latest-
    status``'s legitimate superseded pair vs off-topic spillover), and
    now lexical vs vector on the same real question (cutoff vs RP3's
    uncut informational engine), which is exactly what must never
    collapse into one pooled number.

    Also pins the denominator: each provenance's rate must divide by that
    provenance's own question count, never by the pooled
    ``total_questions`` -- an unsplit denominator silently produces a wrong
    rate for whichever provenance's question count differs from the total,
    the same shape of bug as an unsplit numerator.
    """
    mixed = QUESTIONS + (REAL_QUESTIONS[0],)
    report = _run(questions=mixed)
    provenance = report["summary"]["provenance"]

    synthetic = provenance["synthetic"]
    real = provenance["real"]

    # Baseline synthetic behavior (q04's superseded pair is a genuine hit
    # on both engines at the default fixture corpus): pinned so a
    # regression in the underlying retrieval, not just the split
    # accounting, would also be visible here.
    assert synthetic["lexical_stale_excluded"] == 9
    assert synthetic["vector_stale_excluded"] == 8
    assert synthetic["question_count"] == len(QUESTIONS)
    assert synthetic["lexical_stale_excluded_rate"] == round(9 / len(QUESTIONS), 4)
    assert synthetic["vector_stale_excluded_rate"] == round(8 / len(QUESTIONS), 4)

    # The one real question, engine by engine: the #62 cutoff empties the
    # lexical answer (excluded, 1 of 1, per the documented empty-answer
    # caveat) while the uncut vector spillover still returns the
    # superseded note (not excluded, 0 of 1) -- pinned to the observed,
    # deterministic values, not assumed.
    assert real["question_count"] == 1
    assert real["lexical_stale_excluded"] == 1
    assert real["vector_stale_excluded"] == 0
    assert real["lexical_stale_excluded_rate"] == 1.0
    assert real["vector_stale_excluded_rate"] == 0.0

    # The two provenances must disagree -- proves the rate is not silently
    # reading the same (pooled) number for both.
    assert (
        synthetic["lexical_stale_excluded_rate"] != real["lexical_stale_excluded_rate"]
    )
    assert synthetic["vector_stale_excluded_rate"] != real["vector_stale_excluded_rate"]

    # Pooled continues to sum both, unchanged in meaning.
    assert report["summary"]["lexical_stale_excluded"] == (
        synthetic["lexical_stale_excluded"] + real["lexical_stale_excluded"]
    )
    assert report["summary"]["vector_stale_excluded"] == (
        synthetic["vector_stale_excluded"] + real["vector_stale_excluded"]
    )


def test_provenance_split_covers_no_answer_correctness() -> None:
    """No real D2 note is a ``no-answer`` fixture, so the real provenance
    must report ``None`` ("not asked"), never a fake ``True`` -- while the
    synthetic provenance, which does own ``q10-no-answer``, reports an
    actual bool. Two different *kinds* of value, not just two different
    numbers, so a mutant that pools this field cannot coincidentally still
    pass by returning the same bool for both.
    """
    mixed = QUESTIONS + (REAL_QUESTIONS[0],)
    report = _run(questions=mixed)
    provenance = report["summary"]["provenance"]

    assert provenance["synthetic"]["lexical_no_answer_correct"] is True
    assert provenance["real"]["lexical_no_answer_correct"] is None
    assert report["summary"]["lexical_no_answer_correct"] is True


def test_provenance_with_no_questions_reports_none_not_zero() -> None:
    """A provenance absent from ``questions`` must read as "not asked"
    (``None`` hit rate), never as "asked and failed" (``0.0``) -- the same
    honesty rule ``_hit`` already applies per-question, extended to the
    per-provenance summary block.
    """
    only_synthetic = _run(questions=QUESTIONS)["summary"]["provenance"]
    assert only_synthetic["real"]["scored_questions"] == 0
    assert only_synthetic["real"]["lexical_hit_rate"] is None
    assert only_synthetic["real"]["vector_hit_rate"] is None


def test_per_question_report_carries_its_own_provenance() -> None:
    mixed = QUESTIONS + (_synthetic_question_labeled_real("r-test-mixed"),)
    report = _run(questions=mixed)
    by_id = {q["question_id"]: q for q in report["questions"]}
    assert by_id["r-test-mixed"]["provenance"] == "real"
    for question in QUESTIONS:
        assert by_id[question.question_id]["provenance"] == "synthetic"


def test_report_always_carries_the_known_coverage_gaps() -> None:
    report = _run()
    assert report["coverage_gaps"] == list(KNOWN_COVERAGE_GAPS)
    assert report["coverage_gaps"]


# --- token cost: unmeasured must never look like a measured zero ----------
#
# Coordinator review on issue #26: merging this report with #24's real QMD
# baseline (which *can* read a real note's body) would let a silently-zero
# platform token cost "beat" a real QMD number on every real question --
# a false pass on D1's "context token total <= QMD baseline" gate. These
# tests pin ``None``/``token_cost_measured=False`` instead of ``0``.


def _stand_in_real_member():
    """A ``BundleMember`` at a real §D2 path, entirely test-authored body.

    Not real Wiki content -- the body text below is fiction written for this
    test. It reuses ``REAL_QUESTIONS[0]``'s frozen path and matches its query
    terms only so the harness actually retrieves it, exercising the
    "returned path has no body in CORPUS_BODIES" branch without needing
    ``CKP_PILOT_WIKI_ROOT`` or any live checkout (same no-vendoring pattern
    ``tests/test_pilot_no_vendoring.py`` already uses).
    """
    target = REAL_QUESTIONS[0]
    assert target.provenance == "real"
    body = (
        "# stand-in\n\nTest-authored fixture body, not real Wiki content: "
        + target.query
        + ".\n"
    )
    content = f"---\nprivacy: public\n---\n\n{body}".encode()
    return make_member(target.expected_paths[0], content), target


def test_unmeasured_real_hit_reports_none_not_zero_token_cost() -> None:
    stand_in_member, real_question = _stand_in_real_member()
    members = corpus_members() + (stand_in_member,)
    mixed = QUESTIONS + (real_question,)
    # ``_run`` always builds ``members`` from ``corpus_members()``; this test
    # needs the stand-in member indexed too, so it calls the harness
    # directly.
    from benchmarks.shadow import run_shadow_benchmark

    from ckp.index import InMemoryVectorIndex

    report = run_shadow_benchmark(
        members=members,
        stack=deterministic_stack(),
        index_provider=InMemoryVectorIndex(),
        gate=public_gate(),
        top_k=3,
        questions=mixed,
        trials=1,
    )

    by_id = {q["question_id"]: q for q in report["questions"]}
    entry = by_id[real_question.question_id]
    # The stand-in body is actually indexed and matches its own query, so
    # this must be a real hit, not an accidental miss that never reaches the
    # token-cost branch at all.
    assert real_question.expected_paths[0] in entry["lexical"]["paths"]
    assert entry["lexical"]["token_cost"] is None
    assert entry["lexical"]["token_cost_measured"] is False
    assert real_question.expected_paths[0] in entry["vector"]["paths"]
    assert entry["vector"]["token_cost"] is None
    assert entry["vector"]["token_cost_measured"] is False


def test_unmeasured_questions_are_excluded_from_the_token_cost_total() -> None:
    """The unmeasured question's cost must not be folded into the total as 0.

    Adding the stand-in member changes what both engines index, which in
    turn can change *other* questions' retrieved paths and their token
    costs too (a new document can shift vector ranking) -- so this asserts
    internal consistency of one report (total == sum of measured
    per-question costs; unmeasured count == number of ``measured=False``
    entries) rather than diffing against a separately-run baseline whose
    corpus is not actually the same population.
    """
    stand_in_member, real_question = _stand_in_real_member()
    members = corpus_members() + (stand_in_member,)
    mixed = QUESTIONS + (real_question,)

    from benchmarks.shadow import run_shadow_benchmark

    from ckp.index import InMemoryVectorIndex

    report = run_shadow_benchmark(
        members=members,
        stack=deterministic_stack(),
        index_provider=InMemoryVectorIndex(),
        gate=public_gate(),
        top_k=3,
        questions=mixed,
        trials=1,
    )
    summary = report["summary"]

    lexical_measured_sum = sum(
        q["lexical"]["token_cost"]
        for q in report["questions"]
        if q["lexical"]["token_cost_measured"]
    )
    vector_measured_sum = sum(
        q["vector"]["token_cost"]
        for q in report["questions"]
        if q["vector"]["token_cost_measured"]
    )
    lexical_unmeasured_count = sum(
        1 for q in report["questions"] if not q["lexical"]["token_cost_measured"]
    )
    vector_unmeasured_count = sum(
        1 for q in report["questions"] if not q["vector"]["token_cost_measured"]
    )

    assert summary["lexical_token_cost_total"] == lexical_measured_sum
    assert summary["vector_token_cost_total"] == vector_measured_sum
    assert (
        summary["lexical_token_cost_unmeasured_questions"] == lexical_unmeasured_count
    )
    assert summary["vector_token_cost_unmeasured_questions"] == vector_unmeasured_count
    # The stand-in real question is a genuine hit on both engines (asserted
    # in the previous test), so at least one unmeasured question exists on
    # each side -- this guards against the counters trivially staying 0.
    assert lexical_unmeasured_count >= 1
    assert vector_unmeasured_count >= 1

    real_block = summary["provenance"]["real"]
    assert real_block["lexical_token_cost_unmeasured_questions"] >= 1
    assert real_block["vector_token_cost_unmeasured_questions"] >= 1
    # None of the real block's unmeasured hits contributed to its own total.
    assert real_block["lexical_token_cost_total"] == sum(
        q["lexical"]["token_cost"]
        for q in report["questions"]
        if q["provenance"] == "real" and q["lexical"]["token_cost_measured"]
    )
    assert real_block["vector_token_cost_total"] == sum(
        q["vector"]["token_cost"]
        for q in report["questions"]
        if q["provenance"] == "real" and q["vector"]["token_cost_measured"]
    )


# --- unified token-cost measurement across all three sides (issue #40) ---
#
# #24 gave the QMD side a real body reader; #26 made the platform's own
# ``lexical``/``vector`` sides honestly report ``None`` (never a silent 0)
# for a real-provenance hit they could not measure. #40 closes the gap: a
# ``wiki_root`` passed to ``run_shadow_benchmark`` lets those two sides
# measure the same way QMD always could, via the single shared
# ``benchmarks.token_cost.measure_token_cost`` implementation. These tests
# exercise the *wiring* -- that a real integer actually reaches the report
# -- not just the pure helper (``tests/test_token_cost.py`` covers that).


def test_wiki_root_measures_real_provenance_token_cost_on_lexical_and_vector(
    tmp_path: Path,
) -> None:
    stand_in_member, real_question = _stand_in_real_member()
    members = corpus_members() + (stand_in_member,)
    mixed = QUESTIONS + (real_question,)

    from benchmarks.shadow import run_shadow_benchmark

    from ckp.index import InMemoryVectorIndex

    real_path = real_question.expected_paths[0]
    (tmp_path / real_path).parent.mkdir(parents=True, exist_ok=True)
    # A body deliberately different from the stand-in fixture's indexed
    # text and with a known, non-trivial whitespace-token count -- proves
    # the number in the report was actually read from ``wiki_root``, not
    # from the indexed body or a coincidental match.
    real_body = "alpha beta gamma delta epsilon zeta eta"  # 7 tokens
    (tmp_path / real_path).write_text(
        f"---\nprivacy: internal\n---\n\n{real_body}\n", encoding="utf-8"
    )

    report = run_shadow_benchmark(
        members=members,
        stack=deterministic_stack(),
        index_provider=InMemoryVectorIndex(),
        gate=public_gate(),
        top_k=3,
        questions=mixed,
        trials=1,
        wiki_root=tmp_path,
    )

    by_id = {q["question_id"]: q for q in report["questions"]}
    entry = by_id[real_question.question_id]
    real_tokens = len(real_body.split())

    def _expected_total(paths: list[str]) -> int:
        # ``top_k`` may return other, synthetic-corpus paths alongside the
        # real hit -- with ``wiki_root`` given, *every* returned path is now
        # measurable (the real one from disk, the rest from
        # ``CORPUS_BODIES``), so the honest expected total sums all of them,
        # not just the real path this test planted.
        return sum(
            real_tokens if path == real_path else len(CORPUS_BODIES[path].split())
            for path in paths
        )

    assert real_path in entry["lexical"]["paths"]
    assert entry["lexical"]["token_cost_measured"] is True
    assert entry["lexical"]["token_cost"] == _expected_total(entry["lexical"]["paths"])
    # The real path's own contribution is exactly what was planted on disk,
    # not a coincidental match from the indexed (different) stand-in body.
    assert real_tokens <= entry["lexical"]["token_cost"]

    assert real_path in entry["vector"]["paths"]
    assert entry["vector"]["token_cost_measured"] is True
    assert entry["vector"]["token_cost"] == _expected_total(entry["vector"]["paths"])

    # The provenance-split summary reflects the same measured numbers, not
    # an unmeasured count -- proves the value actually flows into the
    # aggregated report, not just the per-question block. This mixed
    # question set's only "real" question is ``real_question`` itself.
    real_block = report["summary"]["provenance"]["real"]
    assert real_block["lexical_token_cost_unmeasured_questions"] == 0
    assert real_block["vector_token_cost_unmeasured_questions"] == 0
    assert real_block["lexical_token_cost_total"] == entry["lexical"]["token_cost"]
    assert real_block["vector_token_cost_total"] == entry["vector"]["token_cost"]


def test_wiki_root_absent_leaves_real_provenance_unmeasured() -> None:
    """The default (``wiki_root=None``) must be byte-for-byte the pre-#40
    behavior -- CI has no wiki checkout, so this path must stay honest.
    """
    stand_in_member, real_question = _stand_in_real_member()
    members = corpus_members() + (stand_in_member,)
    mixed = QUESTIONS + (real_question,)

    from benchmarks.shadow import run_shadow_benchmark

    from ckp.index import InMemoryVectorIndex

    report = run_shadow_benchmark(
        members=members,
        stack=deterministic_stack(),
        index_provider=InMemoryVectorIndex(),
        gate=public_gate(),
        top_k=3,
        questions=mixed,
        trials=1,
        # wiki_root intentionally omitted -- must default to None.
    )

    by_id = {q["question_id"]: q for q in report["questions"]}
    entry = by_id[real_question.question_id]
    assert entry["lexical"]["token_cost"] is None
    assert entry["lexical"]["token_cost_measured"] is False
    assert entry["vector"]["token_cost"] is None
    assert entry["vector"]["token_cost_measured"] is False


def test_wiki_root_never_leaks_the_real_note_body_into_the_report(
    tmp_path: Path,
) -> None:
    """Complements the QMD-side ``test_qmd_report_never_contains_the_real_
    note_body_text`` guard: the lexical/vector token-cost read must count
    and discard, never surface the body text anywhere in the report.
    """
    stand_in_member, real_question = _stand_in_real_member()
    members = corpus_members() + (stand_in_member,)
    mixed = QUESTIONS + (real_question,)

    from benchmarks.shadow import run_shadow_benchmark

    from ckp.index import InMemoryVectorIndex

    real_path = real_question.expected_paths[0]
    (tmp_path / real_path).parent.mkdir(parents=True, exist_ok=True)
    sentinel = "SENTINEL-LEXICAL-VECTOR-BODY-TOKEN-SHOULD-NEVER-LEAK"
    (tmp_path / real_path).write_text(
        f"---\nprivacy: internal\n---\n\n{sentinel}\n", encoding="utf-8"
    )

    report = run_shadow_benchmark(
        members=members,
        stack=deterministic_stack(),
        index_provider=InMemoryVectorIndex(),
        gate=public_gate(),
        top_k=3,
        questions=mixed,
        trials=1,
        wiki_root=tmp_path,
    )

    serialized = json.dumps(report)
    assert sentinel not in serialized


def test_report_shape_and_honesty_fields() -> None:
    report = _run()
    assert report["schema"] == REPORT_SCHEMA
    assert report["semantic"] is False
    assert "not semantic quality" in report["semantic_note"]
    assert report["index_provider"] == "memory-cosine"
    assert report["token_cost_method"] == TOKEN_COST_METHOD
    assert report["token_cost_note"] == TOKEN_COST_NOTE
    # 11 fixture members, 3 non-public: the rebuild indexed exactly the 8
    # public notes, and the report carries no count of what was refused.
    assert report["rebuild"] == {"point_count": 8}
    assert len(report["questions"]) == len(QUESTIONS)
    assert report["summary"]["scored_questions"] == sum(
        1 for question in QUESTIONS if question.expected_paths
    )


def test_both_engines_answer_the_frozen_corpus() -> None:
    summary = _run()["summary"]
    assert summary["lexical_hit_rate"] == 1.0
    assert summary["vector_hit_rate"] == 1.0
    assert summary["lexical_no_answer_correct"] is True
    assert summary["citation_correct_questions"] == len(QUESTIONS)
    assert summary["privacy_false_negatives"] == 0
    assert "privacy_violations" not in summary


def test_report_is_deterministic_apart_from_latency() -> None:
    left = json.dumps(strip_latency(_run()), sort_keys=True, ensure_ascii=False)
    right = json.dumps(strip_latency(_run()), sort_keys=True, ensure_ascii=False)
    assert left == right
    assert "latency_ms" not in left


def test_no_sentinel_or_non_public_path_ever_reaches_the_report() -> None:
    rendered = json.dumps(_run(), sort_keys=True, ensure_ascii=False)
    for forbidden in (
        "INTERNAL-SENTINEL-ALLOCATION",
        "SENSITIVE-SENTINEL-MEETING",
        "STUDENT-SENTINEL-RECORD",
        *sorted(NON_PUBLIC_PATHS),
    ):
        assert forbidden not in rendered


class _LeakyIndex(InMemoryVectorIndex):
    """A rogue provider that surfaces a non-public path in every search.

    ``never-indexed.md`` never existed anywhere -- it proves leak detection
    is judged against the gated plan, not against a fixture list.
    """

    def search(self, query_vector, *, top_k) -> SearchResult:
        honest = super().search(query_vector, top_k=top_k)
        leaked = (
            SearchHit(relative_path="learner-record.md", score=3.0, rank=0),
            SearchHit(relative_path="never-indexed.md", score=2.0, rank=1),
            *(
                SearchHit(
                    relative_path=hit.relative_path,
                    score=hit.score,
                    rank=hit.rank + 2,
                )
                for hit in honest.hits[: top_k - 2]
            ),
        )
        return SearchResult(composed_revision=honest.composed_revision, hits=leaked)


class _PilotPathSpoofingIndex(InMemoryVectorIndex):
    """A rogue provider that surfaces a *real D2 pilot path* on every search,
    without that path ever actually being part of this run's indexed plan.

    Issue #26 Round 2 (Codex Finding 2 fix, coordinator-mandated mutation
    guard): the leak basis must be "was this path actually in the plan this
    run built", never "is this one of the six paths D2 named" -- a pilot
    path is not pre-approved by identity. This run uses the *default*
    ``index_admissible`` (public-only, see ``_run``'s default), so the real
    path here was never indexed -- surfacing it must count as a leak, the
    same as any other unindexed path, not be silently treated as safe just
    because a human would recognize it as "one of the pilot six".
    """

    def search(self, query_vector, *, top_k) -> SearchResult:
        honest = super().search(query_vector, top_k=top_k)
        spoofed = (
            SearchHit(
                relative_path=REAL_QUESTIONS[0].expected_paths[0], score=3.0, rank=0
            ),
            *(
                SearchHit(
                    relative_path=hit.relative_path,
                    score=hit.score,
                    rank=hit.rank + 1,
                )
                for hit in honest.hits[: top_k - 1]
            ),
        )
        return SearchResult(composed_revision=honest.composed_revision, hits=spoofed)


def test_a_known_pilot_path_not_actually_indexed_this_run_is_still_a_leak() -> None:
    report = _run(index_provider=_PilotPathSpoofingIndex())
    spoofed_path = REAL_QUESTIONS[0].expected_paths[0]
    assert report["summary"]["privacy_false_negatives"] > 0
    rendered = json.dumps(report, sort_keys=True, ensure_ascii=False)
    assert spoofed_path not in rendered


def test_privacy_violations_are_counted_and_redacted() -> None:
    report = _run(index_provider=_LeakyIndex())
    # Two leaked paths per question: one non-public fixture, one invented.
    assert report["summary"]["privacy_false_negatives"] == 2 * len(QUESTIONS)
    rendered = json.dumps(report, sort_keys=True, ensure_ascii=False)
    assert "learner-record.md" not in rendered
    assert "never-indexed.md" not in rendered


def test_provenance_split_covers_privacy_false_negatives_asymmetrically() -> None:
    """Issue #26 Round 3 (Codex audit): ``privacy_false_negatives`` is
    zero-tolerance, so pooling it never hides a *pass/fail* problem the way
    an averaged rate can -- but it was still pooled-only, so a reviewer
    could not tell *which* provenance leaked. Split as a diagnostic
    convenience (documented in ``_provenance_summary``), verified here with
    a fixture where the two provenances get a genuinely different raw
    count: ``_LeakyIndex`` injects the same two leaked hits on every single
    search call, so ten synthetic questions accumulate ten times the leaks
    of the one real question mixed in -- 20 vs 2, never equal, and each
    independently checked against its own question count rather than one
    shared pooled figure.
    """
    mixed = QUESTIONS + (REAL_QUESTIONS[0],)
    report = _run(index_provider=_LeakyIndex(), questions=mixed)
    provenance = report["summary"]["provenance"]

    assert provenance["synthetic"]["privacy_false_negatives"] == 2 * len(QUESTIONS)
    assert provenance["real"]["privacy_false_negatives"] == 2 * 1
    assert (
        provenance["synthetic"]["privacy_false_negatives"]
        != provenance["real"]["privacy_false_negatives"]
    )
    assert report["summary"]["privacy_false_negatives"] == (
        provenance["synthetic"]["privacy_false_negatives"]
        + provenance["real"]["privacy_false_negatives"]
    )


def test_privacy_false_positives_and_false_negatives_stay_separate() -> None:
    """FN (leak, zero-tolerance) and FP (over-blocking) must never merge.

    The frozen corpus has zero false positives (every ``public`` note makes
    it into the gated plan), so this pins the honest-zero shape and the key
    names; a mutant that folds one count into the other is caught by
    ``scripts/test-c5-mutations.sh`` M9b, which forces a false positive to
    exist and asserts it does not inflate ``privacy_false_negatives``.
    """
    summary = _run()["summary"]
    assert summary["privacy_false_positives"] == 0
    assert summary["privacy_false_positive_paths"] == []
    assert summary["privacy_false_negatives"] == 0

    leaky_summary = _run(index_provider=_LeakyIndex())["summary"]
    # The leaky provider only ever manufactures false negatives; it must not
    # move the false-positive count at all.
    assert leaky_summary["privacy_false_positives"] == 0
    assert leaky_summary["privacy_false_negatives"] == 2 * len(QUESTIONS)


def test_privacy_false_positives_do_not_inflate_false_negatives() -> None:
    """Force a real false positive and prove it stays out of the FN count.

    A gate that admits no privacy class at all turns every ``public`` corpus
    note into a false positive (declared public, never indexed) while
    leaking nothing (an empty index returns nothing, so false negatives stay
    zero). This is the scenario that actually distinguishes "FP and FN kept
    separate" from "FP silently folded into FN" -- the frozen corpus alone
    has zero false positives, so a merge bug is invisible without forcing
    one here. ``scripts/test-c5-mutations.sh`` M9b targets exactly this
    test.
    """
    blocking_gate = PrivacyGate(FrontmatterClassifier(UNUSED_ROOT), frozenset())
    report = run_shadow_benchmark(
        members=corpus_members(),
        stack=deterministic_stack(),
        index_provider=InMemoryVectorIndex(),
        gate=blocking_gate,
        top_k=3,
        questions=QUESTIONS,
        trials=1,
    )
    summary = report["summary"]
    assert summary["privacy_false_positives"] == len(PUBLIC_DECLARED_PATHS)
    assert summary["privacy_false_positive_paths"] == sorted(PUBLIC_DECLARED_PATHS)
    assert summary["privacy_false_negatives"] == 0


def test_stale_exclusion_is_tracked_per_engine() -> None:
    report = _run()
    summary = report["summary"]
    total = len(QUESTIONS)
    assert 0 <= summary["lexical_stale_excluded"] <= total
    assert 0 <= summary["vector_stale_excluded"] <= total
    assert summary["lexical_stale_excluded_rate"] == round(
        summary["lexical_stale_excluded"] / total, 4
    )
    assert summary["vector_stale_excluded_rate"] == round(
        summary["vector_stale_excluded"] / total, 4
    )
    for question in report["questions"]:
        assert isinstance(question["lexical"]["stale_returned"], bool)
        assert isinstance(question["vector"]["stale_returned"], bool)

    q04 = next(
        q for q in report["questions"] if q["question_id"] == "q04-latest-status"
    )
    outdated = "platform-status-outdated.md"
    assert (outdated in q04["lexical"]["paths"]) == q04["lexical"]["stale_returned"]
    assert (outdated in q04["vector"]["paths"]) == q04["vector"]["stale_returned"]
    assert outdated in SUPERSEDED_PATHS


def test_percentile_is_nearest_rank_not_mean_or_interpolated() -> None:
    """Pin exact values so a mean, or an interpolating quantile, goes red.

    ``[1, 2, 3, 4, 100]`` is chosen so mean (22.0), nearest-rank P50 (3),
    and nearest-rank P95 (100) are all distinct: any implementation that
    silently degrades to the mean, or to an interpolating percentile
    (``statistics.quantiles`` default method gives P50 == 3 too, but a
    different P95 than nearest-rank's 100), is caught by asserting the
    *specific* nearest-rank numbers, not just "some number came back".
    """
    from benchmarks.shadow import _percentile

    samples = [1.0, 2.0, 3.0, 4.0, 100.0]
    mean = sum(samples) / len(samples)
    assert mean == pytest.approx(22.0)

    assert _percentile(samples, 50) == 3.0
    assert _percentile(samples, 95) == 100.0
    assert _percentile(samples, 50) != mean
    assert _percentile(samples, 95) != mean


def test_percentile_of_a_single_sample_is_that_sample() -> None:
    from benchmarks.shadow import _percentile

    assert _percentile([7.5], 50) == 7.5
    assert _percentile([7.5], 95) == 7.5


def test_latency_block_with_a_single_sample_has_p50_and_p95_equal_the_sample() -> None:
    """The per-engine ``latency_ms`` block, with exactly one pooled sample.

    ``run_shadow_benchmark`` with ``trials=1`` still pools one sample *per
    question* (ten questions -> ten samples), so P50 and P95 are not
    generally equal at the report level. This isolates ``_latency_block``
    itself on the single-sample case the coordinator asked to pin: both
    percentiles must equal the one measurement.
    """
    from benchmarks.shadow import _latency_block

    block = _latency_block([0.0421])
    assert block["p50"] == block["p95"] == round(0.0421 * 1000.0, 3)
    assert block["total"] == block["p50"]


def test_token_cost_is_whitespace_split_and_labeled_accordingly() -> None:
    """Bind the ``token_cost_method`` label to the actual computation.

    A mutant that relabels ``TOKEN_COST_METHOD`` (e.g. to
    ``"model-tokenizer"``) without changing how ``token_cost`` is computed
    must fail this test: both assertions live in the same test, against the
    same known note body, so the label and the arithmetic cannot drift
    apart unnoticed.
    """
    from benchmarks.questions import CORPUS_BODIES

    report = _run()
    assert report["token_cost_method"] == "whitespace-proxy"

    espresso_body = CORPUS_BODIES["espresso-dial-log.md"]
    expected_tokens = len(espresso_body.split())
    assert expected_tokens > 0

    q01 = next(q for q in report["questions"] if q["question_id"] == "q01-espresso")
    assert q01["lexical"]["paths"] == ["espresso-dial-log.md"]
    assert q01["lexical"]["token_cost"] == expected_tokens


def test_latency_reports_trials_total_p50_p95() -> None:
    report = _run(trials=4)
    latency = report["latency_ms"]
    assert latency["trials_per_question"] == 4
    for engine in ("lexical", "vector"):
        block = latency[engine]
        assert set(block) == {"total", "p50", "p95"}
        assert block["p50"] <= block["p95"]
        assert block["total"] >= block["p95"]


def test_latency_default_trials_matches_the_named_constant() -> None:
    from benchmarks.shadow import DEFAULT_LATENCY_TRIALS

    report = run_shadow_benchmark(
        members=corpus_members(),
        stack=deterministic_stack(),
        index_provider=InMemoryVectorIndex(),
        gate=public_gate(),
        top_k=3,
        questions=QUESTIONS,
    )
    assert report["latency_ms"]["trials_per_question"] == DEFAULT_LATENCY_TRIALS


def test_all_required_composition_no_defaults() -> None:
    import inspect

    signature = inspect.signature(run_shadow_benchmark)
    parameters = signature.parameters
    # ``trials``, ``index_admissible``, ``qmd_config`` and ``wiki_root`` are
    # the four intentional exceptions: every *other* composition input the
    # harness depends on to build a correct report still has no default.
    # ``trials`` (contract §5.3 latency dimension) keeps existing callers
    # working unchanged. ``index_admissible`` (#26) defaults to public-only
    # because widening it is only ever a deliberate pilot-benchmark choice,
    # never an accident of a caller that forgot it -- production and pre-#26
    # callers must see zero behaviour change from the parameter existing.
    # ``qmd_config`` (#24) defaults to ``None``, meaning "no QMD comparison
    # attempted" -- the same honest-absence shape, not a silent fallback.
    # ``wiki_root`` (#40) defaults to ``None``, meaning "no live wiki
    # checkout available" -- every real-provenance token cost stays
    # unmeasured, the same honest-absence shape ``qmd_config`` established,
    # and every pre-#40 caller sees zero behaviour change.
    # ``questions`` joined the required-no-default set in #26: which
    # questions to score is a composition input exactly like
    # ``members``/``stack``/``gate``, not a hardcoded import.
    assert set(parameters) - {
        "trials",
        "index_admissible",
        "qmd_config",
        "wiki_root",
    } == {
        "members",
        "stack",
        "index_provider",
        "gate",
        "top_k",
        "questions",
    }
    for name, parameter in parameters.items():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        if name == "trials":
            assert parameter.default == 5
        elif name == "index_admissible":
            assert parameter.default == frozenset({PrivacyClass.PUBLIC})
        elif name in ("qmd_config", "wiki_root"):
            assert parameter.default is None
        else:
            assert parameter.default is inspect.Parameter.empty


def test_report_can_be_written_as_stable_json(tmp_path: Path) -> None:
    report = strip_latency(_run())
    target = tmp_path / "shadow-report.json"
    target.write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assert json.loads(target.read_text(encoding="utf-8")) == report


def test_descriptor_admission_is_enforced_by_the_harness() -> None:
    class NotAProvider:
        pass

    from ckp.index.errors import IndexRefusal

    with pytest.raises(IndexRefusal):
        run_shadow_benchmark(
            members=corpus_members(),
            stack=deterministic_stack(),
            index_provider=NotAProvider(),
            gate=public_gate(),
            top_k=3,
            questions=QUESTIONS,
        )


def test_index_descriptor_of_reference_provider() -> None:
    descriptor = InMemoryVectorIndex().descriptor
    assert descriptor == IndexDescriptor(
        contract_version=INDEX_CONTRACT,
        provider_id="memory-cosine",
        provider_version="1",
        schema_version=INDEX_SCHEMA_VERSION,
    )


def test_rebuild_report_is_a_plain_frozen_record() -> None:
    report = InMemoryVectorIndex().rebuild(
        __import__("ckp.index", fromlist=["plan_rebuild"]).plan_rebuild(
            members=corpus_members(),
            stack=deterministic_stack(),
            gate=public_gate(),
        )
    )
    assert isinstance(report, RebuildReport)
    with pytest.raises(AttributeError):
        report.point_count = 0  # type: ignore[misc]


# --- QMD baseline (issue #24, Epic #21 D3) ------------------------------


def _qmd_config(**overrides) -> QmdBaselineConfig:
    defaults = dict(
        index_name="cyclone-wiki",
        wiki_root=UNUSED_ROOT,
        # A non-empty placeholder: every call site below overrides this
        # explicitly. An empty default here would itself be the exact
        # config-time footgun issue #24 Round 2 exists to reject (Codex
        # Round 1 finding) -- ``QmdBaselineConfig`` no longer accepts it.
        corpus_paths=frozenset({"placeholder-unused.md"}),
    )
    defaults.update(overrides)
    return QmdBaselineConfig(**defaults)


def test_qmd_not_compared_by_default() -> None:
    """No ``qmd_config`` -- the untouched shape every pre-#24 caller gets."""
    report = _run()
    assert report["qmd_compared"] is False
    assert report["qmd_index"] is None
    assert report["qmd_unavailable_reason"] is None
    assert report["summary"]["qmd_hit_rate"] is None
    assert report["summary"]["qmd_token_cost_total"] is None
    assert report["latency_ms"]["qmd"] is None
    for question in report["questions"]:
        assert question["qmd"] is None
    # D2 known coverage gap: an explicit note, always present, never an
    # absent or zero-valued numeric field a gate could misread.
    assert "superseded" in report["qmd_stale_exclusion_note"]
    # Codex Round 1: no comparison happened, so scope-overlap is also
    # unknown -- never False (which would imply a comparison did happen).
    assert report["qmd_scope_overlap"] is None


def test_qmd_baseline_populates_the_report_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real QMD adapter call, if it always finds the expected paths.

    Fakes ``run_qmd_query`` so this test never needs a real subprocess --
    the adapter's own subprocess/JSON contract is covered by
    ``test_qmd_adapter.py``. This test is only about how ``shadow.py``
    wires an available QMD baseline into the report.
    """

    def fake_run_qmd_query(query, *, config, top_k):
        # Echo back whatever the current question expects, as if QMD found
        # it perfectly -- lets this test assert a clean qmd_hit_rate of 1.0
        # for the scored questions without depending on real search ranking.
        for question in QUESTIONS:
            if question.query == query:
                return QmdQueryResult(
                    raw_paths=question.expected_paths,
                    paths=question.expected_paths,
                    hits=(),
                )
        raise AssertionError(f"unexpected query {query!r}")

    monkeypatch.setattr("benchmarks.shadow.run_qmd_query", fake_run_qmd_query)

    config = _qmd_config(corpus_paths=frozenset(PUBLIC_DECLARED_PATHS))
    report = _run(qmd_config=config)

    assert report["qmd_compared"] is True
    assert report["qmd_index"] == "cyclone-wiki"
    assert report["qmd_unavailable_reason"] is None
    assert report["summary"]["qmd_hit_rate"] == 1.0
    assert report["latency_ms"]["qmd"] is not None
    for question in report["questions"]:
        assert question["qmd"] is not None
        assert question["qmd"]["paths"] == list(question.get("expected_paths", []))


def test_qmd_failure_mid_run_discards_the_whole_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D3 "跑不了就整份失敗": one failed call nulls every question's qmd data.

    This is the single most important guard in issue #24 -- it must be
    impossible for a QMD outage partway through a run to leave some
    questions "compared" and others silently blank, which would look like
    a partial success instead of the honest "not compared" the acceptance
    criteria require.
    """
    calls = {"count": 0}

    def flaky_run_qmd_query(query, *, config, top_k):
        calls["count"] += 1
        if calls["count"] >= 3:
            raise QmdUnavailable("simulated qmd outage")
        return QmdQueryResult(raw_paths=(), paths=(), hits=())

    monkeypatch.setattr("benchmarks.shadow.run_qmd_query", flaky_run_qmd_query)

    config = _qmd_config(corpus_paths=frozenset(PUBLIC_DECLARED_PATHS))
    report = _run(qmd_config=config)

    assert report["qmd_compared"] is False
    assert report["qmd_unavailable_reason"] == "simulated qmd outage"
    assert report["summary"]["qmd_hit_rate"] is None
    # A mid-run failure must not leave a valid-looking partial token total
    # behind for #30 to gate against.
    assert report["summary"]["qmd_token_cost_total"] is None
    assert report["latency_ms"]["qmd"] is None
    assert report["qmd_scope_overlap"] is None
    for question in report["questions"]:
        assert question["qmd"] is None
    # The failure must not have quietly stopped the run early: every other
    # engine still answered every question.
    assert len(report["questions"]) == len(QUESTIONS)
    # No further qmd calls after the failure -- exactly 3 attempts (2 ok,
    # 1 failing), not one per remaining question.
    assert calls["count"] == 3


def test_qmd_never_silently_falls_back_to_the_platforms_own_lexical_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A QMD outage must produce ``None``, never the lexical block reused.

    This pins the exact failure mode issue #24 exists to prevent: an
    unavailable external baseline quietly being replaced by the platform's
    own engine, which would make the report look like a real external
    comparison happened when it did not.
    """

    def always_fails(query, *, config, top_k):
        raise QmdUnavailable("qmd not installed")

    monkeypatch.setattr("benchmarks.shadow.run_qmd_query", always_fails)

    config = _qmd_config(corpus_paths=frozenset(PUBLIC_DECLARED_PATHS))
    report = _run(qmd_config=config)

    for question in report["questions"]:
        assert question["qmd"] is None
        assert question["qmd"] != question["lexical"]


def test_qmd_scoping_excludes_hits_outside_the_configured_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report only ever sees what the adapter already scoped (D3).

    ``shadow.py`` trusts ``QmdQueryResult.paths`` as pre-filtered -- this
    test proves that trust is exercised: a hit the fake adapter fabricates
    outside the declared corpus never reaches ``qmd_hit_rate`` as a match.
    """

    def fake_run_qmd_query(query, *, config, top_k):
        return QmdQueryResult(
            raw_paths=("never-in-corpus.md",),
            paths=(),  # a real adapter would have already filtered this out
            hits=(),
        )

    monkeypatch.setattr("benchmarks.shadow.run_qmd_query", fake_run_qmd_query)

    config = _qmd_config(corpus_paths=frozenset({"never-in-corpus.md"}))
    report = _run(qmd_config=config)

    assert report["qmd_compared"] is True
    scored_with_expectations = [q for q in report["questions"] if q["expected_paths"]]
    assert scored_with_expectations  # sanity: some questions do expect paths
    for question in scored_with_expectations:
        assert question["qmd"]["hit"] is False
    assert report["summary"]["qmd_hit_rate"] == 0.0
    # Codex Round 1: this run's shape -- QMD returned real hits every
    # question, none ever landed in scope -- is exactly the "looks like a
    # bad score but is actually a config error" signature. ``False`` here
    # is the whole point of the diagnostic: it must not silently read the
    # same as a genuine 0.0 hit rate.
    assert report["qmd_scope_overlap"] is False


def test_qmd_scope_overlap_is_true_once_any_raw_hit_lands_in_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A correctly-scoped, ordinary comparison must read as ``True``.

    Guards the other direction of the Round 1 diagnostic: it must not fire
    (or worse, force ``False``) on a run that is behaving normally.
    """

    def fake_run_qmd_query(query, *, config, top_k):
        return QmdQueryResult(
            raw_paths=("Core/note.md",), paths=("Core/note.md",), hits=()
        )

    monkeypatch.setattr("benchmarks.shadow.run_qmd_query", fake_run_qmd_query)

    config = _qmd_config(corpus_paths=frozenset({"Core/note.md"}))
    report = _run(qmd_config=config)

    assert report["qmd_compared"] is True
    assert report["qmd_scope_overlap"] is True


def test_qmd_scope_overlap_is_none_when_qmd_never_returns_any_raw_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely empty QMD answer is not a scope-mismatch signal.

    When QMD itself never returns anything (``raw_paths`` always empty),
    there is nothing to tell a scope mismatch apart from an honest "QMD
    found nothing for any of these queries" -- the diagnostic must stay
    ``None`` rather than guessing ``False``.
    """

    def fake_run_qmd_query(query, *, config, top_k):
        return QmdQueryResult(raw_paths=(), paths=(), hits=())

    monkeypatch.setattr("benchmarks.shadow.run_qmd_query", fake_run_qmd_query)

    config = _qmd_config(corpus_paths=frozenset({"Core/note.md"}))
    report = _run(qmd_config=config)

    assert report["qmd_compared"] is True
    assert report["qmd_scope_overlap"] is None


def test_qmd_token_cost_total_matches_the_whitespace_proxy_computation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins an exact, non-trivial number -- catches a hardcoded 0 or a
    hardcoded constant the same way ``test_token_cost_is_whitespace_split_
    and_labeled_accordingly`` pins the lexical/vector side.
    """
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    body = "alpha beta gamma delta epsilon"  # 5 whitespace tokens
    (note_dir / "note.md").write_text(
        f"---\nprivacy: internal\n---\n\n{body}\n", encoding="utf-8"
    )

    def fake_run_qmd_query(query, *, config, top_k):
        # Every question "finds" the same one real note -- lets this test
        # pin the total to exactly 5 tokens * len(QUESTIONS) questions.
        return QmdQueryResult(
            raw_paths=("Core/note.md",), paths=("Core/note.md",), hits=()
        )

    monkeypatch.setattr("benchmarks.shadow.run_qmd_query", fake_run_qmd_query)

    config = _qmd_config(wiki_root=tmp_path, corpus_paths=frozenset({"Core/note.md"}))
    report = _run(qmd_config=config)

    expected_total = len(body.split()) * len(QUESTIONS)
    assert report["summary"]["qmd_token_cost_total"] == expected_total
    for question in report["questions"]:
        assert question["qmd"]["token_cost"] == len(body.split())


def test_qmd_report_never_contains_the_real_note_body_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The token-cost read must count and discard, never surface the body.

    Complements ``test_qmd_token_cost_never_writes_any_file`` in
    ``test_qmd_adapter.py`` (which proves nothing is written to disk) by
    proving the *other* direction: nothing the read touches ends up
    serialized into the report this module returns, the same shape as the
    existing ``test_no_sentinel_or_non_public_path_ever_reaches_the_report``
    guard for the synthetic corpus.
    """
    note_dir = tmp_path / "Core"
    note_dir.mkdir()
    sentinel = "SENTINEL-QMD-BODY-TOKEN-SHOULD-NEVER-LEAK"
    (note_dir / "note.md").write_text(
        f"---\nprivacy: internal\n---\n\n{sentinel}\n", encoding="utf-8"
    )

    def fake_run_qmd_query(query, *, config, top_k):
        return QmdQueryResult(
            raw_paths=("Core/note.md",), paths=("Core/note.md",), hits=()
        )

    monkeypatch.setattr("benchmarks.shadow.run_qmd_query", fake_run_qmd_query)

    config = _qmd_config(wiki_root=tmp_path, corpus_paths=frozenset({"Core/note.md"}))
    report = _run(qmd_config=config)

    rendered = json.dumps(report, ensure_ascii=False)
    assert sentinel not in rendered
    # And nothing was written back to the checkout either.
    assert list((tmp_path / "Core").iterdir()) == [tmp_path / "Core" / "note.md"]
