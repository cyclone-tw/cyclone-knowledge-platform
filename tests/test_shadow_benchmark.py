"""The §5.3 shadow benchmark: determinism, honesty, and zero privacy leaks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.qmd.adapter import QmdBaselineConfig, QmdQueryResult, QmdUnavailable
from benchmarks.questions import (
    NON_PUBLIC_PATHS,
    PUBLIC_DECLARED_PATHS,
    QUESTIONS,
    SUPERSEDED_PATHS,
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
from ckp.privacy import FrontmatterClassifier
from ckp.privacy.gate import PrivacyGate
from index_fixtures import (
    UNUSED_ROOT,
    corpus_members,
    deterministic_stack,
    public_gate,
)


def _run(
    index_provider=None,
    top_k: int = 3,
    trials: int = 1,
    qmd_config: QmdBaselineConfig | None = None,
) -> dict:
    return run_shadow_benchmark(
        members=corpus_members(),
        stack=deterministic_stack(),
        index_provider=index_provider or InMemoryVectorIndex(),
        gate=public_gate(),
        top_k=top_k,
        trials=trials,
        qmd_config=qmd_config,
    )


def test_question_set_covers_the_contract_categories() -> None:
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

    def search(self, query_vector, *, top_k, filter_privacy) -> SearchResult:
        honest = super().search(
            query_vector, top_k=top_k, filter_privacy=filter_privacy
        )
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


def test_privacy_violations_are_counted_and_redacted() -> None:
    report = _run(index_provider=_LeakyIndex())
    # Two leaked paths per question: one non-public fixture, one invented.
    assert report["summary"]["privacy_false_negatives"] == 2 * len(QUESTIONS)
    rendered = json.dumps(report, sort_keys=True, ensure_ascii=False)
    assert "learner-record.md" not in rendered
    assert "never-indexed.md" not in rendered


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
    )
    assert report["latency_ms"]["trials_per_question"] == DEFAULT_LATENCY_TRIALS


def test_all_required_composition_no_defaults() -> None:
    import inspect

    signature = inspect.signature(run_shadow_benchmark)
    parameters = signature.parameters
    # ``trials`` and ``qmd_config`` are the two intentional exceptions:
    # every composition input the harness depends on to build a correct
    # report still has no default, but the latency repeat count and the
    # opt-in QMD baseline (issue #24; ``None`` means "no QMD comparison
    # attempted", the same honest-absence shape as an unconfigured trial
    # count) are allowed documented defaults so existing callers keep
    # working unchanged.
    assert set(parameters) - {"trials", "qmd_config"} == {
        "members",
        "stack",
        "index_provider",
        "gate",
        "top_k",
    }
    for name, parameter in parameters.items():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        if name == "trials":
            assert parameter.default == 5
        elif name == "qmd_config":
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
