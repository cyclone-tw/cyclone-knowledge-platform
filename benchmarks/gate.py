"""P9 threshold gate (issue #30, Epic #21 D1 frozen): report -> pass/fail.

Turns a ``benchmarks.shadow`` report into a reproducible pass/fail verdict
against the seven D1-frozen dimensions, plus the two D3 QMD preconditions
that make the four relative dimensions meaningful at all. This module never
loosens, reorders, or reinterprets a D1 threshold -- D1 is frozen; changing
it means reopening Epic #21's decision comment, not editing this file.

**Real provenance only.** D1: "門檻套用在 real provenance 那組" -- the
synthetic corpus is questions this implementation wrote to be found, and
grading it against its own self-authored answer key is not evidence of
anything. Every threshold below reads ``report["summary"]["provenance"]
["real"]`` (or, for the QMD side -- see next paragraph -- a real-only slice
recomputed from ``report["questions"]``), never the pooled top-level numbers
and never ``["synthetic"]``. ``latency_ms`` is the one deliberate exception:
``benchmarks.shadow``'s own module docstring says latency is pooled-only by
design (a synthetic question cannot cheaply inflate a wall-clock number the
way it can inflate a hit rate), so this gate follows that established
decision rather than inventing a per-provenance latency split nothing else
in the report supports.

**Why this module recomputes real-provenance QMD stats instead of reading
them from the report.** ``benchmarks.shadow.run_shadow_benchmark`` tracks
``qmd_hit_rate`` / ``qmd_token_cost_total`` pooled only -- its own
``_provenance_summary`` helper never grew a QMD half (see that module's
docstring). Splitting it there is out of scope here: this issue's file
ownership is ``benchmarks/gate.py`` plus this module's own tests, and
``benchmarks/shadow.py`` is a different in-flight branch's (#45) territory
this issue must not touch. Every per-question ``report["questions"][i]``
entry already carries both ``"provenance"`` and a ``"qmd"`` block (or
``None``), so the real-provenance QMD hit rate and token cost this gate
needs can be derived here, from data the report already exposes, without
changing what ``run_shadow_benchmark`` returns.

**RP3 -- a hash baseline is never allowed to decide a gate outcome.** The
vector engine's numbers come from the C4 hash embedding, which declares
``semantic=false``: a deterministic, reproducible baseline, not a claim of
retrieval quality (see ``benchmarks.shadow``'s module docstring and Epic
#21's RP3). Letting a report where ``vector_hit_rate`` clears the QMD
baseline but ``lexical_hit_rate`` does not read as "the platform passed"
would be exactly the self-proof-through-newness RP3 exists to forbid. Every
relative dimension below is therefore decided on ``lexical`` alone --
``lexical`` is also the platform's actual served answer (``ckp.gateway``'s
``query_response`` is what a real caller gets back; ``vector`` is a parallel
shadow computation for comparison, never itself served). ``vector`` numbers
are still carried in each item's ``detail`` for visibility, explicitly
labelled informational, but no code path here lets them flip a verdict --
not even when ``report["semantic"]`` is ``true``. If a later phase adds a
real semantic provider and wants vector numbers to count, that is a new,
reviewed decision, not a default this gate should quietly opt into.

**D3 -- no QMD baseline, no verdict.** ``report["qmd_compared"] is False``
makes ``qmd_compared`` itself fail *and* every one of the four relative
dimensions ``not_evaluable`` (there is nothing to compare against) --
never silently skipped, and never treated as "no evidence against it, so
pass". ``report["qmd_scope_overlap"] is False`` is a second, independent
precondition failure (Epic #21, Codex Round 1 on #24): QMD returned real
hits that never once matched ``corpus_paths``, which reads as a scope or
path configuration mistake wearing a "QMD compared and did badly" costume,
and D3 says to fail loud on it rather than let a misconfigured comparison
produce a passing report.

**D2 -- stale exclusion is a structural, not a missing, gap.** The real
pilot corpus (Epic #21 D2, six notes) has no ``superseded`` relationship, so
QMD-side stale exclusion cannot exist, not merely "was not measured this
run". ``stale_exclusion`` is therefore always ``not_evaluable`` here, with
the reason lifted verbatim from ``report["qmd_stale_exclusion_note"]``, and
``report["coverage_gaps"]`` is carried through this gate's output unchanged
so a reader of the *gate* result (not just the underlying report) can still
see what was never verified on real data.

**``not_evaluable`` is a third state, not a synonym for pass.** A dimension
this gate cannot honestly score (missing baseline, unmeasured token totals
on either side, the structural D2 gap above) is reported as
``not_evaluable`` with a stated reason, and does **not** contribute to
``verdict`` either way -- it neither blocks a pass nor manufactures one.
Silently mapping it to ``"pass"`` would launder "nobody checked" into
"checked and fine", which is the exact failure shape issue #26's
coordinator review caught in the shadow report's own token-cost accounting
one layer down.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unicodedata
from typing import Any

from benchmarks.qmd.adapter import QmdBaselineConfig
from benchmarks.questions import CORPUS, PILOT_QUESTIONS, CorpusNote
from benchmarks.shadow import run_shadow_benchmark
from ckp.bundle import AnchoredBundleReader, BundleMember, MemberRefused
from ckp.config import Config, load_config
from ckp.embedding import build_c4_deterministic_stack
from ckp.index import InMemoryVectorIndex
from ckp.privacy import FrontmatterClassifier, PrivacyClass
from ckp.privacy.gate import PrivacyGate

# ``ckp.pilot`` is imported lazily inside ``run_gate`` below, not here at
# module level. It is benchmark-support tooling deliberately excluded from
# the runtime image (issue #23's Dockerfile: ``RUN rm -rf ./src/ckp/pilot``
# in the runtime stage, before the C7/C8 disposable smoke stage ever copies
# ``benchmarks/`` in). This module is never imported from
# ``benchmarks/__init__.py`` and no container-smoke test file currently
# imports it either, so a module-level ``ckp.pilot`` import would not
# actually break anything *today* -- but keeping the import inside the one
# function that needs it means it never can, even if that changes later,
# with no dependency on remembering why.

GATE_SCHEMA = "ckp-gate-report/1"

#: D3 frozen: the named index, never the flagless default mirror. A literal
#: index *name* is not a host path (AGENTS.md §7 governs paths/hosts, not
#: this kind of logical identifier), so it is safe to pin here the same way
#: ``tests/test_qmd_adapter.py``'s own real-QMD smoke check already does.
QMD_INDEX_NAME = "cyclone-wiki"

#: Matches ``run_qmd_query``'s -n and every platform-side ``top_k`` in the
#: pre-existing shadow/QMD test smoke checks.
DEFAULT_TOP_K = 5

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_NOT_EVALUABLE = "not_evaluable"
#: Privacy false-positive: D1 says "記錄，不設門檻" -- recorded, no
#: threshold. A fourth status keeps it visibly distinct from the three
#: verdict-bearing ones above; nothing that reads ``status`` for pass/fail
#: purposes should ever see this value and mistake it for either.
STATUS_RECORDED = "recorded"

ABSOLUTE = "absolute"
RELATIVE = "relative"
PRECONDITION = "precondition"
QUALITY = "quality"


def _item(
    dimension: str,
    threshold_type: str,
    status: str,
    *,
    detail: dict[str, Any],
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "dimension": dimension,
        "threshold_type": threshold_type,
        "status": status,
        "detail": detail,
        "reason": reason,
    }


def _real_questions(report: dict) -> list[dict]:
    return [q for q in report["questions"] if q["provenance"] == "real"]


def _real_qmd_stats(report: dict) -> dict[str, Any]:
    """Recompute the real-provenance slice of the QMD baseline.

    See the module docstring for why this is recomputed here rather than
    read off a ``summary["provenance"]["real"]`` field that does not exist.
    Mirrors ``run_shadow_benchmark``'s own accounting exactly: a question
    only enters ``scored``/``hits`` when it has ``expected_paths`` (an
    ``expected_paths``-empty question is deliberately unscored, same as the
    platform side), and a ``None`` ``token_cost`` is counted as unmeasured,
    never folded into the total as 0.
    """
    scored = 0
    hits = 0
    token_total = 0
    token_unmeasured = 0
    saw_any_qmd_block = False
    for question in _real_questions(report):
        qmd = question.get("qmd")
        if qmd is None:
            continue
        saw_any_qmd_block = True
        if question["expected_paths"]:
            scored += 1
            if qmd["hit"]:
                hits += 1
        cost = qmd["token_cost"]
        if cost is None:
            token_unmeasured += 1
        else:
            token_total += cost
    return {
        "scored_questions": scored,
        "hits": hits,
        "hit_rate": round(hits / scored, 4) if scored else None,
        "token_cost_total": token_total,
        "token_cost_unmeasured_questions": token_unmeasured,
        "saw_any_qmd_block": saw_any_qmd_block,
    }


def _gate_citation_correctness(report: dict) -> dict[str, Any]:
    real = report["summary"]["provenance"]["real"]
    total = real["question_count"]
    correct = real["citation_correct_questions"]
    detail = {"correct_questions": correct, "total_questions": total}
    if total == 0:
        return _item(
            "citation_correctness",
            ABSOLUTE,
            STATUS_NOT_EVALUABLE,
            detail=detail,
            reason="no real-provenance questions in this run",
        )
    status = STATUS_PASS if correct == total else STATUS_FAIL
    reason = (
        None
        if status == STATUS_PASS
        else f"{total - correct}/{total} real-provenance question(s) had an "
        "incorrect citation binding; D1 requires 100%, not 'mostly correct'"
    )
    return _item("citation_correctness", ABSOLUTE, status, detail=detail, reason=reason)


def _gate_privacy_false_negative(report: dict) -> dict[str, Any]:
    real = report["summary"]["provenance"]["real"]
    real_fn = real["privacy_false_negatives"]
    pooled_fn = report["summary"]["privacy_false_negatives"]
    detail = {"real_false_negatives": real_fn, "pooled_false_negatives": pooled_fn}
    status = STATUS_PASS if real_fn == 0 else STATUS_FAIL
    reason = (
        None
        if status == STATUS_PASS
        else f"{real_fn} non-public path(s) leaked in real-provenance "
        "questions -- RP2 is zero-tolerance, this is a hard failure, not a "
        "deduction"
    )
    return _item(
        "privacy_false_negative", ABSOLUTE, status, detail=detail, reason=reason
    )


def _gate_qmd_compared(report: dict) -> dict[str, Any]:
    compared = report["qmd_compared"]
    detail = {
        "qmd_compared": compared,
        "qmd_unavailable_reason": report.get("qmd_unavailable_reason"),
    }
    status = STATUS_PASS if compared else STATUS_FAIL
    reason = (
        None
        if status == STATUS_PASS
        else "no QMD baseline was obtained this run (D3): the four relative "
        "thresholds have nothing to compare against, so this report cannot "
        "pass -- 'did not take the exam' is not a passing grade"
    )
    return _item("qmd_compared", PRECONDITION, status, detail=detail, reason=reason)


def _gate_qmd_scope_overlap(report: dict) -> dict[str, Any]:
    overlap = report.get("qmd_scope_overlap")
    detail = {"qmd_scope_overlap": overlap}
    status = STATUS_FAIL if overlap is False else STATUS_PASS
    reason = None
    if status == STATUS_FAIL:
        reason = (
            "QMD returned real hits but none of them, in the whole run, ever "
            "fell inside corpus_paths -- Codex Round 1 (#24): this reads as a "
            "scope/path configuration error, not a retrieval-quality result "
            "(D3)"
        )
    return _item(
        "qmd_scope_overlap", PRECONDITION, status, detail=detail, reason=reason
    )


def _gate_answerable_hit_rate(report: dict, qmd_stats: dict) -> dict[str, Any]:
    real = report["summary"]["provenance"]["real"]
    lexical_rate = real["lexical_hit_rate"]
    vector_rate = real["vector_hit_rate"]  # informational only -- see RP3 above.
    qmd_rate = qmd_stats["hit_rate"]
    detail = {
        "platform_lexical_hit_rate": lexical_rate,
        "platform_vector_hit_rate_informational": vector_rate,
        "semantic": report.get("semantic"),
        "qmd_real_hit_rate": qmd_rate,
        "qmd_real_scored_questions": qmd_stats["scored_questions"],
        "decided_on": "lexical",
        "vector_excluded_reason": (
            "RP3: a vector engine reporting semantic=false must never decide "
            "a gate outcome, regardless of its own number"
        ),
    }
    if not report["qmd_compared"] or qmd_rate is None or lexical_rate is None:
        return _item(
            "answerable_hit_rate",
            RELATIVE,
            STATUS_NOT_EVALUABLE,
            detail=detail,
            reason="missing QMD real-provenance baseline or no scored "
            "real-provenance platform questions",
        )
    status = STATUS_PASS if lexical_rate >= qmd_rate else STATUS_FAIL
    reason = (
        None
        if status == STATUS_PASS
        else f"platform lexical hit rate {lexical_rate} is below the QMD "
        f"real-provenance baseline {qmd_rate}"
    )
    return _item("answerable_hit_rate", RELATIVE, status, detail=detail, reason=reason)


def _gate_context_token_total(report: dict, qmd_stats: dict) -> dict[str, Any]:
    real = report["summary"]["provenance"]["real"]
    lexical_total = real["lexical_token_cost_total"]
    lexical_unmeasured = real["lexical_token_cost_unmeasured_questions"]
    qmd_total = qmd_stats["token_cost_total"]
    qmd_unmeasured = qmd_stats["token_cost_unmeasured_questions"]
    detail = {
        "platform_lexical_token_cost_total": lexical_total,
        "platform_lexical_token_cost_unmeasured_questions": lexical_unmeasured,
        "platform_vector_token_cost_total_informational": real[
            "vector_token_cost_total"
        ],
        "qmd_real_token_cost_total": qmd_total,
        "qmd_real_token_cost_unmeasured_questions": qmd_unmeasured,
    }
    if not report["qmd_compared"] or lexical_unmeasured > 0 or qmd_unmeasured > 0:
        return _item(
            "context_token_total",
            RELATIVE,
            STATUS_NOT_EVALUABLE,
            detail=detail,
            reason="a residual (partial) total must never be compared -- "
            "either side has at least one unmeasured question, or there is "
            "no QMD baseline at all",
        )
    status = STATUS_PASS if lexical_total <= qmd_total else STATUS_FAIL
    reason = (
        None
        if status == STATUS_PASS
        else f"platform lexical token total {lexical_total} exceeds the QMD "
        f"real-provenance baseline {qmd_total}"
    )
    return _item("context_token_total", RELATIVE, status, detail=detail, reason=reason)


def _gate_latency_p95(report: dict) -> dict[str, Any]:
    """Pooled, not real-only -- ``benchmarks.shadow``'s own documented design.

    See the module docstring's "Real provenance only" section: latency is
    the one dimension the shadow harness deliberately never split by
    provenance, because a synthetic question cannot cheaply inflate a
    wall-clock number the way it can inflate a hit rate. Both platform
    engines are checked (unlike the hit-rate/token gates, this is a speed
    measurement, not a quality claim, so RP3's "vector must never win on
    quality" does not apply) -- the gate fails if *either* exceeds 2x.
    """
    latency = report["latency_ms"]
    qmd_latency = latency.get("qmd")
    lexical_p95 = latency["lexical"]["p95"]
    vector_p95 = latency["vector"]["p95"]
    detail = {
        "platform_lexical_p95_ms": lexical_p95,
        "platform_vector_p95_ms": vector_p95,
    }
    if not report["qmd_compared"] or qmd_latency is None:
        detail["qmd_p95_ms"] = None
        return _item(
            "latency_p95",
            RELATIVE,
            STATUS_NOT_EVALUABLE,
            detail=detail,
            reason="no QMD baseline: there is no P95 to compare against",
        )
    qmd_p95 = qmd_latency["p95"]
    threshold = qmd_p95 * 2
    detail["qmd_p95_ms"] = qmd_p95
    detail["threshold_ms"] = threshold
    lexical_ok = lexical_p95 <= threshold
    vector_ok = vector_p95 <= threshold
    status = STATUS_PASS if lexical_ok and vector_ok else STATUS_FAIL
    reason = None
    if status == STATUS_FAIL:
        offenders = []
        if not lexical_ok:
            offenders.append(f"lexical {lexical_p95}ms")
        if not vector_ok:
            offenders.append(f"vector {vector_p95}ms")
        reason = (
            f"{', '.join(offenders)} exceed(s) 2x the QMD P95 baseline ({threshold}ms)"
        )
    return _item("latency_p95", RELATIVE, status, detail=detail, reason=reason)


def _gate_stale_exclusion(report: dict) -> dict[str, Any]:
    real = report["summary"]["provenance"]["real"]
    detail = {
        "platform_lexical_real_stale_excluded_rate": real[
            "lexical_stale_excluded_rate"
        ],
        "platform_vector_real_stale_excluded_rate_informational": real[
            "vector_stale_excluded_rate"
        ],
    }
    return _item(
        "stale_exclusion",
        RELATIVE,
        STATUS_NOT_EVALUABLE,
        detail=detail,
        reason=report.get(
            "qmd_stale_exclusion_note",
            "QMD side has no stale-exclusion metric on the real pilot corpus "
            "(D2 known coverage gap)",
        ),
    )


def _gate_privacy_false_positive(report: dict) -> dict[str, Any]:
    summary = report["summary"]
    detail = {
        "count": summary["privacy_false_positives"],
        "paths": summary["privacy_false_positive_paths"],
    }
    return _item(
        "privacy_false_positive",
        QUALITY,
        STATUS_RECORDED,
        detail=detail,
        reason="D1: recorded as a quality signal, no threshold set -- "
        "over-blocking is not a privacy incident",
    )


def evaluate_gate(report: dict) -> dict[str, Any]:
    """Apply the D1-frozen thresholds to one ``benchmarks.shadow`` report.

    Pure function over the report dict -- no filesystem, no subprocess, no
    network. This is what every mutation test in
    ``tests/test_gate.py`` exercises directly, and it is what
    ``run_gate``/``main`` below call after actually producing a report.
    """
    qmd_stats = _real_qmd_stats(report)

    # Degenerate-baseline rule (#58, ruled during the first live run): when
    # QMD was compared but landed *zero* hits across every scored real
    # question, the relative dimensions stop meaning anything in either
    # direction -- hit rate "passes" against 0.0 as trivially as the token
    # total "fails" against a measured 0. A baseline that retrieved nothing
    # is morally the same as no baseline (D3: not compared -> fail), so the
    # relative dimensions all become not_evaluable with the reason stated,
    # and the gate fails with `degenerate_qmd_baseline` on the failing list
    # rather than letting one relative dimension pass for free while its
    # mirror image fails.
    degenerate_baseline = (
        report.get("qmd_compared") is True
        and qmd_stats["scored_questions"] > 0
        and qmd_stats["hits"] == 0
    )
    if degenerate_baseline:
        reason = (
            "QMD baseline is degenerate: zero hits across all "
            f"{qmd_stats['scored_questions']} scored real questions, so "
            "every relative comparison is meaningless in both directions "
            "(#58); treated like an absent baseline per D3"
        )
        relative_items = [
            _item(name, RELATIVE, STATUS_NOT_EVALUABLE, detail=None, reason=reason)
            for name in (
                "answerable_hit_rate",
                "context_token_total",
                "latency_p95",
                "stale_exclusion",
            )
        ]
    else:
        relative_items = [
            _gate_answerable_hit_rate(report, qmd_stats),
            _gate_context_token_total(report, qmd_stats),
            _gate_latency_p95(report),
            _gate_stale_exclusion(report),
        ]
    items = [
        _gate_citation_correctness(report),
        _gate_privacy_false_negative(report),
        _gate_qmd_compared(report),
        _gate_qmd_scope_overlap(report),
        *relative_items,
        _gate_privacy_false_positive(report),
    ]
    failing = [item["dimension"] for item in items if item["status"] == STATUS_FAIL]
    if degenerate_baseline:
        failing.append("degenerate_qmd_baseline")
    verdict = STATUS_FAIL if failing else STATUS_PASS
    not_evaluable = [
        item["dimension"] for item in items if item["status"] == STATUS_NOT_EVALUABLE
    ]
    return {
        "schema": GATE_SCHEMA,
        "verdict": verdict,
        "failing_dimensions": failing,
        "not_evaluable_dimensions": not_evaluable,
        "items": items,
        # D2: carried through verbatim so a reader of *this* gate result,
        # not just the underlying shadow report, can see what was never
        # verified on real data -- issue #26 froze this list; #30 must not
        # be the place it quietly stops showing up.
        "coverage_gaps": list(report.get("coverage_gaps", [])),
        "report_schema": report.get("schema"),
        "corpus_version": report.get("corpus_version"),
        "question_set_version": report.get("question_set_version"),
        "semantic": report.get("semantic"),
        "semantic_note": report.get("semantic_note"),
    }


# --- End-to-end runner (issues #24/#40's deferred wiring, this issue) ------


class GateRunnerError(RuntimeError):
    """The end-to-end runner could not produce a report to gate.

    Distinct from :class:`ckp.pilot.manifest.PilotBindingError` /
    :class:`benchmarks.qmd.adapter.QmdUnavailable`, both of which still
    propagate unhandled from here -- this exception covers only failure
    modes specific to assembling the run (e.g. a real note readable enough
    for ``bind_pilot_corpus`` to verify its hash but not for this runner's
    own second read).
    """


def _make_synthetic_member(note: CorpusNote) -> BundleMember:
    content = note.content
    return BundleMember(
        relative_path=note.relative_path,
        digest_key=unicodedata.normalize("NFC", note.relative_path),
        content=content,
        content_sha256=hashlib.sha256(content).hexdigest(),
    )


def _build_synthetic_members() -> tuple[BundleMember, ...]:
    return tuple(_make_synthetic_member(note) for note in CORPUS)


def _build_real_members(wiki_root, paths: tuple[str, ...]) -> tuple[BundleMember, ...]:
    """Read the frozen pilot notes' actual bytes for indexing.

    ``ckp.pilot.bind_pilot_corpus`` (called first, by ``run_gate`` below)
    already verified each note's hash, privacy class, and forbidden-type
    exclusion -- but by design (see its own docstring) it returns only
    path+hash, never a body. Indexing and querying the notes for real needs
    the actual bytes, so this reads each one again with the same anchored,
    symlink-refusing reader (RP1: the rule is "content never enters this
    repo's tracked source or disk", not "content is never held transiently
    in this process's memory" -- the same reasoning
    ``benchmarks.qmd.adapter.qmd_token_cost`` already relies on).
    """
    reader = AnchoredBundleReader(wiki_root)
    members = []
    for path in paths:
        member = reader.read_member(path)
        if isinstance(member, MemberRefused):
            raise GateRunnerError(
                f"gate runner: {path} was verified by bind_pilot_corpus but "
                f"could not be re-read for indexing ({member.reason}) -- the "
                "checkout may have changed between the two reads"
            )
        members.append(member)
    return tuple(members)


def run_gate(*, config: Config, qmd_binary: str) -> tuple[dict, dict]:
    """Bind the real pilot corpus, run the shadow benchmark, and gate it.

    D5 (fail loud, never skip): a missing or unreadable
    ``CKP_PILOT_WIKI_ROOT`` raises immediately -- there is no synthetic-only
    fallback for this end-to-end path, because a report that silently ran
    synthetic-only would look complete while never having touched real data,
    exactly the failure mode D5 exists to forbid.

    The embedding stack is always the deterministic C4 hash provider
    (``semantic=false``), never the semantic provider from issue #25: this
    keeps the real run's own report honest about what it is (D4's own
    fallback story), and it is what the answerable-hit-rate/token-cost gate
    logic above is built to score anyway (RP3: only ``lexical`` decides).
    """
    from ckp.pilot import PILOT_NOTE_PATHS, bind_pilot_corpus

    wiki_root = config.pilot_wiki_root
    if wiki_root is None or not wiki_root.is_dir():
        raise GateRunnerError(
            "CKP_PILOT_WIKI_ROOT is not set to a real directory; the P9 gate "
            "runner refuses to run without a real Wiki checkout (D5) -- "
            "there is no synthetic-only fallback for the end-to-end gate"
        )

    # Raises PilotBindingError, unhandled, on any hash drift, privacy
    # refusal, or forbidden type -- D5 again, one layer down.
    bind_pilot_corpus(config)

    real_members = _build_real_members(wiki_root, PILOT_NOTE_PATHS)
    members = real_members + _build_synthetic_members()

    stack = build_c4_deterministic_stack(
        dimension=256, embedding_name="hash", reranker_name="cosine"
    )
    index_provider = InMemoryVectorIndex()
    admissible = frozenset({PrivacyClass.PUBLIC, PrivacyClass.INTERNAL})
    privacy_gate = PrivacyGate(FrontmatterClassifier(wiki_root), admissible)

    qmd_config = QmdBaselineConfig(
        index_name=QMD_INDEX_NAME,
        wiki_root=wiki_root,
        corpus_paths=frozenset(PILOT_NOTE_PATHS),
        binary=qmd_binary,
    )

    report = run_shadow_benchmark(
        members=members,
        stack=stack,
        index_provider=index_provider,
        gate=privacy_gate,
        top_k=DEFAULT_TOP_K,
        questions=PILOT_QUESTIONS,
        index_admissible=admissible,
        qmd_config=qmd_config,
        wiki_root=wiki_root,
    )
    return report, evaluate_gate(report)


def main(argv: list[str] | None = None) -> int:
    """``python -m benchmarks.gate``: run the real end-to-end gate, print
    the verdict JSON to stdout, exit 0 on pass and 1 on fail (or on any
    runner error -- ``GateRunnerError``, ``PilotBindingError``, or
    ``QmdUnavailable`` propagating means the run itself did not complete,
    which is not a pass).

    Local-only (D5's environment mirrors ``CKP_REQUIRE_QDRANT``/
    ``CKP_REQUIRE_SEMANTIC``): a real run needs a Wiki checkout and the
    ``qmd`` binary, neither of which CI has. CI never invokes this
    ``__main__`` entry point; it only imports and unit-tests
    ``evaluate_gate`` above against synthetic report fixtures.
    """
    del argv  # No CLI flags today; kept for signature symmetry with main()s
    # elsewhere in this repo and so a future flag does not need a signature
    # change at every call site.
    qmd_binary = os.environ.get("CKP_QMD_BINARY", "qmd")
    config = load_config()
    _report, result = run_gate(config=config, qmd_binary=qmd_binary)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result["verdict"] == STATUS_PASS else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ABSOLUTE",
    "DEFAULT_TOP_K",
    "GATE_SCHEMA",
    "PRECONDITION",
    "QMD_INDEX_NAME",
    "QUALITY",
    "RELATIVE",
    "STATUS_FAIL",
    "STATUS_NOT_EVALUABLE",
    "STATUS_PASS",
    "STATUS_RECORDED",
    "GateRunnerError",
    "evaluate_gate",
    "main",
    "run_gate",
]
