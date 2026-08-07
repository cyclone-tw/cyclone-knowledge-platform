"""Tests for the OpenWiki v0.1 -> Cyclone Profile adapter (Epic #21, P7).

Two purposes, per AGENTS.md §9: prove the happy paths work, and prove the
guards actually guard -- a positive test that only checks "a value ended up
in the field" would stay green even if the adapter quietly fabricated that
value. Every honest-metadata assertion below is paired with a mutation
description in its docstring; the mutations were run by hand against a
scratch copy of the adapter during development (see the session report) and
confirmed red before being reverted -- they are not re-run automatically
here because they require editing production source, which is out of scope
for a test file.

One test in this module, ``test_adapter_output_passes_the_real_profile_validator``,
is a LOCAL-ONLY gate and will NEVER run in CI. It shells out to
`cyclone-wiki`'s own ``scripts/profile_validator.py`` to prove the adapter's
output actually passes real Profile validation -- the core P7 acceptance
criterion -- but `cyclone-wiki` is a private repo containing `sensitive` and
`student-private` content, and checking it out on a CI runner is a hard-stop
privacy violation, not an option to weigh. So CI's `.github/workflows/ci.yml`
never sets `CKP_PILOT_WIKI_ROOT`, and that single test is silently absent
from every CI run by design. To make that absence provable rather than
invisible, `CKP_REQUIRE_WIKI_VALIDATOR=1` turns "no wiki checkout found"
into a hard collection-time failure instead of a skip (see the block
comment above that test for the full rationale and the exact command).
Every other assertion in this file -- all the field-mapping, gap, drop, and
unmapped-field behavior -- is synthetic-only and does run in CI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ckp.adapters.openwiki_v01 import (
    OpenWikiConversionError,
    convert_note_text,
    convert_openwiki_v01,
)

# A raw OpenWiki v0.1 export as it actually looks in the wild: date-only
# timestamp, a citations list, and none of Cyclone's own provenance fields
# (created_at, generated). This is deliberately *not* copied from
# `cyclone-wiki` -- it is a synthetic fixture built to match the field shapes
# documented in that repo's `tests/fixtures/profile-v1/PF-21-openwiki/note.mdf`
# (verified by reading that file directly; see the session report for the
# exact fields it carries).
RAW_V01_METADATA = {
    "type": "Concept",
    "title": "Sample Concept",
    "id": "018f6d00-0000-7000-8000-000000000001",
    "status": "draft",
    "workflow_status": "active",
    "privacy": "internal",
    "content_category": "ai",
    "topic_id": "018f6d00-0000-7000-8000-000000000384",
    "timestamp": "2025-05-01",
    "citations": ["https://example.org"],
}


def test_date_only_timestamp_is_not_fabricated_into_created_at() -> None:
    """Honest metadata: a date-only timestamp must not become a created_at.

    Mutation check: replace the adapter's real branch with
    ``output["created_at"] = raw_timestamp`` unconditionally (the "obvious"
    naive implementation) -- this assertion goes red because created_at
    would then equal ``'2025-05-01'`` instead of being absent.
    """
    result = convert_openwiki_v01(RAW_V01_METADATA)
    assert "created_at" not in result.metadata
    assert "timestamp" not in result.metadata  # raw field is not left behind either


def test_date_only_timestamp_is_not_padded_with_a_guessed_time_or_offset() -> None:
    """Honest metadata: no assumed 00:00 or +08:00 anywhere in the output.

    Mutation check: change the fallback to
    ``output["created_at"] = raw_timestamp + "T00:00:00+08:00"`` -- this
    assertion goes red because that fabricated string would appear in the
    output metadata values.
    """
    result = convert_openwiki_v01(RAW_V01_METADATA)
    values = [str(v) for v in result.metadata.values()]
    assert not any("T00:00:00" in v for v in values)
    assert not any("+08:00" in v for v in values)


def test_date_only_timestamp_declares_a_legacy_gap_instead() -> None:
    """The honest fallback is Profile v1 §6's own legacy-gap mechanism."""
    result = convert_openwiki_v01(RAW_V01_METADATA)
    assert result.metadata["legacy"] is True
    assert result.metadata["metadata_gaps"] == ["created_at", "generated"]
    gap_fields = {g.field for g in result.gaps}
    assert gap_fields == {"created_at", "generated"}


def test_dropped_timestamp_is_reported_not_silently_discarded() -> None:
    result = convert_openwiki_v01(RAW_V01_METADATA)
    dropped_fields = {d.field: d for d in result.dropped}
    assert "timestamp" in dropped_fields
    assert dropped_fields["timestamp"].value == "2025-05-01"
    assert "date-only" in dropped_fields["timestamp"].reason.lower() or (
        "time-of-day" in dropped_fields["timestamp"].reason.lower()
    )


def test_citations_maps_to_sources_verbatim() -> None:
    result = convert_openwiki_v01(RAW_V01_METADATA)
    assert result.metadata["sources"] == ["https://example.org"]
    assert "citations" not in result.metadata
    mapped_targets = {m.target_field for m in result.mapped}
    assert "sources" in mapped_targets


def test_full_precision_timestamp_maps_directly_to_created_at() -> None:
    """When the source data is actually sufficient, use it -- do not gap
    a field just because it came from v0.1; honest metadata cuts both ways.
    """
    metadata = dict(RAW_V01_METADATA)
    metadata["timestamp"] = "2025-05-01T09:30:00+08:00"
    result = convert_openwiki_v01(metadata)
    assert result.metadata["created_at"] == "2025-05-01T09:30:00+08:00"
    assert "created_at" not in {g.field for g in result.gaps}


@pytest.mark.parametrize(
    "raw_timestamp",
    [
        pytest.param("2025-05-01T09:30:00+08:00", id="colon-offset"),
        pytest.param("2025-05-01T09:30:00+0800", id="no-colon-offset"),
        pytest.param("2025-05-01T09:30:00+08", id="hour-only-offset"),
        pytest.param("2025-05-01T09:30:00-05:00", id="negative-colon-offset"),
        pytest.param("2025-05-01T09:30:00.123456+08:00", id="fractional-seconds"),
        pytest.param("2025-05-01T01:30:00Z", id="zulu"),
        pytest.param("2025-05-01 09:30:00+08:00", id="space-separator"),
    ],
)
def test_all_legal_iso8601_offset_forms_map_without_a_gap(raw_timestamp: str) -> None:
    """`+0800`, `+08`, and `+08:00` are all legal ISO 8601 offsets (§4.2.5
    permits omitting the colon and/or the minutes) and all carry identical
    evidence. Rejecting the terser legal forms while accepting the verbose
    one would itself be a honesty bug in the other direction: discarding
    evidence that exists and mislabeling it a gap is not more honest than
    fabricating evidence that does not exist -- Codex review round 2 flagged
    exactly this on `+0800`/`+08`.
    """
    metadata = dict(RAW_V01_METADATA)
    metadata["timestamp"] = raw_timestamp
    metadata["generated"] = {"by": "test/fixture", "at": "2026-07-27T10:00:00+08:00"}
    result = convert_openwiki_v01(metadata)
    assert result.metadata["created_at"] == raw_timestamp
    assert "created_at" not in {g.field for g in result.gaps}
    # generated is supplied here specifically so `legacy`/`metadata_gaps`
    # stay fully absent -- proof that a sufficient timestamp produces zero
    # gaps, not just "one fewer gap than the date-only case".
    assert "legacy" not in result.metadata
    assert "metadata_gaps" not in result.metadata
    assert "timestamp" not in result.metadata


def test_existing_created_at_supersedes_v01_timestamp() -> None:
    metadata = dict(RAW_V01_METADATA)
    metadata["created_at"] = "2026-07-27T10:00:00+08:00"
    result = convert_openwiki_v01(metadata)
    assert result.metadata["created_at"] == "2026-07-27T10:00:00+08:00"
    assert "timestamp" not in result.metadata
    dropped_fields = {d.field for d in result.dropped}
    assert "timestamp" in dropped_fields
    # created_at was already valid evidence -- no gap needed for it.
    assert "created_at" not in {g.field for g in result.gaps}


def test_existing_sources_supersedes_v01_citations() -> None:
    metadata = dict(RAW_V01_METADATA)
    metadata["sources"] = ["https://existing.example.org"]
    result = convert_openwiki_v01(metadata)
    assert result.metadata["sources"] == ["https://existing.example.org"]
    dropped_fields = {d.field for d in result.dropped}
    assert "citations" in dropped_fields


def test_duplicate_citations_case_variant_is_dropped_and_recorded() -> None:
    """`citations` and `Citations` both present with identical values.

    Codex review round 2: case variants are legacy data's most common way
    to duplicate a field, and the pre-fix adapter let the non-priority key
    disappear with no trace in any of mapped/dropped/gaps/unmapped. The
    entire point of those four lists is that nothing leaves the note
    unaccounted for -- a silent-loss path here breaks that guarantee no
    matter how unlikely OpenWiki itself is to emit this shape.

    Mutation check: remove the ``dropped.append(...)`` call in the
    duplicate-key branch of ``_convert_citations`` -- this assertion goes
    red because ``"Citations"`` would no longer appear in
    ``result.dropped``.
    """
    metadata = dict(RAW_V01_METADATA)
    metadata["Citations"] = list(metadata["citations"])  # identical duplicate
    result = convert_openwiki_v01(metadata)
    assert result.metadata["sources"] == ["https://example.org"]
    assert "citations" not in result.metadata
    assert "Citations" not in result.metadata
    dropped_fields = {d.field: d for d in result.dropped}
    assert "Citations" in dropped_fields
    assert "duplicate" in dropped_fields["Citations"].reason.lower()
    # It is accounted for, not just absent from the output -- unmapped
    # must not *also* claim credit for it.
    assert "Citations" not in result.unmapped


def test_conflicting_citations_case_variant_raises() -> None:
    """`citations` and `Citations` present with *different* values: refuse
    to guess which legacy declaration is authoritative rather than pick one
    silently.
    """
    metadata = dict(RAW_V01_METADATA)
    metadata["Citations"] = ["https://different.example.org"]
    with pytest.raises(OpenWikiConversionError, match="conflicting"):
        convert_openwiki_v01(metadata)


def test_present_generated_block_is_never_touched() -> None:
    metadata = dict(RAW_V01_METADATA)
    metadata["generated"] = {"by": "test/fixture", "at": "2026-07-27T10:00:00+08:00"}
    result = convert_openwiki_v01(metadata)
    assert result.metadata["generated"] == {
        "by": "test/fixture",
        "at": "2026-07-27T10:00:00+08:00",
    }
    assert "generated" not in {g.field for g in result.gaps}
    # created_at is still the only gap (timestamp is still date-only).
    assert result.metadata["metadata_gaps"] == ["created_at"]


def test_unknown_field_is_never_silently_dropped() -> None:
    """Acceptance: unknown fields get explicit tolerance behavior, not a
    silent disappearance.

    Mutation check: change ``convert_openwiki_v01`` to build ``output`` from
    only a fixed allowlist of keys instead of ``dict(metadata)`` -- this
    assertion goes red because ``custom_openwiki_field`` would vanish from
    ``result.metadata``.
    """
    metadata = dict(RAW_V01_METADATA)
    metadata["custom_openwiki_field"] = "some future OpenWiki concept"
    result = convert_openwiki_v01(metadata)
    assert result.metadata["custom_openwiki_field"] == "some future OpenWiki concept"
    assert "custom_openwiki_field" in result.unmapped


def test_recognized_profile_fields_are_not_flagged_unmapped() -> None:
    result = convert_openwiki_v01(RAW_V01_METADATA)
    assert "type" not in result.unmapped
    assert "title" not in result.unmapped
    assert "topic_id" not in result.unmapped


def test_convert_openwiki_v01_rejects_non_mapping_input() -> None:
    with pytest.raises(OpenWikiConversionError):
        convert_openwiki_v01("not a mapping")  # type: ignore[arg-type]


def test_convert_note_text_preserves_body_and_title_h1() -> None:
    text = (
        "---\n"
        "type: Concept\n"
        "title: Sample Concept\n"
        "id: 018f6d00-0000-7000-8000-000000000001\n"
        "status: draft\n"
        "workflow_status: active\n"
        "privacy: internal\n"
        "content_category: ai\n"
        "topic_id: 018f6d00-0000-7000-8000-000000000384\n"
        "timestamp: '2025-05-01'\n"
        "citations:\n"
        "- https://example.org\n"
        "---\n"
        "\n"
        "# Sample Concept\n"
        "\n"
        "content.\n"
    )
    document, result = convert_note_text(text)
    assert document.startswith("---\n")
    assert "# Sample Concept\n\ncontent.\n" in document
    assert "timestamp" not in document.split("---", 2)[1]
    assert result.metadata["sources"] == ["https://example.org"]
    assert result.metadata["legacy"] is True


def test_convert_note_text_rejects_missing_frontmatter() -> None:
    with pytest.raises(OpenWikiConversionError):
        convert_note_text("# no frontmatter here\n")


def test_convert_note_text_rejects_duplicate_frontmatter_keys() -> None:
    text = "---\ntype: Concept\ntype: Concept\n---\n\nbody\n"
    with pytest.raises(OpenWikiConversionError):
        convert_note_text(text)


# --- end-to-end: adapter output actually passes the real Profile validator --
#
# This block is a LOCAL-ONLY gate. It will never run in CI, and that is
# deliberate, not an oversight: `cyclone-wiki` is a private repo that holds
# `sensitive` and `student-private` content. Checking it out on a GitHub
# Actions runner would be a hard-stop privacy violation (AGENTS.md §5 /
# global CLAUDE.md hard-stop list), so `.github/workflows/ci.yml` never sets
# `CKP_PILOT_WIKI_ROOT` and never will. That means
# `test_adapter_output_passes_the_real_profile_validator` -- the single
# strongest assertion in this file, "the converted note actually passes
# Profile validation" -- is *always* absent from the CI signal. A bare
# `skipif` would let that fact hide inside a green checkmark forever.
#
# The fix is not to make CI reachable; it is to make local absence loud.
# `CKP_REQUIRE_WIKI_VALIDATOR=1` turns "validator not found" into a hard
# `pytest.fail` at collection time (same pattern as `CKP_REQUIRE_QDRANT=1`
# in `tests/test_index_qdrant.py:27,42`), so anyone who needs this specific
# guarantee -- before a release, before trusting the P7 acceptance
# criterion, during this session's own verification -- has a way to prove
# it actually ran rather than silently skipped:
#
#     CKP_PILOT_WIKI_ROOT=/path/to/cyclone-wiki CKP_REQUIRE_WIKI_VALIDATOR=1 \
#         .venv/bin/python -m pytest tests/test_openwiki_v01_adapter.py
#
# Every other assertion in this module -- field mapping, gap declaration,
# drop reporting, unmapped-field reporting, the timestamp honesty regex --
# is covered by the synthetic (non-wiki) tests above this block and runs in
# CI unconditionally. Only "does the real external validator agree" is
# local-only; the conversion logic itself is not.

_WIKI_ROOT_ENV = "CKP_PILOT_WIKI_ROOT"
_REQUIRE_ENV = "CKP_REQUIRE_WIKI_VALIDATOR"


def _wiki_validator_path() -> Path | None:
    """Locate `cyclone-wiki`'s profile_validator.py via the same env-var
    mechanism the pilot binding (#23, D5) uses. Never a hardcoded host path
    (tests/test_portability.py forbids that); unset means the check is
    skipped, not faked.
    """
    root = os.environ.get(_WIKI_ROOT_ENV)
    if not root:
        return None
    candidate = Path(root) / "scripts" / "profile_validator.py"
    return candidate if candidate.is_file() else None


_wiki_validator = _wiki_validator_path()
_wiki_validator_required = os.environ.get(_REQUIRE_ENV) == "1"

if _wiki_validator_required and _wiki_validator is None:
    # Collection-time hard failure, not a runtime skip -- mirrors
    # tests/test_index_qdrant.py:42's pytest.fail() for CKP_REQUIRE_QDRANT.
    pytest.fail(
        f"{_REQUIRE_ENV}=1 but no usable cyclone-wiki checkout was found "
        f"via {_WIKI_ROOT_ENV} (set it to a checkout containing "
        "scripts/profile_validator.py)"
    )

needs_wiki_validator = pytest.mark.skipif(
    _wiki_validator is None,
    reason=f"{_WIKI_ROOT_ENV} not set or cyclone-wiki checkout not found; "
    "set it to a cyclone-wiki checkout to run this against the real "
    f"Profile validator ({_REQUIRE_ENV}=1 turns this into a hard failure "
    "instead of a skip)",
)


@needs_wiki_validator
def test_adapter_output_passes_the_real_profile_validator(tmp_path) -> None:
    text = (
        "---\n"
        "type: Concept\n"
        "title: Sample Concept\n"
        "id: 018f6d00-0000-7000-8000-000000000001\n"
        "status: draft\n"
        "workflow_status: active\n"
        "privacy: internal\n"
        "content_category: ai\n"
        "topic_id: 018f6d00-0000-7000-8000-000000000384\n"
        "timestamp: '2025-05-01'\n"
        "citations:\n"
        "- https://example.org\n"
        "---\n"
        "\n"
        "# Sample Concept\n"
        "\n"
        "content.\n"
    )
    document, result = convert_note_text(text)
    assert result.metadata["legacy"] is True  # sanity: this is the gap path

    note_path = tmp_path / "converted-note.mdf"
    note_path.write_text(document, encoding="utf-8")

    validator = _wiki_validator_path()
    assert validator is not None
    completed = subprocess.run(
        [
            sys.executable,
            str(validator),
            str(note_path),
            "--mode",
            "migration",
            "--level",
            "governed",
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"converted note failed Profile validation:\n{completed.stdout}\n"
        f"{completed.stderr}"
    )


def test_require_flag_fails_hard_instead_of_skipping_when_validator_missing() -> None:
    """CKP_REQUIRE_WIKI_VALIDATOR=1 must fail the run, never quietly skip.

    This is the automated form of the mutation check the coordinator asked
    for: it spawns a real, isolated pytest subprocess with the require flag
    set and `CKP_PILOT_WIKI_ROOT` deliberately unset -- exactly CI's actual
    environment -- and asserts the subprocess run *fails*. It needs no
    cyclone-wiki checkout to run and is therefore itself CI-safe, unlike the
    gate it is checking.

    Mutation check: change the module-level ``pytest.fail(...)`` above back
    to ``pytest.skip(...)`` -- this test goes red, because the subprocess
    would then exit 0 (a clean skip) instead of failing.
    """
    child_env = dict(os.environ)
    child_env[_REQUIRE_ENV] = "1"
    child_env.pop(_WIKI_ROOT_ENV, None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            str(Path(__file__).resolve()),
            "-k",
            "test_adapter_output_passes_the_real_profile_validator",
        ],
        capture_output=True,
        text=True,
        env=child_env,
        cwd=Path(__file__).resolve().parent,
    )
    # The return code is the load-bearing assertion: any CI gate keyed off
    # "pytest exit code == 0" already treats every nonzero exit as a
    # failure, whether that is a proper `pytest.fail()` collection error
    # (exit 2, "1 error during collection") or a mistaken module-level
    # `pytest.skip(..., allow_module_level=True)` that leaves nothing for
    # `-k` to match (exit 5, "no tests ran"). Both were produced and
    # confirmed here while developing this guard -- the correct
    # implementation must additionally produce the collection-error shape,
    # not just "nothing ran", so the second assertion pins that too.
    assert completed.returncode != 0, (
        f"{_REQUIRE_ENV}=1 with no cyclone-wiki checkout must fail the run, "
        f"not skip it -- got exit code {completed.returncode}:\n"
        f"{completed.stdout}\n{completed.stderr}"
    )
    assert "error" in completed.stdout.lower() or "error" in completed.stderr.lower(), (
        f"expected a collection-time ERROR (pytest.fail semantics), not just "
        f"a nonzero exit -- got exit code {completed.returncode}:\n"
        f"{completed.stdout}\n{completed.stderr}"
    )
