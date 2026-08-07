"""Pin ``CORPUS_VERSION`` and ``QUESTION_SET_VERSION`` to the content they
describe (issue #32).

``benchmarks/questions.py`` promises in its module docstring that
``CORPUS_VERSION`` and ``QUESTION_SET_VERSION`` are bumped together
whenever the content they describe changes. Nothing enforced that promise:
edit a note body or a question's expected paths, forget to bump the
version, and no test noticed. The version numbers land in every shadow
report (``corpus_version`` / ``question_set_version``) and #30 uses them
as the audit trail for which corpus/question content a report was run
against -- a version number that can silently lie is worse than no version
number.

This module computes a deterministic content fingerprint for ``CORPUS``
and for the full question set (``QUESTIONS`` + ``REAL_QUESTIONS``, i.e.
``PILOT_QUESTIONS`` -- the module docstring says the corpus *or either*
question tuple changing means both versions describe stale content) and
pins each fingerprint to the version it was frozen at. If content changes
without a version bump, the freshly computed fingerprint stops matching
the frozen one and the test goes red with a message that says which
version to bump -- not just "update the expected fingerprint", which would
turn this file into exactly the kind of rubber-stamp the issue warns
against.

The fingerprint walks dataclass fields via ``dataclasses.fields`` rather
than hand-listing attribute names, so a field added to ``CorpusNote`` or
``BenchmarkQuestion`` later is automatically covered instead of silently
falling outside the fingerprint.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields
from typing import Any

from benchmarks.questions import (
    CORPUS,
    CORPUS_VERSION,
    QUESTION_SET_VERSION,
    QUESTIONS,
    REAL_QUESTIONS,
)


def _content_fingerprint(items: tuple[Any, ...]) -> str:
    """SHA-256 over every dataclass field of every item, in item order.

    Fields are sorted by name within each item (so reordering the field
    declarations in the dataclass cannot change the fingerprint), but items
    are hashed in the order the tuple provides them in -- reordering
    ``CORPUS`` or ``QUESTIONS`` themselves is a content change too.
    """
    hasher = hashlib.sha256()
    for item in items:
        item_fields = sorted(fields(item), key=lambda f: f.name)
        for field in item_fields:
            value = getattr(item, field.name)
            hasher.update(field.name.encode())
            hasher.update(b"=")
            hasher.update(repr(value).encode())
            hasher.update(b"\n")
        hasher.update(b"---\n")
    return hasher.hexdigest()


#: {CORPUS_VERSION: fingerprint of CORPUS at that version}. Add a new entry
#: (and bump CORPUS_VERSION in benchmarks/questions.py) whenever CORPUS
#: content changes -- do not just replace the old entry's value, or this
#: file degrades into the rubber-stamp the issue warns against.
_CORPUS_FINGERPRINTS_BY_VERSION: dict[str, str] = {
    "3": "162357bd35009399717ddce6a27a29f676fb2fa2ae1e8f1c250371b6dc463c10",
}

#: {QUESTION_SET_VERSION: fingerprint of QUESTIONS + REAL_QUESTIONS at that
#: version}. Same update rule as above, but for question content.
_QUESTION_SET_FINGERPRINTS_BY_VERSION: dict[str, str] = {
    "2": "4a7159150ceb2019a5167f9ff09c7f89e2768e11f9b392bb103659f576724af5",
}


def test_corpus_content_matches_pinned_corpus_version() -> None:
    actual = _content_fingerprint(CORPUS)
    expected = _CORPUS_FINGERPRINTS_BY_VERSION.get(CORPUS_VERSION)
    assert expected is not None, (
        f"CORPUS_VERSION={CORPUS_VERSION!r} has no pinned fingerprint in "
        "tests/test_benchmark_versions.py::_CORPUS_FINGERPRINTS_BY_VERSION. "
        "If you just bumped CORPUS_VERSION, add a new entry for it (do not "
        "reuse or delete the old one)."
    )
    assert actual == expected, (
        f"CORPUS content changed but CORPUS_VERSION is still "
        f"{CORPUS_VERSION!r} (benchmarks/questions.py). Bump "
        "CORPUS_VERSION *and* add a new "
        "{new_version: fingerprint} entry to "
        "_CORPUS_FINGERPRINTS_BY_VERSION in this file -- do not just "
        "update the expected fingerprint for the current version, that "
        "would defeat the point of this test."
    )


def test_question_set_content_matches_pinned_question_set_version() -> None:
    actual = _content_fingerprint(QUESTIONS + REAL_QUESTIONS)
    expected = _QUESTION_SET_FINGERPRINTS_BY_VERSION.get(QUESTION_SET_VERSION)
    assert expected is not None, (
        f"QUESTION_SET_VERSION={QUESTION_SET_VERSION!r} has no pinned "
        "fingerprint in tests/test_benchmark_versions.py::"
        "_QUESTION_SET_FINGERPRINTS_BY_VERSION. If you just bumped "
        "QUESTION_SET_VERSION, add a new entry for it (do not reuse or "
        "delete the old one)."
    )
    assert actual == expected, (
        "QUESTIONS or REAL_QUESTIONS content changed but "
        f"QUESTION_SET_VERSION is still {QUESTION_SET_VERSION!r} "
        "(benchmarks/questions.py). Bump QUESTION_SET_VERSION *and* add a "
        "new {new_version: fingerprint} entry to "
        "_QUESTION_SET_FINGERPRINTS_BY_VERSION in this file -- do not just "
        "update the expected fingerprint for the current version, that "
        "would defeat the point of this test."
    )


def test_corpus_and_question_set_versions_are_independent() -> None:
    """Changing CORPUS must not require bumping QUESTION_SET_VERSION, and
    vice versa -- the two fingerprints are computed from disjoint data, so
    nothing here should ever couple them.
    """
    corpus_fp = _content_fingerprint(CORPUS)
    question_fp = _content_fingerprint(QUESTIONS + REAL_QUESTIONS)
    assert corpus_fp != question_fp, (
        "CORPUS and QUESTIONS+REAL_QUESTIONS fingerprints collided, which "
        "would make it impossible to tell the two apart; this almost "
        "certainly means _content_fingerprint stopped depending on its "
        "input."
    )


def test_fingerprint_is_a_known_value_for_a_known_fixed_input() -> None:
    """Gate the gate: assert the fingerprint helper produces a specific,
    pre-computed value for a fully-controlled fixed input (not anything
    imported from ``benchmarks.questions``), so a future edit that turns
    ``_content_fingerprint`` into (for example) ``return "ok"`` cannot pass
    by coincidence -- it has to keep producing *this* exact digest for
    *this* exact input.
    """
    from benchmarks.questions import CorpusNote

    fixed_note = CorpusNote(
        relative_path="fixture.md",
        privacy="public",
        body="fixed fixture body for the fingerprint self-test",
        superseded=False,
    )
    assert _content_fingerprint((fixed_note,)) == (
        "a684490cef3006487128821f3025b6f754c9e98b873ec56c9686cb76ba48ee8a"
    )


def test_fingerprint_changes_when_a_single_field_changes() -> None:
    """Gate the gate (mutation-shaped): a fingerprint computed over an item
    with one field changed must differ from the original, and a fingerprint
    computed twice over unchanged input must be identical. This is the
    property the whole file relies on; if ``_content_fingerprint`` were
    replaced with something that ignores its input (e.g. always returns a
    constant), this test -- not just the version-pin tests above -- would
    catch it directly.
    """
    from dataclasses import replace

    original = QUESTIONS[0]
    mutated = replace(original, query="a completely different query string")

    fp_original_1 = _content_fingerprint((original,))
    fp_original_2 = _content_fingerprint((original,))
    fp_mutated = _content_fingerprint((mutated,))

    assert fp_original_1 == fp_original_2, (
        "_content_fingerprint is not deterministic for identical input."
    )
    assert fp_original_1 != fp_mutated, (
        "_content_fingerprint did not change when a field changed -- it "
        "may have been reduced to a constant or is not reading all fields."
    )
