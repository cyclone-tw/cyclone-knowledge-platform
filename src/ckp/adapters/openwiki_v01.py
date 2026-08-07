"""OpenWiki v0.1 -> Cyclone Profile v1 field-level conversion (Epic #21, P7).

OpenWiki is a knowledge-organizing engine, not a second knowledge base
(`cyclone-wiki` `Core/decision-cyclone-wiki-openwiki-role-boundary.md`). Its
output must pass through an adapter before it is treated as a Cyclone
Profile note (contract §2.6). `cyclone-wiki` Phase 2 already proved the
Profile validator *tolerates* the legacy v0.1 fields without erroring
(`tests/fixtures/profile-v1/PF-21-openwiki/`), but that fixture is a
validator-tolerance test, not a conversion: it ships `created_at` *and*
`timestamp` side by side specifically to prove the extra field is ignored.
Nothing there actually turns `timestamp` into `created_at`. This module is
that missing conversion.

Honest metadata (AGENTS.md §8; `cyclone-wiki`
`Core/decision-cyclone-identity-provenance-v1.md` §2.1): a v0.1 `timestamp`
such as ``'2025-05-01'`` carries a date only -- no time-of-day, no timezone.
That is not sufficient evidence to construct an ISO 8601 `created_at`, and
the identity-provenance decision already forbids backfilling `created_at`
from weaker evidence elsewhere (its own example, IP-23, is refusing to backfill
from the UUIDv7-embedded mint timestamp -- "不得用 `id` 時間回填
`created_at`"). The same principle applies here: when the source data is
insufficient, this adapter must not synthesize a plausible time or offset
(no assumed ``T00:00:00``, no assumed ``+08:00``). Instead it falls back to
Profile v1 §6's own legacy-gap mechanism (`legacy: true` +
`metadata_gaps`), which is the sanctioned way to say "this is missing and
here is why" -- the same shape `cyclone-wiki`'s own
`tests/fixtures/profile-v1/PF-14-legacy-ok/` fixture uses.

Notably, the Profile validator's own ISO 8601 check
(`scripts/profile_validator.py::_iso8601_ok`) is *not* strict enough to
catch a naive ``created_at = timestamp`` substitution on its own --
``datetime.fromisoformat('2025-05-01')`` succeeds, so a date-only value
would sail through as a "valid" `created_at` and the validator would stay
silent. This module's own evidence check (`_offset_aware_datetime_evidence`)
is what actually enforces the "has time-of-day and a real UTC offset" bar;
the external validator cannot be relied on for that distinction (see the
mutation tests in `tests/test_openwiki_v01_adapter.py`).

Two things Codex review round 3 caught that are worth recording here so the
next reader does not reintroduce them:

1. **PyYAML parses unquoted ISO timestamps into `datetime`/`date` objects,
   not strings.** OpenWiki v0.1's real output is produced by
   ``Date.toISOString()``, which is unquoted YAML, so `convert_note_text()`
   -- the actual production entry point -- hands `_convert_timestamp` a
   `datetime` object, never a `str`, for the overwhelmingly common real
   case. A version of this adapter that only branched on `isinstance(...,
   str)` silently mis-declared a `created_at` gap on every real OpenWiki
   note despite the timestamp carrying perfectly good evidence. Both
   `str` and `datetime`/`date` inputs are handled below; the `str` branch
   is not dead code -- YAML still gives you a string for *quoted*
   timestamps, and both `convert_openwiki_v01()`'s dict entry point and
   hand-authored fixtures use quoted strings routinely.
2. **Offset acceptance and offset *range* are separate concerns.** Round 2
   widened acceptance to cover `+0800`/`+08` alongside `+08:00`/`Z`; round
   3 found the widened regex also accepted syntactically-shaped nonsense
   like `+99:00` or month `99`. `datetime.fromisoformat` already performs
   real calendar/clock validation (it raises `ValueError` on
   `month=99`, `hour=25`, offsets `>= 24h`, `2025-02-30`, etc.) more
   correctly than a hand-rolled numeric-range regex would, so it does the
   range-checking here instead of a bigger regex. The one thing Python's
   own offset check does *not* enforce is realism -- it allows any offset
   under 24h, including ones no real timezone uses (`+23:59`) -- so
   `_MIN_UTC_OFFSET`/`_MAX_UTC_OFFSET` narrow that down to the actual
   real-world UTC-12..UTC+14 range, which is *not* symmetric: UTC+14
   (Kiribati's Line Islands) is two hours further from UTC than the
   westernmost real offset, UTC-12. `abs(offset) <= 14h` would silently
   let a nonexistent `UTC-14:00` through (issue #37, found in PR #35
   round 3 too, but not blocking that round); the two-sided bound below
   is what actually matches the comment it lives next to.
3. **`datetime.fromisoformat` accepts any single character as the
   date/time separator**, not just `T` and space -- `'2025-05-01x09:30:00
   +08:00'` parses cleanly. Round 3 judged this non-blocking because no
   real source (`Date.toISOString()`, or a hand-written YAML timestamp)
   ever produces that shape, but issue #37 revisits it: since no real
   producer emits anything but `T` or a space, any other separator is
   evidence of a hand-edited or corrupted value, not a form worth
   accepting on the theory that `fromisoformat` happens to parse it.
   `_has_acceptable_separator` enforces that ahead of parsing.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml

# OpenWiki v0.1 keys this adapter actively understands and transforms.
_TIMESTAMP_KEY = "timestamp"
_CITATIONS_KEYS = ("citations", "Citations")

# The real-world range of UTC offsets: UTC-12 (west) through UTC+14
# (Kiribati's Line Islands, east) -- NOT symmetric. Python's own `datetime`
# only refuses offsets >= 24h, which is wider than any offset a real
# timezone actually uses in either direction.
_MIN_UTC_OFFSET = _dt.timedelta(hours=-12)
_MAX_UTC_OFFSET = _dt.timedelta(hours=14)

# The only date/time separators a real producer emits: `T` (every ISO 8601
# string `Date.toISOString()` produces) or a plain space (YAML convention
# for hand-written timestamps). `datetime.fromisoformat` itself accepts any
# single character here, which would let a corrupted or hand-mangled value
# like `2025-05-01x09:30:00+08:00` parse as if it were legitimate evidence.
_VALID_DATETIME_SEPARATORS = ("T", " ")


def _has_acceptable_separator(value: str) -> bool:
    """True unless ``value`` uses a date/time separator other than `T`/space.

    ISO 8601 dates are a fixed-width ``YYYY-MM-DD`` (10 characters), so the
    separator -- when a time-of-day is present at all -- sits at index 10.
    Values with no character there (date-only, or too short to carry a
    separator) are left for ``datetime.fromisoformat``/the sufficiency check
    to reject on their own merits; this function only ever narrows, never
    widens, what already gets through.
    """
    return len(value) <= 10 or value[10] in _VALID_DATETIME_SEPARATORS


def _offset_aware_datetime_evidence(value: str) -> _dt.datetime | None:
    """Parse ``value`` as a full, valid, offset-bearing ISO 8601 timestamp.

    Returns the parsed timezone-aware ``datetime`` when ``value`` carries
    an explicit time-of-day *and* a real UTC offset (``Z``, ``+HH:MM``,
    ``+HHMM``, or ``+HH`` -- all legal ISO 8601 §4.2.5 forms) within the
    real-world UTC-12..UTC+14 range, separated from the date by `T` or a
    space. Returns ``None`` for anything else: date-only strings, naive
    time-of-day strings with no offset, syntactically-shaped-but-nonexistent
    values (month 99, offset +99:00, February 30th) --
    ``datetime.fromisoformat`` already rejects those with a proper
    calendar-aware ``ValueError`` -- and values that use a separator other
    than `T`/space, which no real producer emits.
    """
    if not _has_acceptable_separator(value):
        return None
    try:
        parsed = _dt.datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    return _sufficient_evidence(parsed)


def _sufficient_evidence(parsed: _dt.datetime) -> _dt.datetime | None:
    """Apply the tz-aware + realistic-offset bar to an already-parsed value."""
    if parsed.tzinfo is None:
        return None
    offset = parsed.utcoffset()
    if offset is None or offset < _MIN_UTC_OFFSET or offset > _MAX_UTC_OFFSET:
        return None
    return parsed


# Fields this adapter recognizes as already Cyclone-Profile-shaped, so it
# does not flag them as "unmapped" in the conversion report. Advisory only
# (drives report quality, not validation) -- loosely mirrors the wiki
# validator's KNOWN_KEYS; drift here degrades reporting, never correctness.
_RECOGNIZED_PROFILE_FIELDS = frozenset(
    {
        "type",
        "title",
        "description",
        "resource",
        "tags",
        "sources",
        "generated",
        "verified",
        "status",
        "stale_after",
        "okf_version",
        "id",
        "created_at",
        "privacy",
        "content_category",
        "workflow_status",
        "category_id",
        "topic_id",
        "module_id",
        "allowed_sinks",
        "legacy",
        "metadata_gaps",
        "series_id",
        "created",
        "updated",
        "url",
    }
)


class OpenWikiConversionError(ValueError):
    """Raised when the adapter receives input it cannot honestly process."""


@dataclass(frozen=True, slots=True)
class FieldMapping:
    """One v0.1 -> Profile field transform the adapter actually performed."""

    source_field: str
    target_field: str
    detail: str


@dataclass(frozen=True, slots=True)
class FieldDrop:
    """One field whose raw value the adapter removed from the output.

    Existing purely so nothing disappears silently: every value the adapter
    takes out of the note is reported here, with a reason, even when the
    reason is "we could not honestly convert this."
    """

    field: str
    value: Any
    reason: str


@dataclass(frozen=True, slots=True)
class GapDeclaration:
    """A required field the adapter could not fabricate evidence for.

    Distinct from :class:`FieldDrop`: nothing was removed here, a field
    that was never present is being honestly declared missing via Profile
    v1 §6's legacy-gap mechanism instead of being invented.
    """

    field: str
    reason: str


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """Adapter output plus a full, honest account of what happened to it."""

    metadata: dict[str, Any]
    mapped: tuple[FieldMapping, ...] = ()
    dropped: tuple[FieldDrop, ...] = ()
    gaps: tuple[GapDeclaration, ...] = ()
    unmapped: tuple[str, ...] = ()


def convert_openwiki_v01(metadata: Mapping[str, Any]) -> ConversionResult:
    """Convert one OpenWiki v0.1 frontmatter mapping into Cyclone Profile shape.

    Pure function: takes a parsed frontmatter mapping, returns a new mapping
    plus a full report of every field the adapter touched. Never mutates
    ``metadata``. Fields this adapter does not recognize are passed through
    into the output untouched (never silently dropped) and are additionally
    listed in ``ConversionResult.unmapped`` so callers can see what the
    adapter did *not* claim to understand.
    """
    if not isinstance(metadata, Mapping):
        raise OpenWikiConversionError(
            f"expected a frontmatter mapping, got {type(metadata).__name__}"
        )

    output: dict[str, Any] = dict(metadata)
    mapped: list[FieldMapping] = []
    dropped: list[FieldDrop] = []
    existing_gaps = {
        g for g in (output.get("metadata_gaps") or ()) if isinstance(g, str)
    }
    new_gaps: dict[str, str] = {}

    _convert_citations(metadata, output, mapped, dropped)
    _convert_timestamp(metadata, output, mapped, dropped, new_gaps)
    _declare_generated_gap(metadata, output, new_gaps)

    all_gap_fields = existing_gaps | set(new_gaps)
    if all_gap_fields:
        output["legacy"] = True
        output["metadata_gaps"] = sorted(all_gap_fields)

    gaps = tuple(
        GapDeclaration(field=name, reason=new_gaps[name]) for name in sorted(new_gaps)
    )

    unmapped = tuple(
        sorted(
            key
            for key in metadata
            if key not in _RECOGNIZED_PROFILE_FIELDS
            and key != _TIMESTAMP_KEY
            and key not in _CITATIONS_KEYS
        )
    )

    return ConversionResult(
        metadata=output,
        mapped=tuple(mapped),
        dropped=tuple(dropped),
        gaps=gaps,
        unmapped=unmapped,
    )


def _convert_citations(
    source: Mapping[str, Any],
    output: dict[str, Any],
    mapped: list[FieldMapping],
    dropped: list[FieldDrop],
) -> None:
    # `_CITATIONS_KEYS` order is the declared priority when both the
    # lowercase and capitalized legacy spellings show up at once (dirty
    # legacy data's most common shape is exactly this kind of case
    # variant). Whichever key is *not* chosen must still be accounted for
    # -- either as a recorded duplicate drop, or as a hard failure -- never
    # as a value that just stops existing.
    present_keys = [k for k in _CITATIONS_KEYS if k in source]
    if not present_keys:
        return
    citations_key, *extra_keys = present_keys
    raw_citations = output.pop(citations_key)

    for extra_key in extra_keys:
        extra_value = output.pop(extra_key)
        if extra_value == raw_citations:
            dropped.append(
                FieldDrop(
                    field=extra_key,
                    value=extra_value,
                    reason=(
                        f"`{extra_key}` duplicates `{citations_key}` with the "
                        "identical value -- both legacy spellings were "
                        f"present; keeping `{citations_key}` (declared "
                        f"priority order: {list(_CITATIONS_KEYS)}) and "
                        "dropping the redundant duplicate rather than "
                        "letting it disappear unrecorded."
                    ),
                )
            )
            continue
        raise OpenWikiConversionError(
            f"conflicting legacy citation fields: `{citations_key}`="
            f"{raw_citations!r} and `{extra_key}`={extra_value!r} are both "
            "present with different values. Guessing which one is "
            "authoritative on legacy data is worse than failing loudly; "
            "resolve the conflict in the source note before converting."
        )

    if source.get("sources") is not None:
        dropped.append(
            FieldDrop(
                field=citations_key,
                value=raw_citations,
                reason=(
                    "note already carries `sources`; v0.1 citations dropped "
                    "as superseded rather than silently merged (no de-dup "
                    "rule exists for combining the two)."
                ),
            )
        )
        return

    if (
        isinstance(raw_citations, list)
        and raw_citations
        and all(isinstance(c, str) and c.strip() for c in raw_citations)
    ):
        output["sources"] = list(raw_citations)
        mapped.append(
            FieldMapping(
                source_field=citations_key,
                target_field="sources",
                detail=(
                    f"{len(raw_citations)} citation URL(s) carried over as "
                    "Profile `sources` entries verbatim."
                ),
            )
        )
        return

    dropped.append(
        FieldDrop(
            field=citations_key,
            value=raw_citations,
            reason=(
                "citations was not a non-empty list of non-empty strings; "
                "mapping it into `sources` would require guessing "
                "structure this adapter has no evidence for."
            ),
        )
    )


def _convert_timestamp(
    source: Mapping[str, Any],
    output: dict[str, Any],
    mapped: list[FieldMapping],
    dropped: list[FieldDrop],
    new_gaps: dict[str, str],
) -> None:
    if _TIMESTAMP_KEY not in source:
        return
    raw_timestamp = output.pop(_TIMESTAMP_KEY)

    if source.get("created_at") is not None:
        dropped.append(
            FieldDrop(
                field=_TIMESTAMP_KEY,
                value=raw_timestamp,
                reason=(
                    "note already carries `created_at`; v0.1 timestamp "
                    "dropped as superseded."
                ),
            )
        )
        return

    # PyYAML gives back a `str` for a *quoted* frontmatter value but a
    # `datetime`/`date` object for an *unquoted* one -- and OpenWiki v0.1's
    # real `Date.toISOString()` output is unquoted, so the datetime branch
    # is the one production traffic actually exercises (Codex review round
    # 3; see the module docstring). Both must apply the identical
    # sufficiency bar, or which branch a given note happens to hit becomes
    # a silent source of inconsistent gap declarations.
    created_at_value: str | None = None
    if isinstance(raw_timestamp, str):
        if _offset_aware_datetime_evidence(raw_timestamp) is not None:
            # Keep the caller's exact original text -- do not reformat a
            # value that was already sufficient evidence.
            created_at_value = raw_timestamp
    elif isinstance(raw_timestamp, _dt.datetime):
        if _sufficient_evidence(raw_timestamp) is not None:
            created_at_value = raw_timestamp.isoformat()
    elif isinstance(raw_timestamp, _dt.date):
        # A bare `date` (PyYAML's parse of an unquoted date-only value,
        # e.g. `timestamp: 2025-05-01`) has no time-of-day and no offset at
        # all -- the same insufficiency as the date-only string case, just
        # arriving as a different Python type.
        pass

    if created_at_value is not None:
        output["created_at"] = created_at_value
        mapped.append(
            FieldMapping(
                source_field=_TIMESTAMP_KEY,
                target_field="created_at",
                detail=(
                    "timestamp already carried an explicit time-of-day and "
                    "a real-world UTC offset; mapped to created_at."
                ),
            )
        )
        return

    reason = (
        f"timestamp {raw_timestamp!r} (type {type(raw_timestamp).__name__}) "
        "lacks an explicit time-of-day and/or a real-world UTC offset "
        "(e.g. a date-only value like '2025-05-01', a naive datetime with "
        "no offset, or an offset outside UTC-12..UTC+14). Honest metadata "
        "forbids fabricating the missing time or offset "
        "(decision-cyclone-identity-provenance-v1 §2.1: the same rule that "
        "forbids backfilling created_at from a UUIDv7 mint timestamp "
        "applies here), so created_at is left as a declared legacy gap "
        "(Profile v1 §6) instead of an invented value."
    )
    dropped.append(FieldDrop(field=_TIMESTAMP_KEY, value=raw_timestamp, reason=reason))
    new_gaps["created_at"] = reason


def _declare_generated_gap(
    source: Mapping[str, Any],
    output: dict[str, Any],
    new_gaps: dict[str, str],
) -> None:
    # Only a gap when the caller genuinely never supplied one -- never
    # overwrite or second-guess a `generated` block that was already there.
    if "generated" in source:
        return
    new_gaps["generated"] = (
        "OpenWiki v0.1 has no actor/timestamp provenance equivalent to "
        "Profile's `generated.by`/`generated.at`; no evidence exists to "
        "synthesize an actor or a mint time, so it is declared a legacy "
        "gap (Profile v1 §6) instead of invented."
    )


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate frontmatter keys (no last-key-wins)."""


def _reject_duplicate_keys(loader: _StrictLoader, node: yaml.MappingNode):
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=False)
        if key in mapping:
            raise OpenWikiConversionError(f"duplicate frontmatter key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=False)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _reject_duplicate_keys
)


def convert_note_text(text: str) -> tuple[str, ConversionResult]:
    """Convert one full ``.mdf``/``.md`` document's frontmatter + body.

    Splits ``text`` into YAML frontmatter and body, runs
    :func:`convert_openwiki_v01` on the frontmatter, and re-serializes.
    The body is passed through byte-for-byte -- this adapter is a metadata
    transform, not a content transform.
    """
    if not text.startswith("---"):
        raise OpenWikiConversionError("no YAML frontmatter block (missing '---')")
    end = text.find("\n---", 3)
    if end == -1:
        raise OpenWikiConversionError("unterminated YAML frontmatter block")
    raw_frontmatter = text[3:end]
    rest = text[end + 4 :]
    body = rest[1:] if rest.startswith("\n") else rest

    parsed = yaml.load(raw_frontmatter, Loader=_StrictLoader)
    if not isinstance(parsed, dict):
        raise OpenWikiConversionError("frontmatter did not parse to a mapping")

    result = convert_openwiki_v01(parsed)
    dumped = yaml.safe_dump(
        result.metadata, sort_keys=True, allow_unicode=True, default_flow_style=False
    )
    document = f"---\n{dumped}---\n\n{body}"
    return document, result
