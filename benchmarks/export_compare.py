"""P8 (issue #29): current export vs generated Catalog comparison.

Independent module, independent CLI entrypoint. Deliberately does not
import or modify ``benchmarks/shadow.py``, ``benchmarks/questions.py``,
``benchmarks/qmd/``, or ``benchmarks/resilience*.py`` -- this batch's file
ownership (AGENTS.md §3, four agents working ``benchmarks/`` in parallel)
puts those under other issues.

## What "current export" is

Investigated, not assumed (issue #29 says this is the biggest unknown):
Cyclone-Dashboard reads a JSON file at
``$CYCLONE_WIKI_ROOT/.cache/wiki-export.v1.json`` (or
``$CYCLONE_WIKI_EXPORT_PATH`` if set) -- see Cyclone-Dashboard
``src/lib/wiki/wiki-adapter.ts`` (``defaultExportPath`` /
``readWikiExport``) and its API route ``src/app/api/wiki/export/route.ts``.
The file is produced by ``cyclone-wiki`` ``scripts/wiki_dashboard_export.py``
(schema ``wiki-export.v1``): a curated dashboard feed, *not* a full note
inventory.

This module never regenerates the export and never invents a substitute.
It reads whatever file is currently on disk at the configured path -- the
same file Dashboard is currently reading from, staleness included -- and
compares it against a freshly generated Catalog scoped to the pilot
manifest (D2/#23's frozen six real notes; scope-matching is the same
discipline D3 applies to the QMD baseline: comparing different scopes is
not a comparison).

## Round 2: structural absence vs stale absence are data, not prose

Round 1 review (Codex, correctly) rejected a version of this module that put
"structural gap vs staleness" only in this docstring while every unmatched
pilot note fell into one undifferentiated ``missing_from_export`` bucket in
the actual report. That claimed a guarantee the code did not implement.

This version makes the distinction a field, computed two ways and never
guessed:

* :func:`zone_eligibility` restates, as data, exactly which
  ``wiki-export.v1`` zone(s) *could ever* represent a given note -- sourced
  from ``cyclone-wiki`` ``scripts/wiki_dashboard_export.py``,
  ``scripts/development_candidates.py``, and ``scripts/wiki_topics.py``
  (each rule cites the function it mirrors). If a note is eligible for zero
  zones, its absence is ``reason: "structural"`` regardless of how fresh
  the export is -- freshness is not even consulted, because it cannot
  change the answer.
* For a zone-eligible note, :func:`freshness_verdict` asks git (via the
  injectable :class:`GitOps`) whether the export's ``commit`` predates the
  note's last change in the same Wiki checkout. ``"stale"`` means yes --
  ``reason: "freshness"``. Anything the git signal cannot establish
  (missing commit info, ambiguous ref, not a git checkout) is
  ``reason: "unknown"``, never guessed into one of the other two buckets
  (round 2 feedback, point 4).

## Report body vs privacy (issue #29 privacy boundary)

The diff report never carries note bodies, excerpts, or metadata *values*
(RP1; ``tests/test_no_wiki_content.py``). It carries only: relative paths
(also present in ``config/pilot-manifest.toml``, already committed), zone
*names* (a fixed vocabulary from ``wiki-export.v1``'s schema, not wiki
content), git commit hashes (repository metadata, not note content), and
metadata *field names* that disagree (``"title"``, ``"status"`` -- never
the disagreeing strings themselves).

## Known limitation carried over from Round 1, not yet fixed

``metadata_mismatch``'s ``title`` comparison has a definitional gap: every
``wiki-export.v1`` zone derives its ``title`` field from the note's first
Markdown ``# `` heading (``wiki_dashboard_export.first_heading_body`` /
``development_candidates.note_title`` both do this), while this module's
Catalog side reads the frontmatter ``title:`` field
(``ckp.catalog.parser.project_member``). A note whose heading text and
frontmatter title differ will show as ``metadata_mismatch`` even against a
perfectly fresh export -- that is a comparison of two different fields
wearing the same name, not evidence of drift. Fixing it needs the Catalog
side to also expose an h1-derived title, which lives in ``src/ckp/``
(outside this issue's file ownership); flagged here rather than silently
producing a false positive category.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from ckp.bundle import (
    AnchoredBundleReader,
    BundleMember,
    BundleSnapshot,
    MemberRefused,
    compute_index_revision_from_members,
)
from ckp.catalog import CatalogBuilder
from ckp.catalog.models import CatalogSnapshot
from ckp.config import Config, load_config
from ckp.pilot.manifest import (
    PILOT_NOTE_PATHS,
    PilotBindingError,
    PilotManifest,
    bind_pilot_corpus,
)
from ckp.privacy import FrontmatterClassifier, PrivacyClass, PrivacyGate

REPORT_SCHEMA = "ckp-export-compare-report/2"

#: Override for where the "current export" lives. Unset means "derive from
#: the pilot Wiki checkout root", mirroring how Cyclone-Dashboard's own
#: ``CYCLONE_WIKI_EXPORT_PATH`` overrides its otherwise-derived default
#: (``wiki-adapter.ts``). This repo does not read Dashboard's env var name
#: directly -- that would silently couple this module's config surface to
#: another repo's -- it defines its own, defaulting to the same file.
#:
#: Deliberately **not** prefixed ``CKP_``: ``ckp.config`` fails closed on
#: every ``CKP_*`` environment variable it does not recognize (its own
#: unknown-key guard), and this module is not part of that schema (it lives
#: in ``benchmarks/``, outside ``src/ckp/`` ownership for this issue) -- a
#: ``CKP_``-prefixed name here would make ``load_config()`` raise
#: ``ConfigError`` the moment both this variable and any real
#: ``CKP_PILOT_*`` variable are set together, which is every real run.
EXPORT_PATH_ENV = "EXPORT_COMPARE_PATH"

#: Same frozen admission set D2 pinned for the pilot corpus
#: (``ckp.pilot.manifest._PILOT_ADMISSIBLE``). Restated rather than imported
#: from that private name: an accidental change to one is then a visible
#: two-place diff instead of a silent desync.
_PILOT_ADMISSIBLE: frozenset[PrivacyClass] = frozenset({PrivacyClass.INTERNAL})

#: wiki-export.v1 zones that carry a per-item ``path`` field addressing a
#: wiki note -- the only zones this comparison can ever match against the
#: pilot corpus. Named explicitly (not "everything that happens to have a
#: path key") so a schema change to wiki_dashboard_export.py is a visible
#: diff here, not a silently-widened or silently-narrowed comparison.
PATH_ADDRESSABLE_ZONES: tuple[str, ...] = (
    "knowledge_feed",
    "projects",
    "development_candidates",
    "life_domains",
    "topics",
)

#: Zones with no per-item note path -- named so their absence from
#: ``PATH_ADDRESSABLE_ZONES`` is documented as deliberate, not overlooked.
_NON_PATH_ZONES: tuple[str, ...] = ("shared_now", "agent_activity")

_EXPECTED_SCHEMA = "wiki-export.v1"

_GIT_TIMEOUT_SECONDS = 5.0


class ExportUnavailable(RuntimeError):
    """The current export could not be read as a comparable artifact.

    Every caller of :func:`load_current_export` must let this propagate
    (or handle it by marking the report ``compared: false`` -- never by
    treating the export as present-but-empty). Issue #29's single named
    failure mode is a caller that quietly does the latter and reports zero
    differences; this exception type exists so that mistake requires
    actively catching and discarding it, which is a visible, greppable act.
    """


@dataclass(frozen=True)
class ExportEntry:
    """One path-addressable item read from the current export.

    Deliberately narrow: zone, path, and the two metadata fields this
    module compares (``title``, ``status``). No excerpt, no body, no other
    frontmatter field -- extending this dataclass to carry more is exactly
    the kind of change that needs a matching look at the privacy boundary
    in this module's docstring.
    """

    zone: str
    path: str
    title: str | None
    status: str | None


def _normalize_path(path: str) -> str:
    """Canonicalize a path for matching only -- never for reporting.

    Strips a leading ``./`` and collapses ``//`` / ``..`` segments via
    ``posixpath.normpath``. Two paths that normalize the same but are not
    byte-identical are flagged ``path_mismatch`` using the *raw* forms.
    """
    value = path.strip()
    if value.startswith("./"):
        value = value[2:]
    if not value:
        return value
    return posixpath.normpath(value)


def default_export_path(wiki_root: Path) -> Path:
    """Same default Cyclone-Dashboard's ``wiki-adapter.ts`` derives."""
    return wiki_root / ".cache" / "wiki-export.v1.json"


def resolve_export_path(wiki_root: Path, env: dict[str, str] | None = None) -> Path:
    active_env = os.environ if env is None else env
    override = active_env.get(EXPORT_PATH_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return default_export_path(wiki_root)


def _extract_path_addressable_entries(zones: dict) -> list[ExportEntry]:
    entries: list[ExportEntry] = []
    for zone_name in PATH_ADDRESSABLE_ZONES:
        items = zones.get(zone_name)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if not isinstance(path, str) or not path:
                continue
            title = item.get("title")
            status = item.get("status")
            entries.append(
                ExportEntry(
                    zone=zone_name,
                    path=path,
                    title=title if isinstance(title, str) else None,
                    status=status if isinstance(status, str) else None,
                )
            )
    return entries


@dataclass(frozen=True)
class CurrentExport:
    path: Path
    exported_at: str | None
    commit: str | None
    entries: tuple[ExportEntry, ...]


def load_current_export(export_path: Path) -> CurrentExport:
    """Read and parse the export file. Raises :class:`ExportUnavailable`.

    Never returns a value representing "no export" -- absence, an
    unreadable file, invalid JSON, and a schema mismatch are all the same
    kind of failure from this comparison's point of view (issue #29: "取不到
    就明確標記，不靜默當作零差異"), so they all raise the same exception
    rather than three of the four raising and the fourth quietly returning
    an empty :class:`CurrentExport`.
    """
    try:
        raw = export_path.read_bytes()
    except OSError as exc:
        raise ExportUnavailable(
            f"current export unreadable at {export_path}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportUnavailable(
            f"current export at {export_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ExportUnavailable(f"current export at {export_path} is not a JSON object")
    if payload.get("schema") != _EXPECTED_SCHEMA:
        raise ExportUnavailable(
            f"current export at {export_path} does not declare "
            f"schema={_EXPECTED_SCHEMA!r} (found {payload.get('schema')!r})"
        )
    zones = payload.get("zones")
    if not isinstance(zones, dict):
        raise ExportUnavailable(
            f"current export at {export_path} has no 'zones' object"
        )

    entries = tuple(_extract_path_addressable_entries(zones))
    exported_at = payload.get("exported_at")
    commit = payload.get("commit")
    return CurrentExport(
        path=export_path,
        exported_at=exported_at if isinstance(exported_at, str) else None,
        commit=commit if isinstance(commit, str) else None,
        entries=entries,
    )


# --- zone eligibility: wiki_dashboard_export.py's selection rules, --------
# --- restated as data (Round 2 requirement) --------------------------------


@dataclass(frozen=True)
class NoteFrontmatterFacts:
    """Only the scalar facts zone eligibility needs -- never surfaced
    verbatim anywhere in the report (only the derived zone *names* are).
    """

    path: str
    status: str | None
    development_candidate: bool
    library_id: str | None
    privacy: str | None


#: ``cyclone-wiki`` ``scripts/wiki_topics.py`` ``TOPIC_PROFILES["owner"]``.
#: The "owner" profile, not the bare script's "core" default, because
#: ``scripts/wiki-dashboard-export.sh`` -- the wrapper that actually
#: produces the file Dashboard reads -- defaults ``topics_profile`` to
#: ``"owner"`` (its own comment: "the local owner-facing dashboard pipeline
#: defaults to owner topics"); only a bare, flagless
#: ``wiki_dashboard_export.py`` invocation defaults to "core". Widest
#: realistic case; verified to have zero effect on any of today's six pilot
#: notes, none of which carry a ``library_id`` at all.
_TOPICS_PRIVACY_ALLOWLIST = frozenset(
    {"public", "internal", "sensitive", "student-private"}
)


#: Every rule below cites the exact function it mirrors in
#: ``cyclone-wiki`` (read 2026-08-07, wiki HEAD ``54040ca``). This is a
#: restatement, never an import -- this repo does not execute another
#: repo's scripts -- so a future change to the source is a silent desync
#: until someone re-reads it and updates this comment+code together; that
#: risk is why each branch names precisely what it mirrors.
def zone_eligibility(facts: NoteFrontmatterFacts) -> tuple[str, ...]:
    """Which ``wiki-export.v1`` zone(s) could *ever* represent this note.

    Path-shape and required-field rules only. Two known narrowings are
    deliberately not modeled because they are moot for the frozen six-note
    pilot corpus (D2 restricts it to ``Core/...`` paths only):

    * ``knowledge_feed`` additionally truncates to the 30
      lexicographically-last filenames in its directory
      (``wiki_dashboard_export.build_export``).
    * ``life_domains`` additionally keeps only the single
      lexicographically-last file per domain (same function).

    Neither pilot note lives under ``Private/_inbox/info-collect/`` or
    ``Private/Life/*/reports/`` at all, so path-shape alone already
    excludes every pilot note from both zones regardless of the cutoff.
    """
    path = PurePosixPath(facts.path)
    parts = path.parts
    eligible: list[str] = []

    # knowledge_feed (wiki_dashboard_export.build_export): direct child of
    # Private/_inbox/info-collect/.
    if len(parts) == 4 and parts[:3] == ("Private", "_inbox", "info-collect"):
        eligible.append("knowledge_feed")

    # life_domains (wiki_dashboard_export.build_export): a
    # Private/Life/<domain>/reports/*.md file.
    if (
        len(parts) == 5
        and parts[0] == "Private"
        and parts[1] == "Life"
        and parts[3] == "reports"
    ):
        eligible.append("life_domains")

    # projects (wiki_dashboard_export.active_projects): direct child of
    # Core/, filename starts with "project-", AND frontmatter
    # status == "active" -- both conditions are in the source function
    # (`core.glob("project-*.md")` then `if fm.get("status") != "active":
    # continue`).
    if (
        len(parts) == 2
        and parts[0] == "Core"
        and path.name.startswith("project-")
        and facts.status == "active"
    ):
        eligible.append("projects")

    # development_candidates (development_candidates.collect_development_
    # candidates -> candidate_record): anywhere under Core/_inbox/ (an
    # `rglob("*.md")`), gated purely by `development_candidate: true`.
    # `candidate_record()` returns a record for any such note regardless of
    # `validate_candidate_fields()`'s findings -- capture_kind /
    # capture_surface only populate that record's `validation_errors` list,
    # they do not gate whether the note is included at all.
    if (
        len(parts) >= 3
        and parts[0] == "Core"
        and parts[1] == "_inbox"
        and facts.development_candidate
    ):
        eligible.append("development_candidates")

    # topics (wiki_topics.collect_topics / _allowed): any Core/ or Private/
    # note with a non-empty library_id and a privacy value inside the
    # configured profile's allowlist.
    if facts.library_id and facts.privacy in _TOPICS_PRIVACY_ALLOWLIST:
        eligible.append("topics")

    return tuple(eligible)


_FRONTMATTER_LINE_RE = re.compile(r"^([A-Za-z0-9_]+):[ \t]*(.*)$")


def _frontmatter_lines(content: bytes) -> list[str] | None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = [line.rstrip("\r") for line in text.split("\n")]
    if not lines or lines[0] != "---":
        return None
    for index in range(1, len(lines)):
        if lines[index] == "---":
            return lines[1:index]
    return None


def extract_note_frontmatter_facts(member: BundleMember) -> NoteFrontmatterFacts:
    """Parse only the scalar fields :func:`zone_eligibility` needs.

    A minimal line-based reader (same shape as
    ``ckp.pilot.manifest._frontmatter_body``), not the strict YAML parser
    ``ckp.catalog.parser`` uses -- this module intentionally does not reach
    into that private parsing internals, and every field read here is a
    short scalar (a status word, a boolean, an id), never a value long
    enough to be note content.
    """
    lines = _frontmatter_lines(member.content) or []
    fields: dict[str, str] = {}
    for line in lines:
        match = _FRONTMATTER_LINE_RE.match(line.strip())
        if match:
            fields[match.group(1)] = match.group(2).strip().strip("'\"")

    return NoteFrontmatterFacts(
        path=member.relative_path,
        status=fields.get("status"),
        development_candidate=fields.get("development_candidate", "").lower() == "true",
        library_id=fields.get("library_id") or None,
        privacy=fields.get("privacy"),
    )


# --- freshness: git metadata only, never note content ----------------------


class GitOps(Protocol):
    """Two git-metadata-only queries against the pilot Wiki checkout.

    Injectable so :func:`freshness_verdict` and its callers are testable
    without a real git repository -- production wiring always uses
    :class:`SubprocessGitOps`.
    """

    def last_touch_commit(self, relative_path: str) -> str | None: ...

    def is_ancestor(self, ancestor: str, descendant: str) -> bool | None: ...


class NullGitOps:
    """Every query answers "unknown". Freshness always comes back
    ``"unknown"`` -- never ``"stale"`` or ``"current"`` by omission."""

    def last_touch_commit(self, relative_path: str) -> str | None:
        return None

    def is_ancestor(self, ancestor: str, descendant: str) -> bool | None:
        return None


@dataclass(frozen=True)
class SubprocessGitOps:
    """Real git introspection against the pilot Wiki checkout.

    Both queries read only commit metadata (hashes, ancestry) -- never a
    note body, mirroring ``ckp.bundle``'s own private ``_git_commit``
    helper's use of ``subprocess.run`` for the bundle's commit stamp,
    restated here (not imported: that helper is private to ``ckp.bundle``,
    and this module stays outside ``src/ckp/`` ownership for this issue).
    """

    wiki_root: Path

    def last_touch_commit(self, relative_path: str) -> str | None:
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.wiki_root),
                    "log",
                    "-1",
                    "--format=%H",
                    "--",
                    relative_path,
                ],
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        commit = result.stdout.strip()
        return commit or None

    def is_ancestor(self, ancestor: str, descendant: str) -> bool | None:
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.wiki_root),
                    "merge-base",
                    "--is-ancestor",
                    ancestor,
                    descendant,
                ],
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        # 128 and friends: unknown ref, not a repository, ambiguous commit
        # -- an honest "unknown", never coerced into True or False.
        return None


def freshness_verdict(
    git_ops: GitOps, export_commit: str | None, relative_path: str
) -> str:
    """``"current"`` | ``"stale"`` | ``"unknown"``. Never guesses.

    ``"stale"`` means the export's own commit predates the note's most
    recent change in the same checkout -- i.e. the export could not
    possibly have seen the note's current content. Any missing or
    ambiguous git signal along the way is ``"unknown"`` (round 2 feedback,
    point 4: "寧可標不明也不要猜一個看起來合理的分類").
    """
    if not export_commit:
        return "unknown"
    last_touch = git_ops.last_touch_commit(relative_path)
    if not last_touch:
        return "unknown"
    if last_touch == export_commit:
        return "current"
    ancestor = git_ops.is_ancestor(export_commit, last_touch)
    if ancestor is None:
        return "unknown"
    return "stale" if ancestor else "current"


def _classify_missing(
    pilot_path: str,
    facts: NoteFrontmatterFacts | None,
    *,
    export_commit: str | None,
    git_ops: GitOps,
) -> dict:
    """One ``missing_from_export`` diff entry, with ``reason`` as data.

    ``facts`` is ``None`` only when the caller could not read the note's
    frontmatter at all (defensive; every real caller in this module always
    has facts for the six frozen pilot paths) -- treated as "cannot
    determine eligibility", i.e. ``"unknown"``, never guessed as structural.
    """
    if facts is None:
        return {
            "category": "missing_from_export",
            "path": pilot_path,
            "reason": "unknown",
            "detail": (
                "note frontmatter unavailable; eligibility could not be determined"
            ),
        }

    eligible = zone_eligibility(facts)
    if not eligible:
        return {
            "category": "missing_from_export",
            "path": pilot_path,
            "reason": "structural",
            "detail": (
                "no wiki-export.v1 zone can ever represent this note's "
                "type/location, regardless of export freshness"
            ),
        }

    verdict = freshness_verdict(git_ops, export_commit, pilot_path)
    if verdict == "stale":
        return {
            "category": "missing_from_export",
            "path": pilot_path,
            "reason": "freshness",
            "eligible_zones": list(eligible),
            "detail": (
                "zone-eligible, but the export's commit predates this "
                "note's last change"
            ),
        }
    return {
        "category": "missing_from_export",
        "path": pilot_path,
        "reason": "unknown",
        "eligible_zones": list(eligible),
        "detail": (
            "zone-eligible and freshness did not confirm staleness, yet the "
            "note is still absent -- cause not determined (e.g. a "
            "selection/limit rule this module does not model), and this is "
            "reported as unknown rather than guessed"
        ),
    }


def build_pilot_catalog(
    config: Config, *, frozen_manifest: PilotManifest | None = None
) -> tuple[CatalogSnapshot, dict[str, NoteFrontmatterFacts]]:
    """Rebuild the generated Catalog from exactly the pilot manifest's six
    real notes -- never a wider glob of the live checkout -- and return the
    zone-eligibility facts read from the same six notes alongside it.

    Delegates hash and privacy verification to :func:`bind_pilot_corpus`
    (P1/#23) and lets :class:`PilotBindingError` propagate unhandled: a
    pilot corpus that fails to bind must fail this comparison loudly, the
    same D5 rule #23 already enforces, not fall back to comparing against
    an empty or partial Catalog.
    """
    bind_pilot_corpus(config, frozen_manifest=frozen_manifest)
    wiki_root = config.pilot_wiki_root
    if wiki_root is None:  # pragma: no cover - bind_pilot_corpus already refused
        raise PilotBindingError("CKP_PILOT_WIKI_ROOT is not set")

    reader = AnchoredBundleReader(wiki_root)
    members = []
    for relative_path in PILOT_NOTE_PATHS:
        member = reader.read_member(relative_path)
        if isinstance(member, MemberRefused):
            raise PilotBindingError(
                f"{relative_path}: unreadable while rebuilding the pilot "
                f"Catalog ({member.reason})"
            )
        members.append(member)
    members_t = tuple(members)

    facts_by_path = {
        member.relative_path: extract_note_frontmatter_facts(member)
        for member in members_t
    }

    classifier = FrontmatterClassifier(reader)
    gate = PrivacyGate(classifier, admissible=_PILOT_ADMISSIBLE)
    snapshot = BundleSnapshot(
        members=members_t,
        profile_version=None,
        bundle_commit=None,
        index_revision=compute_index_revision_from_members(members_t),
        sources={},
        token=(),
    )
    catalog = CatalogBuilder(gate).build(snapshot)
    return catalog, facts_by_path


def diff_export_against_catalog(
    export_entries: tuple[ExportEntry, ...],
    catalog: CatalogSnapshot,
    *,
    facts_by_path: dict[str, NoteFrontmatterFacts],
    export_commit: str | None,
    git_ops: GitOps,
) -> tuple[list[dict], dict]:
    """Classification only: no file I/O of its own (``git_ops`` and
    ``facts_by_path`` are handed in already resolved, so this function is
    pure with respect to its own body -- injecting a fake :class:`GitOps`
    and synthetic facts makes it fully testable without a filesystem).

    Returns ``(diffs, summary)``. Scope is exactly ``PILOT_NOTE_PATHS`` --
    an export entry whose path is not one of those six is out of scope and
    silently excluded from the diff (same reasoning as D3 scoping the QMD
    baseline down to the pilot corpus: comparing against a wider export
    would compare different scopes, which is not a comparison).
    """
    by_norm_path: dict[str, list[ExportEntry]] = defaultdict(list)
    for entry in export_entries:
        by_norm_path[_normalize_path(entry.path)].append(entry)

    catalog_by_path = {entry.citation.path: entry for entry in catalog.entries}

    diffs: list[dict] = []
    counts = {
        "missing_from_export": 0,
        "missing_from_export_structural": 0,
        "missing_from_export_freshness": 0,
        "missing_from_export_unknown": 0,
        "extra_in_export": 0,
        "metadata_mismatch": 0,
        "path_mismatch": 0,
        "matched_clean": 0,
    }

    for pilot_path in PILOT_NOTE_PATHS:
        catalog_entry = catalog_by_path.get(pilot_path)
        if catalog_entry is None:
            # The pilot corpus binds all six paths or fails loudly
            # (build_pilot_catalog / bind_pilot_corpus, D5) -- this branch
            # is defensive, not a normal outcome, and is exercised directly
            # by a unit test that hands diff_export_against_catalog() a
            # deliberately incomplete synthetic CatalogSnapshot.
            diffs.append(
                {
                    "category": "missing_from_catalog",
                    "path": pilot_path,
                    "detail": "pilot path absent from the generated Catalog",
                }
            )
            continue

        norm = _normalize_path(pilot_path)
        matches = by_norm_path.get(norm, [])
        if not matches:
            missing = _classify_missing(
                pilot_path,
                facts_by_path.get(pilot_path),
                export_commit=export_commit,
                git_ops=git_ops,
            )
            diffs.append(missing)
            counts["missing_from_export"] += 1
            counts[f"missing_from_export_{missing['reason']}"] += 1
            continue

        primary = matches[0]
        clean = True

        if len(matches) > 1:
            facts = facts_by_path.get(pilot_path)
            eligible = zone_eligibility(facts) if facts is not None else ()
            extra_zones = [m.zone for m in matches[1:]]
            extra_reasons = [
                "duplicate_in_eligible_zone"
                if zone in eligible
                else "zone_not_eligible"
                for zone in extra_zones
            ]
            diffs.append(
                {
                    "category": "extra_in_export",
                    "path": pilot_path,
                    "zones": extra_zones,
                    "reasons": extra_reasons,
                    "detail": (
                        f"{len(matches) - 1} duplicate zone entr"
                        f"{'y' if len(matches) == 2 else 'ies'} beyond the "
                        "first representation -- 'duplicate_in_eligible_zone' "
                        "means the note legitimately qualifies for that zone "
                        "too, 'zone_not_eligible' means the export placed it "
                        "somewhere wiki_dashboard_export.py's own rules "
                        "would never put it"
                    ),
                }
            )
            counts["extra_in_export"] += 1
            clean = False

        if primary.path != pilot_path:
            diffs.append(
                {
                    "category": "path_mismatch",
                    "path": pilot_path,
                    "export_zone": primary.zone,
                    "detail": (
                        "export path matches only after normalization "
                        "(leading './' or path segment differences)"
                    ),
                }
            )
            counts["path_mismatch"] += 1
            clean = False

        mismatched_fields = []
        if primary.title is not None and primary.title != catalog_entry.title:
            mismatched_fields.append("title")
        if primary.status is not None and primary.status != catalog_entry.status:
            mismatched_fields.append("status")
        if mismatched_fields:
            diffs.append(
                {
                    "category": "metadata_mismatch",
                    "path": pilot_path,
                    "fields": mismatched_fields,
                    "zone": primary.zone,
                }
            )
            counts["metadata_mismatch"] += 1
            clean = False

        if clean:
            counts["matched_clean"] += 1

    summary = {
        "pilot_scope_count": len(PILOT_NOTE_PATHS),
        **counts,
    }
    return diffs, summary


def compare_export_to_catalog(
    config: Config,
    *,
    export_path: Path | None = None,
    frozen_manifest: PilotManifest | None = None,
    git_ops: GitOps | None = None,
) -> dict:
    """Assemble the full report. Never returns a zero-diff report when the
    export could not be read -- ``report["compared"]`` says which case this
    is, and ``"diffs"`` is entirely absent (not ``[]``) when ``compared`` is
    ``False``, so a caller that only checks ``len(diffs) == 0`` cannot
    mistake "did not compare" for "compared, found nothing".

    ``git_ops`` defaults to :class:`SubprocessGitOps` against the pilot
    Wiki checkout -- real freshness introspection in production. Tests
    inject a fake to stay filesystem-free, or a real temporary git repo to
    exercise :class:`SubprocessGitOps` end-to-end.
    """
    # build_pilot_catalog raises PilotBindingError (unhandled, by design) if
    # CKP_PILOT_WIKI_ROOT is unset or the corpus fails to bind -- by the time
    # this returns, config.pilot_wiki_root is guaranteed not None.
    catalog, facts_by_path = build_pilot_catalog(
        config, frozen_manifest=frozen_manifest
    )
    wiki_root = config.pilot_wiki_root
    assert wiki_root is not None
    resolved_export_path = (
        export_path if export_path is not None else resolve_export_path(wiki_root)
    )
    resolved_git_ops = git_ops if git_ops is not None else SubprocessGitOps(wiki_root)

    base = {
        "schema": REPORT_SCHEMA,
        "pilot_note_paths": list(PILOT_NOTE_PATHS),
        "catalog": {
            "index_revision": catalog.index_revision,
            "entry_count": len(catalog.entries),
        },
    }

    try:
        current_export = load_current_export(resolved_export_path)
    except ExportUnavailable as exc:
        return {
            **base,
            "compared": False,
            "export": {
                "status": "unavailable",
                "path": str(resolved_export_path),
                "reason": str(exc),
            },
        }

    diffs, summary = diff_export_against_catalog(
        current_export.entries,
        catalog,
        facts_by_path=facts_by_path,
        export_commit=current_export.commit,
        git_ops=resolved_git_ops,
    )

    return {
        **base,
        "compared": True,
        "export": {
            "status": "ok",
            "path": str(current_export.path),
            "exported_at": current_export.exported_at,
            "commit": current_export.commit,
            "path_addressable_entry_count": len(current_export.entries),
        },
        "diffs": diffs,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    try:
        report = compare_export_to_catalog(config)
    except PilotBindingError as exc:
        print(f"export_compare: pilot corpus binding failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not report["compared"]:
        print(
            "export_compare: current export unavailable -- not compared "
            "(this is a failure, not a zero-diff result)",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPORT_PATH_ENV",
    "PATH_ADDRESSABLE_ZONES",
    "REPORT_SCHEMA",
    "CurrentExport",
    "ExportEntry",
    "ExportUnavailable",
    "GitOps",
    "NoteFrontmatterFacts",
    "NullGitOps",
    "SubprocessGitOps",
    "build_pilot_catalog",
    "compare_export_to_catalog",
    "default_export_path",
    "diff_export_against_catalog",
    "extract_note_frontmatter_facts",
    "freshness_verdict",
    "load_current_export",
    "main",
    "resolve_export_path",
    "zone_eligibility",
]
