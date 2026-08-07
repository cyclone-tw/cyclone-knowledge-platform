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
inventory. Its ``zones`` cover only ``shared_now``, ``knowledge_feed``
(``Private/_inbox/info-collect/*.md``, latest 30), ``agent_activity``
(``Private/agents/*/working.md``), ``life_domains``
(``Private/Life/*/reports/*.md``, latest per domain), ``projects``
(``Core/project-*.md`` with ``status: active``), ``development_candidates``
(``Core/_inbox/agent-captures/*.md`` matching the candidate schema), and
``topics``. There is no zone for ``Procedure`` or ``Decision`` notes at all
-- that is a structural gap in what the export can ever represent, not a
staleness artifact, and this module reports it as ``missing_from_export``
the same as any other gap: the report's job is to say what is true of the
export as it exists, not to explain away entries the export was never
designed to carry.

This module never regenerates the export and never invents a substitute.
It reads whatever file is currently on disk at the configured path -- the
same file Dashboard is currently reading from, staleness included -- and
compares it against a freshly generated Catalog scoped to the pilot
manifest (D2/#23's frozen six real notes; scope-matching is the same
discipline D3 applies to the QMD baseline: comparing different scopes is
not a comparison).

## Report body vs privacy (issue #29 privacy boundary)

The diff report never carries note bodies, excerpts, or metadata *values*
(RP1; ``tests/test_no_wiki_content.py``). It carries only: relative paths
(also present in ``config/pilot-manifest.toml``, already committed), zone
*names* (a fixed vocabulary from ``wiki-export.v1``'s schema, not wiki
content), and metadata *field names* that disagree (``"title"``,
``"status"`` -- never the disagreeing strings themselves).
"""

from __future__ import annotations

import json
import os
import posixpath
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ckp.bundle import (
    AnchoredBundleReader,
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

REPORT_SCHEMA = "ckp-export-compare-report/1"

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


def build_pilot_catalog(
    config: Config, *, frozen_manifest: PilotManifest | None = None
) -> CatalogSnapshot:
    """Rebuild the generated Catalog from exactly the pilot manifest's six
    real notes -- never a wider glob of the live checkout.

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
    return CatalogBuilder(gate).build(snapshot)


def diff_export_against_catalog(
    export_entries: tuple[ExportEntry, ...], catalog: CatalogSnapshot
) -> tuple[list[dict], dict]:
    """Pure classification: no filesystem, no privacy gate, no I/O.

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
            diffs.append({"category": "missing_from_export", "path": pilot_path})
            counts["missing_from_export"] += 1
            continue

        primary = matches[0]
        clean = True

        if len(matches) > 1:
            diffs.append(
                {
                    "category": "extra_in_export",
                    "path": pilot_path,
                    "zones": [m.zone for m in matches[1:]],
                    "detail": (
                        f"{len(matches) - 1} duplicate zone entr"
                        f"{'y' if len(matches) == 2 else 'ies'} beyond the "
                        "first representation"
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
) -> dict:
    """Assemble the full report. Never returns a zero-diff report when the
    export could not be read -- ``report["compared"]`` says which case this
    is, and ``"diffs"`` is entirely absent (not ``[]``) when ``compared`` is
    ``False``, so a caller that only checks ``len(diffs) == 0`` cannot
    mistake "did not compare" for "compared, found nothing".
    """
    # build_pilot_catalog raises PilotBindingError (unhandled, by design) if
    # CKP_PILOT_WIKI_ROOT is unset or the corpus fails to bind -- by the time
    # this returns, config.pilot_wiki_root is guaranteed not None.
    catalog = build_pilot_catalog(config, frozen_manifest=frozen_manifest)
    wiki_root = config.pilot_wiki_root
    assert wiki_root is not None
    resolved_export_path = (
        export_path if export_path is not None else resolve_export_path(wiki_root)
    )

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

    diffs, summary = diff_export_against_catalog(current_export.entries, catalog)

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
    "build_pilot_catalog",
    "compare_export_to_catalog",
    "default_export_path",
    "diff_export_against_catalog",
    "load_current_export",
    "main",
    "resolve_export_path",
]
