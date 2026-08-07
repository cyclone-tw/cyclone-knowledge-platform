"""Pilot corpus binding: a read-only manifest over the real Wiki checkout.

Rule source: Epic #21 D2 (frozen six-note corpus) and D5 (fail loud, never
silently skip). This module adds no new filesystem-safety mechanism -- every
byte crosses `ckp.bundle.AnchoredBundleReader`, the same anchored, symlink-
and hardlink-refusing reader C1-C3 already use, and every privacy decision
crosses `ckp.privacy.PrivacyGate`. This module only adds two things on top:

* the fixed six-path allowlist (`PILOT_NOTE_PATHS`) and the requirement that
  the frozen manifest file names exactly that list, in that order (RP4:
  a glob or a rule scan would silently grow the corpus as the wiki grows,
  which is exactly what D2 forbids);
* the frozen-hash drift check, which fails the whole binding -- naming which
  note -- the moment the live checkout disagrees with the hash recorded when
  the corpus was frozen (catches another session's uncommitted dirty file in
  the shared wiki checkout, and catches the note having legitimately changed
  upstream without a re-freeze PR).

Nothing here returns a note body. `PilotManifestEntry` carries a relative
path and a content hash, nothing else -- there is no code path from this
module that could copy real Wiki content into this repo's return values, let
alone onto disk here.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass

from ckp.bundle import AnchoredBundleReader, BundleMember, MemberRefused
from ckp.config import Config
from ckp.privacy import FrontmatterClassifier, PrivacyClass, PrivacyGate, Refused

# D2, frozen by Epic #21 comment. Six paths, explicit -- not a pattern.
# Growing the pilot corpus means editing this tuple *and* re-freezing
# config/pilot-manifest.toml in the same reviewed PR; it is never picked up
# automatically as the wiki grows (RP4).
PILOT_NOTE_PATHS: tuple[str, ...] = (
    "Core/procedure-agent-wiki-note-retrieval.md",
    "Core/procedure-agent-memory-read-scopes.md",
    "Core/decision-cyclone-wiki-openwiki-role-boundary.md",
    "Core/project-cyclone-okf-knowledge-contract.md",
    "Core/_inbox/agent-captures/2026-07-04-relayapi-repo-analysis.md",
    "Core/_inbox/agent-captures/2026-06-23-hermes-os-2-100-stars-loops.md",
)

# The pilot corpus is deliberately narrower than "everything this repo is
# allowed to reference" -- tests/test_no_wiki_content.py forbids far less
# (only sensitive/restricted/student-private + internal at the repo-content
# level). D2 froze the six notes above at exactly `internal`. `public` is
# deliberately *not* admitted here even though it would be strictly less
# restrictive: a survey of the whole wiki found zero `public` notes in Core,
# so that branch would never be exercised by anything real -- it would only
# be one more path to keep in step with D2 for no benefit. Admissible is
# pinned to what D2 actually froze, nothing wider.
_PILOT_ADMISSIBLE: frozenset[PrivacyClass] = frozenset({PrivacyClass.INTERNAL})

# A second, independent field check, deliberately not folded into the
# privacy gate above: `type: Student Reference` must be refused even if the
# note were (incorrectly) tagged `internal` upstream. Two independent
# signals catch a mistake in either one alone.
_TYPE_KEY = re.compile(r"^type:")
_TYPE_DECLARATION = re.compile(
    r"^type:[ \t]+(?P<quote>['\"]?)(?P<value>[^'\"\n]+?)(?P=quote)[ \t]*$"
)
_FORBIDDEN_TYPES = frozenset({"Student Reference"})


class PilotBindingError(RuntimeError):
    """The pilot corpus could not be bound to a verified live checkout.

    Every failure mode below (missing root, unreadable note, refused
    privacy, forbidden type, hash drift) raises this rather than skipping
    the offending note: a pilot corpus that silently came back with five of
    six notes would look complete to every downstream caller (D5).
    """


@dataclass(frozen=True)
class PilotManifestEntry:
    """One corpus note: where it lives, and what its bytes must hash to.

    Deliberately two fields only (issue #23 acceptance) -- no body, no
    title, no excerpt. Nothing that could let real note content leak into
    this repo through a manifest field nobody thought to scrub.
    """

    relative_path: str
    content_sha256: str


@dataclass(frozen=True)
class PilotManifest:
    entries: tuple[PilotManifestEntry, ...]


def _frontmatter_body(text: str) -> list[str] | None:
    lines = [line.rstrip("\r") for line in text.split("\n")]
    if not lines or lines[0] != "---":
        return None
    for index in range(1, len(lines)):
        if lines[index] == "---":
            return lines[1:index]
    return None


def _reject_forbidden_type(relative_path: str, content: bytes) -> None:
    """Refuse `type: Student Reference`, independent of the privacy class.

    Unreadable/unparsable frontmatter is left to the privacy classifier,
    which already fails closed on it (`Unclassified`) -- this function only
    ever narrows what passes, never widens it, so returning quietly on a
    shape it does not recognise is safe here.
    """
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return
    body = _frontmatter_body(text)
    if body is None:
        return
    for line in body:
        if not _TYPE_KEY.match(line):
            continue
        match = _TYPE_DECLARATION.match(line)
        value = match.group("value") if match else line.split(":", 1)[1].strip()
        if value in _FORBIDDEN_TYPES:
            raise PilotBindingError(
                f"{relative_path}: refused, type declares {value!r}"
            )


def _enforce_allowlist(manifest: PilotManifest, *, source: str) -> None:
    """RP4, enforced on the manifest actually about to be used.

    This must run on **every** path that produces a `PilotManifest`
    `bind_pilot_corpus` might bind against -- both the file loaded from disk
    and any manifest a caller constructs directly (tests use the latter;
    nothing else in this codebase should). Checking only inside
    `load_frozen_manifest` would leave `bind_pilot_corpus(config,
    frozen_manifest=...)` as a second, unchecked read point on the same
    allowlist, which is exactly the kind of second entry point AGENTS.md
    §9's third question exists to catch.
    """
    found = tuple(entry.relative_path for entry in manifest.entries)
    if found != PILOT_NOTE_PATHS:
        raise PilotBindingError(
            f"{source}: manifest paths do not match the frozen "
            f"PILOT_NOTE_PATHS allowlist (RP4); found {found!r}, "
            f"expected {PILOT_NOTE_PATHS!r}"
        )


def load_frozen_manifest(config: Config) -> PilotManifest:
    """Load the committed, hash-pinned manifest.

    Pins the loaded paths to `PILOT_NOTE_PATHS` verbatim and in order: a
    manifest file edited to add a seventh path, reorder, or widen into a
    pattern fails here rather than silently expanding the corpus (RP4).
    """
    manifest_path = config.pilot_manifest_path
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise PilotBindingError(
            f"frozen pilot manifest unreadable at {manifest_path}: {exc}"
        ) from exc
    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise PilotBindingError(
            f"frozen pilot manifest {manifest_path} is not valid TOML: {exc}"
        ) from exc

    notes = parsed.get("note")
    if not isinstance(notes, list) or not notes:
        raise PilotBindingError(f"{manifest_path}: missing [[note]] entries")

    try:
        entries = tuple(
            PilotManifestEntry(
                relative_path=note["relative_path"],
                content_sha256=note["content_sha256"],
            )
            for note in notes
        )
    except (KeyError, TypeError) as exc:
        raise PilotBindingError(
            f"{manifest_path}: each [[note]] needs relative_path and content_sha256"
        ) from exc

    manifest = PilotManifest(entries=entries)
    _enforce_allowlist(manifest, source=str(manifest_path))
    return manifest


def bind_pilot_corpus(
    config: Config, *, frozen_manifest: PilotManifest | None = None
) -> PilotManifest:
    """Verify the live Wiki checkout against the frozen manifest and return it.

    `frozen_manifest` is an injection point for tests only: production
    callers always let this load `config/pilot-manifest.toml` (the default,
    when the parameter is omitted). Tests use it to bind a synthetic root
    against a synthetic manifest, so this module's own test suite never has
    to depend on -- or accidentally vendor -- the real Wiki checkout.

    The RP4 allowlist is re-enforced here on whichever manifest this
    function is actually about to bind against, regardless of whether it
    came from disk or from `frozen_manifest` -- `load_frozen_manifest`
    checking its own output is not enough, because an injected manifest
    never goes through that function at all.
    """
    wiki_root = config.pilot_wiki_root
    if wiki_root is None:
        raise PilotBindingError(
            "CKP_PILOT_WIKI_ROOT is not set; the pilot corpus binding refuses "
            "to run without an explicit Wiki checkout root -- no default "
            "root, no skip (D5, issue #23)"
        )
    if not wiki_root.is_dir():
        raise PilotBindingError(
            f"CKP_PILOT_WIKI_ROOT={wiki_root} is not a directory; the pilot "
            "corpus binding fails rather than silently skipping a missing "
            "checkout (D5, issue #23)"
        )

    manifest = (
        frozen_manifest if frozen_manifest is not None else load_frozen_manifest(config)
    )
    _enforce_allowlist(
        manifest, source="frozen_manifest" if frozen_manifest else "manifest"
    )

    reader = AnchoredBundleReader(wiki_root)
    classifier = FrontmatterClassifier(reader)
    gate = PrivacyGate(classifier, admissible=_PILOT_ADMISSIBLE)

    verified: list[PilotManifestEntry] = []
    for expected in manifest.entries:
        member = reader.read_member(expected.relative_path)
        if isinstance(member, MemberRefused):
            raise PilotBindingError(
                f"{expected.relative_path}: unreadable in the live checkout "
                f"({member.reason})"
            )
        assert isinstance(member, BundleMember)

        _reject_forbidden_type(expected.relative_path, member.content)

        verdict = gate.admit_member(member)
        if isinstance(verdict, Refused):
            raise PilotBindingError(
                f"{expected.relative_path}: privacy gate refused ({verdict.reason})"
            )

        if member.content_sha256 != expected.content_sha256:
            raise PilotBindingError(
                f"{expected.relative_path}: content hash mismatch -- expected "
                f"{expected.content_sha256}, live checkout has "
                f"{member.content_sha256} (checkout may be dirty, or the note "
                "changed upstream since the manifest was frozen)"
            )
        verified.append(
            PilotManifestEntry(
                relative_path=expected.relative_path,
                content_sha256=member.content_sha256,
            )
        )

    return PilotManifest(entries=tuple(verified))


__all__ = [
    "PILOT_NOTE_PATHS",
    "PilotBindingError",
    "PilotManifest",
    "PilotManifestEntry",
    "bind_pilot_corpus",
    "load_frozen_manifest",
]
