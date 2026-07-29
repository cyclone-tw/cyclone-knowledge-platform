"""The four revision fields: derived where it matters, honest where it cannot be.

The digest tests are the substance of this file. Contract §5.5 makes "the
rebuild is not reproducible from the same commit" a rollback trigger, so
reproducibility has to be a property that fails loudly, not a hope.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ckp.config import load_config
from ckp.revision import (
    API_VERSION,
    build_revision,
    bundle_is_readable,
    bundle_member_key,
    compute_index_revision,
    iter_note_paths,
    read_bundle_descriptor,
    resolve_bundle_commit,
    resolve_profile_version,
)
from conftest import REPO_ROOT

GLOB = "**/*.md"
FIXTURE_BUNDLE = REPO_ROOT / "fixtures" / "synthetic-bundle"


def _bundle(root: Path, files: dict[str, bytes]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def _git_available() -> bool:
    try:
        return (
            subprocess.run(
                ["git", "--version"], capture_output=True, check=False
            ).returncode
            == 0
        )
    except OSError:
        return False


#: Skip rather than fail where git is absent. A source export or a slim image
#: has no repository to ask, and "no evidence" is the documented answer there,
#: not a defect. CI has git, so these still run where it matters.
requires_git = pytest.mark.skipif(not _git_available(), reason="git not available")

#: Stricter, and the precise premise for anything asserting that the *shipped*
#: fixture reports a commit: the git binary existing is not the same as this
#: checkout being resolvable. A source export, or a worktree whose .git file
#: points somewhere unmounted, has the binary and still no commit.
requires_bundle_commit = pytest.mark.skipif(
    resolve_bundle_commit(FIXTURE_BUNDLE)[0] is None,
    reason="fixture bundle has no resolvable commit here",
)


def _git_repo(root: Path) -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    common = [
        "git",
        "-C",
        str(root),
        "-c",
        "user.email=ci@example.invalid",
        "-c",
        "user.name=ckp-test",
    ]
    subprocess.run([*common, "add", "-A"], check=True)
    subprocess.run([*common, "commit", "-q", "-m", "fixture"], check=True)
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return head.stdout.strip()


# --- index_revision: derived and reproducible -------------------------------


def test_digest_is_stable_across_repeated_computation(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n", "sub/b.md": b"beta\n"})
    assert compute_index_revision(root, GLOB) == compute_index_revision(root, GLOB)


def test_digest_ignores_the_order_files_were_created(tmp_path) -> None:
    """Filesystem walk order differs between hosts; the digest must not."""
    first = _bundle(tmp_path / "one", {"a.md": b"alpha\n", "b.md": b"beta\n"})
    second = _bundle(tmp_path / "two", {"b.md": b"beta\n", "a.md": b"alpha\n"})
    assert compute_index_revision(first, GLOB) == compute_index_revision(second, GLOB)


def test_digest_changes_when_one_byte_changes(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    before = compute_index_revision(root, GLOB)
    (root / "a.md").write_bytes(b"alphb\n")
    assert compute_index_revision(root, GLOB) != before


def test_digest_changes_when_a_file_is_renamed(tmp_path) -> None:
    """Same bytes at a different path is a different bundle."""
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    before = compute_index_revision(root, GLOB)
    (root / "a.md").rename(root / "renamed.md")
    assert compute_index_revision(root, GLOB) != before


def test_digest_distinguishes_crlf_from_lf(tmp_path) -> None:
    """Regression guard: notes are hashed as bytes, never as decoded text.

    ``Path.read_text`` performs universal-newline translation, which would map
    a CRLF note and an LF note onto the same digest and report two materially
    different bundles as one revision.
    """
    lf = _bundle(tmp_path / "lf", {"a.md": b"line one\nline two\n"})
    crlf = _bundle(tmp_path / "crlf", {"a.md": b"line one\r\nline two\r\n"})
    assert compute_index_revision(lf, GLOB) != compute_index_revision(crlf, GLOB)


def test_digest_covers_nested_paths(tmp_path) -> None:
    flat = _bundle(tmp_path / "flat", {"a.md": b"alpha\n"})
    nested = _bundle(tmp_path / "nested", {"sub/a.md": b"alpha\n"})
    assert compute_index_revision(flat, GLOB) != compute_index_revision(nested, GLOB)


def test_digest_ignores_non_note_files(tmp_path) -> None:
    """Stamping provenance into the bundle must not move the index revision."""
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    before = compute_index_revision(root, GLOB)
    (root / ".bundle-commit").write_text("deadbeef\n", encoding="utf-8")
    (root / "bundle.toml").write_text('profile_version = "x"\n', encoding="utf-8")
    assert compute_index_revision(root, GLOB) == before


def test_digest_is_none_when_the_bundle_is_absent(tmp_path) -> None:
    assert compute_index_revision(tmp_path / "nope", GLOB) is None


def test_empty_bundle_has_no_revision_at_all(tmp_path) -> None:
    """An empty bundle is not "the digest of an empty set".

    This test previously asserted the opposite -- a digest plus an unreadable
    bundle -- which pinned the exact contradiction it should have caught:
    /health said 503 while /revision handed back a revision.
    """
    root = tmp_path / "empty"
    root.mkdir()
    assert compute_index_revision(root, GLOB) is None
    assert bundle_is_readable(root, GLOB) is False


def test_health_and_revision_never_disagree(tmp_path) -> None:
    """One computation answers both, so walk the states and check they track."""
    root = tmp_path / "b"
    root.mkdir()
    assert (compute_index_revision(root, GLOB) is None) is (
        not bundle_is_readable(root, GLOB)
    )
    (root / "a.md").write_bytes(b"alpha\n")
    assert (compute_index_revision(root, GLOB) is not None) is bundle_is_readable(
        root, GLOB
    )
    assert compute_index_revision(tmp_path / "gone", GLOB) is None
    assert bundle_is_readable(tmp_path / "gone", GLOB) is False


def test_note_paths_are_sorted_by_posix_relative_path(tmp_path) -> None:
    root = _bundle(
        tmp_path / "b",
        {"z.md": b"z\n", "a.md": b"a\n", "m/b.md": b"b\n"},
    )
    rel = [p.relative_to(root).as_posix() for p in iter_note_paths(root, GLOB)]
    assert rel == sorted(rel)
    assert rel == ["a.md", "m/b.md", "z.md"]


# --- bundle_commit: evidence, never invention -------------------------------


@requires_git
def test_commit_comes_from_git_when_the_bundle_is_a_work_tree(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    head = _git_repo(root)
    assert resolve_bundle_commit(root) == (head, "git")


def test_commit_falls_back_to_the_stamp_without_git(tmp_path) -> None:
    """The container case: no repository to ask, a build-time stamp instead."""
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / ".bundle-commit").write_text("abc123\n", encoding="utf-8")
    assert resolve_bundle_commit(root) == ("abc123", "stamp")


@requires_git
def test_live_git_wins_over_a_stale_stamp(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / ".bundle-commit").write_text("stale000\n", encoding="utf-8")
    head = _git_repo(root)
    commit, source = resolve_bundle_commit(root)
    assert (commit, source) == (head, "git")
    assert commit != "stale000"


def test_commit_is_unknown_rather_than_invented(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    assert resolve_bundle_commit(root) == (None, "unknown")


def test_empty_stamp_is_unknown_not_empty_string(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / ".bundle-commit").write_text("\n", encoding="utf-8")
    assert resolve_bundle_commit(root) == (None, "unknown")


# --- profile_version: the bundle speaks first -------------------------------


def test_profile_version_prefers_the_bundle_descriptor(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / "bundle.toml").write_text(
        'profile_version = "from-bundle"\n', encoding="utf-8"
    )
    config = load_config(env={"CKP_PROFILE_EXPECTED_VERSION": "from-config"})
    assert resolve_profile_version(root, config) == ("from-bundle", "bundle-descriptor")


def test_profile_version_falls_back_to_config(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    config = load_config(env={"CKP_PROFILE_EXPECTED_VERSION": "from-config"})
    assert resolve_profile_version(root, config) == ("from-config", "config")


def test_profile_version_is_unknown_when_nobody_declares_one(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    assert resolve_profile_version(root, load_config(env={})) == (None, "unknown")


def test_malformed_descriptor_does_not_crash_the_service(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / "bundle.toml").write_text("profile_version = \n", encoding="utf-8")
    assert resolve_profile_version(root, load_config(env={})) == (None, "unknown")


# --- build_revision over the real fixture bundle ----------------------------


@requires_bundle_commit
def test_revision_over_the_shipped_fixture_bundle() -> None:
    config = load_config(env={"CKP_BUNDLE_ROOT": str(FIXTURE_BUNDLE)})
    revision = build_revision(config)

    assert revision.api_version == API_VERSION
    assert revision.sources["api_version"] == "code"
    assert revision.profile_version == "cyclone-profile-v1"
    assert revision.sources["profile_version"] == "bundle-descriptor"
    assert revision.index_revision is not None
    assert revision.index_revision.startswith("sha256:")
    assert revision.sources["index_revision"] == "computed"
    # The fixture lives inside this repository, so git is the evidence here.
    assert revision.sources["bundle_commit"] in {"git", "stamp"}
    assert revision.bundle_commit


def test_revision_reports_unknowns_instead_of_placeholders(tmp_path) -> None:
    """A broken deployment must be distinguishable from a healthy one."""
    config = load_config(env={"CKP_BUNDLE_ROOT": str(tmp_path / "absent")})
    revision = build_revision(config)

    assert revision.profile_version is None
    assert revision.bundle_commit is None
    assert revision.index_revision is None
    assert revision.sources["profile_version"] == "unknown"
    assert revision.sources["bundle_commit"] == "unknown"
    assert revision.sources["index_revision"] == "unknown"
    # api_version is a code constant, so it is knowable even with no bundle.
    assert revision.api_version == API_VERSION


def test_revision_dict_carries_exactly_the_contract_fields() -> None:
    config = load_config(env={"CKP_BUNDLE_ROOT": str(FIXTURE_BUNDLE)})
    payload = build_revision(config).as_dict()
    assert set(payload) == {
        "profile_version",
        "api_version",
        "bundle_commit",
        "index_revision",
        "sources",
    }


# --- stamp reading: undecodable or exotic input is not evidence -------------


def test_binary_stamp_is_not_evidence(tmp_path) -> None:
    """UnicodeDecodeError is a ValueError, so an OSError-only guard misses it."""
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / ".bundle-commit").write_bytes(b"\xff\xfe\x00binary")
    assert resolve_bundle_commit(root) == (None, "unknown")


def test_stamp_takes_the_first_newline_delimited_line(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / ".bundle-commit").write_text("abc123\ntrailing junk\n", encoding="utf-8")
    assert resolve_bundle_commit(root) == ("abc123", "stamp")


def test_stamp_does_not_split_on_exotic_line_separators(tmp_path) -> None:
    """str.splitlines() breaks on U+2028; `head -n 1` does not, and neither do we.

    Silently truncating at a separator the writer never intended would report a
    commit that was never stamped.
    """
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / ".bundle-commit").write_text("abc def\n", encoding="utf-8")
    commit, source = resolve_bundle_commit(root)
    assert (commit, source) == ("abc def", "stamp")


# --- unicode normalisation: one logical name, one digest --------------------


def test_digest_is_stable_across_unicode_normalisation_forms(tmp_path) -> None:
    """macOS stores NFD, ext4 stores what it was given; the digest must not care.

    Without normalisation the same logical bundle digests differently on the
    two hosts, and contract §5.5 calls a non-reproducible rebuild a rollback
    trigger.
    """
    import unicodedata

    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd

    a = _bundle(tmp_path / "nfc", {f"{nfc}.md": b"same content\n"})
    b = _bundle(tmp_path / "nfd", {f"{nfd}.md": b"same content\n"})
    assert compute_index_revision(a, GLOB) == compute_index_revision(b, GLOB)


def test_names_sharing_a_canonical_form_still_order_deterministically(
    tmp_path,
) -> None:
    """Both spellings can coexist on a byte-preserving filesystem."""
    import unicodedata

    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", nfc)
    root = tmp_path / "both"
    root.mkdir()
    (root / f"{nfc}.md").write_bytes(b"one\n")
    try:
        (root / f"{nfd}.md").write_bytes(b"two\n")
    except OSError:  # pragma: no cover - normalising filesystem
        pytest.skip("filesystem normalises filenames; both spellings collapse")
    # Ask the directory, not iter_note_paths: deciding the premise with the
    # function under test lets a broken implementation skip instead of fail.
    if len(os.listdir(root)) < 2:  # pragma: no cover - normalising filesystem
        pytest.skip("filesystem normalises filenames; both spellings collapse")

    assert len(iter_note_paths(root, GLOB)) == 2
    assert compute_index_revision(root, GLOB) == compute_index_revision(root, GLOB)
    ordering = [p.name for p in iter_note_paths(root, GLOB)]
    assert ordering == sorted(ordering)


# --- health must reflect what serving actually requires (Codex r1 #1) -------


def test_unreadable_note_makes_the_bundle_unreadable(tmp_path) -> None:
    """The inconsistency this closes: listable but not readable.

    `is_file()` passes on a note whose permissions deny reading, so a
    list-only check reported the bundle healthy while index_revision was null.
    """
    if os.geteuid() == 0:  # pragma: no cover - root ignores the mode bits
        pytest.skip("running as root; permission bits do not apply")

    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    os.chmod(root / "a.md", 0o000)
    try:
        assert compute_index_revision(root, GLOB) is None
        assert bundle_is_readable(root, GLOB) is False
    finally:
        os.chmod(root / "a.md", 0o644)

    assert bundle_is_readable(root, GLOB) is True


# --- symlinks may not carry the digest outside the bundle (Codex r1 #2) ----


def test_note_symlinked_out_of_the_bundle_is_excluded(tmp_path) -> None:
    """Otherwise the revision depends on state the bundle does not contain."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "external.md").write_bytes(b"not part of the bundle\n")

    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    alone = compute_index_revision(root, GLOB)

    (root / "external.md").symlink_to(outside / "external.md")
    assert [p.name for p in iter_note_paths(root, GLOB)] == ["a.md"]
    assert compute_index_revision(root, GLOB) == alone


def test_symlink_inside_the_bundle_is_also_refused(tmp_path) -> None:
    """Allowing any symlink leaves a swap window between the check and the read.

    An in-bundle link is harmless in principle, but it is worth nothing here
    and permitting the category is what creates the window.
    """
    root = _bundle(tmp_path / "b", {"real/a.md": b"alpha\n"})
    (root / "alias.md").symlink_to(root / "real" / "a.md")
    names = [p.relative_to(root).as_posix() for p in iter_note_paths(root, GLOB)]
    assert names == ["real/a.md"]


def test_bundle_root_may_itself_be_a_symlink(tmp_path) -> None:
    """Mounting a checkout at a stable path is normal and must keep working."""
    real = _bundle(tmp_path / "real", {"a.md": b"alpha\n"})
    link = tmp_path / "mounted"
    link.symlink_to(real, target_is_directory=True)
    assert compute_index_revision(link, GLOB) == compute_index_revision(real, GLOB)
    assert bundle_is_readable(link, GLOB) is True


def test_descriptor_symlinked_out_of_the_bundle_is_ignored(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "bundle.toml").write_text(
        'profile_version = "smuggled"\n', encoding="utf-8"
    )
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / "bundle.toml").symlink_to(outside / "bundle.toml")
    assert read_bundle_descriptor(root) == {}


def test_stamp_symlinked_out_of_the_bundle_is_ignored(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".bundle-commit").write_text("smuggled\n", encoding="utf-8")
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / ".bundle-commit").symlink_to(outside / ".bundle-commit")
    assert resolve_bundle_commit(root) == (None, "unknown")


# --- the digest key must not depend on how the glob reached the note --------


def test_digest_key_ignores_the_route_the_glob_took(tmp_path) -> None:
    """`../bundle/a.md` and `a.md` are one note and must digest identically.

    relative_to() is purely lexical, so a pattern reaching a note by way of
    `..` used to key it under a path that leaves the bundle in the string even
    though it resolves inside. Same content, two revisions.
    """
    root = _bundle(tmp_path / "bundle", {"a.md": b"alpha\n"})
    direct = compute_index_revision(root, "**/*.md")
    roundabout = compute_index_revision(root, "../**/*.md")
    assert direct is not None
    assert roundabout == direct


# --- a golden digest, computed outside the implementation ------------------

#: The digest of the shipped fixture bundle, pinned as a literal. Every other
#: digest assertion compares compute_index_revision() against itself, so
#: swapping sha256 for sha1 while keeping the "sha256:" prefix would satisfy
#: all of them. This one would not.
GOLDEN_FIXTURE_DIGEST = (
    "sha256:754d8514119194d5cb3dbb906ac852282e2e318f3aeae2d5ca5dcd490280509d"
)


def test_shipped_fixture_matches_the_golden_digest() -> None:
    assert compute_index_revision(FIXTURE_BUNDLE, GLOB) == GOLDEN_FIXTURE_DIGEST


def test_golden_digest_is_reproducible_from_the_documented_algorithm(
    tmp_path,
) -> None:
    """Re-derive the digest here, without calling the module under test.

    If this and compute_index_revision ever disagree, one of them changed the
    algorithm; the point is that changing it cannot go unnoticed.
    """
    import hashlib

    outer = hashlib.sha256()
    outer.update(b"ckp-index-v1\n")
    for rel in sorted(
        p.relative_to(FIXTURE_BUNDLE).as_posix()
        for p in FIXTURE_BUNDLE.rglob("*.md")
        if p.is_file()
    ):
        outer.update(rel.encode("utf-8"))
        outer.update(b"\0")
        outer.update(
            hashlib.sha256((FIXTURE_BUNDLE / rel).read_bytes())
            .hexdigest()
            .encode("ascii")
        )
        outer.update(b"\n")
    assert f"sha256:{outer.hexdigest()}" == GOLDEN_FIXTURE_DIGEST


# --- descriptor values that are not versions -------------------------------


@pytest.mark.parametrize(
    "declared",
    ["profile_version = 42", "profile_version = true", 'profile_version = ""'],
)
def test_non_string_or_empty_descriptor_version_is_not_a_declaration(
    tmp_path, declared: str
) -> None:
    """`if declared:` instead of an isinstance check would report 42 as a version."""
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / "bundle.toml").write_text(declared + "\n", encoding="utf-8")
    assert resolve_profile_version(root, load_config(env={})) == (None, "unknown")


def test_whitespace_descriptor_version_falls_back_to_config(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / "bundle.toml").write_text('profile_version = "   "\n', encoding="utf-8")
    config = load_config(env={"CKP_PROFILE_EXPECTED_VERSION": "from-config"})
    assert resolve_profile_version(root, config) == ("from-config", "config")


def test_declared_version_is_stripped(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / "bundle.toml").write_text('profile_version = " v9 "\n', encoding="utf-8")
    assert resolve_profile_version(root, load_config(env={})) == (
        "v9",
        "bundle-descriptor",
    )


# --- symlinked directories, not just symlinked files (Codex r3 #1) ---------


def _symlinked_dir_bundle(tmp_path) -> tuple[Path, Path]:
    root = tmp_path / "bundle"
    (root / "real").mkdir(parents=True)
    (root / "real" / "a.md").write_bytes(b"alpha\n")
    (root / "alias").symlink_to(root / "real", target_is_directory=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.md").write_bytes(b"content from outside the bundle\n")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    return root, outside


def test_pattern_naming_a_symlinked_directory_finds_nothing(tmp_path) -> None:
    """`**` skips symlinked directories, but a pattern naming one walks it."""
    root, _ = _symlinked_dir_bundle(tmp_path)
    assert iter_note_paths(root, "alias/*.md") == []


def test_symlinked_directory_cannot_duplicate_a_note(tmp_path) -> None:
    """The real defect: one note reachable twice, hashed into the digest twice.

    Both routes key to the resolved path, so they are indistinguishable in the
    digest input and the same content is counted more than once.
    """
    root, _ = _symlinked_dir_bundle(tmp_path)
    keys = [bundle_member_key(p, root) for p in iter_note_paths(root, "*/*.md")]
    assert keys == ["real/a.md"]
    assert len(keys) == len(set(keys))


def test_directory_symlink_out_of_the_bundle_stays_excluded(tmp_path) -> None:
    root, _ = _symlinked_dir_bundle(tmp_path)
    for pattern in ("escape/*.md", "*/*.md", "**/*.md"):
        names = [p.name for p in iter_note_paths(root, pattern)]
        assert "x.md" not in names, pattern


# --- members must be regular files (Codex r3 #2) ---------------------------


def test_a_fifo_is_not_a_bundle_member(tmp_path) -> None:
    """Not reproducible as a hang on macOS 3.14 or Linux 3.12, but a FIFO named
    bundle.toml is not a descriptor and reading one is meaningless at best."""
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    os.mkfifo(root / "bundle.toml")
    os.mkfifo(root / ".bundle-commit")
    os.mkfifo(root / "note.md")

    assert bundle_member_key(root / "bundle.toml", root) is None
    assert bundle_member_key(root / ".bundle-commit", root) is None
    assert read_bundle_descriptor(root) == {}
    assert resolve_bundle_commit(root) == (None, "unknown")
    assert [p.name for p in iter_note_paths(root, GLOB)] == ["a.md"]


def test_a_directory_is_not_a_bundle_member(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / "notadir.md").mkdir()
    assert bundle_member_key(root / "notadir.md", root) is None
    assert [p.name for p in iter_note_paths(root, GLOB)] == ["a.md"]


# --- one file, one entry, however many routes reach it (Codex r4 #1) -------


def test_a_pattern_reaching_one_file_twice_counts_it_once(tmp_path) -> None:
    """`**/../**/*.md` finds bundle/../bundle/sub/a.md and bundle/sub/../sub/a.md.

    Both resolve to the same note and key identically, so without dedup the
    digest counted one note twice and disagreed with the same bundle walked by
    a plainer pattern.
    """
    root = tmp_path / "bundle"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "a.md").write_bytes(b"alpha\n")

    plain = compute_index_revision(root, "**/*.md")
    assert len(iter_note_paths(root, "**/../**/*.md")) == 1
    assert compute_index_revision(root, "**/../**/*.md") == plain
    assert compute_index_revision(root, "*/../*/*.md") == plain


def test_dedup_does_not_merge_distinct_files_sharing_a_canonical_key(
    tmp_path,
) -> None:
    """Dedup keys on the resolved path, not on the canonical key.

    Two byte-distinct filenames can normalise to one NFC key while being
    different files; merging them would silently drop content from the digest.
    """
    import unicodedata

    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", nfc)
    root = tmp_path / "both"
    root.mkdir()
    (root / f"{nfc}.md").write_bytes(b"one\n")
    try:
        (root / f"{nfd}.md").write_bytes(b"two\n")
    except OSError:  # pragma: no cover - normalising filesystem
        pytest.skip("filesystem normalises filenames; both spellings collapse")
    # The premise comes from the directory, not from iter_note_paths. Asking
    # the function under test whether its own precondition holds is how a
    # dedup bug turns into a skip instead of a failure -- which is exactly
    # what this assertion caught when it was written the other way round.
    if len(os.listdir(root)) < 2:  # pragma: no cover - normalising filesystem
        pytest.skip("filesystem normalises filenames; both spellings collapse")

    notes = iter_note_paths(root, GLOB)
    assert len(notes) == 2
    assert {p.read_bytes() for p in notes} == {b"one\n", b"two\n"}
