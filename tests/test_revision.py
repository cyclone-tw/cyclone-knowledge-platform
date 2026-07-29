"""The four revision fields: derived where it matters, honest where it cannot be.

The digest tests are the substance of this file. Contract §5.5 makes "the
rebuild is not reproducible from the same commit" a rollback trigger, so
reproducibility has to be a property that fails loudly, not a hope.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ckp.config import load_config
from ckp.revision import (
    API_VERSION,
    build_revision,
    bundle_is_readable,
    compute_index_revision,
    iter_note_paths,
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


def test_empty_bundle_digests_but_is_not_readable(tmp_path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    assert compute_index_revision(root, GLOB) is not None
    assert bundle_is_readable(root, GLOB) is False


def test_note_paths_are_sorted_by_posix_relative_path(tmp_path) -> None:
    root = _bundle(
        tmp_path / "b",
        {"z.md": b"z\n", "a.md": b"a\n", "m/b.md": b"b\n"},
    )
    rel = [p.relative_to(root).as_posix() for p in iter_note_paths(root, GLOB)]
    assert rel == sorted(rel)
    assert rel == ["a.md", "m/b.md", "z.md"]


# --- bundle_commit: evidence, never invention -------------------------------


def test_commit_comes_from_git_when_the_bundle_is_a_work_tree(tmp_path) -> None:
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    head = _git_repo(root)
    assert resolve_bundle_commit(root) == (head, "git")


def test_commit_falls_back_to_the_stamp_without_git(tmp_path) -> None:
    """The container case: no repository to ask, a build-time stamp instead."""
    root = _bundle(tmp_path / "b", {"a.md": b"alpha\n"})
    (root / ".bundle-commit").write_text("abc123\n", encoding="utf-8")
    assert resolve_bundle_commit(root) == ("abc123", "stamp")


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
    if len(iter_note_paths(root, GLOB)) < 2:  # pragma: no cover
        pytest.skip("filesystem normalises filenames; both spellings collapse")

    assert compute_index_revision(root, GLOB) == compute_index_revision(root, GLOB)
    ordering = [p.name for p in iter_note_paths(root, GLOB)]
    assert ordering == sorted(ordering)
