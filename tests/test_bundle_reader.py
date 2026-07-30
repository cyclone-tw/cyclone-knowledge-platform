"""The bundle is read from an anchored directory descriptor, never by path trust."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ckp.bundle import (
    REFUSAL_HARDLINK,
    REFUSAL_NOT_REGULAR,
    REFUSAL_PATH,
    REFUSAL_RACE,
    REFUSAL_SYMLINK,
    AnchoredBundleReader,
    BundleMember,
    MemberRefused,
)


def test_reads_a_regular_member_as_verified_bytes(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_bytes(b"public bytes\n")

    result = AnchoredBundleReader(tmp_path).read_member("note.md")

    assert isinstance(result, BundleMember)
    assert result.relative_path == "note.md"
    assert result.content == b"public bytes\n"
    assert result.content_sha256


@pytest.mark.parametrize(
    "member",
    ["", ".", "..", "../outside.md", "sub/../../outside.md", "/tmp/x.md"],
)
def test_path_traversal_and_non_members_are_refused(
    tmp_path: Path, member: str
) -> None:
    result = AnchoredBundleReader(tmp_path).read_member(member)
    assert isinstance(result, MemberRefused)
    assert result.reason == REFUSAL_PATH


def test_path_traversal_cannot_read_an_existing_outside_file(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (tmp_path / "outside.md").write_bytes(b"outside bytes\n")

    result = AnchoredBundleReader(root).read_member("../outside.md")

    assert isinstance(result, MemberRefused)
    assert result.reason == REFUSAL_PATH


def test_final_symlink_is_refused_even_when_it_points_inside(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_bytes(b"safe\n")
    (tmp_path / "alias.md").symlink_to(tmp_path / "real.md")

    result = AnchoredBundleReader(tmp_path).read_member("alias.md")

    assert isinstance(result, MemberRefused)
    assert result.reason == REFUSAL_SYMLINK


def test_parent_symlink_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "note.md").write_bytes(b"safe\n")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)

    result = AnchoredBundleReader(tmp_path).read_member("alias/note.md")

    assert isinstance(result, MemberRefused)
    assert result.reason == REFUSAL_SYMLINK


def test_bundle_root_itself_may_be_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "note.md").write_bytes(b"safe\n")
    mounted = tmp_path / "mounted"
    mounted.symlink_to(real, target_is_directory=True)

    result = AnchoredBundleReader(mounted).read_member("note.md")

    assert isinstance(result, BundleMember)
    assert result.content == b"safe\n"


def test_non_regular_member_is_refused_without_opening_it(tmp_path: Path) -> None:
    fifo = tmp_path / "note.md"
    os.mkfifo(fifo)

    result = AnchoredBundleReader(tmp_path).read_member("note.md")

    assert isinstance(result, MemberRefused)
    assert result.reason == REFUSAL_NOT_REGULAR


def test_multiple_hardlinks_are_conservatively_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"must not cross the bundle boundary\n")
    root = tmp_path / "bundle"
    root.mkdir()
    os.link(outside, root / "note.md")

    result = AnchoredBundleReader(root).read_member("note.md")

    assert isinstance(result, MemberRefused)
    assert result.reason == REFUSAL_HARDLINK


def test_check_to_open_replacement_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swap the final inode after lstat but immediately before openat."""
    note = tmp_path / "note.md"
    note.write_bytes(b"checked bytes\n")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"different bytes\n")
    original_open = os.open
    swapped = False

    def swap_then_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "note.md" and dir_fd is not None and not swapped:
            swapped = True
            note.rename(tmp_path / "old.md")
            replacement.rename(note)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("ckp.bundle.os.open", swap_then_open)

    result = AnchoredBundleReader(tmp_path).read_member("note.md")

    assert swapped
    assert isinstance(result, MemberRefused)
    assert result.reason == REFUSAL_RACE


def test_parent_replacement_during_walk_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "nested"
    original.mkdir()
    (original / "note.md").write_bytes(b"checked bytes\n")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "note.md").write_bytes(b"different bytes\n")
    original_open = os.open
    swapped = False

    def swap_then_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "nested" and dir_fd is not None and not swapped:
            swapped = True
            original.rename(tmp_path / "old")
            replacement.rename(original)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("ckp.bundle.os.open", swap_then_open)

    result = AnchoredBundleReader(tmp_path).read_member("nested/note.md")

    assert swapped
    assert isinstance(result, MemberRefused)
    assert result.reason == REFUSAL_RACE


def test_capture_uses_only_safe_members_and_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "z.md").write_bytes(b"z\n")
    (tmp_path / "a.md").write_bytes(b"a\n")
    (tmp_path / "target.md").write_bytes(b"target\n")
    (tmp_path / "alias.md").symlink_to(tmp_path / "target.md")

    first = AnchoredBundleReader(tmp_path).capture("**/*.md")
    second = AnchoredBundleReader(tmp_path).capture("**/*.md")

    assert [member.relative_path for member in first.members] == [
        "a.md",
        "target.md",
        "z.md",
    ]
    assert first.members == second.members
    assert first.token == second.token
