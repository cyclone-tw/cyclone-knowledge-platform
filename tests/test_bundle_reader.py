"""The bundle is read from an anchored directory descriptor, never by path trust."""

from __future__ import annotations

import errno
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import ckp.bundle as bundle_module
from ckp.bundle import (
    REFUSAL_HARDLINK,
    REFUSAL_NOT_REGULAR,
    REFUSAL_PATH,
    REFUSAL_RACE,
    REFUSAL_ROOT,
    REFUSAL_SYMLINK,
    REFUSAL_UNREADABLE,
    AnchoredBundleReader,
    BundleMember,
    MemberRefused,
    SnapshotRaceError,
)


def test_reads_a_regular_member_as_verified_bytes(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_bytes(b"public bytes\n")

    result = AnchoredBundleReader(tmp_path).read_member("note.md")

    assert isinstance(result, BundleMember)
    assert result.relative_path == "note.md"
    assert result.content == b"public bytes\n"
    assert result.content_sha256


def test_root_fd_must_be_a_directory_even_if_open_flags_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_file = tmp_path / "not-a-directory"
    root_file.write_bytes(b"regular file\n")
    original_open = os.open

    def open_root_without_directory_enforcement(
        path, flags, mode=0o777, *, dir_fd=None
    ):
        if path == root_file and dir_fd is None:
            return original_open(path, os.O_RDONLY)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("ckp.bundle.os.open", open_root_without_directory_enforcement)

    result = AnchoredBundleReader(root_file)._open_root()

    assert isinstance(result, MemberRefused)
    assert result.reason == REFUSAL_ROOT


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


def test_posix_backslash_filename_is_a_literal_bundle_member(tmp_path: Path) -> None:
    note = tmp_path / "literal\\name.md"
    note.write_bytes(b"safe\n")

    result = AnchoredBundleReader(tmp_path).read_member("literal\\name.md")

    assert isinstance(result, BundleMember)
    assert result.relative_path == "literal\\name.md"


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


def test_post_read_file_type_or_link_change_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note = tmp_path / "note.md"
    note.write_bytes(b"checked bytes\n")
    inode = note.stat().st_ino
    original_reason = bundle_module._reason_for_stat
    calls = 0

    def changed_after_read(value):
        nonlocal calls
        if value.st_ino == inode:
            calls += 1
            if calls == 3:
                return REFUSAL_HARDLINK
        return original_reason(value)

    monkeypatch.setattr(bundle_module, "_reason_for_stat", changed_after_read)

    result = AnchoredBundleReader(tmp_path).read_member("note.md")

    assert calls == 3
    assert isinstance(result, MemberRefused)
    assert result.reason == REFUSAL_RACE


def test_post_read_fstat_signature_change_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note = tmp_path / "note.md"
    note.write_bytes(b"checked bytes\n")
    inode = note.stat().st_ino
    original_fstat = os.fstat
    file_fstats = 0

    def changed_after_read(fd):
        nonlocal file_fstats
        value = original_fstat(fd)
        if value.st_ino == inode:
            file_fstats += 1
            if file_fstats == 2:
                return SimpleNamespace(
                    st_dev=value.st_dev,
                    st_ino=value.st_ino,
                    st_mode=value.st_mode,
                    st_nlink=value.st_nlink,
                    st_size=value.st_size + 1,
                    st_mtime_ns=value.st_mtime_ns,
                    st_ctime_ns=value.st_ctime_ns,
                )
        return value

    monkeypatch.setattr("ckp.bundle.os.fstat", changed_after_read)

    result = AnchoredBundleReader(tmp_path).read_member("note.md")

    assert file_fstats == 2
    assert isinstance(result, MemberRefused)
    assert result.reason == REFUSAL_RACE


def test_post_read_namespace_replacement_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note = tmp_path / "note.md"
    note.write_bytes(b"checked bytes\n")
    inode = note.stat().st_ino
    original_signature = bundle_module._file_signature
    calls = 0

    def changed_namespace(value):
        nonlocal calls
        signature = original_signature(value)
        if value.st_ino == inode:
            calls += 1
            if calls == 4:
                return replace(signature, inode=signature.inode + 1)
        return signature

    monkeypatch.setattr(bundle_module, "_file_signature", changed_namespace)

    result = AnchoredBundleReader(tmp_path).read_member("note.md")

    assert calls == 4
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


def test_parent_replacement_after_file_read_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "note.md").write_bytes(b"checked bytes\n")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "note.md").write_bytes(b"different bytes\n")
    original_stat = os.stat
    nested_stats = 0
    swapped = False

    def swap_before_lineage_check(path, *args, **kwargs):
        nonlocal nested_stats, swapped
        if path == "nested" and kwargs.get("dir_fd") is not None:
            nested_stats += 1
            if nested_stats == 2:
                nested.rename(tmp_path / "old")
                replacement.rename(nested)
                swapped = True
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr("ckp.bundle.os.stat", swap_before_lineage_check)

    result = AnchoredBundleReader(tmp_path).read_member("nested/note.md")

    assert swapped
    assert isinstance(result, MemberRefused)
    assert result.reason == REFUSAL_RACE


def test_capture_rejects_a_stale_member_probe(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_bytes(b"probed bytes\n")
    reader = AnchoredBundleReader(tmp_path)
    probe = reader.probe("**/*.md")
    note.rename(tmp_path / "old.md")
    note.write_bytes(b"replacement bytes\n")

    with pytest.raises(SnapshotRaceError):
        reader.capture("**/*.md", probe)


def test_capture_rejects_a_retargeted_root_symlink(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    mounted = tmp_path / "mounted"
    mounted.symlink_to(first, target_is_directory=True)
    reader = AnchoredBundleReader(mounted)
    probe = reader.probe("**/*.md")
    mounted.unlink()
    mounted.symlink_to(second, target_is_directory=True)

    with pytest.raises(SnapshotRaceError):
        reader.capture("**/*.md", probe)


def test_capture_marks_an_open_refusal_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "safe.md").write_bytes(b"safe bytes\n")
    (tmp_path / "blocked.md").write_bytes(b"blocked bytes\n")
    original_open = os.open

    def refuse_blocked(path, flags, mode=0o777, *, dir_fd=None):
        if path == "blocked.md" and dir_fd is not None:
            raise PermissionError(errno.EACCES, "synthetic refusal", path)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("ckp.bundle.os.open", refuse_blocked)

    capture = AnchoredBundleReader(tmp_path).capture("**/*.md")

    assert [member.relative_path for member in capture.members] == ["safe.md"]
    assert capture.readable is False
    probe = AnchoredBundleReader(tmp_path).probe("**/*.md")
    assert (
        next(
            item for item in probe.members if item.relative_path == "blocked.md"
        ).refusal
        == REFUSAL_UNREADABLE
    )


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
    # C1 compatibility: symlink/non-regular candidates are excluded from the
    # note set. Hardlink, unreadable, and unstable candidates fail the whole
    # snapshot instead.
    assert first.readable is True
