"""P1 pilot corpus binding (issue #23): fixed six-path manifest, fail-closed.

Every test here uses a synthetic root under ``tmp_path`` and a synthetic
``frozen_manifest`` injected into ``bind_pilot_corpus`` -- never the real
Cyclone-Wiki checkout, so this suite never has to depend on (or accidentally
vendor) real note content. The real checkout is exercised only by a manual
smoke check (issue #23 "Runtime smoke"), outside pytest.
"""

from __future__ import annotations

import hashlib

import pytest

from ckp.config import load_config
from ckp.pilot.manifest import (
    PILOT_NOTE_PATHS,
    PilotBindingError,
    PilotManifest,
    PilotManifestEntry,
    bind_pilot_corpus,
    load_frozen_manifest,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write(root, relative_path: str, text: str) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _note(
    privacy: str = "internal", note_type: str = "Procedure", body: str = "hi"
) -> str:
    return f"---\ntype: {note_type}\nprivacy: {privacy}\n---\n\n# note\n\n{body}\n"


def _synthetic_manifest(entries: dict[str, str]) -> PilotManifest:
    """Build a manifest matching ``PILOT_NOTE_PATHS`` from ``{path: text}``."""
    return PilotManifest(
        entries=tuple(
            PilotManifestEntry(
                relative_path=path, content_sha256=_sha256(entries[path])
            )
            for path in PILOT_NOTE_PATHS
        )
    )


def _populate_all(root, text_by_path: dict[str, str]) -> None:
    for path in PILOT_NOTE_PATHS:
        _write(root, path, text_by_path[path])


def _all_internal(tmp_path) -> dict[str, str]:
    return {path: _note(body=f"body for {path}") for path in PILOT_NOTE_PATHS}


# --- RP4: the corpus is an explicit list, not a glob or a rule scan --------


def test_pilot_note_paths_is_exactly_six_explicit_relative_paths() -> None:
    assert len(PILOT_NOTE_PATHS) == 6
    for path in PILOT_NOTE_PATHS:
        assert "*" not in path
        assert not path.startswith("/")
        assert path.endswith(".md")


def test_load_frozen_manifest_rejects_paths_that_drift_from_the_allowlist(
    tmp_path,
) -> None:
    """RP4 in the manifest file itself: editing paths must fail, not expand."""
    manifest_file = tmp_path / "pilot-manifest.toml"
    manifest_file.write_text(
        '[[note]]\nrelative_path = "Core/not-the-frozen-list.md"\n'
        'content_sha256 = "' + "0" * 64 + '"\n',
        encoding="utf-8",
    )
    config = load_config(env={"CKP_PILOT_MANIFEST_PATH": str(manifest_file)})
    with pytest.raises(PilotBindingError, match="PILOT_NOTE_PATHS"):
        load_frozen_manifest(config)


@pytest.mark.parametrize(
    "value_toml,type_name",
    [
        ('{ body = "leaked note body" }', "dict"),
        ('["leaked", "note", "body"]', "list"),
        ("42", "int"),
    ],
    ids=["inline-table", "array", "int"],
)
def test_load_frozen_manifest_rejects_non_string_field_values(
    tmp_path, value_toml: str, type_name: str
) -> None:
    """The third layer of the same hole (Codex round 1 on #36).

    #33 guarded the note-level keys, this ticket guarded the top-level keys,
    and a TOML *value* was still a place to smuggle content: an inline table
    under an allowed key passed both allowlists. Values must be plain
    strings, and the refusal must not echo the value -- the manifest sits
    outside the Markdown privacy scanner, so an error message that repeats
    the payload is itself a leak path.
    """
    manifest_file = tmp_path / "pilot-manifest.toml"
    lines = "\n".join(
        f'[[note]]\nrelative_path = "{path}"\ncontent_sha256 = '
        + (value_toml if index == 0 else f'"{"0" * 64}"')
        for index, path in enumerate(PILOT_NOTE_PATHS)
    )
    manifest_file.write_text(lines + "\n", encoding="utf-8")

    config = load_config(env={"CKP_PILOT_MANIFEST_PATH": str(manifest_file)})
    with pytest.raises(PilotBindingError) as excinfo:
        load_frozen_manifest(config)
    message = str(excinfo.value)
    assert "content_sha256" in message
    assert type_name in message
    assert "leaked" not in message  # the value itself must never be echoed


@pytest.mark.parametrize(
    "extra_key,extra_value",
    [
        ("body", '"leaked content"'),
        ("content", '"leaked content"'),
        ("privacy", '"public"'),
        ("note_body_b64", '"bGVha2Vk"'),
    ],
)
def test_load_frozen_manifest_rejects_any_undeclared_note_field(
    tmp_path,
    extra_key: str,
    extra_value: str,
) -> None:
    """RP1 at the file-format layer, not just the dataclass layer.

    `PilotManifestEntry` has no `body` field, so a naive schema loader would
    silently *accept and ignore* a manifest carrying one -- and a manifest
    file that can carry `body = "..."` at all means RP1 was never actually
    enforced on the file format, only on the in-memory type (Codex review
    finding). Every real `[[note]]` key here is otherwise valid.

    Parametrised beyond `body` on purpose: the rule is "no undeclared key",
    not "no key literally named body". A loader hardened only against the
    one field this review happened to name would pass a body-only test while
    still accepting `content` or a base64 smuggling field.
    """
    manifest_file = tmp_path / "pilot-manifest.toml"
    lines = "\n".join(
        f'[[note]]\nrelative_path = "{path}"\ncontent_sha256 = "{"0" * 64}"'
        + (f"\n{extra_key} = {extra_value}" if index == 0 else "")
        for index, path in enumerate(PILOT_NOTE_PATHS)
    )
    manifest_file.write_text(lines + "\n", encoding="utf-8")

    config = load_config(env={"CKP_PILOT_MANIFEST_PATH": str(manifest_file)})
    with pytest.raises(PilotBindingError) as excinfo:
        load_frozen_manifest(config)
    message = str(excinfo.value)
    # Match the offending key, not the boilerplate: the error text explains
    # the rule with `body` as its example, so `match="body"` would pass for
    # every parameter regardless of which key actually tripped the check.
    assert repr(extra_key) in message, message
    assert "#0" in message


@pytest.mark.parametrize(
    "extra_key,extra_value",
    [
        ("body", '"leaked note body"'),
        ("content", '"leaked note body"'),
        ("privacy", '"public"'),
        ("note_body_b64", '"bGVha2Vk"'),
    ],
)
def test_load_frozen_manifest_rejects_any_undeclared_top_level_field(
    tmp_path,
    extra_key: str,
    extra_value: str,
) -> None:
    """RP1 at the file-format layer, one level above `[[note]]` (issue #36).

    PR #33 hardened the schema inside each `[[note]]` table but left the
    manifest file's top level unguarded: Codex Round 2 found that a bare
    `body = "leaked note body"` line at the top of `pilot-manifest.toml`
    still loaded successfully, because it never reaches the `[[note]]` loop
    at all. Every real `[[note]]` entry here is otherwise valid.

    Parametrised beyond `body` on purpose, same reasoning as the `[[note]]`
    field test above: the rule is "no undeclared top-level key", not "no key
    literally named body". A loader hardened only against the one field this
    review happened to name would pass a body-only test while still
    accepting `content` or a base64 smuggling field at the top level.
    """
    manifest_file = tmp_path / "pilot-manifest.toml"
    notes = "\n".join(
        f'[[note]]\nrelative_path = "{path}"\ncontent_sha256 = "{"0" * 64}"'
        for path in PILOT_NOTE_PATHS
    )
    manifest_file.write_text(
        f"{extra_key} = {extra_value}\n\n{notes}\n", encoding="utf-8"
    )

    config = load_config(env={"CKP_PILOT_MANIFEST_PATH": str(manifest_file)})
    with pytest.raises(PilotBindingError) as excinfo:
        load_frozen_manifest(config)
    message = str(excinfo.value)
    # Match the offending key, not the boilerplate: the error text explains
    # the rule with `body` as its example, so `match="body"` would pass for
    # every parameter regardless of which key actually tripped the check.
    assert repr(extra_key) in message, message


def test_bind_pilot_corpus_rejects_an_injected_manifest_outside_the_allowlist(
    tmp_path,
) -> None:
    """Regression test for the coordinator's exploit run against this branch.

    `load_frozen_manifest` checking its own output was not enough: a caller
    that builds a ``PilotManifest`` directly and passes it in as
    ``frozen_manifest`` used to skip the RP4 check entirely and reach the
    privacy gate with an out-of-allowlist note. A note that would pass the
    privacy gate on its own (`internal`, readable, correctly hashed) must
    still be refused here because its path was never in `PILOT_NOTE_PATHS`.
    """
    rogue_path = "Core/agent-capabilities.md"
    assert rogue_path not in PILOT_NOTE_PATHS  # the whole point of the exploit

    text = _note(privacy="internal")
    _write(tmp_path, rogue_path, text)
    rogue_manifest = PilotManifest(
        entries=(
            PilotManifestEntry(relative_path=rogue_path, content_sha256=_sha256(text)),
        )
    )

    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(tmp_path)})
    with pytest.raises(PilotBindingError, match="PILOT_NOTE_PATHS"):
        bind_pilot_corpus(config, frozen_manifest=rogue_manifest)


def test_real_frozen_manifest_matches_the_allowlist() -> None:
    """The committed config/pilot-manifest.toml must name exactly D2's six."""
    config = load_config(env={})
    manifest = load_frozen_manifest(config)
    assert tuple(entry.relative_path for entry in manifest.entries) == PILOT_NOTE_PATHS
    for entry in manifest.entries:
        assert len(entry.content_sha256) == 64
        int(entry.content_sha256, 16)  # hex, not a placeholder string


# --- D5: checkout missing fails, it never skips -----------------------------


def test_missing_wiki_root_env_var_fails_with_a_named_message() -> None:
    config = load_config(env={})
    with pytest.raises(PilotBindingError, match="CKP_PILOT_WIKI_ROOT"):
        bind_pilot_corpus(config)


def test_wiki_root_pointing_at_a_nonexistent_directory_fails(tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(missing)})
    with pytest.raises(PilotBindingError, match="not a directory"):
        bind_pilot_corpus(config)


def test_wiki_root_pointing_at_a_file_not_a_directory_fails(tmp_path) -> None:
    not_a_dir = tmp_path / "wiki-root-is-a-file"
    not_a_dir.write_text("not a checkout", encoding="utf-8")
    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(not_a_dir)})
    with pytest.raises(PilotBindingError, match="not a directory"):
        bind_pilot_corpus(config)


# --- content hash drift: fails and names the offending note ----------------


def test_content_hash_mismatch_fails_and_names_the_note(tmp_path) -> None:
    texts = _all_internal(tmp_path)
    _populate_all(tmp_path, texts)
    manifest = _synthetic_manifest(texts)

    # Simulate another session's uncommitted dirty file: rewrite one note
    # after the manifest was frozen against the original bytes.
    dirty_path = PILOT_NOTE_PATHS[2]
    _write(tmp_path, dirty_path, _note(body="a dirty session touched this"))

    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(tmp_path)})
    with pytest.raises(PilotBindingError, match="content hash mismatch") as excinfo:
        bind_pilot_corpus(config, frozen_manifest=manifest)
    assert dirty_path in str(excinfo.value)


# --- privacy allowlist: fail-closed -----------------------------------------


def test_sensitive_note_in_the_live_checkout_is_refused(tmp_path) -> None:
    texts = _all_internal(tmp_path)
    texts[PILOT_NOTE_PATHS[0]] = _note(privacy="sensitive")
    _populate_all(tmp_path, texts)
    manifest = _synthetic_manifest(texts)

    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(tmp_path)})
    with pytest.raises(PilotBindingError, match="privacy gate refused"):
        bind_pilot_corpus(config, frozen_manifest=manifest)


def test_student_private_note_in_the_live_checkout_is_refused(tmp_path) -> None:
    texts = _all_internal(tmp_path)
    texts[PILOT_NOTE_PATHS[0]] = _note(privacy="student-private")
    _populate_all(tmp_path, texts)
    manifest = _synthetic_manifest(texts)

    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(tmp_path)})
    with pytest.raises(PilotBindingError, match="privacy gate refused"):
        bind_pilot_corpus(config, frozen_manifest=manifest)


def test_restricted_note_in_the_live_checkout_is_refused(tmp_path) -> None:
    texts = _all_internal(tmp_path)
    texts[PILOT_NOTE_PATHS[0]] = _note(privacy="restricted")
    _populate_all(tmp_path, texts)
    manifest = _synthetic_manifest(texts)

    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(tmp_path)})
    with pytest.raises(PilotBindingError, match="privacy gate refused"):
        bind_pilot_corpus(config, frozen_manifest=manifest)


def test_public_note_in_the_live_checkout_is_refused(tmp_path) -> None:
    """Admissible is pinned to exactly what D2 froze (`internal`), not wider.

    `public` is strictly less restrictive than `internal`, but D2 never
    admitted it and the wiki survey found zero `public` notes in Core, so
    a `public` note here is refused rather than silently let through a
    branch nothing real would ever exercise.
    """
    texts = _all_internal(tmp_path)
    texts[PILOT_NOTE_PATHS[0]] = _note(privacy="public")
    _populate_all(tmp_path, texts)
    manifest = _synthetic_manifest(texts)

    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(tmp_path)})
    with pytest.raises(PilotBindingError, match="privacy gate refused"):
        bind_pilot_corpus(config, frozen_manifest=manifest)


def test_pilot_admissible_is_pinned_to_internal_only() -> None:
    from ckp.pilot.manifest import _PILOT_ADMISSIBLE
    from ckp.privacy import PrivacyClass

    assert frozenset({PrivacyClass.INTERNAL}) == _PILOT_ADMISSIBLE


def test_note_with_no_parseable_privacy_class_is_refused(tmp_path) -> None:
    texts = _all_internal(tmp_path)
    texts[PILOT_NOTE_PATHS[0]] = "---\ntype: Procedure\n---\n\n# no privacy line\n"
    _populate_all(tmp_path, texts)
    manifest = _synthetic_manifest(texts)

    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(tmp_path)})
    with pytest.raises(PilotBindingError, match="privacy gate refused"):
        bind_pilot_corpus(config, frozen_manifest=manifest)


def test_student_reference_type_is_refused_even_when_privacy_says_internal(
    tmp_path,
) -> None:
    """Second, independent guard: a mistagged `internal` note must still fail."""
    texts = _all_internal(tmp_path)
    texts[PILOT_NOTE_PATHS[0]] = _note(
        privacy="internal", note_type="Student Reference"
    )
    _populate_all(tmp_path, texts)
    manifest = _synthetic_manifest(texts)

    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(tmp_path)})
    with pytest.raises(PilotBindingError, match="Student Reference"):
        bind_pilot_corpus(config, frozen_manifest=manifest)


# --- happy path: manifest carries no body -----------------------------------


def test_successful_binding_returns_only_path_and_hash_no_body(tmp_path) -> None:
    texts = _all_internal(tmp_path)
    _populate_all(tmp_path, texts)
    manifest = _synthetic_manifest(texts)

    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(tmp_path)})
    result = bind_pilot_corpus(config, frozen_manifest=manifest)

    assert tuple(entry.relative_path for entry in result.entries) == PILOT_NOTE_PATHS
    for entry, path in zip(result.entries, PILOT_NOTE_PATHS, strict=True):
        assert entry.content_sha256 == _sha256(texts[path])
        # Structural: an entry has exactly these two fields, nothing else.
        assert {f for f in entry.__dataclass_fields__} == {
            "relative_path",
            "content_sha256",
        }
        for value in entry.__dict__.values():
            assert isinstance(value, str)
            # Neither field is long enough to be a note body -- the hash is
            # fixed-width hex, the path is a short repo-relative string.
        assert texts[path] not in repr(entry)


def test_missing_note_in_checkout_fails_and_names_it(tmp_path) -> None:
    texts = _all_internal(tmp_path)
    _populate_all(tmp_path, texts)
    (tmp_path / PILOT_NOTE_PATHS[-1]).unlink()
    manifest = _synthetic_manifest(texts)

    config = load_config(env={"CKP_PILOT_WIKI_ROOT": str(tmp_path)})
    with pytest.raises(PilotBindingError, match="unreadable") as excinfo:
        bind_pilot_corpus(config, frozen_manifest=manifest)
    assert PILOT_NOTE_PATHS[-1] in str(excinfo.value)
