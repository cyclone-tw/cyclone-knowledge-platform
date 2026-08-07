"""Config layering, and what happens when an override is wrong."""

from __future__ import annotations

import re

import pytest

from ckp.config import (
    RESERVED_ENV,
    ConfigError,
    _coerce,
    _env_name,
    _load_defaults,
    load_config,
)
from conftest import REPO_ROOT


def test_defaults_only_records_one_layer() -> None:
    config = load_config(env={})
    assert config.layers == ("defaults",)
    assert config.server_port == 8080
    assert config.bundle_note_glob == "**/*.md"


def test_file_layer_overrides_defaults(tmp_path) -> None:
    cfg = tmp_path / "ckp.toml"
    cfg.write_text("[server]\nport = 9001\n", encoding="utf-8")

    config = load_config(config_file=cfg, env={})

    assert config.server_port == 9001
    assert config.layers == ("defaults", f"file:{cfg}")
    # Untouched keys keep the default rather than disappearing.
    assert config.server_host == "127.0.0.1"


def test_env_overrides_file_overrides_defaults(tmp_path) -> None:
    """The whole point of layering: the most specific source wins."""
    cfg = tmp_path / "ckp.toml"
    cfg.write_text('[server]\nport = 9001\nhost = "10.0.0.1"\n', encoding="utf-8")

    config = load_config(config_file=cfg, env={"CKP_SERVER_PORT": "9002"})

    assert config.server_port == 9002  # env beat the file
    assert config.server_host == "10.0.0.1"  # file beat the default
    assert config.bundle_note_glob == "**/*.md"  # default survived untouched
    assert config.layers == ("defaults", f"file:{cfg}", "env")


def test_config_file_can_come_from_the_environment(tmp_path) -> None:
    cfg = tmp_path / "ckp.toml"
    cfg.write_text("[server]\nport = 9003\n", encoding="utf-8")

    config = load_config(env={"CKP_CONFIG_FILE": str(cfg)})

    assert config.server_port == 9003
    # CKP_CONFIG_FILE selects a source; it must not be mistaken for a value
    # override and it must not register as an env layer on its own.
    assert config.layers == ("defaults", f"file:{cfg}")


def test_unknown_environment_variable_fails_closed() -> None:
    """A typo'd override that silently does nothing is the expensive bug."""
    with pytest.raises(ConfigError, match="CKP_SERVER_PORTT"):
        load_config(env={"CKP_SERVER_PORTT": "9000"})


def test_unrelated_environment_variables_are_ignored() -> None:
    config = load_config(env={"PATH": "/usr/bin", "HOME": "/somewhere"})
    assert config.layers == ("defaults",)


def test_unknown_key_in_config_file_fails_closed(tmp_path) -> None:
    cfg = tmp_path / "ckp.toml"
    cfg.write_text("[server]\nprot = 9001\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="server.prot"):
        load_config(config_file=cfg, env={})


def test_unknown_section_in_config_file_fails_closed(tmp_path) -> None:
    cfg = tmp_path / "ckp.toml"
    cfg.write_text('[qdrant]\nurl = "http://localhost:6333"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\[qdrant\]"):
        load_config(config_file=cfg, env={})


def test_malformed_config_file_fails_closed(tmp_path) -> None:
    cfg = tmp_path / "ckp.toml"
    cfg.write_text("[server\nport = 9001\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(config_file=cfg, env={})


def test_missing_config_file_fails_closed(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not readable"):
        load_config(config_file=tmp_path / "absent.toml", env={})


def test_env_value_is_coerced_to_the_declared_type() -> None:
    config = load_config(env={"CKP_SERVER_PORT": "9100"})
    assert config.server_port == 9100
    assert isinstance(config.server_port, int)


def test_uncoercible_env_value_fails_closed() -> None:
    with pytest.raises(ConfigError, match="not an integer"):
        load_config(env={"CKP_SERVER_PORT": "eight thousand"})


def test_wrong_type_in_config_file_fails_closed(tmp_path) -> None:
    cfg = tmp_path / "ckp.toml"
    cfg.write_text("[server]\nport = true\n", encoding="utf-8")
    # bool is a subclass of int, so a naive isinstance check would accept this.
    with pytest.raises(ConfigError, match="must be int"):
        load_config(config_file=cfg, env={})


def test_unsupported_default_type_is_rejected_rather_than_guessed() -> None:
    """New default types arrive with their branch and their test, not before."""
    with pytest.raises(ConfigError, match="unsupported default type"):
        _coerce("1.5", 1.5, "synthetic")


def test_empty_expected_profile_version_reads_as_unset() -> None:
    assert load_config(env={}).profile_expected_version is None
    config = load_config(env={"CKP_PROFILE_EXPECTED_VERSION": "cyclone-profile-v1"})
    assert config.profile_expected_version == "cyclone-profile-v1"


# --- paths that only fail outside OSError (Codex r1 findings 1-2, plus the
# --- same root cause at the bundle-root read point)


def test_config_file_with_a_null_byte_raises_config_error() -> None:
    """A null byte only fails at the syscall, far from where it was accepted."""
    with pytest.raises(ConfigError, match="null byte"):
        load_config(config_file="\x00bad", env={})


def test_config_file_with_an_unknown_user_home_raises_config_error() -> None:
    """``expanduser`` raises RuntimeError here, which no OSError guard catches."""
    with pytest.raises(ConfigError, match="cannot resolve path"):
        load_config(config_file="~definitely-no-such-user-ckp/f.toml", env={})


def test_bad_bundle_root_fails_at_startup_not_at_request_time() -> None:
    """The worst shape of this bug: config loads, then a request 500s.

    Resolving bundle.root during load means the process refuses to start
    instead of serving a broken /health.
    """
    with pytest.raises(ConfigError, match="bundle.root"):
        load_config(env={"CKP_BUNDLE_ROOT": "~definitely-no-such-user-ckp/bundle"})
    with pytest.raises(ConfigError, match="null byte"):
        load_config(env={"CKP_BUNDLE_ROOT": "bad\x00root"})


def test_bundle_root_is_expanded_once_at_load_time(tmp_path) -> None:
    config = load_config(env={"CKP_BUNDLE_ROOT": str(tmp_path)})
    assert config.bundle_root == tmp_path
    # The resolved Path is what callers use; the raw string stays inspectable.
    assert config.get("bundle", "root") == str(tmp_path)


def test_home_relative_bundle_root_expands() -> None:
    from pathlib import Path

    config = load_config(env={"CKP_BUNDLE_ROOT": "~/ckp-bundle"})
    assert config.bundle_root == Path.home() / "ckp-bundle"
    assert "~" not in str(config.bundle_root)


# --- the same class as the path bug, one level up: a value the path layer
# --- will reject must be rejected here, not at request time (Codex r1 #3)


def test_empty_note_glob_is_rejected_at_load() -> None:
    """Path.glob("") raises ValueError from the call itself, not on iteration."""
    with pytest.raises(ConfigError, match="pattern is empty"):
        load_config(env={"CKP_BUNDLE_NOTE_GLOB": ""})


def test_whitespace_note_glob_is_rejected_at_load() -> None:
    with pytest.raises(ConfigError, match="pattern is empty"):
        load_config(env={"CKP_BUNDLE_NOTE_GLOB": "   "})


def test_absolute_note_glob_is_rejected_at_load() -> None:
    """Path.glob raises NotImplementedError for a non-relative pattern."""
    with pytest.raises(ConfigError, match="unusable pattern"):
        load_config(env={"CKP_BUNDLE_NOTE_GLOB": "/etc/*.md"})


def test_usable_note_glob_survives_validation() -> None:
    config = load_config(env={"CKP_BUNDLE_NOTE_GLOB": "notes/**/*.md"})
    assert config.bundle_note_glob == "notes/**/*.md"


def _ci_workflow_env_names() -> set[str]:
    """Every ``CKP_*`` name the CI workflow sets, read from the YAML itself.

    Parsed with a regex rather than a YAML loader on purpose: the assertion is
    about names appearing anywhere in the file, so it must not depend on where
    in the job tree they sit, and PyYAML is not a test dependency here.
    """
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    # Matches the name wherever it appears as a mapping key: bare, single- or
    # double-quoted, and inside a flow mapping (`{CKP_X: "1"}`), which are all
    # legal YAML for the same thing. Anchoring on the prefix rather than on
    # line structure keeps a formatting choice from silently shrinking what
    # this guard sees.
    return set(re.findall(r"""['"{,\s](CKP_[A-Za-z0-9_]+)['"]?\s*:""", workflow))


def test_every_ci_env_name_is_a_config_key_or_explicitly_reserved() -> None:
    """Issue #46: CI set three ``CKP_*`` harness switches config had never
    heard of, so any test reading the real environment blew up there with a
    message blaming config -- on the first unrelated PR that happened to add
    such a test, months after the switches landed.

    A harness switch is a legitimate thing to have. Silently colliding with a
    fail-closed allowlist is not. This fails at the moment a new name is added
    without a decision about which side it belongs on.
    """
    defaults = _load_defaults()
    config_keys = {
        _env_name(section, key)
        for section, entries in defaults.items()
        for key in entries
    }
    for name in sorted(_ci_workflow_env_names()):
        assert name in config_keys or name in RESERVED_ENV, (
            f"{name} is set by .github/workflows/ci.yml but is neither a "
            f"config key from defaults.toml nor listed in RESERVED_ENV; "
            f"loading config over the real CI environment would raise "
            f"ConfigError. Register it in defaults.toml if it is a config "
            f"value, or add it to RESERVED_ENV if it is a harness switch."
        )


def test_no_config_key_is_also_reserved() -> None:
    """A real config key wrongly added to ``RESERVED_ENV`` fails silently.

    The override loop skips reserved names before the lookup, so the variable
    would simply stop working -- no error, no warning, just a setting that
    quietly ignores its environment override. That is worse than the
    ConfigError this issue was about, because nothing points at it at all.
    """
    defaults = _load_defaults()
    config_keys = {
        _env_name(section, key)
        for section, entries in defaults.items()
        for key in entries
    }
    overlap = sorted(config_keys & RESERVED_ENV)
    assert not overlap, (
        f"{overlap} appear both as config keys and in RESERVED_ENV; the "
        f"override loop skips reserved names, so these would silently stop "
        f"reading their environment variable"
    )


def test_the_ci_workflow_actually_sets_some_ckp_names() -> None:
    """Guards the guard: a regex that silently matched nothing would make the
    check above pass for every possible workflow.
    """
    assert len(_ci_workflow_env_names()) >= 3
