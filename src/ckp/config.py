"""Layered configuration.

Three layers, each overriding the one before it:

0. ``defaults.toml`` shipped inside the package -- also the key schema.
1. an optional TOML file, from the ``config_file`` argument or ``CKP_CONFIG_FILE``.
2. environment variables, ``CKP_<SECTION>_<KEY>``.

Unknown keys fail closed at every layer. A mistyped override that silently
does nothing is the config bug that costs the most to find, so an override
naming a key the schema does not define is an error, not a no-op.

Which layers actually contributed is recorded on the resulting ``Config`` --
layering that cannot be observed cannot be tested.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

ENV_PREFIX = "CKP_"

# `CKP_*` names that are deliberately NOT config keys, so the unknown-key check
# below cannot mistake them for typos.
#
# `CKP_CONFIG_FILE` selects *where* config comes from; not a config value.
#
# The rest are test-harness switches read straight from `os.environ` by the
# test modules that own them -- they never travel through config layering. CI
# sets all three, which meant any test calling `load_config()` over the real
# environment failed there with a message pointing at config (issue #46). The
# failure was invisible on main because no test read the real environment, so
# it surfaced as someone else's bug on the first PR that did.
# `tests/test_config.py` pins that every `CKP_*` name in the CI workflow is
# either a config key or listed here, so the next harness switch cannot
# reintroduce this silently.
RESERVED_ENV = frozenset(
    {
        "CKP_CONFIG_FILE",
        "CKP_REQUIRE_QDRANT",
        "CKP_REQUIRE_SEMANTIC",
        "CKP_SEMANTIC_MODEL_DIR",
    }
)


class ConfigError(ValueError):
    """Raised for an unknown key, an unreadable file, or an uncoercible value."""


def _load_defaults() -> dict[str, dict[str, Any]]:
    raw = resources.files("ckp").joinpath("defaults.toml").read_bytes()
    return tomllib.loads(raw.decode("utf-8"))


def _env_name(section: str, key: str) -> str:
    return f"{ENV_PREFIX}{section.upper()}_{key.upper()}"


def _expand_path(value: str, where: str) -> Path:
    """Turn a configured path string into a Path, or raise ``ConfigError``.

    Both failure modes here escape as something other than ``OSError``, so
    neither is caught by the ordinary read guards downstream:

    * a null byte only fails at the syscall, arbitrarily far from here;
    * ``expanduser`` on an unknown ``~user`` raises ``RuntimeError``.

    Resolving every configured path through this function at load time means a
    bad path fails closed while the process is still starting, instead of
    surfacing later as a 500 from a request handler.
    """
    if "\x00" in value:
        raise ConfigError(f"{where}: path contains a null byte")
    try:
        return Path(value).expanduser()
    except (RuntimeError, ValueError, OSError) as exc:
        raise ConfigError(f"{where}: cannot resolve path {value!r}: {exc}") from exc


def _validate_glob(pattern: str, where: str) -> str:
    """Reject a glob the path layer will refuse, while we can still say why.

    Same shape as the path problem above and it needs the same treatment: an
    empty pattern raises ``ValueError`` and an absolute one raises
    ``NotImplementedError``. Left unvalidated they are accepted at startup and
    surface as a 500 from whichever request first walks the bundle.

    **The iterator must be consumed.** On Python 3.12 -- the floor this project
    supports, and what the container runs -- ``Path.glob`` builds a lazy
    iterator and raises only when it is advanced; on 3.13+ it raises from the
    call. Validating without consuming therefore passes on a newer interpreter
    and lets the bug straight through on the supported one, which is exactly
    what happened here. The walk runs against an empty temporary directory so
    consuming it costs nothing and cannot depend on the working directory.
    """
    if not pattern.strip():
        raise ConfigError(f"{where}: pattern is empty")
    try:
        with tempfile.TemporaryDirectory() as probe:
            next(Path(probe).glob(pattern), None)
    except (ValueError, NotImplementedError, OSError) as exc:
        raise ConfigError(f"{where}: unusable pattern {pattern!r}: {exc}") from exc
    return pattern


def _coerce(value: str, template: Any, where: str) -> Any:
    """Coerce a string override to the type its default declares.

    Only the types the schema currently uses are handled. A new default type
    must arrive together with its branch and its test rather than sitting here
    untested waiting for a caller.
    """
    if isinstance(template, int) and not isinstance(template, bool):
        try:
            return int(value)
        except ValueError as exc:
            raise ConfigError(f"{where}: {value!r} is not an integer") from exc
    if isinstance(template, str):
        return value
    raise ConfigError(f"{where}: unsupported default type {type(template).__name__}")


def _merge_file(
    merged: dict[str, dict[str, Any]],
    defaults: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    try:
        raw = path.read_bytes()
    except (OSError, ValueError) as exc:
        # ValueError as well as OSError: a path the filesystem layer rejects
        # outright (an embedded null byte, say) raises the former.
        raise ConfigError(f"config file {path} is not readable: {exc}") from exc
    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"config file {path} is not valid TOML: {exc}") from exc

    for section, entries in parsed.items():
        if section not in defaults:
            raise ConfigError(f"{path}: unknown config section [{section}]")
        if not isinstance(entries, dict):
            raise ConfigError(f"{path}: [{section}] must be a table")
        for key, value in entries.items():
            if key not in defaults[section]:
                raise ConfigError(f"{path}: unknown config key {section}.{key}")
            template = defaults[section][key]
            if isinstance(value, str) and not isinstance(template, str):
                value = _coerce(value, template, f"{path}: {section}.{key}")
            elif isinstance(value, bool) is not isinstance(template, bool):
                # bool is a subclass of int, so `isinstance(True, int)` would
                # wave `port = true` through the check below.
                raise ConfigError(
                    f"{path}: {section}.{key} must be "
                    f"{type(template).__name__}, got {type(value).__name__}"
                )
            elif not isinstance(value, type(template)):
                raise ConfigError(
                    f"{path}: {section}.{key} must be "
                    f"{type(template).__name__}, got {type(value).__name__}"
                )
            merged[section][key] = value


def _merge_env(
    merged: dict[str, dict[str, Any]],
    defaults: Mapping[str, Mapping[str, Any]],
    env: Mapping[str, str],
) -> bool:
    known = {
        _env_name(section, key): (section, key)
        for section, entries in defaults.items()
        for key in entries
    }
    applied = False
    for name, value in env.items():
        if not name.startswith(ENV_PREFIX) or name in RESERVED_ENV:
            continue
        target = known.get(name)
        if target is None:
            raise ConfigError(
                f"unknown config environment variable {name}; "
                f"known names: {', '.join(sorted(known))}"
            )
        section, key = target
        merged[section][key] = _coerce(value, defaults[section][key], name)
        applied = True
    return applied


@dataclass(frozen=True)
class Config:
    """A resolved configuration plus the record of how it was resolved."""

    values: Mapping[str, Mapping[str, Any]]
    layers: tuple[str, ...]
    #: Resolved once at load time rather than on each access, so a path the
    #: system cannot expand is a startup failure and not a 500 mid-request.
    bundle_root: Path

    def get(self, section: str, key: str) -> Any:
        try:
            return self.values[section][key]
        except KeyError as exc:
            raise ConfigError(f"no such config key {section}.{key}") from exc

    @property
    def bundle_note_glob(self) -> str:
        return self.get("bundle", "note_glob")

    @property
    def profile_expected_version(self) -> str | None:
        declared = self.get("profile", "expected_version")
        return declared or None

    @property
    def server_host(self) -> str:
        return self.get("server", "host")

    @property
    def server_port(self) -> int:
        return self.get("server", "port")

    @property
    def pilot_wiki_root(self) -> Path | None:
        """The real Wiki checkout root for the pilot corpus, or ``None``.

        Empty is the deliberate "not configured" sentinel, same pattern as
        ``profile.expected_version``. Unlike ``bundle.root`` this is not
        resolved eagerly at load time: a missing pilot checkout is a P1
        binding-time failure (D5), not a service-startup failure, because the
        rest of this package must keep working with no pilot corpus at all.
        """
        raw = self.get("pilot", "wiki_root")
        return Path(raw).expanduser() if raw else None

    @property
    def pilot_manifest_path(self) -> Path:
        return Path(self.get("pilot", "manifest_path")).expanduser()


def load_config(
    config_file: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Resolve the three config layers, most specific last."""
    env = os.environ if env is None else env
    defaults = _load_defaults()
    merged = {section: dict(entries) for section, entries in defaults.items()}
    layers = ["defaults"]

    chosen = config_file if config_file is not None else env.get("CKP_CONFIG_FILE")
    if chosen:
        path = _expand_path(str(chosen), "CKP_CONFIG_FILE")
        _merge_file(merged, defaults, path)
        layers.append(f"file:{path}")

    if _merge_env(merged, defaults, env):
        layers.append("env")

    frozen = {section: dict(entries) for section, entries in merged.items()}
    # Everything the bundle layer will hand to the filesystem is validated here,
    # while ConfigError is still the contract. Anything left unchecked at this
    # boundary becomes a request-time 500 instead of a refusal to start.
    _validate_glob(frozen["bundle"]["note_glob"], "bundle.note_glob")
    return Config(
        values=frozen,
        layers=tuple(layers),
        bundle_root=_expand_path(frozen["bundle"]["root"], "bundle.root"),
    )
