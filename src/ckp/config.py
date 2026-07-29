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
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

ENV_PREFIX = "CKP_"

# Selects *where* config comes from; not a config value. Kept separate so the
# unknown-key check below cannot mistake it for a typo.
RESERVED_ENV = frozenset({"CKP_CONFIG_FILE"})


class ConfigError(ValueError):
    """Raised for an unknown key, an unreadable file, or an uncoercible value."""


def _load_defaults() -> dict[str, dict[str, Any]]:
    raw = resources.files("ckp").joinpath("defaults.toml").read_bytes()
    return tomllib.loads(raw.decode("utf-8"))


def _env_name(section: str, key: str) -> str:
    return f"{ENV_PREFIX}{section.upper()}_{key.upper()}"


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
    except OSError as exc:
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

    def get(self, section: str, key: str) -> Any:
        try:
            return self.values[section][key]
        except KeyError as exc:
            raise ConfigError(f"no such config key {section}.{key}") from exc

    @property
    def bundle_root(self) -> Path:
        return Path(self.get("bundle", "root")).expanduser()

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
        path = Path(chosen).expanduser()
        _merge_file(merged, defaults, path)
        layers.append(f"file:{path}")

    if _merge_env(merged, defaults, env):
        layers.append("env")

    frozen = {section: dict(entries) for section, entries in merged.items()}
    return Config(values=frozen, layers=tuple(layers))
