from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _expand_env_refs(value: str, values: Mapping[str, str]) -> str:
    expanded = value
    for key, replacement in {**os.environ, **values}.items():
        expanded = expanded.replace("${" + key + ":-}", replacement)
        expanded = expanded.replace("${" + key + "}", replacement)
        expanded = expanded.replace("$" + key, replacement)
    return expanded


def _merge_path_value(key: str, value: str) -> str:
    if key not in {"PATH", "PYTHONPATH"}:
        return value
    existing = os.environ.get(key, "")
    if not existing:
        return value
    parts = [part for part in value.split(":") if part]
    for part in existing.split(":"):
        if part and part not in parts:
            parts.append(part)
    return ":".join(parts)


def load_env_file(path: str | Path, *, update_os: bool = False) -> dict[str, str]:
    env_path = Path(path)
    values: dict[str, str] = {}
    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _expand_env_refs(_strip_quotes(value.strip()), values)
        values[key] = value
        if update_os:
            os.environ[key] = _merge_path_value(key, value)
    return values


def merged_env(file_values: Mapping[str, str]) -> dict[str, str]:
    merged = dict(os.environ)
    merged.update(file_values)
    return merged


def get_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    value = env.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_list(env: Mapping[str, str], key: str, default: str = "") -> list[str]:
    raw = env.get(key, default)
    return [part.strip() for part in raw.split(",") if part.strip()]
