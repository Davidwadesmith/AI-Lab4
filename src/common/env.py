from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


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
        value = _strip_quotes(value.strip())
        values[key] = value
        if update_os:
            os.environ[key] = value
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
