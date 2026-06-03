from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Any


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    jsonl_path = Path(path)
    ensure_dir(jsonl_path.parent)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return jsonl_path


def write_text(path: str | Path, content: str) -> Path:
    text_path = Path(path)
    ensure_dir(text_path.parent)
    text_path.write_text(content, encoding="utf-8")
    return text_path
