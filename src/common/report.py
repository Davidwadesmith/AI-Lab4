from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Any

from src.common.io import write_text


def markdown_table(rows: Iterable[Mapping[str, Any]], columns: list[str]) -> str:
    rows = list(rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, separator, *body])


def write_report(path: str | Path, title: str, sections: Mapping[str, str]) -> Path:
    parts = [f"# {title}", ""]
    for heading, body in sections.items():
        parts.extend([f"## {heading}", "", body.strip(), ""])
    return write_text(path, "\n".join(parts).rstrip() + "\n")
