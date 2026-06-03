from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from src.common.io import read_jsonl, write_jsonl


def convert_xhs_records(records: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, object]]:
    for record in records:
        if record.get("repo_name") != "xhs/xhs":
            continue
        instruction = str(record.get("instruction", "")).strip()
        output = str(record.get("output", "")).strip()
        if not instruction or not output:
            continue
        yield {
            "meta_prompt": [],
            "data": [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}],
        }


def convert_jsonl_file(input_path: str | Path, output_path: str | Path) -> Path:
    return write_jsonl(output_path, convert_xhs_records(read_jsonl(input_path)))


def load_dataset_records(dataset_path: str, *, split: str = "train"):
    from datasets import load_dataset

    dataset = load_dataset(dataset_path)
    if isinstance(dataset, dict):
        return dataset[split]
    return dataset


def convert_dataset_path(dataset_path: str, output_path: str | Path, *, split: str = "train") -> Path:
    return write_jsonl(output_path, convert_xhs_records(load_dataset_records(dataset_path, split=split)))


def should_convert_mcore(path: str | Path, *, force: bool) -> bool:
    return force or not Path(path).exists()
