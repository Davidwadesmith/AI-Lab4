from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from src.common.io import read_jsonl, write_jsonl


def convert_xhs_records(records: Iterable[Mapping[str, Any]], *, max_records: int | None = None) -> Iterator[dict[str, object]]:
    written = 0
    for record in records:
        if record.get("repo_name") != "xhs/xhs":
            continue
        instruction = str(record.get("instruction", "")).strip()
        output = str(record.get("output", "")).strip()
        if not instruction or not output:
            continue
        if max_records is not None and written >= max_records:
            break
        written += 1
        yield {
            "meta_prompt": [],
            "data": [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}],
        }


def convert_jsonl_file(input_path: str | Path, output_path: str | Path, *, max_records: int | None = None) -> Path:
    return write_jsonl(output_path, convert_xhs_records(read_jsonl(input_path), max_records=max_records))


def load_dataset_records(dataset_path: str, *, split: str = "train", streaming: bool = False):
    from datasets import load_dataset

    dataset = load_dataset(dataset_path, split=split if streaming else None, streaming=streaming)
    if streaming:
        return dataset
    if isinstance(dataset, dict):
        return dataset[split]
    return dataset


def convert_dataset_path(
    dataset_path: str,
    output_path: str | Path,
    *,
    split: str = "train",
    streaming: bool = False,
    max_records: int | None = None,
) -> Path:
    records = load_dataset_records(dataset_path, split=split, streaming=streaming)
    return write_jsonl(output_path, convert_xhs_records(records, max_records=max_records))


def should_convert_mcore(path: str | Path, *, force: bool) -> bool:
    return force or not Path(path).exists()
