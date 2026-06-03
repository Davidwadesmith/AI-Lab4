from __future__ import annotations

from pathlib import Path


def build_preprocess_command(
    *,
    code_root: str,
    input_dir: str,
    tokenizer_path: str,
    output_prefix: str,
    seq_length: int,
    workers: int = 4,
) -> str:
    return (
        f"cd {code_root} && python ./preprocess_data.py "
        f"--input {input_dir} "
        f"--tokenizer-name-or-path {tokenizer_path} "
        f"--output-prefix {output_prefix} "
        f"--workers {workers} "
        "--tokenizer-type PretrainedFromHF "
        "--handler-name PanguInstructionHandler "
        f"--seq-length {seq_length} "
        "--pack"
    )


def cache_exists(output_prefix: str) -> bool:
    prefix = Path(output_prefix)
    return any(prefix.parent.glob(prefix.name + "*.bin")) and any(prefix.parent.glob(prefix.name + "*.idx"))
