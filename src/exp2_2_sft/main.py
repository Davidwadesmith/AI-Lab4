from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from src.common.env import get_bool, get_list, load_env_file, merged_env
from src.common.io import ensure_dir, write_jsonl, write_text
from src.common.report import markdown_table, write_report
from src.exp2_2_sft.convert_ckpt import build_hf_to_mcore_command, build_mcore_to_hf_command
from src.exp2_2_sft.infer_eval import build_copywriting_eval_set, evaluate_copywriting
from src.exp2_2_sft.parse_logs import parse_training_log
from src.exp2_2_sft.prepare_data import convert_dataset_path, convert_jsonl_file, should_convert_mcore
from src.exp2_2_sft.preprocess import build_preprocess_command, cache_exists
from src.exp2_2_sft.train import build_run_matrix, build_train_command


def _run(command: str, *, dry_run: bool) -> None:
    print(command)
    if not dry_run:
        subprocess.run(command, shell=True, check=True)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _join_config_path(root: str, *parts: str) -> str:
    if root.startswith("/"):
        return "/".join([root.rstrip("/"), *[part.strip("/") for part in parts]])
    return str(Path(root, *parts))


def _ensure_preprocess_output_dir(output_prefix: str) -> Path:
    return ensure_dir(Path(output_prefix).parent)


def _get_optional_int(env: dict[str, str], key: str) -> int | None:
    value = env.get(key, "").strip()
    return int(value) if value else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run openPangu MindSpeed SFT experiment.")
    parser.add_argument("--env", required=True, help="Path to exp2_2 env file.")
    args = parser.parse_args(argv)

    env = merged_env(load_env_file(args.env, update_os=True))
    output_root = Path(env.get("OUTPUT_ROOT", "outputs/exp2_2"))
    run_mode = env.get("RUN_MODE", "smoke")
    dry_run = run_mode == "smoke" or get_bool(env, "DRY_RUN", False)
    ensure_dir(output_root)

    source_jsonl = env.get("SOURCE_JSONL", "")
    source_dataset_path = env.get("SOURCE_DATASET_PATH", "")
    max_train_records = _get_optional_int(env, "MAX_TRAIN_RECORDS")
    train_jsonl = output_root / "data" / "train_xhs_dataset.jsonl"
    if run_mode == "smoke":
        sample = [
            {
                "meta_prompt": [],
                "data": [
                    {"role": "user", "content": "请为花果香香水写一段小红书文案"},
                    {"role": "assistant", "content": "清新的葡萄柚和梨一上身就很明亮，茉莉与牡丹让整体更温柔。"},
                ],
            }
        ]
        write_jsonl(train_jsonl, sample)
    elif source_jsonl and Path(source_jsonl).exists():
        convert_jsonl_file(source_jsonl, train_jsonl, max_records=max_train_records)
    elif source_dataset_path:
        convert_dataset_path(
            source_dataset_path,
            train_jsonl,
            split=env.get("SOURCE_DATASET_SPLIT", "train"),
            streaming=get_bool(env, "SOURCE_DATASET_STREAMING", False),
            max_records=max_train_records,
        )
    else:
        raise ValueError("full mode requires SOURCE_JSONL or SOURCE_DATASET_PATH")

    code_root = env.get("MINDSPEED_LLM_ROOT", "/home/ma-user/work/MindSpeed-LLM")
    seq_length = int(env.get("SEQ_LENGTH", "8192"))
    cache_prefix = env.get("DATA_CACHE_PREFIX", str(output_root / "cache" / "sft"))
    _ensure_preprocess_output_dir(cache_prefix)
    commands = []

    preprocess_command = build_preprocess_command(
        code_root=code_root,
        input_dir=str(train_jsonl.parent.resolve()),
        tokenizer_path=env.get("HF_MODEL_PATH", ""),
        output_prefix=cache_prefix,
        seq_length=seq_length,
        workers=int(env.get("PREPROCESS_WORKERS", "4")),
    )
    commands.append(("preprocess", preprocess_command))
    if not cache_exists(cache_prefix) or get_bool(env, "FORCE_PREPROCESS", False):
        _run(preprocess_command, dry_run=dry_run)

    hf_to_mcore = build_hf_to_mcore_command(
        code_root=code_root,
        load_hf_dir=env.get("HF_MODEL_PATH", ""),
        save_mcore_dir=env.get("MCORE_MODEL_PATH", str(output_root / "mcore_base")),
        tokenizer_model=env.get("TOKENIZER_MODEL", env.get("HF_MODEL_PATH", "")),
        tensor_parallel_size=int(env.get("NPUS_PER_NODE", "4")),
    )
    commands.append(("hf_to_mcore", hf_to_mcore))
    if should_convert_mcore(env.get("MCORE_MODEL_PATH", str(output_root / "mcore_base")), force=get_bool(env, "FORCE_CONVERT", False)):
        _run(hf_to_mcore, dry_run=dry_run)

    learning_rates = get_list(env, "LEARNING_RATES", "1e-5,5e-6,6e-7")
    global_batch_sizes = [int(value) for value in get_list(env, "GLOBAL_BATCH_SIZES", "1,2,4")]
    if run_mode == "smoke":
        learning_rates = learning_rates[:1]
        global_batch_sizes = global_batch_sizes[:1]
    runs = build_run_matrix(run_mode, learning_rates=learning_rates, global_batch_sizes=global_batch_sizes)

    all_metrics = []
    for index, run in enumerate(runs):
        ckpt_save_dir = _join_config_path(env.get("OUTPUT_ROOT", "outputs/exp2_2"), "checkpoints", str(run["name"]))
        log_path = _join_config_path(env.get("OUTPUT_ROOT", "outputs/exp2_2"), "logs", f"{run['name']}.log")
        train_iters = int(env.get("TRAIN_ITERS", "3200"))
        train_command = build_train_command(
            code_root=code_root,
            npu_devices=env.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3"),
            npus_per_node=int(env.get("NPUS_PER_NODE", "4")),
            master_port=int(env.get("MASTER_PORT", str(6000 + index))),
            data_path=cache_prefix,
            tokenizer_model=env.get("TOKENIZER_MODEL", env.get("HF_MODEL_PATH", "")),
            ckpt_load_dir=env.get("MCORE_MODEL_PATH", str(output_root / "mcore_base")),
            ckpt_save_dir=ckpt_save_dir,
            log_path=log_path,
            learning_rate=str(run["learning_rate"]),
            global_batch_size=int(run["global_batch_size"]),
            train_iters=train_iters,
            seq_length=seq_length,
            save_interval=int(env.get("SAVE_INTERVAL", str(train_iters))),
            lr_warmup_iters=int(env.get("LR_WARMUP_ITERS", "200")),
            no_save_optim=get_bool(env, "NO_SAVE_OPTIM", False),
            accumulate_allreduce_grads_in_fp32=get_bool(env, "ACCUMULATE_ALLREDUCE_GRADS_IN_FP32", False),
            reuse_fp32_param=get_bool(env, "REUSE_FP32_PARAM", True),
        )
        commands.append((f"train_{run['name']}", train_command))
        _run(train_command, dry_run=dry_run)
        log_file = Path(log_path)
        if log_file.exists():
            for row in parse_training_log(log_file):
                row["run_name"] = run["name"]
                all_metrics.append(row)

    metrics_path = output_root / "metrics" / "train_metrics.csv"
    _write_csv(metrics_path, all_metrics)

    mcore_to_hf = build_mcore_to_hf_command(
        code_root=code_root,
        load_mcore_dir=_join_config_path(env.get("OUTPUT_ROOT", "outputs/exp2_2"), "checkpoints", str(runs[0]["name"])),
        save_hf_dir=env.get("SFT_HF_OUTPUT_PATH", str(output_root / "hf_export")),
    )
    commands.append(("mcore_to_hf", mcore_to_hf))
    if get_bool(env, "EXPORT_HF", False):
        _run(mcore_to_hf, dry_run=dry_run)

    command_text = "\n\n".join(f"## {name}\n{command}" for name, command in commands)
    commands_path = write_text(output_root / "commands.md", command_text)

    eval_rows = build_copywriting_eval_set()
    for row in eval_rows:
        row["generation"] = f"{row['platform']}文案示例：{row['product']}"
    eval_path = write_jsonl(output_root / "inference" / "copywriting_eval.jsonl", eval_rows)
    eval_metrics_path = output_root / "metrics" / "copywriting_metrics.csv"
    _write_csv(eval_metrics_path, evaluate_copywriting(eval_rows))

    report_path = write_report(
        output_root / "report_exp2_2.md",
        "Experiment 2-2 SFT Report",
        {
            "Run Settings": f"- run_mode: `{run_mode}`\n- code_root: `{code_root}`\n- seq_length: `{seq_length}`",
            "Artifacts": (
                f"- train_jsonl: `{train_jsonl}`\n"
                f"- commands: `{commands_path}`\n"
                f"- train_metrics: `{metrics_path}`\n"
                f"- eval: `{eval_path}`\n"
                f"- eval_metrics: `{eval_metrics_path}`"
            ),
            "Runs": markdown_table(runs, ["name", "learning_rate", "global_batch_size"]),
        },
    )
    print(f"wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
