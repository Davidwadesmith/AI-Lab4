from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from src.common.env import load_env_file, merged_env
from src.common.io import ensure_dir, write_jsonl
from src.common.report import markdown_table, write_report
from src.exp2_1_prompt.data import build_demo_inputs
from src.exp2_1_prompt.infer import run_prompt_variants
from src.exp2_1_prompt.judge import call_openai_compatible_judge, judge_enabled
from src.exp2_1_prompt.metrics import evaluate_generations


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run openPangu prompt engineering experiment.")
    parser.add_argument("--env", required=True, help="Path to exp2_1 env file.")
    args = parser.parse_args(argv)

    env = merged_env(load_env_file(args.env, update_os=True))
    run_mode = env.get("RUN_MODE", "smoke")
    output_root = Path(env.get("OUTPUT_ROOT", "outputs/exp2_1"))
    model_path = env.get("MODEL_PATH", "")
    items = build_demo_inputs()

    rows = run_prompt_variants(
        items,
        model_path=model_path,
        run_mode=run_mode,
        max_input_length=int(env.get("MAX_INPUT_LENGTH", "8192")),
        max_new_tokens=int(env.get("MAX_NEW_TOKENS", "2048")),
    )

    generations_path = write_jsonl(output_root / "generations" / "prompt_generations.jsonl", rows)

    metric_rows = []
    for variant_name in sorted({row["prompt_variant"] for row in rows}):
        variant_rows = [row for row in rows if row["prompt_variant"] == variant_name]
        metrics = evaluate_generations(items, [str(row["generation"]) for row in variant_rows])
        summary = {key: value for key, value in metrics.items() if key != "rows"}
        summary["prompt_variant"] = variant_name
        metric_rows.append(summary)
    metrics_path = output_root / "metrics" / "prompt_metrics.csv"
    _write_csv(metrics_path, metric_rows)

    manual_rows = [
        {
            "title": row["title"],
            "prompt_variant": row["prompt_variant"],
            "generation": row["generation"],
            "readability_1_5": "",
            "relevance_1_5": "",
            "hallucination_yes_no": "",
            "comments": "",
        }
        for row in rows
    ]
    manual_path = write_jsonl(output_root / "manual_review" / "manual_review_template.jsonl", manual_rows)

    judge_path = ""
    if judge_enabled(env):
        judge_rows = [call_openai_compatible_judge(row, env) for row in rows]
        judge_path = str(write_jsonl(output_root / "judge" / "judge_scores.jsonl", judge_rows))

    report_path = write_report(
        output_root / "report_exp2_1.md",
        "Experiment 2-1 Prompt Engineering Report",
        {
            "Run Settings": f"- run_mode: `{run_mode}`\n- model_path: `{model_path}`",
            "Artifacts": (
                f"- generations: `{generations_path}`\n"
                f"- metrics: `{metrics_path}`\n"
                f"- manual review: `{manual_path}`\n"
                f"- judge scores: `{judge_path or 'not configured'}`"
            ),
            "Metrics": markdown_table(metric_rows, ["prompt_variant", "count", "length_compliance_rate", "avg_keyword_coverage", "avg_repetition_3gram"]),
            "Prompt Sensitivity Notes": (
                "Compare each prompt variant by length compliance, keyword coverage, repetition, and judge/manual scores. "
                "The anti-hallucination prompt isolates the effect of factual constraints; the few-shot prompt isolates style examples."
            ),
        },
    )
    print(f"wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
