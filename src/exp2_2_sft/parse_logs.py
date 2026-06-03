from __future__ import annotations

import re
from pathlib import Path


LOSS_RE = re.compile(r"lm loss:\s*([0-9.eE+-]+)")
STEP_RE = re.compile(r"(?:iteration|step)\s+([0-9]+)", re.IGNORECASE)
TOKENS_RE = re.compile(r"tokens/s:\s*([0-9.eE+-]+)")
SAMPLES_RE = re.compile(r"samples/s:\s*([0-9.eE+-]+)")


def parse_training_log(path: str | Path) -> list[dict[str, float | int]]:
    rows = []
    fallback_step = 1
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        loss_match = LOSS_RE.search(line)
        if not loss_match:
            continue
        step_match = STEP_RE.search(line)
        row: dict[str, float | int] = {
            "step": int(step_match.group(1)) if step_match else fallback_step,
            "lm_loss": float(loss_match.group(1)),
        }
        tokens_match = TOKENS_RE.search(line)
        samples_match = SAMPLES_RE.search(line)
        if tokens_match:
            row["tokens_per_second"] = float(tokens_match.group(1))
        if samples_match:
            row["samples_per_second"] = float(samples_match.group(1))
        rows.append(row)
        fallback_step += 1
    return rows
