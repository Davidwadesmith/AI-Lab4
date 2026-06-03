from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class NpuStatus:
    available: bool
    command: str
    output: str


def inspect_npu() -> NpuStatus:
    executable = shutil.which("npu-smi")
    if not executable:
        return NpuStatus(False, "npu-smi info", "npu-smi not found")
    command = [executable, "info"]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    output = completed.stdout.strip() or completed.stderr.strip()
    return NpuStatus(completed.returncode == 0, " ".join(command), output)
