from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Mapping

from src.common.env import load_env_file, merged_env


@dataclass(frozen=True)
class BootstrapPlan:
    commands: list[str]


def _enabled(env: Mapping[str, str], key: str, default: str = "0") -> bool:
    return env.get(key, default).strip().lower() in {"1", "true", "yes", "on"}


def _quoted(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def _path_exists_guard(path: str, command: str) -> str:
    return f"if [[ ! -e {_quoted(path)} ]]; then {command}; fi"


def build_bootstrap_plan(env: Mapping[str, str], *, experiment: str) -> BootstrapPlan:
    if not _enabled(env, "AUTO_SETUP_ENV", "1"):
        return BootstrapPlan([])

    commands: list[str] = []

    ascend_env = env.get("ASCEND_TOOLKIT_ENV", "").strip()
    if ascend_env:
        commands.append(f"source {ascend_env}")

    conda_env = env.get("CONDA_ENV_NAME", "").strip()
    venv_dir = env.get("VENV_DIR", ".venv").strip()
    python_bin = env.get("PYTHON_BIN", "python3.10").strip()

    if conda_env:
        commands.append(f"eval \"$(conda shell.bash hook)\" && conda activate {_quoted(conda_env)}")
    elif venv_dir:
        commands.append(f"if [[ ! -d {_quoted(venv_dir)} ]]; then {python_bin} -m venv {_quoted(venv_dir)}; fi")
        commands.append(f"source {_quoted(venv_dir.rstrip('/') + '/bin/activate')}")

    commands.append("python -m pip install --upgrade pip setuptools wheel")

    torch_install = env.get("TORCH_INSTALL_COMMAND", "").strip()
    torch_npu_install = env.get("TORCH_NPU_INSTALL_COMMAND", "").strip()
    if torch_install:
        commands.append(torch_install)
    if torch_npu_install:
        commands.append(torch_npu_install)

    model_download = env.get("MODEL_DOWNLOAD_COMMAND", "").strip()
    if model_download:
        model_path = env.get("MODEL_PATH" if experiment == "exp2_1" else "HF_MODEL_PATH", "").strip()
        if model_path:
            commands.append(_path_exists_guard(model_path, model_download))

    data_download = env.get("DATA_DOWNLOAD_COMMAND", "").strip()
    if data_download and experiment == "exp2_2":
        source_jsonl = env.get("SOURCE_JSONL", "").strip()
        source_dataset_path = env.get("SOURCE_DATASET_PATH", "").strip()
        guard_path = source_jsonl or source_dataset_path
        if guard_path:
            commands.append(_path_exists_guard(guard_path, data_download))
        else:
            commands.append(data_download)

    if experiment == "exp2_2" and _enabled(env, "AUTO_CLONE_MINDSPEED", "0"):
        root = env.get("MINDSPEED_LLM_ROOT", "").strip()
        repo = env.get("MINDSPEED_LLM_REPO", "").strip()
        ref = env.get("MINDSPEED_LLM_REF", "").strip()
        if root and repo:
            commands.append(f"if [[ ! -d {_quoted(root.rstrip('/') + '/.git')} ]]; then git clone {_quoted(repo)} {_quoted(root)}; fi")
            if ref:
                commands.append(f"git -C {_quoted(root)} checkout {_quoted(ref)}")

    return BootstrapPlan(commands)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print shell commands for experiment environment bootstrap.")
    parser.add_argument("--env", required=True)
    parser.add_argument("--experiment", choices=["exp2_1", "exp2_2"], required=True)
    args = parser.parse_args(argv)

    env = merged_env(load_env_file(args.env))
    plan = build_bootstrap_plan(env, experiment=args.experiment)
    for command in plan.commands:
        print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
