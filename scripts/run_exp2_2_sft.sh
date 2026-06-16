#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-configs/exp2_2.env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Config file not found: ${ENV_FILE}" >&2
  echo "Create one from configs/exp2_2.env.example" >&2
  exit 2
fi

set -a
source "${ENV_FILE}"
set +a

echo "[exp2_2] run mode: ${RUN_MODE:-smoke}"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-python3.10}"
if ! command -v "${BOOTSTRAP_PYTHON}" >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="python3"
fi

BOOTSTRAP_COMMANDS="$("${BOOTSTRAP_PYTHON}" -m src.common.bootstrap --env "${ENV_FILE}" --experiment exp2_2)"
if [[ -n "${BOOTSTRAP_COMMANDS}" ]]; then
  echo "[exp2_2] bootstrapping environment"
  eval "${BOOTSTRAP_COMMANDS}"
fi

python --version

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install -r requirements/exp2_2.txt
fi

if [[ "${INSTALL_MINDSPEED:-1}" == "1" && -d "${MINDSPEED_LLM_ROOT:-}/MindSpeed" ]]; then
  python -m pip install -e "${MINDSPEED_LLM_ROOT}/MindSpeed"
fi

NINJA_TARGET_DIR="${NINJA_TARGET_DIR:-${USER_ROOT:-${REPO_ROOT}}/python_pkgs}"
if [[ -d "${NINJA_TARGET_DIR}" ]]; then
  export PYTHONPATH="${NINJA_TARGET_DIR}:${PYTHONPATH:-}"
fi

_add_ninja_to_path() {
  local ninja_bin_dir
  ninja_bin_dir="$(python - <<'PY'
try:
    import ninja
except Exception:
    print("")
else:
    print(getattr(ninja, "BIN_DIR", ""))
PY
)"
  if [[ -n "${ninja_bin_dir}" ]]; then
    export PATH="${ninja_bin_dir}:${PATH}"
  fi
}

_add_ninja_to_path
if ! command -v ninja >/dev/null 2>&1; then
  if [[ -n "${NINJA_INSTALL_COMMAND:-}" ]]; then
    eval "${NINJA_INSTALL_COMMAND}"
  else
    python -m pip install --target "${NINJA_TARGET_DIR}" ninja
    export PYTHONPATH="${NINJA_TARGET_DIR}:${PYTHONPATH:-}"
  fi
  _add_ninja_to_path
fi
if ! command -v ninja >/dev/null 2>&1; then
  echo "[exp2_2] ninja is required by MindSpeed C++ extension builds but was not found" >&2
  exit 3
fi

if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info || true
else
  echo "[exp2_2] npu-smi not found; continuing for smoke/local validation"
fi

python -m src.exp2_2_sft.main --env "${ENV_FILE}"
