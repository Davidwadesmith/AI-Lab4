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

if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info || true
else
  echo "[exp2_2] npu-smi not found; continuing for smoke/local validation"
fi

python -m src.exp2_2_sft.main --env "${ENV_FILE}"
