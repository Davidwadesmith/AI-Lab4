#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-configs/exp2_1.env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Config file not found: ${ENV_FILE}" >&2
  echo "Create one from configs/exp2_1.env.example" >&2
  exit 2
fi

set -a
source "${ENV_FILE}"
set +a

echo "[exp2_1] run mode: ${RUN_MODE:-smoke}"
python --version

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install -r requirements/exp2_1.txt
fi

if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info || true
else
  echo "[exp2_1] npu-smi not found; continuing for smoke/local validation"
fi

python -m src.exp2_1_prompt.main --env "${ENV_FILE}"
