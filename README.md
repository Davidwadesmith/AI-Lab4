# AI-Lab4 OpenPangu Experiments

This repository keeps only the runnable parts needed for the two notebook labs:

- Experiment 2-1: Prompt Engineering for controlled Chinese movie-review generation.
- Experiment 2-2: openPangu-7B SFT for copywriting generation and simple accuracy metrics.

The notebook after-class homework sections are intentionally not implemented.

## Server Assumptions

The scripts can prepare user-space Python packages, clone MindSpeed-LLM, download the openPangu checkpoint, stream the dataset, run preprocessing, train, export, and write reports.

They cannot install system-level Ascend driver, firmware, or CANN without administrator privileges. Before running full mode on the server, check:

```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 - <<'PY'
import torch
import torch_npu
print(torch.__version__)
print("torch_npu ok")
PY
```

## Fresh Server Setup

Clone the public repo:

```bash
git clone https://github.com/Davidwadesmith/AI-Lab4.git
cd AI-Lab4
```

All large resources and outputs default to:

```bash
/home/ma-user/work/openpangu_lab
```

Change only `WORK_ROOT` in the copied env files if your server uses a different writable disk.

## Experiment 2-1

Prepare config and run a smoke check:

```bash
cp configs/exp2_1.env.example configs/exp2_1.env
bash scripts/run_exp2_1_prompt.sh configs/exp2_1.env
```

For real inference, set this in `configs/exp2_1.env`:

```bash
RUN_MODE=full
```

Then rerun:

```bash
bash scripts/run_exp2_1_prompt.sh configs/exp2_1.env
```

Outputs:

- `${WORK_ROOT}/outputs/exp2_1/generations/prompt_generations.jsonl`
- `${WORK_ROOT}/outputs/exp2_1/metrics/prompt_metrics.csv`
- `${WORK_ROOT}/outputs/exp2_1/manual_review/manual_review_template.jsonl`
- `${WORK_ROOT}/outputs/exp2_1/report_exp2_1.md`

## Experiment 2-2

Prepare config and run a smoke check:

```bash
cp configs/exp2_2.env.example configs/exp2_2.env
bash scripts/run_exp2_2_sft.sh configs/exp2_2.env
```

For the homework run, set this in `configs/exp2_2.env`:

```bash
RUN_MODE=full
```

Then run in the foreground:

```bash
bash scripts/run_exp2_2_sft.sh configs/exp2_2.env
```

The default full run is intentionally small:

- `TRAIN_ITERS=100`
- `SAVE_INTERVAL=100`
- `NO_SAVE_OPTIM=1`
- `LEARNING_RATES=5e-5`
- `GLOBAL_BATCH_SIZES=4`
- `MAX_TRAIN_RECORDS=2000`

Outputs:

- `${WORK_ROOT}/outputs/exp2_2/data/train_xhs_dataset.jsonl`
- `${WORK_ROOT}/outputs/exp2_2/commands.md`
- `${WORK_ROOT}/outputs/exp2_2/logs/`
- `${WORK_ROOT}/outputs/exp2_2/checkpoints/`
- `${WORK_ROOT}/outputs/exp2_2/metrics/train_metrics.csv`
- `${WORK_ROOT}/outputs/exp2_2/metrics/copywriting_metrics.csv`
- `${WORK_ROOT}/outputs/exp2_2/report_exp2_2.md`

## What Gets Downloaded

The default configs download:

- Model: `FreedomIntelligence/openPangu-Embedded-7B-V1.1`
- Dataset: `Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT`, streamed and filtered to `repo_name == "xhs/xhs"`
- Code: `https://github.com/Ascend/MindSpeed-LLM.git`

If your course image requires a specific MindSpeed-LLM commit or branch, set `MINDSPEED_LLM_REF` in `configs/exp2_2.env`.

## Common Knobs

- `WORK_ROOT`: single writable root for models, code, cache, checkpoints, and reports.
- `VENV_SYSTEM_SITE_PACKAGES=1`: lets the venv reuse server-provided `torch` and `torch_npu`.
- `TORCH_INSTALL_COMMAND` / `TORCH_NPU_INSTALL_COMMAND`: fill only when your server image does not already provide compatible NPU packages.
- `FORCE_PREPROCESS=1`: rebuild tokenized data cache.
- `FORCE_CONVERT=1`: rerun HF to mcore conversion.
- `EXPORT_HF=1`: convert the trained checkpoint back to HF format for inference artifacts.

## Local Validation

The local machine does not need a GPU/NPU:

```bash
python -m unittest discover -s tests -v
```

Do not commit downloaded models, raw datasets, MindSpeed code, checkpoints, logs, or generated outputs.
