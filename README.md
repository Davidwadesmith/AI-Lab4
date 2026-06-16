# AI-Lab4 OpenPangu Experiments

This repository automates two openPangu labs from the notebooks:

- Experiment 2-1: Prompt Engineering and movie-review data synthesis with openPangu-7B.
- Experiment 2-2: MindSpeed-LLM SFT and copywriting evaluation with openPangu-7B.

The local machine does not need a GPU/NPU. Local validation uses smoke mode and unit tests. Full inference and training run after cloning this repository onto Ascend 910B servers.

The one-click scripts include environment bootstrap: they can create a Python virtual environment, load Ascend toolkit variables, install Python packages, run optional torch/torch-npu install commands, download model/data through user-provided commands, and clone MindSpeed-LLM. System-level Ascend driver, firmware, and CANN installation still need to exist in the server image or be installed by an administrator.

## Local Validation

```bash
python -m unittest discover -s tests -v
python -m src.exp2_1_prompt.main --env configs/exp2_1.env.example
python -m src.exp2_2_sft.main --env configs/exp2_2.env.example
```

## Experiment 2-1 Server Run

Use this on the single-card inference server.

```bash
git clone <your-repo-url>
cd AI-Lab4
cp configs/exp2_1.env.example configs/exp2_1.env
```

Edit `configs/exp2_1.env`:

- `RUN_MODE=full`
- `MODEL_PATH=/path/to/openPangu-Embedded-7B-V1.1`
- Optional judge API variables.

For a first server run, keep `RUN_MODE=smoke` and configure bootstrap:

```bash
AUTO_SETUP_ENV=1
PYTHON_BIN=python3.10
VENV_DIR=.venv
ASCEND_TOOLKIT_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
SETUPTOOLS_INSTALL_COMMAND=

# Fill these only if your image does not already contain compatible packages.
TORCH_INSTALL_COMMAND="pip install torch==2.1.0"
TORCH_NPU_INSTALL_COMMAND="pip install torch-npu==2.1.0"

# Optional. Use any command that creates MODEL_PATH.
MODEL_PATH=/home/ma-user/work/models/openPangu-Embedded-7B-V1.1
MODEL_DOWNLOAD_COMMAND="git clone <model-repo-url> /home/ma-user/work/models/openPangu-Embedded-7B-V1.1"
```

Values containing spaces must be quoted because the Bash script sources the `.env` file.

Run:

```bash
bash scripts/run_exp2_1_prompt.sh configs/exp2_1.env
```

Main outputs:

- `outputs/exp2_1/generations/prompt_generations.jsonl`
- `outputs/exp2_1/metrics/prompt_metrics.csv`
- `outputs/exp2_1/manual_review/manual_review_template.jsonl`
- `outputs/exp2_1/report_exp2_1.md`

## Experiment 2-2 Server Run

Use this on the four-card MindSpeed-LLM SFT server.

```bash
git clone <your-repo-url>
cd AI-Lab4
cp configs/exp2_2.env.example configs/exp2_2.env
```

Edit `configs/exp2_2.env`:

- `RUN_MODE=smoke` for the first environment check, then `RUN_MODE=full`.
- `MINDSPEED_LLM_ROOT=/path/to/MindSpeed-LLM`
- `SOURCE_JSONL=/path/to/source.jsonl`, or `SOURCE_DATASET_PATH=/path/to/dataset` for the notebook-style `datasets.load_dataset(path)` flow.
- `HF_MODEL_PATH=/path/to/openPangu_7B_hf`
- `TOKENIZER_MODEL=/path/to/tokenizer`
- `MCORE_MODEL_PATH=/cache/ckpts/openPangu_7B_mcore`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`

Configure bootstrap on the training server:

```bash
AUTO_SETUP_ENV=1
PYTHON_BIN=python3.10
VENV_DIR=.venv
ASCEND_TOOLKIT_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
SETUPTOOLS_INSTALL_COMMAND=

# Fill these only if your image does not already contain compatible packages.
TORCH_INSTALL_COMMAND="pip install torch==2.1.0"
TORCH_NPU_INSTALL_COMMAND="pip install torch-npu==2.1.0"

# Clone MindSpeed-LLM automatically when MINDSPEED_LLM_ROOT does not exist.
AUTO_CLONE_MINDSPEED=1
MINDSPEED_LLM_ROOT=/home/ma-user/work/MindSpeed-LLM
MINDSPEED_LLM_REPO=https://gitee.com/ascend/MindSpeed-LLM.git
MINDSPEED_LLM_REF=

# Optional. Use commands that create the configured paths.
HF_MODEL_PATH=/home/ma-user/work/models/openPangu-7B-hf
MODEL_DOWNLOAD_COMMAND="git clone <model-repo-url> /home/ma-user/work/models/openPangu-7B-hf"
SOURCE_JSONL=/home/ma-user/work/data/chinese_deepseek_r1_distill.jsonl
DATA_DOWNLOAD_COMMAND="python scripts/download_data.py"
```

Values containing spaces must be quoted because the Bash script sources the `.env` file.

Run smoke mode first:

```bash
bash scripts/run_exp2_2_sft.sh configs/exp2_2.env
```

Then set `RUN_MODE=full` and rerun. Use `FORCE_PREPROCESS=1`, `FORCE_CONVERT=1`, or `EXPORT_HF=1` only when those stages should be rerun.

Main outputs:

- `outputs/exp2_2/data/train_xhs_dataset.jsonl`
- `outputs/exp2_2/commands.md`
- `outputs/exp2_2/logs/`
- `outputs/exp2_2/checkpoints/`
- `outputs/exp2_2/metrics/train_metrics.csv`
- `outputs/exp2_2/metrics/copywriting_metrics.csv`
- `outputs/exp2_2/report_exp2_2.md`

### Shared `/src/init/Shared` Course Environment

If the server already provides the shared course resources, use the shared template instead of downloading model/data/MindSpeed again:

```bash
git clone https://github.com/Davidwadesmith/AI-Lab4.git
cd AI-Lab4
cp configs/exp2_2_shared.env.example configs/exp2_2.env
```

Edit `configs/exp2_2.env` and replace every `/src/init/09024126hcc` with your personal writable directory. Keep shared resources read-only:

```bash
MINDSPEED_LLM_ROOT=/src/init/Shared/MindSpeed-LLM
HF_MODEL_PATH=/src/init/Shared/openPangu-Embedded-7B-V1.1
TOKENIZER_MODEL=/src/init/Shared/openPangu-Embedded-7B-V1.1
SOURCE_JSONL=/src/init/Shared/data/chinese_deepseek_r1_distill.jsonl

OUTPUT_ROOT=/src/init/<your-user>/outputs/exp2_2
VENV_DIR=/src/init/<your-user>/.venv
MCORE_MODEL_PATH=/src/init/<your-user>/ckpts/openPangu_7B_mcore
SFT_HF_OUTPUT_PATH=/src/init/<your-user>/hf_export/openPangu_sft_hf
DATA_CACHE_PREFIX=/src/init/<your-user>/cache/finetune_pangu7b_xhs_seqlen8k/sft
```

The shared template uses space-saving defaults for the 500G environment:

```bash
SETUPTOOLS_INSTALL_COMMAND=python -m pip install setuptools==69.5.1
TRAIN_ITERS=100
SAVE_INTERVAL=100
LR_WARMUP_ITERS=20
NO_SAVE_OPTIM=1
LEARNING_RATES=1e-5,5e-6,6e-7
GLOBAL_BATCH_SIZES=4
```

These settings save only the final checkpoint for each run and skip optimizer-state saving. Do not set `SFT_HF_OUTPUT_PATH` or checkpoint paths under `/src/init/Shared`.

## Notes

Do not commit model weights, raw datasets, MindSpeed source trees, checkpoints, or generated outputs. They are intentionally ignored by `.gitignore`.

The scripts cannot install kernel drivers, firmware, or CANN system packages without the correct server image and privileges. Verify these first when using a fresh server:

```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python - <<'PY'
try:
    import torch
    import torch_npu
    print(torch.__version__)
    print("torch_npu ok")
except Exception as exc:
    print(exc)
    raise
PY
```
