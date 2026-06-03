# AI-Lab4 OpenPangu Experiments

This repository automates two openPangu labs from the notebooks:

- Experiment 2-1: Prompt Engineering and movie-review data synthesis with openPangu-7B.
- Experiment 2-2: MindSpeed-LLM SFT and copywriting evaluation with openPangu-7B.

The local machine does not need a GPU/NPU. Local validation uses smoke mode and unit tests. Full inference and training run after cloning this repository onto Ascend 910B servers.

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

## Notes

Do not commit model weights, raw datasets, MindSpeed source trees, checkpoints, or generated outputs. They are intentionally ignored by `.gitignore`.
