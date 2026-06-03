# OpenPangu Experiment Automation Design

## Goal

Build a GitHub-ready experiment repository for two Ascend 910B openPangu labs. The local machine has no GPU/NPU, so local work must focus on scripts, validation, data shaping, metrics, and report generation. Full model execution happens after cloning the repository onto the target servers.

## Experiment 2-1

Experiment 2-1 runs openPangu-7B inference for Prompt Engineering and data synthesis on a single Ascend 910B server. The one-click entrypoint is `scripts/run_exp2_1_prompt.sh`.

The workflow is:

1. Check Python, package versions, NPU visibility, and model path.
2. Install light Python dependencies.
3. Build the five movie-review structured inputs from the notebook.
4. Load openPangu-7B from a configurable Hugging Face model directory.
5. Run baseline and improved prompts.
6. Remove slow-thinking content between `[unused16]` and `[unused17]`.
7. Compute automatic metrics: length compliance, n-gram repetition, keyword coverage, and rough sentiment consistency.
8. Optionally run LLM-as-Judge when judge API environment variables are configured.
9. Write JSONL, CSV, manual-review templates, and a Markdown report.

## Experiment 2-2

Experiment 2-2 runs MindSpeed-LLM SFT and evaluation on a separate four-card Ascend 910B server. The one-click entrypoint is `scripts/run_exp2_2_sft.sh`.

The workflow is:

1. Check Python, torchrun, MindSpeed-LLM, NPU visibility, model paths, and data paths.
2. Install MindSpeed in editable mode when available.
3. Convert the source dataset into MindSpeed instruction JSONL.
4. Run MindSpeed `preprocess_data.py` to create packed `.bin/.idx` cache.
5. Convert openPangu-7B HF weights to mcore if the mcore output is missing or `FORCE_CONVERT=1`.
6. Launch one or more SFT runs with isolated log and checkpoint directories.
7. Parse training logs for loss and optional throughput fields.
8. Convert the selected mcore checkpoint back to HF format.
9. Run generation on a fixed copywriting evaluation set.
10. Compute automatic copywriting metrics and write an experiment report.

## Repository Structure

```text
configs/
  exp2_1.env.example
  exp2_2.env.example
requirements/
  exp2_1.txt
  exp2_2.txt
scripts/
  run_exp2_1_prompt.sh
  run_exp2_2_sft.sh
src/
  common/
  exp2_1_prompt/
  exp2_2_sft/
tests/
outputs/
```

## Operational Rules

Large model files, raw datasets, MindSpeed source trees, checkpoints, and generated outputs are not committed. They are passed through environment variables or generated under `outputs/`.

All long-running stages are idempotent. Existing output files are reused unless the corresponding force flag is set.

Smoke mode must be available for both experiments. It validates environment, data preparation, and report paths without requiring a full training or inference run.

## Validation

Local validation uses Python unit tests for path parsing, metrics, dataset conversion, command construction, and log parsing. Server validation uses the one-click scripts in `RUN_MODE=smoke` before full execution.
