# OpenPangu Experiment Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a GitHub-ready repository with two one-click scripts for the openPangu Prompt Engineering and SFT notebooks.

**Architecture:** Bash entrypoints load `.env` files and call focused Python modules. Python modules own deterministic data shaping, metrics, command generation, log parsing, and reports; server-only model execution is guarded by smoke/full modes.

**Tech Stack:** Bash, Python 3.10, standard library, optional `torch`, `transformers`, `datasets`, `jieba`, `pandas`, `matplotlib`, and MindSpeed-LLM on Ascend servers.

---

### Task 1: Test Core Utility Behavior

**Files:**
- Create: `tests/test_common.py`
- Create: `tests/test_exp2_1.py`
- Create: `tests/test_exp2_2.py`

- [ ] Write tests for `.env` loading, JSONL IO, thought-chain removal, n-gram repetition, keyword coverage, MindSpeed JSONL conversion, and training log parsing.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Confirm the tests fail because the modules do not exist yet.

### Task 2: Implement Shared Utilities

**Files:**
- Create: `src/common/env.py`
- Create: `src/common/io.py`
- Create: `src/common/report.py`
- Create: `src/common/npu_check.py`

- [ ] Add `.env` parsing with quoted values and comments.
- [ ] Add JSONL read/write helpers.
- [ ] Add Markdown report writing helpers.
- [ ] Add NPU environment reporting that does not fail on local non-NPU machines.
- [ ] Run `python -m unittest discover -s tests -v`.

### Task 3: Implement Experiment 2-1 Modules

**Files:**
- Create: `src/exp2_1_prompt/data.py`
- Create: `src/exp2_1_prompt/prompts.py`
- Create: `src/exp2_1_prompt/metrics.py`
- Create: `src/exp2_1_prompt/infer.py`
- Create: `src/exp2_1_prompt/judge.py`
- Create: `src/exp2_1_prompt/main.py`

- [ ] Add notebook-compatible movie-review inputs.
- [ ] Add baseline and improved prompt variants.
- [ ] Add generation wrappers with smoke-mode fake outputs.
- [ ] Add automatic metrics and manual review template generation.
- [ ] Add optional LLM-as-Judge hook.
- [ ] Add Markdown report generation.
- [ ] Run unit tests and a smoke command.

### Task 4: Implement Experiment 2-2 Modules

**Files:**
- Create: `src/exp2_2_sft/prepare_data.py`
- Create: `src/exp2_2_sft/preprocess.py`
- Create: `src/exp2_2_sft/convert_ckpt.py`
- Create: `src/exp2_2_sft/train.py`
- Create: `src/exp2_2_sft/parse_logs.py`
- Create: `src/exp2_2_sft/infer_eval.py`
- Create: `src/exp2_2_sft/main.py`

- [ ] Add dataset conversion to MindSpeed instruction JSONL.
- [ ] Add command builders for preprocess, HF to mcore conversion, training, and mcore to HF conversion.
- [ ] Add isolated run naming for learning-rate and global-batch experiments.
- [ ] Add log parsing and report generation.
- [ ] Add smoke mode that writes commands without running training.
- [ ] Run unit tests and a smoke command.

### Task 5: Add Scripts and Configs

**Files:**
- Create: `scripts/run_exp2_1_prompt.sh`
- Create: `scripts/run_exp2_2_sft.sh`
- Create: `configs/exp2_1.env.example`
- Create: `configs/exp2_2.env.example`
- Create: `requirements/exp2_1.txt`
- Create: `requirements/exp2_2.txt`
- Create: `.gitignore`
- Create: `README.md`

- [ ] Add strict Bash entrypoints that accept a config path.
- [ ] Add environment templates with path variables and safe defaults.
- [ ] Add usage instructions for local smoke checks and server full runs.
- [ ] Run syntax and unit validation.
