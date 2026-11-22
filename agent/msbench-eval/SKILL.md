---
name: msbench-eval
description: Evaluate a text-to-video system on MSBench (multi-shot scripts) with this repository - set up environments, check inputs, run each stage, resume after failures and read the scores.
---

# Evaluate a system on MSBench

Work from the repository root. Read `README.md` (Environment, Download, VLM Judge, Quick Start) first.

## 1. Check the inputs

- Videos: `videos/<model>/<sample_id>.mp4`, keys from `data/benchmark.json`. List keys without a video and tell the
  user; a system that cannot generate 15 s videos is evaluated on the 250 base scripts only.
- Environments `msbench`, `msbench-qalign`, `msbench-deqa` exist; `models/` and `third_party/` follow their READMEs.
- Judge: either `DASHSCOPE_API_KEY` in `.env` (paid API) or a user-provided OpenAI-compatible URL. **Before any judge
  stage, tell the user the backend, the request count (`--dry-run` prints it) and that API calls cost money, and wait
  for approval.** Never deploy a model server unless asked.

## 2. Run

Prefer `bash scripts/evaluate.sh <model> videos/<model> [judge options]`. Stage by stage, keep this order:
`prepare` → `transitions omnishotcut` → `transitions transvlm` → `transitions type` → `transitions plan` →
`shots` (qalign in msbench-qalign, deqa in msbench-deqa, devil, filmops) → `face detect` → `face som` →
`face score` → `judge --metric pp|csec|csbc|vqa` → `aggregate`.

Long stages belong in `tmux`/`nohup`. Every stage resumes: on failure, read the error, fix the cause, rerun the same
command. Use `--keys <id>` to debug one video and `--shard k/n` to spread GPU stages over several GPUs.

## 3. Report

`aggregate` prints the model-level table and lists videos with missing sub-metrics (`MISSING ...`): rerun the
corresponding stage for them before reporting. Report the base-set (4 s + 8 s) row next to the leaderboard in
`README.md`, and the 15 s row separately.
