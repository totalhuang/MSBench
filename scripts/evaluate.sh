#!/usr/bin/env bash
# Full MSBench evaluation of one system.
#   bash scripts/evaluate.sh <model_name> <video_dir> [extra judge options, e.g. --judge-backend openai --judge-url http://IP:PORT/v1]
# Environments: MSB_ENV_MAIN / MSB_ENV_QALIGN / MSB_ENV_DEQA are shell commands that activate each environment
# (default: conda activate msbench / msbench-qalign / msbench-deqa). Every stage is resumable: rerun to continue.
set -euo pipefail
NAME=$1; VIDEOS=$2; shift 2; JUDGE=("$@")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
RES="results/$NAME"
MAIN="${MSB_ENV_MAIN:-conda activate msbench}"; QALIGN="${MSB_ENV_QALIGN:-conda activate msbench-qalign}"; DEQA="${MSB_ENV_DEQA:-conda activate msbench-deqa}"
eval "$(conda shell.bash hook 2>/dev/null || true)"
in_env() { local env="$1"; shift; ( eval "$env"; "$@" ); }
in_env "$MAIN" python -m msbench prepare --videos "$VIDEOS" --results "$RES"
in_env "$MAIN" python -m msbench transitions omnishotcut --results "$RES"
in_env "$MAIN" python -m msbench transitions transvlm --results "$RES"
in_env "$MAIN" python -m msbench transitions type --results "$RES" "${JUDGE[@]}"     # VLM judge
in_env "$MAIN" python -m msbench transitions plan --results "$RES"
in_env "$QALIGN" python -m msbench shots --method qalign --results "$RES"
in_env "$DEQA" python -m msbench shots --method deqa --results "$RES"
in_env "$MAIN" python -m msbench shots --method devil --results "$RES"
in_env "$MAIN" python -m msbench shots --method filmops --results "$RES"
in_env "$MAIN" python -m msbench face detect --results "$RES"
in_env "$MAIN" python -m msbench face som --results "$RES" "${JUDGE[@]}"              # VLM judge
in_env "$MAIN" python -m msbench face score --results "$RES"
for m in pp csec csbc vqa; do
  in_env "$MAIN" python -m msbench judge --metric "$m" --results "$RES" "${JUDGE[@]}"   # VLM judge
done
in_env "$MAIN" python -m msbench aggregate --results "$RES"
