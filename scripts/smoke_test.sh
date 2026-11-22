#!/usr/bin/env bash
# End-to-end smoke test: full pipeline on the two example videos (videos/example), then a check against the paper run.
#   bash scripts/smoke_test.sh [judge options, e.g. --judge-backend openai --judge-url http://IP:PORT/v1]
# Uses the GPU and the VLM judge: about 14 video requests and ~100 short image requests (face identification). Passing means every stage ran and produced every sub-metric.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
bash scripts/evaluate.sh smoke_example videos/example "$@"
MAIN="${MSB_ENV_MAIN:-conda activate msbench}"
eval "$(conda shell.bash hook 2>/dev/null || true)"
( eval "$MAIN"; python tests/check_smoke.py results/smoke_example )
