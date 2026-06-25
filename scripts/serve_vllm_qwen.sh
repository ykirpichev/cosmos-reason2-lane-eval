#!/usr/bin/env bash
# Serve the Qwen3.5-MoE vision-language model (Qwen/Qwen3.6-35B-A3B-FP8) with vLLM,
# bare-metal, for the lane-behavior eval.
#
# Reuses the same vLLM 0.21 venv as the Cosmos 3 server ($COSMOS3_VENV); that vLLM
# natively registers Qwen3_5MoeForConditionalGeneration, so no --hf-overrides are
# needed. The model is FP8 (A3B MoE, ~3B active) and fits a single GB10's 128 GB
# unified memory. The model emits <think>...</think> reasoning inline in `content`;
# scripts/run_batch.py and scripts/exp_roi8.py parse JSON after the final </think>,
# so no server-side reasoning parser is required.
#
#   scripts/serve_vllm_qwen.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && set -a && . "$REPO_ROOT/.env" && set +a

MODEL="${QWEN_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
PORT="${VLLM_PORT:-8000}"
VENV="${COSMOS3_VENV:-$HOME/sources/cosmos3-server/.venv}"
MEDIA_ROOT="${LANE_MEDIA_ROOT:-$REPO_ROOT}"
MAXLEN="${QWEN_MAX_LEN:-32768}"
GPUUTIL="${QWEN_GPU_UTIL:-0.92}"

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "ERROR: vllm not found at $VENV/bin/vllm (set COSMOS3_VENV)" >&2
  exit 1
fi

# vLLM JIT-compiles the FlashInfer sampler kernel during memory profiling and
# needs `ninja` on PATH; it ships in the venv's bin, which is not otherwise on
# PATH when launched from a bare shell.
export PATH="$VENV/bin:$PATH"

ARGS=(
  "$VENV/bin/vllm" serve "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --allowed-local-media-path "$MEDIA_ROOT"
  --media-io-kwargs '{"video": {"num_frames": -1}}'
  --max-model-len "$MAXLEN"
  --gpu-memory-utilization "$GPUUTIL"
  --async-scheduling
)

echo "Serving $MODEL on :$PORT  media=$MEDIA_ROOT  max_len=$MAXLEN"
exec "${ARGS[@]}"
