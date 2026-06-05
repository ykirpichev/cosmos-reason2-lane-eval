#!/usr/bin/env bash
# Serve the Cosmos3-Super *reasoner* (understanding tower) with vLLM, bare-metal.
#
# Unlike serve_vllm.sh (Docker, Cosmos-Reason2), Cosmos 3 needs the vllm-cosmos3
# plugin + vLLM 0.21 (NVIDIA's release-tested combo). That lives in a dedicated
# venv ($COSMOS3_VENV). The plugin subclasses Qwen3VLForConditionalGeneration and
# loads ONLY the understanding tower (~65 GB BF16) out of the 129 GB MoT shards,
# so it fits a single GB10's 128 GB unified memory. Set COSMOS3_QUANT=fp8 to halve
# the weights (~33 GB) if you hit OOM.
#
#   scripts/serve_vllm_cosmos3.sh                 # BF16
#   COSMOS3_QUANT=fp8 scripts/serve_vllm_cosmos3.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && set -a && . "$REPO_ROOT/.env" && set +a

MODEL="${COSMOS3_MODEL:-nvidia/Cosmos3-Super}"
PORT="${VLLM_PORT:-8000}"
VENV="${COSMOS3_VENV:-$HOME/sources/cosmos3-server/.venv}"
MEDIA_ROOT="${LANE_MEDIA_ROOT:-$REPO_ROOT}"
MAXLEN="${COSMOS3_MAX_LEN:-32768}"
GPUUTIL="${COSMOS3_GPU_UTIL:-0.92}"
QUANT="${COSMOS3_QUANT:-}"

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "ERROR: vllm not found at $VENV/bin/vllm (set COSMOS3_VENV)" >&2
  exit 1
fi

ARGS=(
  "$VENV/bin/vllm" serve "$MODEL"
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}'
  --host 0.0.0.0 --port "$PORT"
  --allowed-local-media-path "$MEDIA_ROOT"
  --media-io-kwargs '{"video": {"num_frames": -1}}'
  --max-model-len "$MAXLEN"
  --gpu-memory-utilization "$GPUUTIL"
  --async-scheduling
)
[ -n "$QUANT" ] && ARGS+=(--quantization "$QUANT")

echo "Serving $MODEL (reasoner) on :$PORT  media=$MEDIA_ROOT  max_len=$MAXLEN  quant=${QUANT:-none}"
exec "${ARGS[@]}"
