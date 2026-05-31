#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for Cosmos-Reason2 in Docker.
#
# The media root (cache dir, or repo root in the legacy layout) is mounted at
# VLLM_MEDIA_PATH_PREFIX so run_batch.py can pass file:// URLs for local clips.
#
#   scripts/serve_vllm.sh                # start (foreground logs via docker logs)
#   VLLM_PORT=8001 scripts/serve_vllm.sh # custom port
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && set -a && . "$REPO_ROOT/.env" && set +a

MODEL="${COSMOS_MODEL:-nvidia/Cosmos-Reason2-32B}"
PORT="${VLLM_PORT:-8000}"
MEDIA_PREFIX="${VLLM_MEDIA_PATH_PREFIX:-/workspace}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
CONTAINER="${VLLM_CONTAINER:-cosmos-vllm}"
IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.04-py3}"

# Resolve the media root the same way scripts/config.py does.
if [ -n "${LANE_CACHE_DIR:-}" ]; then
  MEDIA_ROOT="$LANE_CACHE_DIR"
elif [ -d "$REPO_ROOT/clips" ]; then
  MEDIA_ROOT="$REPO_ROOT"            # legacy layout
else
  MEDIA_ROOT="$REPO_ROOT/cache"
fi
mkdir -p "$MEDIA_ROOT"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "vLLM container '${CONTAINER}' already running."
  exit 0
fi
docker rm -f "$CONTAINER" 2>/dev/null || true

echo "Serving $MODEL on :$PORT  (media root $MEDIA_ROOT -> $MEDIA_PREFIX)"
docker run -d --name "$CONTAINER" \
  --gpus all --ipc=host --network host \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$MEDIA_ROOT:$MEDIA_PREFIX" \
  -w "$MEDIA_PREFIX" \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --allowed-local-media-path "$MEDIA_PREFIX" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --reasoning-parser qwen3

echo "Waiting for vLLM to become ready (model load can take several minutes)..."
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready at http://127.0.0.1:${PORT}"
    exit 0
  fi
  sleep 10
done
echo "ERROR: vLLM did not become ready. Check: docker logs $CONTAINER" >&2
exit 1
