#!/usr/bin/env bash
# Swap GPU from bare-metal Cosmos 3 to Docker Cosmos 2, then run the matched
# ladder (8fps native, 8fps + whole-frame 2x, ROI-zoom) on the 27 ground-truth
# clips. Unattended; sentinels: "COSMOS2 ALL DONE" / "COSMOS2 FAILED".
set -uo pipefail
cd /home/ykirpichev/sources/cosmos-reason-lane-test
ROOT="$PWD"
M="nvidia/Cosmos-Reason2-32B"

echo "[$(date +%T)] stopping Cosmos 3 bare-metal server"
pkill -f "cosmos3-server/.venv" 2>/dev/null || true
sleep 25
# wait for port 8000 to stop answering (old server gone)
for _ in $(seq 1 24); do
  curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 || break
  sleep 5
done
if curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
  echo "[$(date +%T)] WARN port 8000 still up; force kill"
  pkill -9 -f "cosmos3-server/.venv" 2>/dev/null || true
  sleep 20
fi

echo "[$(date +%T)] starting Cosmos 2 Docker server (32B weight load, may take minutes)"
VLLM_MEDIA_PATH_PREFIX="$ROOT" ./scripts/serve_vllm.sh
if ! curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
  echo "COSMOS2 FAILED (server not ready)"; docker logs --tail 30 cosmos-vllm 2>&1 || true; exit 1
fi
SERVED=$(curl -s http://localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
echo "[$(date +%T)] Cosmos 2 ready, serving: $SERVED"

echo "[$(date +%T)] (1/3) 8 fps native"
.venv/bin/python scripts/run_batch.py --manifest clips/manifest_final27_8fps_native.json \
  --model "$M" --fps 8 --media-path-prefix "$ROOT" \
  --output results/cosmos2_final_8fps_native > results/c2_8fps_native.out 2>&1
echo "    -> rc=$? $(tail -1 results/c2_8fps_native.out)"

echo "[$(date +%T)] (2/3) 8 fps + whole-frame 2x upscale"
.venv/bin/python scripts/run_batch.py --manifest clips/manifest_final27.json \
  --model "$M" --fps 8 --media-path-prefix "$ROOT" \
  --output results/cosmos2_final_8fps2x > results/c2_8fps2x.out 2>&1
echo "    -> rc=$? $(tail -1 results/c2_8fps2x.out)"

echo "[$(date +%T)] (3/3) ROI-crop + zoom"
.venv/bin/python scripts/exp_roi8.py --model "$M" \
  --output results/cosmos2_roi8 > results/c2_roi8.out 2>&1
echo "    -> rc=$? $(grep -A12 'ROI8 RESULT' results/c2_roi8.out | tail -13)"

echo "[$(date +%T)] normalizing stored predictions"
.venv/bin/python scripts/normalize_results.py > /dev/null 2>&1 || true
echo "COSMOS2 ALL DONE"
