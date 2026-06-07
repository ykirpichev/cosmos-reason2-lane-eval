#!/usr/bin/env bash
# After the 27-clip ladder finishes, run the FINAL config (ROI-crop + zoom @ 8fps
# greedy) on the full 159-clip manifest for BOTH models, scored against openpilot
# pseudo-labels (with the 27 human clips broken out). Cosmos 2 is already loaded,
# so it goes first; then swap back to Cosmos 3.
# Sentinels: "FULL159 ALL DONE" / "FULL159 FAILED".
set -uo pipefail
cd /home/ykirpichev/sources/cosmos-reason-lane-test
ROOT="$PWD"

echo "[$(date +%T)] waiting for 27-clip ladder to finish..."
for _ in $(seq 1 240); do
  grep -qE "COSMOS2 ALL DONE|COSMOS2 FAILED" results/cosmos2_swap.out 2>/dev/null && break
  sleep 30
done
grep -q "COSMOS2 ALL DONE" results/cosmos2_swap.out 2>/dev/null || { echo "FULL159 FAILED (ladder did not finish)"; exit 1; }

# ---- Cosmos 2 (already loaded) on full 159 ----
if ! curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then echo "FULL159 FAILED (no server for C2)"; exit 1; fi
echo "[$(date +%T)] Cosmos 2 full-159 ROI-zoom (builds ~132 new clips + inference)"
.venv/bin/python scripts/exp_roi8.py --model "nvidia/Cosmos-Reason2-32B" \
  --output results/cosmos2_roi8_full159 --clips all > results/c2_roi8_full159.out 2>&1
echo "    -> rc=$? $(grep -A14 'ROI8 RESULT' results/c2_roi8_full159.out | tail -15)"

# ---- swap back to Cosmos 3 bare-metal ----
echo "[$(date +%T)] stopping Cosmos 2 Docker"
docker rm -f cosmos-vllm 2>/dev/null || true
sleep 20
echo "[$(date +%T)] starting Cosmos 3 bare-metal server"
nohup ./scripts/serve_vllm_cosmos3.sh > /tmp/cosmos3_serve.log 2>&1 &
for _ in $(seq 1 150); do
  curl -sf http://localhost:8000/v1/models 2>/dev/null | grep -q Cosmos3 && break
  sleep 10
done
if ! curl -sf http://localhost:8000/v1/models 2>/dev/null | grep -q Cosmos3; then
  echo "FULL159 FAILED (Cosmos 3 not ready)"; tail -20 /tmp/cosmos3_serve.log; exit 1
fi
echo "[$(date +%T)] Cosmos 3 ready"

# ---- Cosmos 3 on full 159 (ROI clips already built by the C2 pass) ----
echo "[$(date +%T)] Cosmos 3 full-159 ROI-zoom"
.venv/bin/python scripts/exp_roi8.py --model "nvidia/Cosmos3-Super" \
  --output results/cosmos3_roi8_full159 --clips all > results/c3_roi8_full159.out 2>&1
echo "    -> rc=$? $(grep -A14 'ROI8 RESULT' results/c3_roi8_full159.out | tail -15)"

echo "FULL159 ALL DONE"
