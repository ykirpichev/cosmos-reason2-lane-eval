# Cosmos-Reason2 Lane-Behavior Eval

Evaluate **NVIDIA Cosmos-Reason2-32B** (a video reasoning VLM) on ego-vehicle
lane behavior, using real dashcam clips with lane signals mined from
[BATON-Sample](https://huggingface.co/datasets/HenryYHW/BATON-Sample) (openpilot's
production lane model). The pipeline mines 12 s / 4 Hz clips, derives
offset-based pseudo-labels, runs Cosmos inference, and provides Streamlit apps for
human labeling and disagreement review.

## Taxonomy

Each clip is described by a **road geometry** and a **time-ordered list of lane
events**, each event being one of three behaviors:

| Behavior | Meaning |
|---|---|
| `keep_within_lane` | Stays inside the lane; never crosses a line. |
| `lane_change` | Crosses a line and settles in a **different** lane. |
| `lane_wandering` | Crosses/rides a line but returns to the **same** lane (drift-and-return, straddle, weave). |

Geometry is `straight` or `curved`. `overall_behavior` reduces the events to the
single most significant one (`lane_change`/`lane_wandering` outrank `keep_within_lane`).

## Repository layout

```
prompts/lane_behavior.yaml     # Cosmos prompt (geometry + multi-event schema)
scripts/
  config.py                    # central paths + cache dir + helpers
  ingest_baton.py              # mine clips from BATON-Sample (offset signal)
  ingest_openpilot.py          # adapter for ADAS-TO / OpenLKA style data
  remap_pseudo_3class.py       # offset -> 3-class pseudo-label (artifact-gated)
  run_batch.py                 # batch Cosmos inference via vLLM OpenAI API
  serve_vllm.sh                # start the Cosmos-Reason2 vLLM server (Docker)
  video_utils.py               # browser-safe H.264 transcode helpers
apps/
  view_examples.py             # dashboard: predictions + labels + metrics
  label_clips.py               # blind human labeling
  review_disagreements.py      # video + events + offset trace for disagreements
requirements.txt
.env.example
```

The repo is **code only**. All data lives in a cache directory (see below) and is
git-ignored.

## Data & the cache directory

Every large/generated artifact — raw datasets, extracted clips, predictions,
transcoded videos, logs — lives under a single cache dir so the project is
reproducible and relocatable. Point it anywhere (e.g. a shared volume so a Cursor
cloud/local agent can run it from anywhere):

```bash
export LANE_CACHE_DIR=/mnt/volume/lane-eval-cache   # default: ./cache
```

Layout under the cache dir:

```
<cache>/datasets/    # raw source datasets
<cache>/clips/       # extracted clips + manifest_all.json (+ pseudo_3class)
<cache>/results/     # summary.json, human_labels.json, logs/, video_cache/
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # optional: edit cache dir / model / ports
```

## Pipeline

```bash
# 1. Mine clips from BATON-Sample (accept terms once on the HF dataset page;
#    huggingface_hub uses ~/.cache/huggingface/token or HF_TOKEN).
.venv/bin/python scripts/ingest_baton.py --per-category 30

# 2. Derive offset-based 3-class pseudo-labels (artifact-gated).
.venv/bin/python scripts/remap_pseudo_3class.py

# 3. Start the Cosmos-Reason2 vLLM server (Docker, GPU).
scripts/serve_vllm.sh

# 4. Run Cosmos inference over all clips.
.venv/bin/python scripts/run_batch.py        # defaults come from scripts/config.py

# 5. Explore / label / review.
.venv/bin/streamlit run apps/view_examples.py          # dashboard
.venv/bin/streamlit run apps/label_clips.py            # blind human labeling
.venv/bin/streamlit run apps/review_disagreements.py   # disagreement review
```

`ingest_baton.py` flags: `--per-category N`, `--scan-only` (yield report only),
`--max-routes K` (quick run).

## Prediction schema

`run_batch.py` writes `results/summary.json` incrementally; each clip's parsed
prediction looks like:

```json
{
  "road_geometry": "straight",
  "events": [
    {"behavior": "keep_within_lane", "time_of_event_sec": 0.0, "confidence": 0.95, "description": "..."},
    {"behavior": "lane_change",      "time_of_event_sec": 5.0, "confidence": 0.9,  "description": "..."}
  ],
  "overall_behavior": "lane_change"
}
```

Full raw model responses are saved under `results/logs/<clip_id>.log`.

## Frame rate (important)

Clips are authored at **4 Hz (48 frames / 12 s)** and inference requests the
server at `--fps 4.0`. Sampling lower (e.g. 1 Hz) makes Cosmos miss short
maneuvers — a lane change spanning ~3 s gets smeared into 2–3 frames and is
reported as `keep_within_lane`. Keep the request fps equal to the clip fps.

## Pseudo-labels: caveats

`pseudo_3class` is derived from openpilot's `signed_lateral_m` offset trace with
artifact gating (impossible >5 m/s lateral jumps from lane-line re-detection are
removed). Because openpilot re-centers on the new lane after a change,
`lane_change` vs `lane_wandering` is only weakly separable from offset alone —
treat the pseudo-label as a rough reference. **Human labels + Cosmos are the
ground truth** for evaluation.

## DGX / Spark notes

- Image `nvcr.io/nvidia/vllm:26.04-py3`, `--gpu-memory-utilization 0.90`.
- Set `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.
- The cache/media root is mounted at `--allowed-local-media-path` (`/workspace`).
- 32B model load takes several minutes; ~90 s per 12 s clip at fps=4.
