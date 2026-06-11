# Cosmos-Reason Lane-Behavior Eval (Cosmos 2 & Cosmos 3)

Evaluate NVIDIA's **Cosmos** video reasoning VLMs — **Cosmos-Reason2-32B**
("Cosmos 2") and **Cosmos3-Super** ("Cosmos 3") — on ego-vehicle lane behavior,
using real dashcam clips with lane signals mined from
[BATON-Sample](https://huggingface.co/datasets/HenryYHW/BATON-Sample) (openpilot's
production lane model) and from [nuScenes](https://www.nuscenes.org/) (HD-map +
ego-pose projection). The pipeline mines 12 s / 4 Hz clips, derives pseudo-labels,
runs inference on **either model**, and provides Streamlit apps for human labeling
and disagreement review. The two models are compared head-to-head in the
[reports below](#reports--key-findings).

Cosmos 2 is served via Docker vLLM (`scripts/serve_vllm.sh`); Cosmos 3 is served
bare-metal via vLLM (`scripts/serve_vllm_cosmos3.sh`). They share one GPU and cannot
co-reside, so cross-model runs are sequential.

Clips can be a single forward camera (`front_only`, BATON/openpilot) or a
multi-camera **mosaic** (`front_mosaic3`, nuScenes): `CAM_FRONT` on top at higher
resolution, with `CAM_FRONT_LEFT | CAM_FRONT_RIGHT` below at lower resolution.

## Reports & key findings

Two written studies (figures + analysis) live in [`docs/`](docs/):

| Report | What it shows |
|---|---|
| **[Cosmos 3 — staged diagnosis](docs/cosmos3_report.md)** | Improving Cosmos 3-Super from 0.56 to **0.93 accuracy** on 27 human-labeled clips by adjusting **temporal sampling** (4→8 fps, greedy) and then **targeting spatial tokens** (ROI-crop + zoom). Includes an ablation showing that *targeted* spatial tokens (ROI) outperform a uniform whole-frame upscale. |
| **[Cosmos 2 — frame-rate study](docs/cosmos2_report.md)** | Frame rate is the dominant input lever for Cosmos-Reason2-32B: **1 fps → 4 fps roughly doubles lane-change recall** (0.15 → 0.38). Cosmos 2 scores highest at native 4 fps (0.74); additional frames/pixels did not improve it in our experiments. |

**Headline result (27 human-labeled clips, accuracy):**

| config | Cosmos 2 | Cosmos 3 |
|---|---|---|
| 4 fps native | **0.74** | 0.56 |
| 8 fps native | 0.67 | 0.78 |
| 8 fps + whole-frame 2× | 0.69 | 0.74 |
| 8 fps + ROI-crop + zoom | 0.67 | **0.93** |

The two models respond differently to the same input-budget levers — the
adjustments are model-specific. A consolidated, reproducible scoring of every run
is in `results/headtohead.json` (`scripts/headtohead.py`).

**At scale (full 150-clip BATON set, openpilot pseudo-labels):** agreement is much
lower (Cosmos 2 ≈ 0.52) than on human ground truth, because the pseudo-labels are
noisy — so the full-set run is best read as a *pseudo-label agreement check* rather
than a model score. Human labels are the more reliable metric.

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
prompts/lane_behavior.yaml         # Cosmos prompt for single-camera clips
prompts/lane_behavior_mosaic.yaml  # Cosmos prompt for 3-pane mosaic clips
scripts/
  config.py                    # central paths + cache dir + camera layouts + helpers
  ingest_baton.py              # mine clips from BATON-Sample (offset signal)
  ingest_openpilot.py          # adapter for ADAS-TO / OpenLKA style data
  ingest_nuscenes.py           # mine mosaic clips from nuScenes (map + ego pose)
  mosaic_utils.py              # compose CAM_FRONT/FRONT_LEFT/FRONT_RIGHT into a mosaic
  remap_pseudo_3class.py       # offset -> 3-class pseudo-label (artifact-gated)
  run_batch.py                 # batch Cosmos inference via vLLM OpenAI API (--model picks C2/C3)
  serve_vllm.sh                # start the Cosmos-Reason2 (Cosmos 2) vLLM server (Docker)
  serve_vllm_cosmos3.sh        # start the Cosmos3-Super (Cosmos 3) vLLM server (bare-metal)
  headtohead.py                # consolidate all runs -> results/headtohead.json
  make_report_figs.py          # figures for docs/cosmos3_report.md
  make_cosmos2_figs.py         # figures for docs/cosmos2_report.md
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

# 3. Start a Cosmos vLLM server (GPU). Pick ONE (they can't co-reside):
scripts/serve_vllm.sh                          # Cosmos 2 (Reason2-32B, Docker)
scripts/serve_vllm_cosmos3.sh                  # Cosmos 3 (Cosmos3-Super, bare-metal)

# 4. Run inference over all clips (choose the model matching the server above).
.venv/bin/python scripts/run_batch.py        # Cosmos 2 default; config.py for defaults
.venv/bin/python scripts/run_batch.py --model nvidia/Cosmos3-Super --fps 8   # Cosmos 3

# 5. Explore / label / review.
.venv/bin/streamlit run apps/view_examples.py          # dashboard
.venv/bin/streamlit run apps/label_clips.py            # blind human labeling
.venv/bin/streamlit run apps/review_disagreements.py   # disagreement review
```

`ingest_baton.py` flags: `--per-category N`, `--scan-only` (yield report only),
`--max-routes K` (quick run).

## Other datasets

### nuScenes (multi-camera mosaic)

`ingest_nuscenes.py` mines mosaic clips from a local nuScenes install (defaults to
`v1.0-mini` under `<cache>/datasets/nuscenes`, override with `NUSCENES_DATAROOT` /
`NUSCENES_VERSION`). It needs the unzipped map-expansion under `maps/expansion`.

```bash
.venv/bin/python scripts/ingest_nuscenes.py --per-category 5   # merges into manifest_all.json
.venv/bin/python scripts/ingest_nuscenes.py --scan-only        # candidate yield only
.venv/bin/python scripts/ingest_nuscenes.py --layout front_only  # single CAM_FRONT instead of mosaic
```

Because nuScenes has no production lane signal, the lateral offset is derived by
projecting the ego pose onto its current lane's map centerline. To avoid the
nearest-centerline "snapping" between parallel lanes that caused phantom
violations, the reference lane is tracked with hysteresis (only re-anchored once
the ego is comfortably centered, `|offset| < 0.7 m`) and extended along lane
connectivity; impossible >5 m/s lateral jumps are gated out. The resulting
`signed_lateral_m` feeds the same mining classifier and 3-class remap as BATON.
These remain **pseudo-labels** — verify them with `apps/label_clips.py` (it has a
**Dataset** filter so you can label just the nuScenes clips).

nuScenes clips are tagged `dataset: "nuscenes"` / `camera_layout: "front_mosaic3"`.
`run_batch.py` auto-selects `prompts/lane_behavior_mosaic.yaml` for mosaic clips
and `prompts/lane_behavior.yaml` for single-camera clips, so one run handles a
mixed manifest.

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
maneuvers — a lane change spanning ~1 s gets smeared into 1–2 frames and is
reported as `keep_within_lane`. Keep the request fps equal to the clip fps.

This is quantified in the [Cosmos 2 report](docs/cosmos2_report.md): **1 fps → 4 fps
roughly doubles lane-change recall** (0.15 → 0.38 on human-labeled clips). The
optimal rate is **model-specific**, though — Cosmos 3 keeps improving up to **8 fps**
(then regresses at higher rates); see the [Cosmos 3 report](docs/cosmos3_report.md).

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
