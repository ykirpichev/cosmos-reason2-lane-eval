# Frame Rate Is the Dominant Lever for Cosmos 2 on Ego-Lane Behavior — and Where It Plateaus

*A companion study to the Cosmos 3 report. We isolate the single most important
input knob for **Cosmos-Reason2-32B** on ego-lane behavior recognition — the video
frame rate — and show that, unlike Cosmos 3, Cosmos 2 saturates early: 4 fps native
is its best configuration, and the spatial levers that lift Cosmos 3 to 0.93 leave
Cosmos 2 flat or worse.*

> **Reproducibility status.** All numbers are recomputed from committed artifacts by
> `scripts/make_cosmos2_figs.py` and `scripts/headtohead.py`. The controlled
> frame-rate experiment uses a **matched prompt** (`results/summary_fps1.json` at
> 1 fps vs `results/summary_oldtaxonomy.json` at 4 fps); the config ladder uses the
> taxonomy-normalized runs in `results/headtohead.json`.

---

## Abstract

We study **Cosmos-Reason2-32B** ("Cosmos 2") on ego-lane behavior recognition —
classifying a 12-second dashcam clip as *lane keep*, *lane change*, or *lane
wandering*. The practically dangerous error is the **silent miss**: declaring
`keep_within_lane` on a clip that contains a real lane change. We show that for
Cosmos 2 the dominant driver of this error is the **temporal sampling rate**. In a
controlled, prompt-matched experiment, raising the frame rate **1 fps → 4 fps**
nearly doubles lane-change recall (0.15 → 0.38 on 27 human-labeled clips; 0.07 →
0.18 on the full 150-clip set) and lifts accuracy 0.59 → 0.70 — because the
≈1-second crossing event is simply not sampled densely enough at 1 fps. We then show
the **opposite** of the Cosmos 3 result: pushing further (8 fps), or spending more
*spatial* tokens (whole-frame 2× upscale, ROI-crop + zoom), does **not** help
Cosmos 2 — its best configuration is **native 4 fps (accuracy 0.74)**, and every
additional lever is flat or slightly worse. Cosmos 2's ceiling on this task is a
**capability** limit, not a conditioning gap; the same input-budget fixes that are
decisive for Cosmos 3 (see the Cosmos 3 report) do not transfer.

---

## 1. Introduction

Reasoning VLMs are attractive zero-shot classifiers for driving-log mining, but
their accuracy is highly sensitive to how the video is *presented* to the model, not
just to model scale. This report isolates that effect for the previous-generation
**Cosmos 2** and answers two questions:

1. **(§3) How much does frame rate matter?** A controlled 1 fps vs 4 fps comparison
   (identical clips, prompt, and decoding) shows frame rate is the dominant lever:
   at 1 fps the brief lane-crossing event is undersampled and recall collapses.
2. **(§4–5) Does Cosmos 2 keep improving with more budget?** No. Beyond 4 fps,
   neither more frames nor more (even targeted) pixels help. Cosmos 2 plateaus at
   its native 4 fps configuration.

This is the mirror image of the Cosmos 3 finding (`docs/cosmos3_report.md`), where
8 fps + ROI-crop zoom is decisive. Read together, the two reports make a single
point: **input-budget fixes are model-specific and must be validated per model.**

---

## 2. Task and dataset

Three mutually exclusive behaviors:

| behavior | definition |
|---|---|
| `keep_within_lane` | stays inside the lane; never crosses a lane line |
| `lane_change` | crosses a line and settles in a different lane |
| `lane_wandering` | crosses/rides a line but returns to the same lane |

**Clips.** 150 single-camera BATON clips (openpilot `qcamera`, native **526×330**,
12 s). **Ground truth** is **27 human-labeled clips** (13 lane-crossings, 14
lane-keeps); the full 150-clip set is scored only against noisy openpilot
**pseudo-labels** (lateral-offset derived) and is used for scale/consistency checks.
The positive class for precision/recall/F1 is `lane_change` ("crossing").

**Model and serving.** `nvidia/Cosmos-Reason2-32B`, served via Docker vLLM
(`scripts/serve_vllm.sh`, 32k context, `--reasoning-parser qwen3`).

---

## 3. The frame-rate study: 1 fps vs 4 fps

A lane change occupies roughly **1 second** of a 12-second clip. At **1 fps** that
event is carried by ≈1 frame, so the model rarely sees the line actually being
crossed. We compare 1 fps and 4 fps under an **identical prompt and decoding**
(the only change is how many frames are sampled), scoring against human ground truth.

![Cosmos 2: 1 fps vs 4 fps](assets/cosmos2/fig_c2_fps.png)

**Figure 1.** Controlled frame-rate experiment on the 27 human-labeled clips
(matched prompt). Raising 1 → 4 fps lifts accuracy 0.59 → 0.70 and **doubles
lane-change recall** (0.15 → 0.38), i.e. from catching 2/13 crossings to 5/13, at
unchanged (perfect) precision.

| Cosmos 2 (matched prompt) | accuracy | crossing P | R | F1 | crossings caught |
|---|---|---|---|---|---|
| **1 fps** | 0.59 | 1.00 | 0.15 | 0.27 | 2 / 13 |
| **4 fps** | **0.70** | 1.00 | **0.38** | **0.56** | 5 / 13 |

The same effect holds at scale on the full 150-clip set (openpilot pseudo-labels;
noisier, but directionally identical):

| Cosmos 2 (matched prompt, full set) | n | accuracy | crossing recall |
|---|---|---|---|
| 1 fps | 150 | 0.42 | 0.07 (4 / 60) |
| 4 fps | 149 | 0.48 | 0.18 (11 / 60) |

**Finding.** Frame rate is the dominant lever for Cosmos 2. At 1 fps the model is
*structurally* unable to see most crossings; 4 fps recovers a large fraction of them
without any change to the model, prompt, or labels. We therefore use **4 fps** as the
Cosmos 2 reference rate.

---

## 4. The plateau: levers beyond 4 fps do not help

Having established that 4 fps beats 1 fps, the natural next step — and the one that
*works for Cosmos 3* — is to push the budget further: more frames (8 fps), more
pixels (whole-frame 2× upscale), or *targeted* pixels (ROI-crop + zoom on the road
band; see the Cosmos 3 report §4.5 for the mechanics). For Cosmos 2, none of them
help. All runs below use greedy decoding and the taxonomy-normalized scorer.

![Cosmos 2 plateaus across configs](assets/cosmos2/fig_c2_ladder.png)

**Figure 2.** Cosmos 2 across the config ladder (27 human-labeled clips; accuracy =
bars, lane-change recall = line). Native **4 fps is the ceiling (0.74)**; 8 fps
*regresses*, the whole-frame 2× upscale does not recover it, and ROI-crop + zoom —
the decisive lever for Cosmos 3 — drops crossing recall to its lowest (0.27).

| config (27 clips, greedy) | accuracy | crossing P | R | F1 | false-pos |
|---|---|---|---|---|---|
| **4 fps native (best)** | **0.74** | 1.00 | 0.46 | 0.63 | 0 |
| 8 fps native | 0.67 | 0.83 | 0.38 | 0.53 | 1 |
| 8 fps + whole-frame 2× | 0.69¹ | 0.83 | 0.42 | 0.56 | 1 |
| 8 fps + ROI-crop + zoom | 0.67² | 1.00 | 0.27 | 0.43 | 0 |

¹ n=26, ² n=24 — a few Cosmos 2 generations returned unparseable JSON and are
excluded; the trend is unaffected. *Source: `results/headtohead.json`.*

For Cosmos 2, raising the frame rate past 4 fps actually **hurts** (0.74 → 0.67), and
the ROI-zoom that lifts Cosmos 3 by +0.15 instead drops Cosmos 2 to its **worst**
crossing recall (0.27): it keeps calling real crossings `keep_within_lane` no matter
how the spatial budget is spent.

---

## 5. Why Cosmos 2 plateaus

The error signature is consistent across every configuration: **high precision,
collapsed recall.** Cosmos 2 almost never *invents* a lane change (precision stays at
0.83–1.00), but it systematically *misses* them. Unlike Cosmos 3 — whose misses were
recoverable by giving it more legible road pixels — Cosmos 2's misses do **not**
respond to the spatial budget: enlarging the lane markings (ROI-zoom) does not
convert them into detections. This is the fingerprint of a **capability ceiling** on
this fine-grained temporal task rather than a presentation/conditioning gap.

On the full 150-clip set at the final ROI configuration, Cosmos 2 lands at **accuracy
0.52, crossing recall 0.26** (pseudo-labels) — mirroring its 27-clip behavior at
scale.

---

## 6. Relation to Cosmos 3

The two models respond *oppositely* to the same levers (matched ladder, 27 clips):

| config | Cosmos 2 acc | Cosmos 3 acc |
|---|---|---|
| 4 fps native | **0.74** | 0.56 |
| 8 fps native | 0.67 | 0.78 |
| 8 fps + whole-frame 2× | 0.69 | 0.74 |
| 8 fps + ROI-crop + zoom | 0.67 | **0.93** |

- **Cosmos 2** peaks at **native 4 fps** and degrades with more budget.
- **Cosmos 3** starts *below* Cosmos 2 at its native rate but, once correctly
  conditioned (8 fps + ROI-zoom), reaches **0.93** — a **+0.19 accuracy / +0.39
  crossing-recall** win over the best Cosmos 2 config.

The takeaway is not "Cosmos 3 > Cosmos 2" but that **the right input budget is
model-specific**: Cosmos 2 needs 4 fps and is then saturated; Cosmos 3 needs more
frames *and* targeted spatial tokens. See `docs/cosmos3_report.md` §5.1 for the full
head-to-head figure and discussion.

---

## 7. Reproducibility

```bash
# Cosmos 2 figures (1 fps vs 4 fps; the config-ladder plateau)
.venv/bin/python scripts/make_cosmos2_figs.py     # -> docs/assets/cosmos2/

# Controlled frame-rate runs (matched prompt; full BATON set)
#   1 fps -> results/summary_fps1.json
#   4 fps -> results/summary_oldtaxonomy.json
.venv/bin/python scripts/run_batch.py \
  --manifest clips/manifest_all.json \
  --model nvidia/Cosmos-Reason2-32B --fps 1 \
  --media-path-prefix "$PWD" --output results/<run>   # (rename summary as above)

# Config ladder (8 fps native, whole-frame 2x, ROI-zoom) + consolidation
bash scripts/_run_cosmos2.sh
.venv/bin/python scripts/headtohead.py            # -> results/headtohead.json

# Inspect individual cases (run/mode/clip deep-linked)
.venv/bin/streamlit run apps/review_disagreements.py --server.port 8503
```

---

## 8. Limitations

- **Small ground-truth set.** Headline metrics are on 27 clips (13 crossings); ±1–2
  detections move F1 by ~0.1. The full-set numbers use noisy pseudo-labels.
- **Prompt-version caveat.** The controlled 1-vs-4 fps experiment (§3) uses the
  earlier single-label prompt for *both* rates (a clean A/B); the config ladder (§4)
  uses the later taxonomy-normalized prompt. The tuned prompt lifts 4 fps slightly
  (0.70 → 0.74), which is why §4's best 4 fps number is 0.74. The 1-vs-4 *difference*
  is unaffected because both sides share the prompt.
- **Unparseable generations.** A few Cosmos 2 outputs at 8 fps + 2× / ROI failed JSON
  parsing and are excluded (n=26 / n=24), marked explicitly.

---

## 9. Conclusion

For Cosmos 2 on ego-lane behavior, **frame rate is the dominant input lever**:
moving from 1 fps to 4 fps roughly doubles lane-change recall by sampling the brief
crossing event densely enough to see it. But Cosmos 2 **saturates at native 4 fps** —
more frames regress, and more pixels (even ROI-targeted) do not convert its
systematic misses into detections, indicating a capability ceiling rather than a
conditioning gap. This is the opposite of Cosmos 3, which only reaches its potential
once given a higher frame rate *and* a targeted spatial token budget. The practical
lesson for deploying reasoning VLMs on video: **profile the frame-rate sensitivity of
each model first, and never assume an input-budget fix transfers across model
generations.**
