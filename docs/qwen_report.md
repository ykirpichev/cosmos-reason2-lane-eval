# A Third Reasoner on Ego-Lane Behavior: Qwen Across the Matched Input-Budget Ladder

*Running an open-weight Qwen vision–language MoE through the same matched
configuration ladder used for Cosmos 2 and Cosmos 3, to test whether the
input-budget levers that lifted Cosmos 3 to 0.93 transfer to a different model
family.*

| | |
|---|---|
| **Version** | 1.0 — June 25, 2026 |
| **Repository** | [`ykirpichev/cosmos-reason2-lane-eval`](https://github.com/ykirpichev/cosmos-reason2-lane-eval) |
| **Model** | `Qwen/Qwen3.6-35B-A3B-FP8` (FP8 MoE, ~3B active params), served bare-metal via vLLM (`scripts/serve_vllm_qwen.sh`); labeled **`Qwen3.6-35B-A3B-FP8`** (short: "Qwen 3.6") in figures/`headtohead.json` |
| **Evaluation set** | 27 human-labeled BATON dashcam clips (13 lane-crossings, 14 lane-keeps); 150-clip pseudo-label scale check |
| **Companion reports** | [Cosmos 3 — staged diagnosis](cosmos3_report.md) · [Cosmos 2 — frame-rate study](cosmos2_report.md) |

> **Executive summary.** Qwen, an open-weight reasoning VLM, was run through the
> **identical matched ladder** (4 fps native, 8 fps native, whole-frame 2×,
> ROI-crop + zoom; greedy decoding) used to diagnose Cosmos 3. It reaches its best
> accuracy of **0.69 (F1 0.56)** in the **ROI-zoom** configuration — the same
> winning lever as Cosmos 3 — and it shares Cosmos 3's *shape* (whole-frame
> upscaling hurts; targeted ROI tokens help most). But its ceiling is well below:
> Qwen lands **third** behind Cosmos 3 (0.93) and Cosmos 2 (0.74). The reason is a
> strong **conservative bias** — Qwen has perfect lane-change *precision* (1.00,
> zero false positives in all four 27-clip configs) but **collapsed recall**
> (≤0.39): it systematically under-calls the brief crossing event and defaults to
> `keep_within_lane`. The input-budget levers move recall a little (0.23 → 0.39 via
> ROI-zoom) but do not break the bias. This reinforces the central finding of the
> companion reports: **the same input-budget adjustments are model-specific**, and
> a model's prior toward the majority class can cap accuracy regardless of how the
> video is presented. Small-sample caveats apply (27 clips, 13 crossings; ±0.1
> noise floor).

> **Reproducibility status.** All numbers are final and reproducible from committed
> artifacts. Qwen was evaluated under the full matched ladder on the 27
> human-labeled clips (`results/qwen_4fps_native`, `results/qwen_8fps_native`,
> `results/qwen_8fps2x`, `results/qwen_roi8`), consolidated by
> `scripts/headtohead.py` into `results/headtohead.json`. The ROI config is
> additionally run on the full 150-clip BATON set against openpilot pseudo-labels
> (`results/qwen_roi8_full159`). The ladder is orchestrated by
> `scripts/_run_qwen.sh`; figures by `scripts/make_qwen_figs.py`.

---

## 1. Why a third model

The [Cosmos 3 report](cosmos3_report.md) showed that input conditioning — temporal
sampling rate plus *targeted* spatial tokens (ROI-crop + zoom) — explained most of
the gap between two NVIDIA models, and that a **matched ladder on Cosmos 2** proved
those levers were **model-specific** (they helped Cosmos 3, not Cosmos 2). That
raises an obvious external-validity question: are these levers a property of the
*task* (lane behavior on low-resolution dashcam) or of the *Cosmos family*?

To probe this we add a third, architecturally different reasoner — an open-weight
**Qwen** vision–language MoE (`Qwen/Qwen3.6-35B-A3B-FP8`, ~3B active parameters,
FP8) — and run it through the **exact same ladder, clips, prompt, greedy decoding,
ROI crop, and label-hygiene pass**. Nothing model-specific is tuned. The model
emits `<think>…</think>` reasoning inline; the run scripts parse the JSON after the
final `</think>`, so no server-side reasoning parser is required
(`scripts/serve_vllm_qwen.sh`).

---

## 2. Task, dataset, and serving

The task and dataset are unchanged from the companion reports: classify a 12-second
dashcam clip as `keep_within_lane`, `lane_change`, or `lane_wandering`. Headline
metrics are on the **27 human-labeled BATON clips** (14 keeps, 13 crossings); the
positive class for P/R/F1 is `lane_change`. The full 150-clip BATON set is scored
against openpilot pseudo-labels as a noisy agreement check only (see §2 of the
Cosmos 3 report for caveats).

**Serving.** Qwen runs bare-metal via vLLM, reusing the Cosmos 3 server's vLLM venv
(it natively registers the Qwen MoE architecture). It is FP8 and fits a single
GB10's 128 GB unified memory. All three models share one GPU and cannot co-reside,
so the Qwen ladder ran after the Cosmos runs. Inference used **greedy decoding**
(`temperature 0`) and clip-parallel requests (`--concurrency 8`, batched
server-side by vLLM) with a 10k-token output budget to accommodate Qwen's verbose
chain-of-thought.

---

## 3. Qwen across the ladder

The four matched configurations on the 27 human-labeled clips (greedy; positive
class = `lane_change`):

| config (27 clips, greedy) | accuracy | lane-change P | R | F1 | false-pos |
|---|---|---|---|---|---|
| 4 fps native | 0.63 | 1.00 | 0.23 | 0.38 | 0 |
| 8 fps native | 0.63 | 1.00 | 0.23 | 0.38 | 0 |
| 8 fps + whole-frame 2× | 0.59 | 1.00 | 0.15 | 0.27 | 0 |
| **8 fps + ROI-crop + zoom** | **0.69**¹ | **1.00** | **0.39** | **0.56** | **0** |

¹ n=26 — one Qwen generation did not return parseable JSON and is excluded; the
trend is unaffected. *Source: `results/headtohead.json`.*

![Qwen across the matched input-budget ladder](assets/qwen/fig_qwen_ladder.png)

**Figure 1.** Qwen across the four matched configurations (accuracy = bars,
lane-change recall = line). The qualitative *shape* matches Cosmos 3: the
whole-frame 2× upscale is the worst configuration, and the targeted ROI-crop + zoom
is the best. But every bar sits low, and the recall line never clears 0.4 — Qwen
catches at most 5 of the 13 crossings.

**Reading.**
- **Frame rate (4 → 8 fps) is flat** for Qwen (0.63 → 0.63, identical recall),
  unlike Cosmos 3 where it was the decisive Stage-1 lever (0.56 → 0.78). Qwen's
  bottleneck is not temporal sampling.
- **Whole-frame 2× upscale hurts** (0.63 → 0.59, recall 0.23 → 0.15) — the same
  negative result seen on Cosmos 3, and for the same likely reason: the extra
  spatial budget is spent on sky/hood rather than the lane cue.
- **ROI-crop + zoom is the best lever** (0.69, recall 0.39) — again matching the
  Cosmos 3 ordering. Concentrating tokens on the road band recovers two more
  crossings than the native configs. But the gain (+0.06 accuracy, +0.16 recall) is
  far smaller than Cosmos 3's (+0.15 accuracy, +0.31 recall).

---

## 4. The conservative-bias ceiling

Across **all four** configurations, Qwen's lane-change **precision is 1.00** — when
it does call a crossing, it is always right — with **zero false positives** on the
27-clip set. The entire error budget is **missed crossings** (false negatives): even
in its best (ROI-zoom) configuration, Qwen reports `keep_within_lane` on **8 of 13**
real crossings, including clips that Cosmos 3 reads correctly at ROI-zoom
(`lane_recovery__17`, `lane_violation_left__14`, `lane_violation_right__13`,
`lane_violation_right__16`). It also misses `lane_violation_left__18`/`__20`, the
pair every model misses (flagged for label re-review in the Cosmos 3 report).

This is a **prior toward the majority class**, not a resolution problem: Qwen
defaults to "nothing happened" and only overrides that default for the most blatant
crossings. The input-budget levers nudge this — ROI-zoom recovers a couple of
borderline cases — but they do not change Qwen's fundamental reluctance to commit to
the rare event. Because 14 of 27 clips are keeps, this bias guarantees a respectable
floor (everything-is-a-keep would score 0.52) but caps the ceiling well below the
models that will commit to a crossing.

---

## 5. Head-to-head: three reasoners, same ladder

![Three-way accuracy comparison across matched configs](assets/qwen/fig_qwen_3way.png)

**Figure 2.** Accuracy of all three reasoners across the four matched
configurations (27 human-labeled clips). The input-budget response is visibly
**model-specific**: Cosmos 2 is best at its native 4 fps and degrades as levers are
added; Cosmos 3 climbs steeply to 0.93 with frames + ROI tokens; Qwen tracks Cosmos
3's *shape* (ROI-zoom best, whole-frame worst) but at a much lower level.

**Best configuration per model (27 human-labeled clips):**

| model | best config | accuracy | lane-change R | F1 | false-pos |
|---|---|---|---|---|---|
| **Cosmos 3-Super** | 8 fps + ROI-zoom | **0.93** | 0.85 | 0.92 | 0 |
| Cosmos-Reason2-32B | 4 fps native | 0.74 | 0.46 | 0.63 | 0 |
| Qwen (`Qwen3.6-35B-A3B-FP8`) | 8 fps + ROI-zoom | 0.69 | 0.39 | 0.56 | 0 |

Qwen lands **third**. Notably, its best configuration is the *same* as Cosmos 3's
(8 fps + ROI-zoom), reinforcing that the ROI-crop lever is task-useful across model
families — it just cannot overcome Qwen's conservative prior the way it complements
Cosmos 3's already-decent recall.

---

## 6. Scale check on the full BATON set

The final ROI config run on the **full 150-clip BATON set**, scored against noisy
openpilot pseudo-labels (agreement, not ground truth — see Cosmos 3 report §2):

| full-set ROI-zoom (pseudo-labels) | n | accuracy | crossing recall |
|---|---|---|---|
| Cosmos 3 | 150 | 0.55 | 0.40 |
| Cosmos 2 | 142 | 0.52 | 0.26 |
| **Qwen** | 145 | **0.46** | **0.13** |

The ordering from the human-labeled set is preserved (Qwen lowest), and Qwen's
crossing recall collapses further at scale (0.13) — consistent with the
conservative bias of §4. Re-scoring the same full-set run on the 27 human-labeled
subset gives Qwen 0.63 accuracy / 0.30 crossing recall, close to its dedicated
27-clip run (0.69 / 0.39), within the small-sample / serving-nondeterminism noise.

---

## 7. Conclusion

Adding a third, open-weight reasoner to the matched ladder sharpens the central
finding of the companion reports. **The input-budget levers are model-specific in
*magnitude* but partly shared in *direction*:** for both Cosmos 3 and Qwen, the
ROI-crop + zoom is the best configuration and the whole-frame 2× upscale is the
worst — so "spend the spatial budget where the cue is" appears to be a property of
the *task*, not just the Cosmos family. What does **not** transfer is the payoff.
Qwen enters with a strong prior toward `keep_within_lane` (perfect precision, zero
false positives, but recall ≤0.39), and no amount of input re-budgeting converts
that conservatism into the crossing detections needed to reach Cosmos 3's 0.93.

Practical takeaways for evaluating reasoning VLMs on video, extending the companion
reports: **(1)** profile the temporal and spatial budgets independently and with a
matched ladder — but **(2)** also profile the model's *class prior*: a
majority-class bias can cap accuracy on a rare-event task regardless of input
conditioning, and it shows up clearly as high precision with collapsed recall;
**(3)** the targeted-ROI spatial lever generalizes across model families on this
task, while its benefit does not. On this evaluation set the ranking is
**Cosmos 3 (0.93) > Cosmos 2 (0.74) > Qwen (0.69)**, subject to the 27-clip
small-sample caveats.
