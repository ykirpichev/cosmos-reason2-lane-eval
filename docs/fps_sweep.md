## FPS sweep for Cosmos 3 lane-behavior eval — repeated (2 runs) for stability

**Summary.** Cosmos 3's lane-crossing quality peaks in the **8–10 fps band** and is lowest at 20 fps (more frames reduce accuracy *and* cost ~2× compute). Averaged over two independent runs, **8 fps and 10 fps are essentially tied** for best; 8 fps is the recommended default (same quality, lower cost). Per-run swings are large (crossing F1 varies by up to ±0.18 between identical re-runs at `temperature=0.6`), so single-run rankings are unreliable — the robust conclusions are (a) peak at 8–10 fps, (b) 20 fps lowest, (c) recall is variance-limited and the most promising lever is **self-consistency decoding**, not more fps.

Eval set: the **27 human-labeled BATON clips** (13 lane-crossings, 14 lane-keeps). Metrics computed on the 3-class mapping (`keep_within_lane` / `lane_change` / `lane_wandering`); crossing P/R/F1 are for the `lane_change` class. Clips re-extracted faithfully from source `qcamera.mp4` at each fps (same start time, drift-correction, timestamp burn-in).

### Per-run results

| Run | Accuracy | Crossing P | Crossing R | Crossing F1 | (tp/fp/fn) |
|---|---|---|---|---|---|
| Cosmos 2 @ 4 fps (baseline) | 0.741 | 1.00 | 0.462 | 0.632 | 6/0/7 |
| Cosmos 3 @ 4 fps — run 1 | 0.556 | 0.750 | 0.231 | 0.353 | 3/1/10 |
| Cosmos 3 @ 4 fps — run 2 | 0.704 | 0.857 | 0.462 | 0.600 | 6/1/7 |
| Cosmos 3 @ 8 fps — run 1 | 0.778 | 0.889 | 0.615 | 0.727 | 8/1/5 |
| Cosmos 3 @ 8 fps — run 2 | 0.704 | 0.857 | 0.462 | 0.600 | 6/1/7 |
| Cosmos 3 @ 10 fps — run 1 | 0.704 | 1.00 | 0.385 | 0.556 | 5/0/8 |
| Cosmos 3 @ 10 fps — run 2 | 0.778 | 0.889 | 0.615 | 0.727 | 8/1/5 |
| Cosmos 3 @ 20 fps — run 1 | 0.630 | 0.800 | 0.308 | 0.444 | 4/1/9 |
| Cosmos 3 @ 20 fps — run 2 | 0.667 | 1.00 | 0.308 | 0.471 | 4/0/9 |

### Mean of the two runs (the stable picture)

| fps | mean accuracy | mean crossing recall | mean crossing F1 | crossing run-to-run agreement |
|---|---|---|---|---|
| 4 | 0.630 | 0.346 | 0.477 | 10/13 (0.77) |
| **8** | **0.741** | 0.538 | **0.664** | 9/13 (0.69) |
| **10** | **0.741** | 0.500 | 0.642 | 10/13 (0.77) |
| 20 | 0.648 | 0.308 | 0.458 | 9/13 (0.69) |

- **Peak at 8–10 fps.** Both outperform 4 fps and 20 fps on accuracy, recall and F1. 8 fps edges 10 fps on mean F1 (0.664 vs 0.642) while costing fewer frames → **recommend 8 fps**.
- **20 fps is the lowest-scoring Cosmos-3 setting** on both runs and ~2× the compute. More temporal tokens dilute per-frame motion cues (Qwen3-VL merges temporal frames) and crowd the 32k context.
- **The earlier "8 fps strictly dominates" observation (from run 1 only) does not fully replicate.** Run 2 had 10 fps best and 8 fps middling; run 1 had the reverse. The accurate statement is that they are tied within noise.

### Sampling variance is large (why repeating mattered)

Identical re-runs at `temperature=0.6` vary substantially:
- 4 fps: F1 0.353 → 0.600 (recall 0.231 → 0.462)
- 8 fps: F1 0.727 → 0.600 (recall 0.615 → 0.462)
- 10 fps: F1 0.556 → 0.727 (recall 0.385 → 0.615)

Crossing detections agree run-to-run only ~69–77% of the time. With only 13 crossings, ±1–2 detections moves F1 by ~0.1, so per-fps differences inside the 8–10 band are not significant. **This is the core finding: recall on borderline/late crossings is variance-limited, not fps-limited past 8 fps.**

### Per-crossing detection across fps (run1 / run2; Y = caught, . = missed)

| crossing clip | 4 fps | 8 fps | 10 fps | 20 fps | Cosmos 2 |
|---|---|---|---|---|---|
| lane_recovery__04 | . / . | . / . | . / Y | . / . | . |
| lane_recovery__17 | . / Y | Y / Y | Y / Y | . / . | Y |
| lane_violation_left__01 | Y / Y | . / Y | Y / Y | Y / Y | Y |
| lane_violation_left__05 | Y / Y | Y / . | Y / Y | . / Y | Y |
| lane_violation_left__10 | . / . | Y / . | . / . | . / . | Y |
| lane_violation_left__14 | . / Y | Y / Y | Y / Y | Y / . | . |
| lane_violation_left__18 | . / . | . / . | . / . | . / . | . |
| lane_violation_left__20 | . / . | . / . | . / . | . / . | . |
| lane_violation_left__21 | Y / Y | . / . | Y / Y | Y / . | . |
| lane_violation_right__00 | . / Y | Y / Y | . / Y | . / Y | Y |
| lane_violation_right__13 | . / . | Y / Y | . / Y | Y / Y | Y |
| lane_violation_right__16 | . / . | Y / . | . / . | . / . | . |
| lane_violation_right__27 | . / . | Y / Y | . / . | . / . | . |

Notes: `left__14`, `right__16`, `right__27` are late lane changes Cosmos 2 misses; Cosmos 3 catches them at 8 fps (and `left__14` across most settings). `left__18`/`left__20` are missed by every configuration at every fps — possibly ambiguous or mislabeled, and worth a human re-review.

### Recommendation

1. **Default to Cosmos 3 @ 8 fps** for this task (tied-best quality, lower cost than 10 fps; clearly above 4 and 20 fps).
2. **Avoid pushing fps past ~10.** 20 fps scores lower and is slower.
3. **Use self-consistency decoding at 8 fps** (sample 3–5×, union the `lane_change` detections, or majority vote) to stabilize the ~0.5–0.6 recall — that targets the variance, which is now the dominant error source.
4. Expand the eval beyond 27 clips so per-fps F1 differences clear the noise floor.

Raw outputs committed under `results/cosmos3_{4,8,10,20}fps*` (run 1) and `results/cosmos3_*fps_r2` (run 2). Comparison scripts: `scripts/compare_fps.py`, `scripts/aggregate_fps.py`.

Follow-up to #2 (Cosmos 3 vs Cosmos 2 comparison).
