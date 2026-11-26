# 3-class lane-behavior evaluation

`scripts/evaluate.py` scores a single Cosmos-Reason run on the three behaviors
the eval cares about:

| class | meaning | model behaviors that map here |
|---|---|---|
| `lane_keep` | ego stays inside its lane | `keep_within_lane`, `lane_keeping` |
| `lane_change` | ego crosses a line and settles in a different lane | `lane_change`, `lane_violation_left`, `lane_violation_right` |
| `other` | **everything else** driving behavior | `lane_wandering`, `lane_recovery`, turns, ramps, merges, … |

The `other` bucket is a catch-all: any predicted or reference behavior that is
not clearly keep/change collapses into it (`to_eval()` defaults unknown labels
to `other`). So if the prompt taxonomy is later widened to emit richer
behaviors, the scorer already buckets them correctly without changes.

## Inputs

* **Predictions** — a run's `summary.json` (written by `scripts/run_batch.py`).
  Each clip's `parsed.overall_behavior` is mapped into the 3-class taxonomy.
* **Ground truth** — one of:
  * `--ground-truth human` (default): blind human labels
    (`results/human_labels_old_taxonomy.json`); clips marked `unclear` are
    dropped unless `--include-unclear` is given.
  * `--ground-truth reference`: the mined `ground_truth` label baked into each
    `summary.json` entry. This covers every clip in the run (not just the
    human-labeled subset) and is the only source that exercises the `other`
    class today, since the current human labels collapse to keep/change.

## Usage

```bash
# vs human labels, markdown report
python scripts/evaluate.py --run results/cosmos3_10fps_r2/summary.json \
    --ground-truth human --format markdown

# vs the mined reference label (full coverage, exercises `other`)
python scripts/evaluate.py --run results/cosmos3_10fps_r2/summary.json \
    --ground-truth reference

# machine-readable, also written to disk
python scripts/evaluate.py --run results/summary.json \
    --format json --out results/eval_3class.json
```

`--format` is `plain` (default), `markdown`, or `json`. The report contains
overall accuracy, macro / weighted F1, per-class precision / recall / F1 /
support, the full confusion matrix, and the list of misclassified clips.

## Example: Cosmos-Reason2-32B frame-rate sweep (vs human, n=24)

Scoring the existing BATON runs shows accuracy peaks at 10 Hz sampling:

| run | accuracy | macro F1 |
|---|---|---|
| `cosmos3` (4 Hz) | 0.583 | 0.542 |
| `cosmos3_4fps_r2` | 0.708 | 0.709 |
| `cosmos3_8fps_r2` | 0.750 | 0.762 |
| `cosmos3_10fps_r2` | **0.792** | **0.788** |
| `cosmos3_20fps_r2` | 0.667 | 0.637 |

The dominant error mode is missed `lane_change` (predicted as `lane_keep`):
Cosmos tends to under-call gradual crossings. Use `--format markdown` to see the
per-clip misclassification table.
