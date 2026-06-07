#!/usr/bin/env python3
"""Score one Cosmos-Reason run on the 3-class lane-behavior task.

The three classes are the user-facing taxonomy:

* ``lane_keep``   — the ego stays inside its lane (model ``keep_within_lane``).
* ``lane_change`` — the ego crosses a line and settles in a different lane.
* ``other``       — *everything else* driving behavior (drift/wander/straddle,
                    recoveries, turns, ramps, merges, ...). Any predicted or
                    reference behavior that is not clearly keep/change collapses
                    here, so the bucket already absorbs richer labels if the
                    prompt taxonomy is later widened.

Predictions come from a run's ``summary.json`` (the ``overall_behavior`` of each
clip). Ground truth comes either from blind human labels or from each clip's
mined reference label. Both sides are mapped into the 3-class taxonomy and
scored with a full multi-class report: confusion matrix, per-class
precision / recall / F1 / support, overall accuracy, and macro / weighted F1.

Usage:
  # vs human labels (default), markdown report
  python scripts/evaluate.py --run results/cosmos3_10fps_r2/summary.json \
      --ground-truth human --format markdown

  # vs the mined reference label baked into each summary entry (full coverage)
  python scripts/evaluate.py --run results/cosmos3_10fps_r2/summary.json \
      --ground-truth reference

  # machine-readable, also written to disk
  python scripts/evaluate.py --run results/summary.json \
      --format json --out results/eval_3class.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# The 3-class taxonomy, in display order.
EVAL_CLASSES = ["lane_keep", "lane_change", "other"]

# Collapse every behavior vocabulary we know about into the 3-class taxonomy.
# Anything not listed here is "everything else" -> ``other`` (see ``to_eval``).
TO_EVAL: dict[str, str] = {
    # current model / new taxonomy
    "keep_within_lane": "lane_keep",
    "lane_change": "lane_change",
    "lane_wandering": "other",
    # old mined / human taxonomy
    "lane_keeping": "lane_keep",
    "lane_violation_left": "lane_change",
    "lane_violation_right": "lane_change",
    "lane_recovery": "other",
    # identity (already collapsed)
    "lane_keep": "lane_keep",
    "other": "other",
}


def to_eval(label: str | None) -> str | None:
    """Map any behavior label into the 3-class taxonomy.

    Unknown / unmapped behaviors are treated as ``other`` ("everything else").
    ``None`` (no usable label) stays ``None`` so it can be filtered out.
    """
    if not label:
        return None
    return TO_EVAL.get(label.strip(), "other")


def load_predictions(path: Path) -> tuple[dict[str, str], int]:
    """Return ``{clip_id: eval_class}`` plus a count of clips with no prediction."""
    data = json.loads(path.read_text())
    preds: dict[str, str] = {}
    missing = 0
    for entry in data:
        parsed = entry.get("parsed")
        behavior = config.overall_behavior(parsed) if parsed else None
        mapped = to_eval(behavior)
        if mapped is None:
            missing += 1
            continue
        preds[entry["id"]] = mapped
    return preds, missing


def load_ground_truth(
    args: argparse.Namespace,
) -> tuple[dict[str, str], set[str]]:
    """Return ``{clip_id: eval_class}`` ground truth and the set of unclear ids."""
    unclear: set[str] = set()
    truth: dict[str, str] = {}

    if args.ground_truth == "human":
        labels = json.loads(args.human.read_text())
        for cid, value in labels.items():
            if value.get("unclear"):
                unclear.add(cid)
            mapped = to_eval(value.get("behavior"))
            if mapped is not None:
                truth[cid] = mapped
    else:  # "reference": use the mined ground-truth label baked into the run
        data = json.loads(args.run.read_text())
        for entry in data:
            # ground_truth looks like "lane_keeping / curved"; take the behavior.
            raw = (entry.get("ground_truth") or "").split("/")[0].strip()
            mapped = to_eval(raw)
            if mapped is not None:
                truth[entry["id"]] = mapped
    return truth, unclear


def confusion_matrix(
    pairs: list[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    """Build ``conf[actual][pred]`` over the 3 classes."""
    conf = {a: {p: 0 for p in EVAL_CLASSES} for a in EVAL_CLASSES}
    for actual, pred in pairs:
        conf[actual][pred] += 1
    return conf


def per_class_metrics(conf: dict[str, dict[str, int]]) -> dict[str, dict]:
    """Precision / recall / F1 / support per class from a confusion matrix."""
    out: dict[str, dict] = {}
    for c in EVAL_CLASSES:
        tp = conf[c][c]
        support = sum(conf[c][p] for p in EVAL_CLASSES)          # actual == c
        predicted = sum(conf[a][c] for a in EVAL_CLASSES)        # pred == c
        fp = predicted - tp
        fn = support - tp
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        if precision and recall:
            f1 = 2 * precision * recall / (precision + recall)
        elif support == 0 and predicted == 0:
            f1 = None  # class absent from both sides
        else:
            f1 = 0.0
        out[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def build_report(args: argparse.Namespace) -> dict:
    preds, missing = load_predictions(args.run)
    truth, unclear = load_ground_truth(args)

    drop_unclear = args.ground_truth == "human" and not args.include_unclear

    ids = sorted(c for c in truth if c in preds)
    if drop_unclear:
        ids = [c for c in ids if c not in unclear]

    pairs = [(truth[c], preds[c]) for c in ids]
    conf = confusion_matrix(pairs)
    per_class = per_class_metrics(conf)

    n = len(ids)
    correct = sum(1 for a, p in pairs if a == p)
    accuracy = correct / n if n else None

    # macro: unweighted mean of per-class F1 over classes present in truth.
    present = [c for c in EVAL_CLASSES if per_class[c]["support"] > 0]
    macro_f1 = (
        sum((per_class[c]["f1"] or 0.0) for c in present) / len(present)
        if present
        else None
    )
    weighted_f1 = (
        sum((per_class[c]["f1"] or 0.0) * per_class[c]["support"] for c in present) / n
        if n
        else None
    )

    errors = [
        {"id": c, "actual": truth[c], "pred": preds[c]}
        for c in ids
        if truth[c] != preds[c]
    ]

    return {
        "run": str(args.run),
        "ground_truth": args.ground_truth,
        "classes": EVAL_CLASSES,
        "n_scored": n,
        "n_predictions_missing": missing,
        "n_unclear_dropped": len([c for c in (unclear if drop_unclear else set()) if c in truth and c in preds]),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion": conf,
        "errors": errors,
    }


# --- rendering ---------------------------------------------------------------

def _fmt(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def render_plain(rep: dict) -> str:
    lines = [
        f"run={rep['run']}  vs {rep['ground_truth']}  "
        f"(n={rep['n_scored']}, missing_pred={rep['n_predictions_missing']}, "
        f"unclear_dropped={rep['n_unclear_dropped']})",
        f"accuracy={_fmt(rep['accuracy'])}  "
        f"macro_f1={_fmt(rep['macro_f1'])}  weighted_f1={_fmt(rep['weighted_f1'])}",
        "",
        f"{'class':<12} {'P':>6} {'R':>6} {'F1':>6} {'support':>8}",
    ]
    for c in EVAL_CLASSES:
        m = rep["per_class"][c]
        lines.append(
            f"{c:<12} {_fmt(m['precision']):>6} {_fmt(m['recall']):>6} "
            f"{_fmt(m['f1']):>6} {m['support']:>8}"
        )
    lines += ["", "confusion (actual \\ pred):"]
    header = " " * 14 + "".join(f"{c:>12}" for c in EVAL_CLASSES)
    lines.append(header)
    for a in EVAL_CLASSES:
        row = "".join(f"{rep['confusion'][a][p]:>12}" for p in EVAL_CLASSES)
        lines.append(f"{a:<14}{row}")
    if rep["errors"]:
        lines += ["", f"misclassified ({len(rep['errors'])}):"]
        for e in rep["errors"]:
            lines.append(f"  {e['id']:<32} actual={e['actual']:<11} pred={e['pred']}")
    return "\n".join(lines)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(rep: dict) -> str:
    blocks: list[str] = []
    blocks.append(
        f"## Cosmos lane-behavior eval (3-class) — vs {rep['ground_truth']}\n\n"
        f"- run: `{rep['run']}`\n"
        f"- scored clips: **{rep['n_scored']}** "
        f"(missing predictions: {rep['n_predictions_missing']}, "
        f"unclear dropped: {rep['n_unclear_dropped']})\n"
        f"- **accuracy: {_fmt(rep['accuracy'])}**, "
        f"macro F1: {_fmt(rep['macro_f1'])}, "
        f"weighted F1: {_fmt(rep['weighted_f1'])}"
    )

    rows = []
    for c in EVAL_CLASSES:
        m = rep["per_class"][c]
        rows.append([f"`{c}`", _fmt(m["precision"]), _fmt(m["recall"]),
                     _fmt(m["f1"]), str(m["support"])])
    blocks.append("### Per-class metrics\n\n"
                  + _md_table(["class", "precision", "recall", "F1", "support"], rows))

    crows = [[f"`{a}`"] + [str(rep["confusion"][a][p]) for p in EVAL_CLASSES]
             for a in EVAL_CLASSES]
    blocks.append("### Confusion matrix (actual \\ pred)\n\n"
                  + _md_table(["actual \\ pred"] + EVAL_CLASSES, crows))

    if rep["errors"]:
        erows = [[f"`{e['id']}`", e["actual"], e["pred"]] for e in rep["errors"]]
        blocks.append(f"### Misclassified ({len(rep['errors'])})\n\n"
                      + _md_table(["clip", "actual", "pred"], erows))
    return "\n\n".join(blocks)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, default=config.SUMMARY,
                    help="run summary.json with parsed predictions")
    ap.add_argument("--ground-truth", choices=["human", "reference"], default="human",
                    help="'human' = blind human labels; 'reference' = mined label "
                         "baked into each summary entry")
    ap.add_argument("--human", type=Path,
                    default=config.RESULTS_DIR / "human_labels_old_taxonomy.json",
                    help="human label file (used when --ground-truth human)")
    ap.add_argument("--include-unclear", action="store_true",
                    help="keep clips the human marked 'unclear' (default: drop them)")
    ap.add_argument("--format", choices=["plain", "markdown", "json"], default="plain")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the full JSON report here")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run.exists():
        print(f"error: run summary not found: {args.run}", file=sys.stderr)
        return 1
    if args.ground_truth == "human" and not args.human.exists():
        print(f"error: human labels not found: {args.human}", file=sys.stderr)
        return 1

    rep = build_report(args)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2))

    if args.format == "json":
        print(json.dumps(rep, indent=2))
    elif args.format == "markdown":
        print(render_markdown(rep))
    else:
        print(render_plain(rep))

    if args.out and args.format != "json":
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
