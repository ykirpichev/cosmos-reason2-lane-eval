#!/usr/bin/env python3
"""Consolidate the Cosmos 2 vs Cosmos 3 lane-behavior results into one JSON.

Scores every available run against (a) the 27 human-labeled clips and (b) the
full 159-clip manifest's openpilot pseudo-labels, using the SAME taxonomy
normalization / precedence as scoring (config.overall_behavior). Writes
results/headtohead.json, which docs/cosmos3_report.md and make_report_figs.py
consume so every number in the report is reproducible from disk.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

HM = {
    "lane_keeping": "keep_within_lane",
    "lane_recovery": "lane_wandering",
    "lane_violation_left": "lane_change",
    "lane_violation_right": "lane_change",
}
ROOT = Path(__file__).resolve().parent.parent


def metrics(pred: dict, gt: dict) -> dict:
    ids = [i for i in gt if pred.get(i)]
    correct = sum(1 for i in ids if pred[i] == gt[i])
    tp = sum(1 for i in ids if gt[i] == "lane_change" and pred[i] == "lane_change")
    fn = sum(1 for i in ids if gt[i] == "lane_change" and pred[i] != "lane_change")
    fp = sum(1 for i in ids if gt[i] != "lane_change" and pred[i] == "lane_change")
    rec = tp / (tp + fn) if (tp + fn) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {
        "n": len(ids),
        "accuracy": round(correct / len(ids), 3) if ids else None,
        "lane_change": {
            "tp": tp, "fn": fn, "fp": fp,
            "recall": round(rec, 3) if rec is not None else None,
            "precision": round(prec, 3) if prec is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
        },
    }


def preds_from_summary(path: Path) -> dict:
    data = json.load(open(path))
    return {e["id"]: config.overall_behavior(e.get("parsed") or {}) for e in data}


def preds_from_results(path: Path) -> dict:
    return json.load(open(path)).get("preds", {})


def main() -> None:
    hr = json.load(open(ROOT / "results/human_labels_old_taxonomy.json"))
    human = {k: HM[v["behavior"]] for k, v in hr.items() if v.get("behavior") in HM}
    man = {c["id"]: c for c in json.load(open(ROOT / "clips/manifest_all.json"))["clips"]}
    pseudo = {cid: HM[c["behavior"]] for cid, c in man.items() if c.get("behavior") in HM}

    # (model, config, path, kind)
    runs = [
        ("Cosmos 2", "4 fps native", "results/summary.json", "summary"),
        ("Cosmos 2", "8 fps native", "results/cosmos2_final_8fps_native/summary.json", "summary"),
        ("Cosmos 2", "8 fps + whole-frame 2x", "results/cosmos2_final_8fps2x/summary.json", "summary"),
        ("Cosmos 2", "8 fps + ROI-zoom", "results/cosmos2_roi8/results.json", "results"),
        ("Cosmos 3", "4 fps native", "results/cosmos3/summary.json", "summary"),
        ("Cosmos 3", "8 fps native", "results/cosmos3_final_8fps_native/summary.json", "summary"),
        ("Cosmos 3", "8 fps + whole-frame 2x", "results/cosmos3_final_8fps2x/summary.json", "summary"),
        ("Cosmos 3", "8 fps + ROI-zoom", "results/exp_roi8/results.json", "results"),
        ("Qwen 3.5", "4 fps native", "results/qwen_4fps_native/summary.json", "summary"),
        ("Qwen 3.5", "8 fps native", "results/qwen_8fps_native/summary.json", "summary"),
        ("Qwen 3.5", "8 fps + whole-frame 2x", "results/qwen_8fps2x/summary.json", "summary"),
        ("Qwen 3.5", "8 fps + ROI-zoom", "results/qwen_roi8/results.json", "results"),
    ]
    full = [
        ("Cosmos 2", "8 fps + ROI-zoom (full set)", "results/cosmos2_roi8_full159/results.json"),
        ("Cosmos 3", "8 fps + ROI-zoom (full set)", "results/cosmos3_roi8_full159/results.json"),
        ("Qwen 3.5", "8 fps + ROI-zoom (full set)", "results/qwen_roi8_full159/results.json"),
    ]

    out = {"human27": [], "full159": []}
    for model, cfg, rel, kind in runs:
        p = ROOT / rel
        if not p.exists():
            continue
        pred = preds_from_summary(p) if kind == "summary" else preds_from_results(p)
        out["human27"].append({"model": model, "config": cfg, **metrics(pred, human)})

    for model, cfg, rel in full:
        p = ROOT / rel
        if not p.exists():
            continue
        pred = preds_from_results(p)
        row = {"model": model, "config": cfg, "pseudo": metrics(pred, pseudo),
               "human27": metrics(pred, human)}
        out["full159"].append(row)

    (ROOT / "results/headtohead.json").write_text(json.dumps(out, indent=2))
    print("=== 27 human-labeled clips ===")
    print(f"{'model':9s} {'config':26s} {'n':>3s} {'acc':>6s} {'LCrec':>6s} {'LCf1':>6s}")
    for r in out["human27"]:
        lc = r["lane_change"]
        print(f"{r['model']:9s} {r['config']:26s} {r['n']:3d} {r['accuracy']!s:>6s} "
              f"{lc['recall']!s:>6s} {lc['f1']!s:>6s}")
    if out["full159"]:
        print("\n=== full 159-clip (openpilot pseudo-labels) ===")
        for r in out["full159"]:
            ps = r["pseudo"]
            print(f"{r['model']:9s} {r['config']:26s} n={ps['n']:3d} acc={ps['accuracy']} "
                  f"LCrec={ps['lane_change']['recall']}")
    print("\nwrote results/headtohead.json")


if __name__ == "__main__":
    main()
