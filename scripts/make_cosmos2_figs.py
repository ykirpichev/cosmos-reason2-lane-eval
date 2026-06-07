"""Figures for docs/cosmos2_report.md -> docs/assets/cosmos2/.

fig_c2_fps.png    - controlled 1 fps vs 4 fps experiment (matched prompt/old taxonomy):
                    accuracy + lane-change recall on the 27 human-labeled clips.
fig_c2_ladder.png - the config ladder (4 fps native, 8 fps, 8 fps+2x, ROI-zoom):
                    Cosmos 2 plateaus/regresses (new-taxonomy runs via headtohead.json).

All numbers are recomputed from results/*.json so the figures are reproducible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import config  # noqa: E402

OUT = ROOT / "docs" / "assets" / "cosmos2"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1b2a4a"
ACCENT = "#2c6fbb"
GOOD = "#2e8b57"
BAD = "#c0392b"
MUTED = "#7f8c8d"
AMBER = "#e08e0b"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

HM = {"lane_keeping": "keep_within_lane", "lane_recovery": "lane_wandering",
      "lane_violation_left": "lane_change", "lane_violation_right": "lane_change"}


def _human() -> dict:
    hr = json.load(open(ROOT / "results/human_labels_old_taxonomy.json"))
    return {k: HM[v["behavior"]] for k, v in hr.items() if v.get("behavior") in HM}


def _metrics(pred: dict, gt: dict) -> dict:
    ids = [i for i in gt if pred.get(i)]
    acc = sum(1 for i in ids if pred[i] == gt[i]) / len(ids) if ids else 0
    tp = sum(1 for i in ids if gt[i] == "lane_change" and pred[i] == "lane_change")
    fn = sum(1 for i in ids if gt[i] == "lane_change" and pred[i] != "lane_change")
    rec = tp / (tp + fn) if (tp + fn) else 0
    return {"acc": acc, "recall": rec, "tp": tp, "fn": fn, "n": len(ids)}


def _preds_old(path: Path) -> dict:
    """Old-taxonomy single-label runs (summary_fps1, summary_oldtaxonomy)."""
    return {e["id"]: HM.get((e.get("parsed") or {}).get("behavior"))
            for e in json.load(open(path))}


def fig_c2_fps() -> None:
    human = _human()
    one = _metrics(_preds_old(ROOT / "results/summary_fps1.json"), human)
    four = _metrics(_preds_old(ROOT / "results/summary_oldtaxonomy.json"), human)

    labels = ["1 fps", "4 fps"]
    acc = [one["acc"], four["acc"]]
    rec = [one["recall"], four["recall"]]
    x = np.arange(2)
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    b1 = ax.bar(x - w / 2, acc, w, color=ACCENT, edgecolor="white", lw=1.2,
                label="accuracy")
    b2 = ax.bar(x + w / 2, rec, w, color=BAD, edgecolor="white", lw=1.2,
                label="lane-change recall")
    for bars, vals, src in [(b1, acc, [one, four]), (b2, rec, [one, four])]:
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=10, fontweight="bold", color=INK)
    # recall fraction callouts
    ax.annotate(f"{one['tp']}/{one['tp']+one['fn']} crossings", (1 - 0 + w / 2 - 1, rec[0]),
                textcoords="offset points", xytext=(0, 18), ha="center",
                fontsize=8.2, color=BAD)
    ax.annotate(f"{four['tp']}/{four['tp']+four['fn']} crossings", (1 + w / 2, rec[1]),
                textcoords="offset points", xytext=(0, 18), ha="center",
                fontsize=8.2, color=BAD)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("score (27 human-labeled clips)", fontweight="bold")
    ax.set_title("Cosmos 2: 1 fps undersamples the ~1 s crossing event\n"
                 "4 fps roughly doubles lane-change recall (matched prompt)",
                 fontsize=10.8, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_c2_fps.png", bbox_inches="tight")
    plt.close(fig)


def fig_c2_ladder() -> None:
    h2h = ROOT / "results/headtohead.json"
    if not h2h.exists():
        return
    rows = {r["config"]: r for r in json.load(open(h2h))["human27"]
            if r["model"] == "Cosmos 2"}
    order = ["4 fps native", "8 fps native",
             "8 fps + whole-frame 2x", "8 fps + ROI-zoom"]
    short = ["4 fps\nnative", "8 fps\nnative", "8 fps\n+2x blur", "8 fps\n+ROI-zoom"]
    acc = [rows[c]["accuracy"] for c in order]
    rec = [rows[c]["lane_change"]["recall"] for c in order]
    colors = [GOOD, MUTED, AMBER, MUTED]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.bar(short, acc, width=0.6, color=colors, edgecolor="white", lw=1.2)
    ax.plot(short, rec, "-s", color=BAD, lw=2, ms=7, label="lane-change recall")
    ax.axhline(acc[0], color=GOOD, ls="--", lw=1.2, zorder=0)
    ax.annotate("native 4 fps is the ceiling", (3.4, acc[0]),
                textcoords="offset points", xytext=(0, 5), ha="right",
                fontsize=8.4, color=GOOD)
    for b, v in zip(bars, acc):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=10, fontweight="bold", color=INK)
    for xi, v in zip(range(len(rec)), rec):
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8.6, color=BAD)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (27 human-labeled clips)", fontweight="bold")
    ax.set_title("Cosmos 2 plateaus: every lever beyond native 4 fps is flat or worse\n"
                 "(accuracy = bars, lane-change recall = line)",
                 fontsize=10.8, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_c2_ladder.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_c2_fps()
    fig_c2_ladder()
    print(f"wrote cosmos 2 figures to {OUT}")


if __name__ == "__main__":
    main()
