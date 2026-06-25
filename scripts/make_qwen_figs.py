"""Figures for docs/qwen_report.md -> docs/assets/qwen/.

fig_qwen_ladder.png - Qwen 3.5 across the matched config ladder (4 fps native,
                      8 fps native, 8 fps + whole-frame 2x, 8 fps + ROI-zoom):
                      accuracy (bars) + lane-change recall (line).
fig_qwen_3way.png   - 3-way accuracy comparison Cosmos 2 vs Cosmos 3 vs Qwen 3.5
                      across the same four configs.

All numbers are read from results/headtohead.json so the figures are reproducible
(run scripts/headtohead.py first).
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

OUT = ROOT / "docs" / "assets" / "qwen"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1b2a4a"
ACCENT = "#2c6fbb"
GOOD = "#2e8b57"
BAD = "#c0392b"
MUTED = "#7f8c8d"
AMBER = "#e08e0b"
PURPLE = "#7d3c98"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

CONFIGS = ["4 fps native", "8 fps native",
           "8 fps + whole-frame 2x", "8 fps + ROI-zoom"]
SHORT = ["4 fps\nnative", "8 fps\nnative", "8 fps\n+2x upscale", "8 fps\n+ROI-zoom"]


def _rows() -> list[dict]:
    return json.load(open(ROOT / "results/headtohead.json"))["human27"]


def _get(rows: list[dict], model: str, cfg: str) -> dict | None:
    for r in rows:
        if r["model"] == model and r["config"] == cfg:
            return r
    return None


def fig_qwen_ladder() -> None:
    rows = _rows()
    runs = [_get(rows, "Qwen 3.5", c) for c in CONFIGS]
    acc = [r["accuracy"] if r else 0 for r in runs]
    rec = [(r["lane_change"]["recall"] or 0) if r else 0 for r in runs]

    best = int(np.argmax([a or 0 for a in acc]))
    colors = [GOOD if i == best else MUTED for i in range(len(CONFIGS))]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.bar(SHORT, acc, width=0.6, color=colors, edgecolor="white", lw=1.2)
    ax.plot(SHORT, rec, "-s", color=BAD, lw=2, ms=7, label="lane-change recall")
    ax.axhline(acc[best], color=GOOD, ls="--", lw=1.2, zorder=0)
    for b, v in zip(bars, acc):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=10, fontweight="bold", color=INK)
    for xi, v in zip(range(len(rec)), rec):
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8.6, color=BAD)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (27 human-labeled clips)", fontweight="bold")
    ax.set_title("Qwen 3.5 across the matched input-budget ladder\n"
                 "(accuracy = bars, lane-change recall = line)",
                 fontsize=10.8, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_qwen_ladder.png", bbox_inches="tight")
    plt.close(fig)


def fig_qwen_3way() -> None:
    rows = _rows()
    models = [("Cosmos 2", MUTED), ("Cosmos 3", ACCENT), ("Qwen 3.5", PURPLE)]
    vals = {m: [(_get(rows, m, c) or {}).get("accuracy") or 0 for c in CONFIGS]
            for m, _ in models}

    x = np.arange(len(CONFIGS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for i, (m, col) in enumerate(models):
        off = (i - 1) * w
        bars = ax.bar(x + off, vals[m], w, color=col, edgecolor="white", lw=1.0,
                      label=m)
        for b, v in zip(bars, vals[m]):
            if v:
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=8.2, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy (27 human-labeled clips)", fontweight="bold")
    ax.set_title("Lane-behavior accuracy across three reasoning VLMs\n"
                 "same matched configs; input-budget response is model-specific",
                 fontsize=10.8, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_qwen_3way.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_qwen_ladder()
    fig_qwen_3way()
    print(f"wrote qwen figures to {OUT}")


if __name__ == "__main__":
    main()
