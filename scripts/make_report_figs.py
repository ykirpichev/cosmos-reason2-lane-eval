"""Generate the schematic/quantitative figures for docs/cosmos3_report.md.

Figures (saved to docs/assets/cosmos3/):
  fig_budget.png   - the two-budget (temporal x spatial) thesis schematic
  fig_fps.png      - fps sweep: accuracy & lane_change F1 vs fps (+ cosmos2 baseline)
  fig_stages.png   - staged accuracy progression (final bar = pending target)

All numbers are sourced from docs/fps_sweep.md and results/cosmos_comparison.json.
Re-run after the final eval lands to refresh fig_stages with the realized number.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "cosmos3"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1b2a4a"
ACCENT = "#2c6fbb"
GOOD = "#2e8b57"
BAD = "#c0392b"
MUTED = "#7f8c8d"
plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def fig_budget() -> None:
    """Two input budgets. The y-axis is the *effective* spatial budget spent on the
    lane cue: a naive whole-frame upscale wastes tokens on sky/blur (stays low),
    while ROI-crop+zoom concentrates them on the road (reaches the sweet spot)."""
    AMBER = "#e08e0b"
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    ax.axvspan(7, 11, color=GOOD, alpha=0.06, zorder=0)
    ax.add_patch(plt.Rectangle((7, 0.66), 4, 0.36, color=GOOD, alpha=0.10,
                               zorder=0, ec="none"))

    # (label, fps, effective-spatial-y, accuracy, color, note, dx, dy, ha)
    pts = [
        ("native, 4 fps", 4, 0.20, 0.56, BAD, "below baseline", 10, -30, "center"),
        ("+fps & greedy, 8 fps", 8, 0.20, 0.78, ACCENT, "beats baseline", 14, -22, "left"),
        ("naive 2x upscale, 8 fps", 8, 0.52, 0.74, AMBER, "whole-frame blur, regresses", 16, 4, "left"),
        ("ROI-crop + zoom, 8 fps", 8, 0.82, 0.93, GOOD, "near-saturated", 16, 4, "left"),
    ]
    for label, x, y, acc, c, note, dx, dy, ha in pts:
        ax.scatter([x], [y], s=260, color=c, zorder=5, edgecolor="white", lw=1.5)
        ax.annotate(f"{label}  (acc {acc:.2f})", (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=9.0, fontweight="bold", color=c)
        ax.annotate(note, (x, y), textcoords="offset points",
                    xytext=(dx, dy - 12), ha=ha, fontsize=8.2, color=c, style="italic")

    # Stage 1 arrow (temporal); Stage 2 arrow offset to the left of the x=8 stack
    ax.add_patch(FancyArrowPatch((4, 0.20), (8, 0.20), arrowstyle="-|>",
                 mutation_scale=16, lw=2, color=INK, alpha=0.5, zorder=4,
                 shrinkA=15, shrinkB=15))
    ax.add_patch(FancyArrowPatch((7.5, 0.30), (7.5, 0.74), arrowstyle="-|>",
                 mutation_scale=16, lw=2, color=INK, alpha=0.5, zorder=4,
                 shrinkA=6, shrinkB=6))
    ax.text(6, 0.265, "Stage 1: temporal", ha="center", fontsize=8.5, color=INK)
    ax.text(7.25, 0.54, "Stage 2:\ntargeted\nspatial", ha="right", fontsize=8.5, color=INK)
    ax.text(10.4, 0.99, "sweet spot", ha="center", va="top", fontsize=9,
            color=GOOD, fontweight="bold")

    ax.set_xlim(2.2, 22)
    ax.set_ylim(0.02, 1.05)
    ax.set_xticks([4, 8, 10, 20])
    ax.set_xlabel("Temporal token budget  →  frames per second", fontweight="bold")
    ax.set_ylabel("Effective spatial budget  →  tokens on the lane cue",
                  fontweight="bold")
    ax.set_yticks([])
    ax.set_title("Two input budgets: over-sampled past 8 fps, "
                 "under-budgeted unless tokens hit the road",
                 fontsize=10.5, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_budget.png", bbox_inches="tight")
    plt.close(fig)


def fig_fps() -> None:
    """fps sweep, mean of two runs (docs/fps_sweep.md)."""
    fps = [4, 8, 10, 20]
    acc = [0.630, 0.741, 0.741, 0.648]
    f1 = [0.477, 0.664, 0.642, 0.458]
    c2_acc = 0.741  # cosmos2 native baseline

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axhline(c2_acc, color=MUTED, ls="--", lw=1.4, zorder=1)
    ax.annotate("Cosmos 2 baseline (acc 0.74)", (20, c2_acc),
                textcoords="offset points", xytext=(-4, 6), ha="right",
                fontsize=8.5, color=MUTED)
    ax.axvspan(7.4, 10.6, color=GOOD, alpha=0.08, zorder=0)
    ax.annotate("peak 8-10 fps", (9, 0.70), ha="center", fontsize=9,
                color=GOOD, fontweight="bold")

    ax.plot(fps, acc, "-o", color=ACCENT, lw=2.2, ms=8, label="accuracy", zorder=3)
    ax.plot(fps, f1, "-s", color=BAD, lw=2.2, ms=7,
            label="lane_change F1", zorder=3)
    for x, y in zip(fps, acc):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5, color=ACCENT)
    for x, y in zip(fps, f1):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8.5, color=BAD)

    ax.set_xticks(fps)
    ax.set_xlabel("frames per second (temporal token budget)", fontweight="bold")
    ax.set_ylabel("score (27 human-labeled clips)", fontweight="bold")
    ax.set_ylim(0.3, 0.85)
    ax.set_title("Stage 1: correcting the frame rate erases the Cosmos 3 deficit\n"
                 "mean of two runs; 20 fps is worst and ~2x the compute",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower center", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fps.png", bbox_inches="tight")
    plt.close(fig)


def fig_stages(final_acc: float | None = None) -> None:
    """Staged accuracy progression with the realized numbers."""
    AMBER = "#e08e0b"
    roi = final_acc if final_acc is not None else 0.926
    labels = ["Cosmos 2\nnative", "Cosmos 3\nnative\n(4 fps)",
              "Cosmos 3\n+fps & greedy\n(8 fps)", "Cosmos 3\n+naive 2x\nupscale",
              "Cosmos 3 final\nROI-zoom\n(8 fps)"]
    vals = [0.741, 0.556, 0.778, 0.741, roi]
    colors = [MUTED, BAD, ACCENT, AMBER, GOOD]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, vals, color=colors, width=0.66,
                  edgecolor="white", lw=1.2)
    ax.axhline(0.741, color=MUTED, ls="--", lw=1.2, zorder=0)
    ax.annotate("Cosmos 2 baseline", (4.45, 0.741), textcoords="offset points",
                xytext=(0, 4), ha="right", fontsize=8, color=MUTED)

    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=10, fontweight="bold", color=INK)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy (27 human-labeled clips)", fontweight="bold")
    ax.set_title("Staged recovery: a conditioning gap, not a capability gap",
                 fontsize=11, fontweight="bold")
    ax.annotate("worse than\nprevious gen", (1, 0.556),
                textcoords="offset points", xytext=(0, 22), ha="center",
                fontsize=8.3, color=BAD, fontweight="bold")
    ax.annotate("naive upscale\nregresses (+1 FP)", (3, 0.741),
                textcoords="offset points", xytext=(0, 22), ha="center",
                fontsize=8.3, color=AMBER, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stages.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # realized final accuracy = ROI-crop+zoom run (the genuine spatial fix)
    final_acc = None
    roi = ROOT / "results" / "exp_roi8" / "results.json"
    if roi.exists():
        try:
            final_acc = json.load(open(roi)).get("accuracy")
        except Exception:
            pass

    fig_budget()
    fig_fps()
    fig_stages(final_acc)
    print(f"wrote figures to {OUT}  (final_acc={'pending' if final_acc is None else final_acc})")


if __name__ == "__main__":
    main()
