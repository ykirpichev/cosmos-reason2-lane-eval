#!/usr/bin/env python3
"""Derive a best-effort 3-class pseudo-label from the stored openpilot offset trace.

Classes (lateral behavior only):
  - keep_within_lane : never crosses a lane line
  - lane_change      : crosses a line and settles in a different lane (sustained shift)
  - lane_wandering   : crosses/rides a line but returns to the original lane

The raw openpilot ``signed_lateral_m`` trace (offset to nearest lane line, +right)
contains end-of-clip line-swap artifacts: physically impossible single-sample jumps
(>~5 m/s lateral). We gate those out before classifying. Because openpilot recenters
on the new lane after a change, change-vs-wander is only weakly separable from offset
alone, so this label is a rough reference -- Cosmos + human labels are ground truth.

Writes ``pseudo_3class`` onto each clip in clips/manifest_all.json.

Run:
    .venv/bin/python scripts/remap_pseudo_3class.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

MANIFEST = config.MANIFEST

DT = 0.25  # s between samples (4 Hz)
MAX_LAT_VEL = 5.0  # m/s; |Δoffset|/DT above this is a lane-detection artifact
CROSS_M = 1.50  # |offset| at/above this = riding/crossing a lane line
CHANGE_NET_M = 1.20  # sustained start->end shift that implies a completed lane change


def degate(offset: np.ndarray) -> np.ndarray:
    """Replace impossible single-sample jumps with the last physically-plausible value."""
    o = offset.astype(float).copy()
    max_step = MAX_LAT_VEL * DT
    for i in range(1, len(o)):
        if abs(o[i] - o[i - 1]) > max_step:
            o[i] = o[i - 1]  # forward-fill across the artifact
    return o


def classify(offset: np.ndarray) -> str:
    if offset.size < 8:
        return "keep_within_lane"
    o = degate(offset)
    peak = float(np.max(np.abs(o)))
    if peak < CROSS_M:
        return "keep_within_lane"
    start = float(np.median(o[:4]))
    end = float(np.median(o[-4:]))
    if abs(end - start) >= CHANGE_NET_M:
        return "lane_change"
    return "lane_wandering"


def main() -> int:
    data = json.loads(MANIFEST.read_text())
    counts: dict[str, int] = {}
    for c in data["clips"]:
        sig = (c.get("metrics", {}) or {}).get("signed_lateral_m", [])
        label = classify(np.asarray(sig, dtype=float)) if sig else "keep_within_lane"
        c["pseudo_3class"] = label
        counts[label] = counts.get(label, 0) + 1
    MANIFEST.write_text(json.dumps(data, indent=2))
    print(f"Wrote pseudo_3class for {len(data['clips'])} clips -> {MANIFEST.name}")
    for k in ("keep_within_lane", "lane_change", "lane_wandering"):
        print(f"  {k}: {counts.get(k, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
