#!/usr/bin/env python3
"""Temporal-windowing experiment: split each 12 s clip into 3 overlapping 6 s
windows (t=0-6, 3-9, 6-12), run Cosmos on each (greedy), and aggregate the
per-window behaviors via severity-max (lane_change/lane_wandering outrank
keep_within_lane).

Compares the aggregate against the single-shot 12 s prediction. Tests:
  - a known miss   (lane_violation_left__14, true lane_change, late crossing)
  - a regression   (lane_recovery__17,        true lane_change)
  - a false-pos check (lane_keeping__straight__02, true keep_within_lane)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from exp_variant import make_variant, run_clip, OUTDIR as _VOUT  # noqa: E402

OUTDIR = config.RESULTS_DIR / "exp_split"
SEV = config.BEHAVIOR_SEVERITY

# clip id -> (route, t0, truth)
CLIPS = {
    "lane_violation_left__14": ("driver_97/route_21", 2168.0, "lane_change"),
    "lane_recovery__17":       ("driver_97/route_21", 3698.0, "lane_change"),
    "lane_keeping__straight__02": (None, None, "keep_within_lane"),
}

WIN_DUR = 6.0
WIN_FPS = 4.0
WIN_STARTS = [0.0, 3.0, 6.0]  # overlapping 6 s windows over a 12 s clip


def aggregate(behaviors: list[str]) -> str:
    valid = [b for b in behaviors if b]
    if not valid:
        return "—"
    return max(valid, key=lambda b: SEV.get(b, 0))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    man = {c["id"]: c for c in json.loads(config.MANIFEST.read_text())["clips"]}
    results = {}
    for cid, (route, t0, truth) in CLIPS.items():
        if route is None:  # pull route/t0 from manifest
            c = man[cid]
            route = c["scene"]
            t0 = c["start_timestamp_us"] / 1e6
        print(f"\n=== {cid}  (truth={truth}, route={route}, t0={t0}) ===", flush=True)
        win_overall = []
        for i, ws in enumerate(WIN_STARTS):
            out = OUTDIR / f"{cid}__w{i}.mp4"
            if not make_variant(route, t0 + ws, WIN_DUR, WIN_FPS, out):
                print(f"  w{i}: BUILD_FAILED"); win_overall.append(None); continue
            r = run_clip(out, WIN_DUR, WIN_FPS)
            win_overall.append(r["overall"])
            print(f"  w{i} t={ws:.0f}-{ws+WIN_DUR:.0f}s -> {r['overall']} ({r['elapsed']}s)", flush=True)
        agg = aggregate(win_overall)
        mark = "OK" if agg == truth else "xx"
        results[cid] = {"truth": truth, "windows": win_overall, "aggregate": agg}
        print(f"  AGGREGATE: {mark} {agg}   (windows={win_overall})", flush=True)
    (OUTDIR / "results.json").write_text(json.dumps(results, indent=2))
    print("\nSummary:")
    for cid, r in results.items():
        mark = "OK" if r["aggregate"] == r["truth"] else "xx"
        print(f"  {mark} {cid:30s} truth={r['truth']:16s} split={r['aggregate']:16s} windows={r['windows']}")


if __name__ == "__main__":
    main()
