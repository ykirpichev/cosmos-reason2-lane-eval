#!/usr/bin/env python3
"""Mine lane-behavior clips from the BATON-Sample dataset (HenryYHW/BATON-Sample).

BATON exposes openpilot's *production* lane model in ``planning.csv`` -
``laneLeft_y`` / ``laneRight_y`` give the lateral distance to each lane line in
the ego frame, so the signed offset of the car from lane center is a real,
continuous signal (unlike the nuScenes nearest-lane-centerline snapping that
produced phantom violations). We slide a 12 s window over each route, classify
the window from that offset signal, balance categories, then cut the matching
slice of the synchronized ``qcamera.mp4`` into a 12 s @ 4 Hz browser-playable
clip.

Categories produced (behavior / geometry):
  - lane_keeping            : stays near center, low curvature
  - lane_keeping (curved)   : stays near center while turning  -> road_geometry=curved
  - lane_recovery           : drifts toward a line then returns to center
  - lane_violation_left     : sustained offset over the left line
  - lane_violation_right    : sustained offset over the right line
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from video_utils import transcode_browser_mp4  # noqa: E402

REPO = "HenryYHW/BATON-Sample"
CLIPS_DIR = config.CLIPS_DIR
VIDEO_OUT = CLIPS_DIR / "baton"
MANIFEST = config.MANIFEST

CLIP_SECONDS = config.CLIP_SECONDS
CLIP_FPS = config.CLIP_FPS
FRAMES_PER_CLIP = config.FRAMES_PER_CLIP  # 48
SAMPLE_DT = 1.0 / CLIP_FPS  # 0.25 s
SCAN_STRIDE_S = 2.0

# Lane-offset thresholds (meters). Lane half-width ~1.6 m.
KEEP_PEAK_M = 0.35
RECOVERY_PEAK_MIN_M = 0.75
RECOVERY_END_MAX_M = 0.35
RECOVERY_START_MAX_M = 0.45
VIOLATION_PEAK_M = 1.00
VIOLATION_END_M = 0.70
CURVED_KAPPA = 0.0030  # 1/m mean |desiredCurvature|
MIN_SPEED_MPS = 5.0

# Set to -1.0 if a manual video check shows left/right are swapped.
OFFSET_SIGN = 1.0  # positive offset => drift toward the RIGHT line


@dataclass
class RouteSignals:
    route: str
    car_model: str
    t: np.ndarray
    offset: np.ndarray  # signed lateral offset from lane center (m), +right
    kappa: np.ndarray  # |desiredCurvature| (1/m)
    kappa_signed: np.ndarray  # signed desiredCurvature (1/m)
    laneL_prob: np.ndarray
    laneR_prob: np.ndarray
    speed: np.ndarray  # m/s
    video_path: str
    video_dur_s: float
    plan_dur_s: float


@dataclass
class Candidate:
    route: str
    car_model: str
    t0: float
    behavior: str
    geometry: str
    score: float
    samples_t: np.ndarray
    offset: np.ndarray
    steering: np.ndarray
    speed: np.ndarray
    kappa: np.ndarray
    metrics: dict = field(default_factory=dict)

    @property
    def category(self) -> str:
        if self.behavior == "lane_keeping":
            return f"lane_keeping__{self.geometry}"
        return self.behavior


def list_video_routes() -> list[str]:
    fs = HfFileSystem()
    base = f"datasets/{REPO}"
    routes: list[str] = []
    for d in fs.ls(base, detail=False):
        name = d.split("/")[-1]
        if not name.startswith("driver_"):
            continue
        for r in fs.ls(d, detail=False):
            files = {x.split("/")[-1] for x in fs.ls(r, detail=False)}
            if {"qcamera.mp4", "planning.csv"} <= files:
                routes.append(f"{name}/{r.split('/')[-1]}")
    return sorted(routes)


def _interp(t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(v_src)
    if mask.sum() < 2:
        return np.full_like(t_dst, np.nan, dtype=float)
    return np.interp(t_dst, t_src[mask], v_src[mask])


def load_route(route: str) -> RouteSignals | None:
    try:
        plan = pd.read_csv(hf_hub_download(REPO, f"{route}/planning.csv", repo_type="dataset"))
    except Exception:
        return None
    if "laneLeft_y" not in plan or plan["laneLeft_y"].isna().all():
        return None
    meta = json.loads(Path(hf_hub_download(REPO, f"{route}/metadata.json", repo_type="dataset")).read_text())
    video_path = hf_hub_download(REPO, f"{route}/qcamera.mp4", repo_type="dataset")

    t = plan["time_s"].to_numpy(dtype=float)
    offset = OFFSET_SIGN * (-(plan["laneLeft_y"] + plan["laneRight_y"]) / 2.0).to_numpy(dtype=float)
    kappa_signed = plan["model_desiredCurvature"].to_numpy(dtype=float)
    kappa = np.abs(kappa_signed)
    laneL = plan.get("laneLeft_prob", pd.Series(np.ones(len(plan)))).to_numpy(dtype=float)
    laneR = plan.get("laneRight_prob", pd.Series(np.ones(len(plan)))).to_numpy(dtype=float)

    # speed from localization velocity (high rate); fall back to gps.
    speed_t, speed_v = t, np.full_like(t, np.nan)
    try:
        loc = pd.read_csv(hf_hub_download(REPO, f"{route}/localization.csv", repo_type="dataset"))
        speed_t = loc["time_s"].to_numpy(dtype=float)
        speed_v = np.sqrt(loc["vel_x"].to_numpy(float) ** 2 + loc["vel_y"].to_numpy(float) ** 2)
    except Exception:
        try:
            gps = pd.read_csv(hf_hub_download(REPO, f"{route}/gps.csv", repo_type="dataset"))
            speed_t = gps["time_s"].to_numpy(dtype=float)
            speed_v = gps["gps_speed"].to_numpy(dtype=float)
        except Exception:
            pass
    speed = _interp(speed_t, speed_v, t)

    cap = cv2.VideoCapture(video_path)
    video_dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1.0)
    cap.release()

    return RouteSignals(
        route=route,
        car_model=meta.get("car_model", "?"),
        t=t,
        offset=offset,
        kappa=kappa,
        kappa_signed=kappa_signed,
        laneL_prob=laneL,
        laneR_prob=laneR,
        speed=speed,
        video_path=video_path,
        video_dur_s=float(video_dur),
        plan_dur_s=float(t[-1]),
    )


def classify(off: np.ndarray, kap: np.ndarray, spd: np.ndarray) -> tuple[str, str, float] | None:
    if np.isnan(off).mean() > 0.2:
        return None
    mean_speed = float(np.nanmean(spd))
    if not np.isnan(mean_speed) and mean_speed < MIN_SPEED_MPS:
        return None  # skip stopped / parking-lot crawling

    peak_i = int(np.nanargmax(np.abs(off)))
    peak = float(abs(off[peak_i]))
    end = float(abs(np.nanmean(off[-6:])))
    start = float(abs(np.nanmean(off[:6])))
    mean_k = float(np.nanmean(kap))
    curved = mean_k >= CURVED_KAPPA
    side = "right" if off[peak_i] >= 0 else "left"

    # Priority: violation > recovery > curved-keeping > straight-keeping.
    if peak >= VIOLATION_PEAK_M and end >= VIOLATION_END_M:
        return f"lane_violation_{side}", ("curved" if curved else "straight"), peak
    if peak >= RECOVERY_PEAK_MIN_M and end <= RECOVERY_END_MAX_M and start <= RECOVERY_START_MAX_M:
        return "lane_recovery", ("curved" if curved else "straight"), (peak - end)
    if curved and peak < 0.6:
        return "lane_keeping", "curved", mean_k * 100
    if peak < KEEP_PEAK_M and not curved:
        return "lane_keeping", "straight", (KEEP_PEAK_M - peak)
    return None


def scan_route(rs: RouteSignals) -> list[Candidate]:
    cands: list[Candidate] = []
    t0 = float(rs.t[0])
    t_end = rs.plan_dur_s - CLIP_SECONDS
    while t0 <= t_end:
        samples_t = t0 + np.arange(FRAMES_PER_CLIP) * SAMPLE_DT
        off = _interp(rs.t, rs.offset, samples_t)
        kap = _interp(rs.t, rs.kappa, samples_t)
        spd = _interp(rs.t, rs.speed, samples_t)
        res = classify(off, kap, spd)
        if res is not None:
            behavior, geometry, score = res
            cands.append(
                Candidate(
                    route=rs.route,
                    car_model=rs.car_model,
                    t0=t0,
                    behavior=behavior,
                    geometry=geometry,
                    score=float(score),
                    samples_t=samples_t,
                    offset=off,
                    steering=_interp(rs.t, rs.kappa_signed, samples_t),
                    speed=spd,
                    kappa=kap,
                )
            )
        t0 += SCAN_STRIDE_S
    return cands


def pick_balanced(cands: list[Candidate], per_cat: int) -> list[Candidate]:
    by_cat: dict[str, list[Candidate]] = {}
    for c in cands:
        by_cat.setdefault(c.category, []).append(c)

    chosen: list[Candidate] = []
    for cat, items in by_cat.items():
        # Highest score first, but spread across routes and non-overlapping.
        items.sort(key=lambda c: c.score, reverse=True)
        picked: list[Candidate] = []
        per_route: dict[str, int] = {}
        max_per_route = max(2, per_cat // 4)
        for c in items:
            if len(picked) >= per_cat:
                break
            if per_route.get(c.route, 0) >= max_per_route:
                continue
            if any(p.route == c.route and abs(p.t0 - c.t0) < CLIP_SECONDS for p in picked):
                continue
            picked.append(c)
            per_route[c.route] = per_route.get(c.route, 0) + 1
        # second pass relaxing per-route cap if short
        if len(picked) < per_cat:
            for c in items:
                if len(picked) >= per_cat:
                    break
                if c in picked:
                    continue
                if any(p.route == c.route and abs(p.t0 - c.t0) < CLIP_SECONDS for p in picked):
                    continue
                picked.append(c)
        chosen.extend(picked)
    return chosen


def build_metrics(rs_video_dur: float, rs_plan_dur: float, c: Candidate) -> dict:
    d = c.offset
    peak = float(np.nanmax(np.abs(d)))
    end = float(abs(np.nanmean(d[-6:])))
    recovery = float((peak - end) / peak) if peak > 1e-6 else 0.0
    return {
        "scene_name": c.route,
        "reference_source": "baton_openpilot_lane_model",
        "scene_description": f"BATON {c.route} · {c.car_model} · t0={c.t0:.0f}s",
        "signed_lateral_m": [round(float(x), 4) for x in d],
        "steering": [round(float(x), 5) for x in c.steering],
        "speed_mps": [round(float(x), 3) for x in c.speed],
        "lateral_peak_m": round(peak, 4),
        "lateral_drift_m": round(end, 4),
        "recovery_score": round(recovery, 4),
        "mean_curvature": round(float(np.nanmean(c.kappa)), 6),
        "mean_speed_mps": round(float(np.nanmean(c.speed)), 3),
        "behavior": c.behavior,
        "road_geometry": c.geometry,
    }


def extract_clip(rs: RouteSignals, c: Candidate, out_path: Path) -> bool:
    cap = cv2.VideoCapture(rs.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # plan time -> video time (corrects the ~0.5% drift)
    scale = rs.video_dur_s / rs.plan_dur_s if rs.plan_dur_s > 0 else 1.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (w, h))
    ok_all = True
    for k in range(FRAMES_PER_CLIP):
        vt = (c.t0 + k * SAMPLE_DT) * scale
        fidx = int(round(vt * fps))
        if fidx >= nframes:
            fidx = int(nframes) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            ok_all = False
            break
        # Timestamp-only overlay (no label/offset leak for fair VLM eval).
        label = f"t={k * SAMPLE_DT:04.1f}s"
        cv2.rectangle(frame, (0, h - 16), (70, h), (0, 0, 0), -1)
        cv2.putText(frame, label, (4, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    cap.release()
    if not ok_all:
        tmp.unlink(missing_ok=True)
        return False
    transcode_browser_mp4(tmp, out_path)
    tmp.unlink(missing_ok=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=30)
    ap.add_argument("--max-routes", type=int, default=0, help="limit routes (0=all) for quick runs")
    ap.add_argument("--scan-only", action="store_true", help="only report candidate yield")
    args = ap.parse_args()

    routes = list_video_routes()
    if args.max_routes:
        routes = routes[: args.max_routes]
    print(f"Scanning {len(routes)} BATON video routes…", flush=True)

    all_cands: list[Candidate] = []
    route_cache: dict[str, RouteSignals] = {}
    for i, route in enumerate(routes, 1):
        rs = load_route(route)
        if rs is None:
            print(f"  [{i}/{len(routes)}] {route}: skipped (no lane model)", flush=True)
            continue
        route_cache[route] = rs
        cs = scan_route(rs)
        all_cands.extend(cs)
        from collections import Counter

        cc = Counter(c.category for c in cs)
        print(f"  [{i}/{len(routes)}] {route}: {dict(cc)}", flush=True)

    from collections import Counter

    print("Total candidates:", dict(Counter(c.category for c in all_cands)), flush=True)
    if args.scan_only:
        return

    chosen = pick_balanced(all_cands, args.per_category)
    print("Chosen:", dict(Counter(c.category for c in chosen)), flush=True)

    VIDEO_OUT.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = {}
    clips: list[dict] = []
    for c in sorted(chosen, key=lambda c: (c.category, c.route, c.t0)):
        n = counters.get(c.category, 0)
        counters[c.category] = n + 1
        clip_id = f"{c.category}__{n:02d}"
        out_path = VIDEO_OUT / f"{clip_id}.mp4"
        rs = route_cache[c.route]
        try:
            if not extract_clip(rs, c, out_path):
                print(f"  ! failed to extract {clip_id}", flush=True)
                continue
        except Exception as exc:
            print(f"  ! error extracting {clip_id}: {exc}", flush=True)
            continue
        metrics = build_metrics(rs.video_dur_s, rs.plan_dur_s, c)
        target = f"{c.behavior} / {c.geometry}"
        clips.append(
            {
                "id": clip_id,
                "behavior": c.behavior,
                "road_geometry": c.geometry,
                "target_label": target,
                "ground_truth_label": target,
                "map_label_at_mine": target,
                "label_matches_target": True,
                "video": str(out_path.relative_to(config.MEDIA_ROOT)),
                "scene": c.route,
                "start_timestamp_us": int(c.t0 * 1e6),
                "scene_description": metrics["scene_description"],
                "metrics": metrics,
            }
        )
        print(f"  + {clip_id}  ({c.route} t0={c.t0:.0f}s peak={metrics['lateral_peak_m']}m)", flush=True)

    manifest = {
        "clip_seconds": int(CLIP_SECONDS),
        "clip_fps": CLIP_FPS,
        "frames_per_clip": FRAMES_PER_CLIP,
        "cameras": ["CAM_FRONT (qcamera)"],
        "label_source": "BATON-Sample openpilot production lane model (laneLeft_y/laneRight_y)",
        "clips_per_scenario": args.per_category,
        "clips": clips,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        backup = MANIFEST.with_name("manifest_all.backup.json")
        backup.write_text(MANIFEST.read_text())
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(clips)} clips -> {MANIFEST}", flush=True)
    print("Per category:", dict(Counter(c["behavior"] + "/" + c["road_geometry"] for c in clips)), flush=True)


if __name__ == "__main__":
    main()
