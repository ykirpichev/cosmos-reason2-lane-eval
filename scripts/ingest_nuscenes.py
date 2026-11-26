#!/usr/bin/env python3
"""Ingest nuScenes scenes into the lane-behavior eval set as multi-camera mosaics.

Unlike BATON (which exposes openpilot's production lane model as a continuous
lateral offset), nuScenes only gives us the HD map plus the ego pose. We derive a
*best-effort* lateral-offset signal by projecting the ego position onto the
centerline of the lane it is driving in, using the map-expansion API:

  - find the closest forward-aligned lane to the ego at each 4 Hz sample,
  - build an extended reference centerline along that lane's connectivity so the
    perpendicular projection stays valid as the car advances,
  - measure the signed perpendicular offset (+right) from that centerline,
  - track the SAME lane while the car is off-center and only re-anchor once it is
    comfortably centered again (hysteresis). This avoids the nearest-centerline
    "snapping" between parallel lanes that produced phantom violations, while still
    letting a genuine lane change show up as a sustained start->end shift,
  - gate physically impossible (>5 m/s lateral) single-sample jumps.

The resulting ``signed_lateral_m`` trace feeds the SAME mining classifier and the
SAME 3-class remap used by BATON. These labels are PSEUDO labels only -- the map
projection is noisy at intersections/merges, so Cosmos + human labels remain the
ground truth (use ``apps/label_clips.py``).

Each clip is composed into a 2-row mosaic (CAM_FRONT on top at higher res,
CAM_FRONT_LEFT | CAM_FRONT_RIGHT below at lower res; see ``mosaic_utils``) at
12 s @ 4 Hz, then merged into ``clips/manifest_all.json`` tagged
``dataset=nuscenes`` / ``camera_layout=front_mosaic3``.

Usage:
    .venv/bin/python scripts/ingest_nuscenes.py --per-category 5
    .venv/bin/python scripts/ingest_nuscenes.py --scan-only
    .venv/bin/python scripts/ingest_nuscenes.py --layout front_only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from mosaic_utils import compose_front_mosaic  # noqa: E402
from remap_pseudo_3class import classify as classify_3class  # noqa: E402
from remap_pseudo_3class import degate  # noqa: E402
from video_utils import transcode_browser_mp4  # noqa: E402

CLIPS_DIR = config.CLIPS_DIR
VIDEO_OUT = CLIPS_DIR / "nuscenes"
MANIFEST = config.MANIFEST

CLIP_SECONDS = config.CLIP_SECONDS
CLIP_FPS = config.CLIP_FPS
FRAMES_PER_CLIP = config.FRAMES_PER_CLIP
SAMPLE_DT = 1.0 / CLIP_FPS

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
SCAN_STRIDE_S = 6.0

# Lane-offset thresholds (meters) -- mirror scripts/ingest_baton.py.
KEEP_PEAK_M = 0.35
RECOVERY_PEAK_MIN_M = 0.75
RECOVERY_END_MAX_M = 0.35
RECOVERY_START_MAX_M = 0.45
VIOLATION_PEAK_M = 1.00
VIOLATION_END_M = 0.70
MIN_SPEED_MPS = 2.0  # nuScenes urban driving is slower than highway BATON

# Map-projection parameters.
LANE_SEARCH_RADIUS_M = 4.0
LANE_RESOLUTION_M = 1.0
REFERENCE_LENGTH_M = 150.0  # cover 12 s even at higher urban speeds
RECENTER_M = 0.7  # re-anchor the reference lane when |offset| is below this
# Beyond ~1.5 lane widths the single-lane reference is unreliable (the ego has
# outrun the centerline or latched a turning lane-connector at an intersection),
# so we re-anchor and, if still implausible, drop the sample rather than emit a
# phantom multi-meter "violation".
MAX_PLAUSIBLE_M = 5.5
FORWARD_COS_MIN = 0.3  # reject lanes whose heading is >~72 deg off ego heading
CURVED_YAW_RAD = 0.20  # total heading change over the window => curved


def _import_nuscenes():
    try:
        from nuscenes.map_expansion import arcline_path_utils
        from nuscenes.map_expansion.map_api import NuScenesMap
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "nuscenes-devkit is required for nuScenes ingest.\n"
            "Install with: pip install nuscenes-devkit"
        ) from exc
    return NuScenes, NuScenesMap, arcline_path_utils


@dataclass
class Frame:
    t: float  # seconds, scene-relative (CAM_FRONT clock)
    x: float
    y: float
    yaw: float
    cam_tokens: dict  # camera -> sample_data token


@dataclass
class SceneSignals:
    name: str
    location: str
    frames: list  # list[Frame] sampled along CAM_FRONT
    side_chains: dict  # camera -> (ts[np], filename[list], None)


@dataclass
class Candidate:
    scene: str
    location: str
    t0: float
    start_idx: int
    behavior: str
    geometry: str
    score: float
    offset: np.ndarray
    yaw_rate: np.ndarray
    speed: np.ndarray
    metrics: dict = field(default_factory=dict)

    @property
    def category(self) -> str:
        if self.behavior == "lane_keeping":
            return f"lane_keeping__{self.geometry}"
        return self.behavior


# --- geometry helpers --------------------------------------------------------
def quaternion_yaw(q: list[float]) -> float:
    """Yaw (rad) from a nuScenes [w, x, y, z] quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _nearest_vertex(poly: np.ndarray, x: float, y: float) -> tuple[int, float]:
    d = np.hypot(poly[:, 0] - x, poly[:, 1] - y)
    i = int(np.argmin(d))
    return i, float(d[i])


def signed_offset(poly: np.ndarray, x: float, y: float) -> float:
    """Signed perpendicular distance from (x, y) to a centerline polyline (+right)."""
    i, _ = _nearest_vertex(poly, x, y)
    th = poly[i, 2]
    # right normal = (sin th, -cos th)
    return (x - poly[i, 0]) * math.sin(th) - (y - poly[i, 1]) * math.cos(th)


# --- map / lane reference ----------------------------------------------------
class LaneRef:
    """Caches discretized lanes and builds extended reference centerlines."""

    def __init__(self, nmap, arcline_utils):
        self.nmap = nmap
        self.arc = arcline_utils
        self._disc: dict[str, np.ndarray] = {}

    def discretize(self, token: str) -> np.ndarray | None:
        if token in self._disc:
            return self._disc[token]
        try:
            path = self.nmap.get_arcline_path(token)
            pts = self.arc.discretize_lane(path, LANE_RESOLUTION_M)
        except Exception:
            self._disc[token] = None
            return None
        arr = np.asarray(pts, dtype=float) if pts else None
        self._disc[token] = arr
        return arr

    def closest_forward(self, x: float, y: float, yaw: float) -> str | None:
        try:
            recs = self.nmap.get_records_in_radius(
                x, y, LANE_SEARCH_RADIUS_M, ["lane", "lane_connector"]
            )
        except Exception:
            return None
        best, best_d = None, float("inf")
        for tok in recs.get("lane", []) + recs.get("lane_connector", []):
            poly = self.discretize(tok)
            if poly is None:
                continue
            i, d = _nearest_vertex(poly, x, y)
            if math.cos(yaw - poly[i, 2]) < FORWARD_COS_MIN:
                continue
            if d < best_d:
                best, best_d = tok, d
        return best

    def reference(self, token: str) -> np.ndarray | None:
        """Concatenate the lane and its forward connectivity into one polyline."""
        parts: list[np.ndarray] = []
        seen: set[str] = set()
        tok: str | None = token
        total = 0.0
        while tok and tok not in seen and total < REFERENCE_LENGTH_M:
            seen.add(tok)
            seg = self.discretize(tok)
            if seg is None or len(seg) == 0:
                break
            parts.append(seg)
            total += float(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1])).sum())
            try:
                outs = self.nmap.get_outgoing_lane_ids(tok)
            except Exception:
                outs = []
            tok = self._best_successor(seg[-1, 2], outs)
        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    def _best_successor(self, last_yaw: float, outs: list[str]) -> str | None:
        best, best_c = None, -2.0
        for o in outs:
            seg = self.discretize(o)
            if seg is None or len(seg) == 0:
                continue
            c = math.cos(last_yaw - seg[0, 2])
            if c > best_c:
                best, best_c = o, c
        return best


def offset_trace(lane_ref: LaneRef, frames: list[Frame]) -> np.ndarray:
    """Lateral-offset trace (+right) with lane-tracking hysteresis and gating."""
    out = np.full(len(frames), np.nan, dtype=float)
    tracked: np.ndarray | None = None
    for k, fr in enumerate(frames):
        if tracked is None:
            tok = lane_ref.closest_forward(fr.x, fr.y, fr.yaw)
            tracked = lane_ref.reference(tok) if tok else None
        if tracked is None:
            continue
        off = signed_offset(tracked, fr.x, fr.y)
        # Re-anchor when comfortably centered (follow lane connectivity) or when the
        # offset is implausibly large (reference outrun / intersection connector).
        if abs(off) < RECENTER_M or abs(off) > MAX_PLAUSIBLE_M:
            tok = lane_ref.closest_forward(fr.x, fr.y, fr.yaw)
            ref = lane_ref.reference(tok) if tok else None
            if ref is not None:
                tracked = ref
                off = signed_offset(tracked, fr.x, fr.y)
        if abs(off) > MAX_PLAUSIBLE_M:
            tracked = None  # give up this lane; re-acquire fresh next frame
            continue
        out[k] = off
    return degate(out)


# --- scene loading -----------------------------------------------------------
def _sensor_chain(nusc, first_sd_token: str):
    ts, files, tokens, poses = [], [], [], []
    tok = first_sd_token
    while tok:
        sd = nusc.get("sample_data", tok)
        ts.append(sd["timestamp"])
        files.append(sd["filename"])
        tokens.append(tok)
        poses.append(sd["ego_pose_token"])
        tok = sd["next"]
    return np.asarray(ts, dtype=np.int64), files, tokens, poses


def load_scene(nusc, scene) -> SceneSignals | None:
    log = nusc.get("log", scene["log_token"])
    location = log["location"]
    first_sample = nusc.get("sample", scene["first_sample_token"])
    if "CAM_FRONT" not in first_sample["data"]:
        return None

    front_ts, front_files, front_tokens, front_poses = _sensor_chain(
        nusc, first_sample["data"]["CAM_FRONT"]
    )
    if len(front_ts) < FRAMES_PER_CLIP:
        return None
    t0_us = front_ts[0]
    t_sec = (front_ts - t0_us) / 1e6

    frames: list[Frame] = []
    for i, tok in enumerate(front_tokens):
        ego = nusc.get("ego_pose", front_poses[i])
        x, y = ego["translation"][0], ego["translation"][1]
        yaw = quaternion_yaw(ego["rotation"])
        frames.append(Frame(t=float(t_sec[i]), x=x, y=y, yaw=yaw,
                            cam_tokens={"CAM_FRONT": tok}))

    side_chains: dict = {}
    for cam in ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"):
        if cam not in first_sample["data"]:
            side_chains[cam] = (np.zeros(0, dtype=np.int64), [], [])
            continue
        cts, cfiles, ctokens, _ = _sensor_chain(nusc, first_sample["data"][cam])
        side_chains[cam] = (cts, cfiles, ctokens)

    return SceneSignals(name=scene["name"], location=location, frames=frames,
                        side_chains=side_chains)


# --- mining classifier (mirrors BATON) ---------------------------------------
def classify(off: np.ndarray, yaw_rate: np.ndarray, spd: np.ndarray,
             yaw_change: float) -> tuple[str, str, float] | None:
    if np.isnan(off).mean() > 0.3:
        return None
    mean_speed = float(np.nanmean(spd))
    if not np.isnan(mean_speed) and mean_speed < MIN_SPEED_MPS:
        return None

    peak_i = int(np.nanargmax(np.abs(off)))
    peak = float(abs(off[peak_i]))
    end = float(abs(np.nanmean(off[-6:])))
    start = float(abs(np.nanmean(off[:6])))
    curved = abs(yaw_change) >= CURVED_YAW_RAD
    side = "right" if off[peak_i] >= 0 else "left"

    if peak >= VIOLATION_PEAK_M and end >= VIOLATION_END_M:
        return f"lane_violation_{side}", ("curved" if curved else "straight"), peak
    if peak >= RECOVERY_PEAK_MIN_M and end <= RECOVERY_END_MAX_M and start <= RECOVERY_START_MAX_M:
        return "lane_recovery", ("curved" if curved else "straight"), (peak - end)
    if curved and peak < 0.6:
        return "lane_keeping", "curved", abs(yaw_change)
    if peak < KEEP_PEAK_M and not curved:
        return "lane_keeping", "straight", (KEEP_PEAK_M - peak)
    return None


def window_frames(scene: SceneSignals, start_idx: int) -> list[Frame] | None:
    """Resample 48 frames at 4 Hz starting near scene-relative time of start_idx."""
    front = scene.frames
    t_start = front[start_idx].t
    fts = np.asarray([f.t for f in front])
    out: list[Frame] = []
    for k in range(FRAMES_PER_CLIP):
        tt = t_start + k * SAMPLE_DT
        fi = int(np.argmin(np.abs(fts - tt)))
        base = front[fi]
        tokens = {"CAM_FRONT": base.cam_tokens["CAM_FRONT"]}
        for cam in ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"):
            cts, cfiles, ctokens = scene.side_chains.get(cam, (np.zeros(0), [], []))
            if len(cts):
                # Side ts are absolute microseconds; match against the front frame's
                # scene-relative time (side and front chains start at ~the same epoch).
                ci = int(np.argmin(np.abs((cts - cts[0]) / 1e6 - base.t)))
                tokens[cam] = ctokens[ci]
        out.append(Frame(t=k * SAMPLE_DT, x=base.x, y=base.y, yaw=base.yaw,
                        cam_tokens=tokens))
    return out


def scan_scene(scene: SceneSignals, lane_ref: LaneRef) -> list[Candidate]:
    cands: list[Candidate] = []
    front = scene.frames
    duration = front[-1].t
    t0 = 0.0
    fts = np.asarray([f.t for f in front])
    while t0 <= duration - CLIP_SECONDS + 1e-6:
        start_idx = int(np.argmin(np.abs(fts - t0)))
        frames = window_frames(scene, start_idx)
        off = offset_trace(lane_ref, frames)
        yaws = np.unwrap([f.yaw for f in frames])
        yaw_rate = np.gradient(yaws, SAMPLE_DT)
        xs = np.asarray([f.x for f in frames])
        ys = np.asarray([f.y for f in frames])
        speed = np.hypot(np.gradient(xs, SAMPLE_DT), np.gradient(ys, SAMPLE_DT))
        yaw_change = float(yaws[-1] - yaws[0])
        res = classify(off, yaw_rate, speed, yaw_change)
        if res is not None:
            behavior, geometry, score = res
            cands.append(Candidate(
                scene=scene.name, location=scene.location, t0=t0, start_idx=start_idx,
                behavior=behavior, geometry=geometry, score=float(score),
                offset=off, yaw_rate=yaw_rate, speed=speed,
                metrics={"yaw_change_rad": yaw_change},
            ))
        t0 += SCAN_STRIDE_S
    return cands


def pick_balanced(cands: list[Candidate], per_cat: int) -> list[Candidate]:
    by_cat: dict[str, list[Candidate]] = {}
    for c in cands:
        by_cat.setdefault(c.category, []).append(c)
    chosen: list[Candidate] = []
    for items in by_cat.values():
        items.sort(key=lambda c: c.score, reverse=True)
        picked: list[Candidate] = []
        for c in items:
            if len(picked) >= per_cat:
                break
            if any(p.scene == c.scene and abs(p.t0 - c.t0) < CLIP_SECONDS for p in picked):
                continue
            picked.append(c)
        chosen.extend(picked)
    return chosen


# --- video extraction --------------------------------------------------------
def _read_image(nusc, token: str | None):
    if not token:
        return None
    sd = nusc.get("sample_data", token)
    path = Path(nusc.dataroot) / sd["filename"]
    img = cv2.imread(str(path))
    return img


def extract(nusc, scene: SceneSignals, c: Candidate, out_path: Path, layout: str) -> bool:
    frames = window_frames(scene, c.start_idx)
    if layout == "front_only":
        w, h = 1280, 720
    else:
        w, h = config.MOSAIC_WIDTH, config.MOSAIC_HEIGHT
    tmp = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (w, h))
    ok_all = True
    for k, fr in enumerate(frames):
        front = _read_image(nusc, fr.cam_tokens.get("CAM_FRONT"))
        if front is None:
            ok_all = False
            break
        ts = k * SAMPLE_DT
        if layout == "front_only":
            canvas = cv2.resize(front, (w, h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(canvas, (0, h - 18), (78, h), (0, 0, 0), -1)
            cv2.putText(canvas, f"t={ts:04.1f}s", (4, h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            left = _read_image(nusc, fr.cam_tokens.get("CAM_FRONT_LEFT"))
            right = _read_image(nusc, fr.cam_tokens.get("CAM_FRONT_RIGHT"))
            canvas = compose_front_mosaic(front, left, right, timestamp_s=ts)
        vw.write(canvas)
    vw.release()
    if not ok_all:
        tmp.unlink(missing_ok=True)
        return False
    transcode_browser_mp4(tmp, out_path)
    tmp.unlink(missing_ok=True)
    return True


def build_metrics(c: Candidate) -> dict:
    d = c.offset
    peak = float(np.nanmax(np.abs(d))) if not np.isnan(d).all() else 0.0
    end = float(abs(np.nanmean(d[-6:]))) if not np.isnan(d).all() else 0.0
    recovery = float((peak - end) / peak) if peak > 1e-6 else 0.0
    return {
        "scene_name": c.scene,
        "map_name": c.location,
        "reference_source": "nuscenes_map_expansion",
        "scene_description": f"nuScenes {c.scene} · {c.location} · t0={c.t0:.0f}s",
        "signed_lateral_m": [None if np.isnan(x) else round(float(x), 4) for x in d],
        "steering": [round(float(x), 5) for x in c.yaw_rate],
        "speed_mps": [round(float(x), 3) for x in c.speed],
        "lateral_peak_m": round(peak, 4),
        "lateral_drift_m": round(end, 4),
        "recovery_score": round(recovery, 4),
        "mean_curvature": round(float(abs(c.metrics.get("yaw_change_rad", 0.0))), 6),
        "mean_speed_mps": round(float(np.nanmean(c.speed)), 3),
        "behavior": c.behavior,
        "road_geometry": c.geometry,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataroot", type=Path, default=config.NUSCENES_DATAROOT)
    ap.add_argument("--version", default=config.NUSCENES_VERSION)
    ap.add_argument("--per-category", type=int, default=5)
    ap.add_argument("--max-scenes", type=int, default=0, help="limit scenes (0=all)")
    ap.add_argument("--layout", default="front_mosaic3", choices=config.CAMERA_LAYOUTS)
    ap.add_argument("--scan-only", action="store_true", help="report candidate yield only")
    args = ap.parse_args()

    if not args.dataroot.exists():
        print(f"nuScenes dataroot not found: {args.dataroot}")
        return 1

    NuScenes, NuScenesMap, arcline_utils = _import_nuscenes()
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    scenes = nusc.scene[: args.max_scenes] if args.max_scenes else nusc.scene
    print(f"Loaded {len(nusc.scene)} scenes (using {len(scenes)})", flush=True)

    maps: dict[str, object] = {}
    lane_refs: dict[str, LaneRef] = {}

    all_cands: list[Candidate] = []
    scene_cache: dict[str, SceneSignals] = {}
    for i, scene in enumerate(scenes, 1):
        sig = load_scene(nusc, scene)
        if sig is None:
            print(f"  [{i}/{len(scenes)}] {scene['name']}: skipped", flush=True)
            continue
        if sig.location not in lane_refs:
            maps[sig.location] = NuScenesMap(dataroot=str(args.dataroot), map_name=sig.location)
            lane_refs[sig.location] = LaneRef(maps[sig.location], arcline_utils)
        scene_cache[sig.name] = sig
        cs = scan_scene(sig, lane_refs[sig.location])
        all_cands.extend(cs)
        print(f"  [{i}/{len(scenes)}] {scene['name']} ({sig.location}): "
              f"{dict(Counter(c.category for c in cs))}", flush=True)

    print("Total candidates:", dict(Counter(c.category for c in all_cands)), flush=True)
    if args.scan_only:
        return 0

    chosen = pick_balanced(all_cands, args.per_category)
    print("Chosen:", dict(Counter(c.category for c in chosen)), flush=True)

    VIDEO_OUT.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = {}
    clips: list[dict] = []
    for c in sorted(chosen, key=lambda c: (c.category, c.scene, c.t0)):
        n = counters.get(c.category, 0)
        counters[c.category] = n + 1
        clip_id = f"nuscenes__{c.category}__{n:02d}"
        out_path = VIDEO_OUT / f"{clip_id}.mp4"
        try:
            if not extract(nusc, scene_cache[c.scene], c, out_path, args.layout):
                print(f"  ! failed to extract {clip_id}", flush=True)
                continue
        except Exception as exc:
            print(f"  ! error extracting {clip_id}: {exc}", flush=True)
            continue
        metrics = build_metrics(c)
        target = f"{c.behavior} / {c.geometry}"
        sig_arr = np.asarray(
            [np.nan if v is None else v for v in metrics["signed_lateral_m"]], dtype=float
        )
        clips.append({
            "id": clip_id,
            "dataset": "nuscenes",
            "camera_layout": args.layout,
            "behavior": c.behavior,
            "road_geometry": c.geometry,
            "target_label": target,
            "ground_truth_label": target,
            "map_label_at_mine": target,
            "label_matches_target": True,
            "video": str(out_path.relative_to(config.MEDIA_ROOT)),
            "scene": c.scene,
            "start_keyframe_idx": c.start_idx,
            "start_timestamp_us": int(c.t0 * 1e6),
            "scene_description": metrics["scene_description"],
            "metrics": metrics,
            "pseudo_3class": classify_3class(sig_arr[~np.isnan(sig_arr)]) if np.isfinite(sig_arr).any() else "keep_within_lane",
        })
        print(f"  + {clip_id}  ({c.scene} t0={c.t0:.0f}s peak={metrics['lateral_peak_m']}m)", flush=True)

    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())
        backup = MANIFEST.with_name("manifest_all.backup.json")
        backup.write_text(json.dumps(man, indent=2))
    else:
        man = {"clip_seconds": int(CLIP_SECONDS), "clip_fps": CLIP_FPS,
               "frames_per_clip": FRAMES_PER_CLIP, "clips": []}
    existing = {c["id"]: c for c in man.get("clips", [])}
    for c in clips:
        existing[c["id"]] = c
    man["clips"] = list(existing.values())
    cams = list(CAMERAS) if args.layout == "front_mosaic3" else ["CAM_FRONT"]
    man["cameras"] = sorted(set(man.get("cameras", []) or []) | set(cams)) or cams
    sources = [s.strip() for s in (man.get("label_source", "") or "").split("+") if s.strip()]
    if "nuscenes_map_expansion" not in sources:
        sources.append("nuscenes_map_expansion")
    man["label_source"] = " + ".join(sources)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(f"\nMerged {len(clips)} nuScenes clips -> {MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
