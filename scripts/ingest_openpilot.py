#!/usr/bin/env python3
"""Ingest openpilot-style LKA clips (ADAS-TO / OpenLKA-Failure) into the eval set.

Unlike BATON (which exposes ``laneLeft_y``/``laneRight_y`` -> a continuous
lateral offset), the ADAS-TO reduced CSVs expose only lane-line *probabilities*
(``laneLineMeta.leftProb``/``rightProb``) plus steering, curvature, ADAS
engagement and ``alertText``. Each clip is a ±10 s window around an ADAS
ON->OFF takeover, so we derive lane-departure labels from:

  - lane-line confidence collapse near the event (a line is lost),
  - which side's probability drops (left vs right) -> violation side,
  - corrective steering / driver override after the event -> recovery vs violation,
  - openpilot ``alertText`` lane/steer keywords,
  - mean |desiredCurvature| -> straight vs curved.

This is heuristic (no ground-truth lateral position) and intended to be spot-
checked against the video. It reads a local directory of clips laid out like
ADAS-TO::

    <clips_dir>/<CAR>/<driver>/<route>/<clip_id>/{takeover.mp4,carState.csv,
                                                  controlsState.csv,drivingModelData.csv,meta.json}

Usage:
    python scripts/ingest_openpilot.py --clips-dir /path/to/ADAS-TO --per-category 30
    python scripts/ingest_openpilot.py --clips-dir /path/to/OpenLKA-Failure --source openlka
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from video_utils import transcode_browser_mp4  # noqa: E402

CLIPS_DIR = config.CLIPS_DIR
VIDEO_OUT = CLIPS_DIR / "openpilot"
MANIFEST = config.MANIFEST

CLIP_SECONDS = config.CLIP_SECONDS
CLIP_FPS = config.CLIP_FPS
FRAMES_PER_CLIP = config.FRAMES_PER_CLIP
SAMPLE_DT = 1.0 / CLIP_FPS

LANE_LOST_PROB = 0.30
CURVED_KAPPA = 0.0030
LANE_ALERT_RE = re.compile(r"lane|steer|take control|keep|departure", re.I)


def _find_clip_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("meta.json")})


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _resample(t_src: np.ndarray, v: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    m = ~np.isnan(v)
    if m.sum() < 2:
        return np.full_like(t_dst, np.nan, dtype=float)
    return np.interp(t_dst, t_src[m], v[m])


def classify_clip(clip_dir: Path) -> dict | None:
    meta = json.loads((clip_dir / "meta.json").read_text())
    cs = _read_csv(clip_dir / "carState.csv")
    ct = _read_csv(clip_dir / "controlsState.csv")
    dm = _read_csv(clip_dir / "drivingModelData.csv")
    if cs is None:
        return None

    dur = float(meta.get("clip_dur_s", CLIP_SECONDS))
    event_t = float(meta.get("video_time_s", 0) - meta.get("clip_start_s", 0)) if "clip_start_s" in meta else dur / 2

    t = (_col(cs, "time_s") or pd.Series(np.arange(len(cs)) * 0.1)).to_numpy(float)
    t = t - t[0]
    speed = (_col(cs, "vEgo") or pd.Series(np.full(len(cs), np.nan))).to_numpy(float)
    steer = (_col(cs, "steeringAngleDeg") or pd.Series(np.zeros(len(cs)))).to_numpy(float)

    leftp = rightp = None
    if dm is not None:
        lp = _col(dm, "laneLineMeta.leftProb", "laneLeft_prob")
        rp = _col(dm, "laneLineMeta.rightProb", "laneRight_prob")
        tdm = (_col(dm, "time_s") or pd.Series(np.linspace(0, dur, len(dm)))).to_numpy(float)
        tdm = tdm - tdm[0]
        if lp is not None:
            leftp = _resample(tdm, lp.to_numpy(float), t)
        if rp is not None:
            rightp = _resample(tdm, rp.to_numpy(float), t)

    kappa = np.array([np.nan])
    alert_text = ""
    if ct is not None:
        kc = _col(ct, "desiredCurvature", "curvature")
        if kc is not None:
            kappa = np.abs(kc.to_numpy(float))
        for c in ("alertText1", "alertText2"):
            s = _col(ct, c)
            if s is not None:
                alert_text += " " + " ".join(str(x) for x in s.dropna().unique())

    # Evidence near the takeover (±2 s).
    near = (t >= event_t - 2) & (t <= event_t + 2)
    if near.sum() < 3:
        near = np.ones_like(t, dtype=bool)
    min_left = float(np.nanmin(leftp[near])) if leftp is not None else 1.0
    min_right = float(np.nanmin(rightp[near])) if rightp is not None else 1.0
    mean_speed = float(np.nanmean(speed))
    lane_alert = bool(LANE_ALERT_RE.search(alert_text))
    line_lost = min(min_left, min_right) < LANE_LOST_PROB

    if not (lane_alert or line_lost):
        return None  # not a lane-keeping failure

    side = "left" if min_left <= min_right else "right"
    # Corrective steering after event opposite to drift => recovery; else violation.
    post = t > event_t
    pre = t <= event_t
    steer_swing = float(np.nanmax(steer[post]) - np.nanmin(steer[post])) if post.any() else 0.0
    returned = (
        leftp is not None
        and rightp is not None
        and np.nanmean(np.minimum(leftp, rightp)[post][-8:] if post.sum() >= 8 else np.minimum(leftp, rightp)[post]) > 0.5
    )
    behavior = "lane_recovery" if (returned and steer_swing > 5.0) else f"lane_violation_{side}"
    geometry = "curved" if (np.nanmean(kappa) >= CURVED_KAPPA) else "straight"

    return {
        "clip_dir": clip_dir,
        "video": clip_dir / ("takeover.mp4" if (clip_dir / "takeover.mp4").exists() else next(iter(clip_dir.glob("*.mp4")), Path())) ,
        "event_t": event_t,
        "behavior": behavior,
        "geometry": geometry,
        "mean_speed": mean_speed,
        "min_left": min_left,
        "min_right": min_right,
        "lane_alert": lane_alert,
        "alert_text": alert_text.strip()[:200],
        "car_model": meta.get("car_model", "?"),
        "t": t,
        "speed": speed,
        "steer": steer,
        "kappa": kappa,
        "score": (1.0 - min(min_left, min_right)) + (1.0 if lane_alert else 0.0) + steer_swing / 30.0,
    }


def extract(video: Path, event_t: float, out_path: Path, label: str) -> bool:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0 or nframes == 0:
        cap.release()
        return False
    start_t = max(0.0, event_t - CLIP_SECONDS / 2)
    tmp = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (w, h))
    ok_all = True
    for k in range(FRAMES_PER_CLIP):
        fidx = int(round((start_t + k * SAMPLE_DT) * fps))
        if fidx >= nframes:
            fidx = int(nframes) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            ok_all = False
            break
        cv2.rectangle(frame, (0, 0), (w, 18), (0, 0, 0), -1)
        cv2.putText(frame, f"{label} t={k * SAMPLE_DT:04.1f}s", (4, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
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
    ap.add_argument("--clips-dir", required=True, type=Path)
    ap.add_argument("--source", default="adasto", choices=["adasto", "openlka"])
    ap.add_argument("--per-category", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.clips_dir.exists():
        print(f"clips dir not found: {args.clips_dir}")
        return

    clip_dirs = _find_clip_dirs(args.clips_dir)
    print(f"Found {len(clip_dirs)} clip dirs under {args.clips_dir}")
    cands = []
    for cd in clip_dirs:
        try:
            r = classify_clip(cd)
        except Exception as exc:
            print(f"  ! {cd}: {exc}")
            continue
        if r:
            cands.append(r)
    from collections import Counter

    print("Lane-failure candidates:", dict(Counter(c["behavior"] for c in cands)))
    if args.dry_run:
        return

    by_cat: dict[str, list[dict]] = {}
    for c in cands:
        by_cat.setdefault(c["behavior"], []).append(c)

    VIDEO_OUT.mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []
    for cat, items in by_cat.items():
        items.sort(key=lambda c: c["score"], reverse=True)
        for i, c in enumerate(items[: args.per_category]):
            clip_id = f"{args.source}__{cat}__{i:02d}"
            out_path = VIDEO_OUT / f"{clip_id}.mp4"
            video = c["video"]
            if not video or not Path(video).exists():
                continue
            label = f"{c['behavior']}/{c['geometry']}"
            try:
                if not extract(Path(video), c["event_t"], out_path, label):
                    continue
            except Exception as exc:
                print(f"  ! extract {clip_id}: {exc}")
                continue
            t = c["t"]
            samples_t = c["event_t"] - CLIP_SECONDS / 2 + np.arange(FRAMES_PER_CLIP) * SAMPLE_DT
            target = f"{c['behavior']} / {c['geometry']}"
            clips.append(
                {
                    "id": clip_id,
                    "behavior": c["behavior"],
                    "road_geometry": c["geometry"],
                    "target_label": target,
                    "ground_truth_label": target,
                    "map_label_at_mine": target,
                    "label_matches_target": True,
                    "video": str(out_path.relative_to(config.MEDIA_ROOT)),
                    "scene": str(Path(video).parent.relative_to(args.clips_dir)),
                    "scene_description": f"{args.source} {c['car_model']} · alert='{c['alert_text']}'",
                    "metrics": {
                        "scene_name": str(Path(video).parent.name),
                        "reference_source": f"{args.source}_lka_failure",
                        "signed_lateral_m": [0.0] * FRAMES_PER_CLIP,
                        "steering": [round(float(x), 3) for x in _resample(t, c["steer"], samples_t)],
                        "speed_mps": [round(float(x), 3) for x in _resample(t, c["speed"], samples_t)],
                        "lateral_peak_m": 0.0,
                        "lateral_drift_m": 0.0,
                        "recovery_score": 0.0,
                        "mean_curvature": round(float(np.nanmean(c["kappa"])), 6) if not np.isnan(c["kappa"]).all() else 0.0,
                        "mean_speed_mps": round(c["mean_speed"], 3),
                        "min_lane_prob_left": round(c["min_left"], 3),
                        "min_lane_prob_right": round(c["min_right"], 3),
                        "behavior": c["behavior"],
                        "road_geometry": c["geometry"],
                    },
                }
            )
            print(f"  + {clip_id}")

    # merge into manifest_all.json
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())
    else:
        man = {"clip_seconds": int(CLIP_SECONDS), "clip_fps": CLIP_FPS, "clips": []}
    existing = {c["id"]: c for c in man.get("clips", [])}
    for c in clips:
        existing[c["id"]] = c
    man["clips"] = list(existing.values())
    man["label_source"] = (man.get("label_source", "") + f" + {args.source}").strip(" +")
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(f"\nMerged {len(clips)} {args.source} clips -> {MANIFEST}")


if __name__ == "__main__":
    main()
