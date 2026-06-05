#!/usr/bin/env python3
"""Build clip variants (resolution / length / centering) from the cached BATON
source route and run Cosmos inference on each, to probe what fixes missed
lane changes. Greedy decoding (temp 0) for determinism.

Each variant re-cuts from the original 526x330 @ ~20 fps qcamera.mp4, optionally
crops a road ROI and/or upscales, re-times to the requested fps, burns the
bottom-left timestamp, and renders a prompt whose duration/fps/frame-count match.
"""
from __future__ import annotations

import glob
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from video_utils import transcode_browser_mp4  # noqa: E402
from run_batch import build_messages, extract_json_block, media_url  # noqa: E402
from openai import OpenAI  # noqa: E402

SNAP = "/home/ykirpichev/.cache/huggingface/hub/datasets--HenryYHW--BATON-Sample/snapshots/892890326ee6aa5561305566be0b5b2937b0fd5f"
OUTDIR = config.RESULTS_DIR / "exp_variants"
MODEL = "nvidia/Cosmos3-Super"
PFX = "/home/ykirpichev/sources/cosmos-reason-lane-test"
import yaml  # noqa: E402
BASE_PROMPT = yaml.safe_load(config.PROMPT_FILE.read_text())


def src_paths(route: str):
    base = f"{SNAP}/{route}"
    return f"{base}/qcamera.mp4", f"{base}/planning.csv"


def render_prompt(dur: float, fps: float) -> tuple[str, str]:
    nframes = int(round(dur * fps))
    sysp = BASE_PROMPT["system_prompt"]
    usr = BASE_PROMPT["user_prompt"]
    sysp = sysp.replace("12-second", f"{dur:g}-second")
    sysp = re.sub(r"4 Hz \(48 frames\)", f"{fps:g} Hz ({nframes} frames)", sysp)
    usr = usr.replace("12 seconds", f"{dur:g} seconds").replace("full 12 seconds", f"full {dur:g} seconds")
    return sysp, usr


def make_variant(route: str, t0: float, dur: float, fps: float, out: Path,
                 roi: tuple[float, float] | None = None, scale_w: int | None = None) -> bool:
    """roi=(y_top_frac,y_bot_frac) crops vertical band; scale_w sets output width."""
    vpath, ppath = src_paths(route)
    plan = pd.read_csv(ppath)
    plan_dur = float(plan["time_s"].to_numpy(float)[-1])
    cap = cv2.VideoCapture(vpath)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vdur = nfr / max(vfps, 1.0)
    tscale = vdur / plan_dur if plan_dur > 0 else 1.0
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_out = int(round(dur * fps))
    dt = 1.0 / fps
    # determine output size
    if roi:
        y0, y1 = int(roi[0] * h0), int(roi[1] * h0)
    else:
        y0, y1 = 0, h0
    crop_h, crop_w = y1 - y0, w0
    out_w = scale_w or crop_w
    out_h = int(round(crop_h * out_w / crop_w))
    tmp = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    for k in range(n_out):
        vt = (t0 + k * dt) * tscale
        fidx = min(int(round(vt * vfps)), int(nfr) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, fr = cap.read()
        if not ok:
            vw.release(); cap.release(); tmp.unlink(missing_ok=True); return False
        fr = fr[y0:y1, :]
        if (out_w, out_h) != (crop_w, crop_h):
            fr = cv2.resize(fr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        label = f"t={k * dt:04.1f}s"
        bw = int(out_w * 0.14)
        cv2.rectangle(fr, (0, out_h - 18), (bw, out_h), (0, 0, 0), -1)
        cv2.putText(fr, label, (4, out_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(fr)
    vw.release(); cap.release()
    transcode_browser_mp4(tmp, out)
    tmp.unlink(missing_ok=True)
    return True


def run_clip(video: Path, dur: float, fps: float) -> dict:
    sysp, usr = render_prompt(dur, fps)
    usr = usr.strip() + "\n\nAnswer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag."
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    msgs = build_messages(sysp, usr, str(video), Path(PFX), PFX)
    t0 = time.time()
    comp = client.chat.completions.create(
        model=MODEL, messages=msgs, max_tokens=4096, temperature=0.0, top_p=1.0,
        extra_body={"mm_processor_kwargs": {"fps": fps, "do_sample_frames": True}},
    )
    msg = comp.choices[0].message
    text = msg.content or ""
    dump = msg.model_dump() if hasattr(msg, "model_dump") else {}
    reasoning = dump.get("reasoning") or ""
    full = text if not reasoning else f"<think>\n{reasoning}\n</think>\n\n{text}"
    parsed = extract_json_block(full) or {}
    (OUTDIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUTDIR / "logs" / f"{video.stem}.log").write_text(full)
    return {"overall": parsed.get("overall_behavior"), "elapsed": round(time.time() - t0, 1),
            "events": parsed.get("events")}


VARIANTS = [
    # name, route, t0, dur, fps, roi, scale_w
    ("lvl14_native12",  "driver_97/route_21", 2168.0,   12, 4, None,        None),
    ("lvl14_res2x",     "driver_97/route_21", 2168.0,   12, 4, None,        1052),
    ("lvl14_roizoom",   "driver_97/route_21", 2168.0,   12, 4, (0.36, 0.95), 1052),
    ("lvl14_center12",  "driver_97/route_21", 2172.75,  12, 4, None,        None),
    ("lvl14_short8c",   "driver_97/route_21", 2174.75,  8,  4, None,        None),
    ("lvl14_roi_center","driver_97/route_21", 2172.75,  12, 4, (0.36, 0.95), 1052),
]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"{'variant':20s} {'overall':18s} {'dur/fps':9s} {'sec':>5s}")
    results = {}
    for name, route, t0, dur, fps, roi, sw in VARIANTS:
        out = OUTDIR / f"{name}.mp4"
        if not make_variant(route, t0, dur, fps, out, roi, sw):
            print(f"{name:20s} BUILD_FAILED"); continue
        cap = cv2.VideoCapture(str(out)); res = f"{int(cap.get(3))}x{int(cap.get(4))}"; cap.release()
        r = run_clip(out, dur, fps)
        results[name] = r
        truth = "lane_change"
        mark = "OK" if r["overall"] == truth else "xx"
        print(f"{name:20s} {mark} {str(r['overall']):15s} {dur:g}/{fps:g}  {r['elapsed']:>5}  res={res}", flush=True)
    (OUTDIR / "results.json").write_text(json.dumps(results, indent=2))
    print("truth = lane_change (right change late in window)")


if __name__ == "__main__":
    main()
