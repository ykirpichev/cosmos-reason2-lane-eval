#!/usr/bin/env python3
"""Test whether raising the per-frame token budget via mm_processor_kwargs
(min_pixels / max_pixels) reproduces the physical 2x-upscale fix WITHOUT
re-encoding the clip. Feeds the original native 526x330 clips and only varies
the processor pixel budget. Greedy decoding (temp 0).

native frame = 526*330 = 173,580 px ;  2x upscale = 1052*660 = 694,320 px
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from exp_variant import render_prompt  # noqa: E402
from run_batch import build_messages, extract_json_block  # noqa: E402
from openai import OpenAI  # noqa: E402

MODEL = "nvidia/Cosmos3-Super"
PFX = "/home/ykirpichev/sources/cosmos-reason-lane-test"
OUTDIR = config.RESULTS_DIR / "exp_maxpixels"
PX = 526 * 330            # native per-frame pixels
PX2 = (526 * 2) * (330 * 2)  # 2x upscale target

# settings: name -> extra mm_processor_kwargs merged with {fps, do_sample_frames}
SETTINGS = {
    "native_default":   {},
    "maxpix_only":      {"max_pixels": 2_000_000},
    "minfloor_2x":      {"min_pixels": PX2, "max_pixels": 2_000_000},
    "minfloor_3x":      {"min_pixels": PX * 9, "max_pixels": 3_000_000},
}

# clip id -> truth
CLIPS = {
    "lane_violation_left__14": "lane_change",       # hard miss (only resolution fixed it)
    "lane_keeping__straight__02": "keep_within_lane",  # false-positive control
}


def run(video: Path, dur, fps, extra) -> dict:
    sysp, usr = render_prompt(dur, fps)
    usr = usr.strip() + "\n\nAnswer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag."
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    msgs = build_messages(sysp, usr, str(video), Path(PFX), PFX)
    mmkw = {"fps": fps, "do_sample_frames": True, **extra}
    t0 = time.time()
    comp = client.chat.completions.create(
        model=MODEL, messages=msgs, max_tokens=4096, temperature=0.0, top_p=1.0,
        extra_body={"mm_processor_kwargs": mmkw},
    )
    msg = comp.choices[0].message
    text = msg.content or ""
    dump = msg.model_dump() if hasattr(msg, "model_dump") else {}
    reasoning = dump.get("reasoning") or ""
    full = text if not reasoning else f"<think>\n{reasoning}\n</think>\n\n{text}"
    parsed = extract_json_block(full) or {}
    return {"overall": parsed.get("overall_behavior"), "elapsed": round(time.time() - t0, 1)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    man = {c["id"]: c for c in json.loads(config.MANIFEST.read_text())["clips"]}
    results = {}
    for cid, truth in CLIPS.items():
        video = config.resolve_media(man[cid]["video"])
        print(f"\n=== {cid}  (truth={truth})  {video.name} ===", flush=True)
        results[cid] = {"truth": truth, "settings": {}}
        for name, extra in SETTINGS.items():
            try:
                r = run(video, 12, 4, extra)
            except Exception as e:
                print(f"  {name:16s} ERROR {e}"); results[cid]["settings"][name] = "ERROR"; continue
            mark = "OK" if r["overall"] == truth else "xx"
            print(f"  {name:16s} {mark} {str(r['overall']):18s} ({r['elapsed']}s)  extra={extra}", flush=True)
            results[cid]["settings"][name] = r["overall"]
    (OUTDIR / "results.json").write_text(json.dumps(results, indent=2))
    print("\nSummary (native clip, only processor pixel budget varies):")
    for cid, r in results.items():
        print(f"  {cid} truth={r['truth']}: {r['settings']}")


if __name__ == "__main__":
    main()
