#!/usr/bin/env python3
"""Test the ROI-crop + zoom spatial lever at 8 fps on the 27 human-labeled clips.

Unlike a whole-frame upscale (which blurs everything and regressed), this re-cuts
from the 526x330 source, crops to the road band and zooms it, spending the token
budget on the lane markings the model must actually track. Greedy decoding.

Writes results/exp_roi8/results.json and prints accuracy + lane_change P/R/F1 vs
the human labels (old taxonomy -> 3-class).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from exp_variant import make_variant, render_prompt  # noqa: E402
from run_batch import build_messages, extract_json_block  # noqa: E402
from openai import OpenAI  # noqa: E402
import time  # noqa: E402

OUTDIR = config.RESULTS_DIR / "exp_roi8"
CLIPDIR = config.CLIPS_DIR / "baton_roi8"
MODEL = "nvidia/Cosmos3-Super"
PFX = "/home/ykirpichev/sources/cosmos-reason-lane-test"
ROI = (0.36, 0.95)
SCALE_W = 1052
DUR, FPS = 12.0, 8.0

HUMAN_MAP = {
    "lane_keeping": "keep_within_lane",
    "lane_recovery": "lane_wandering",
    "lane_violation_left": "lane_change",
    "lane_violation_right": "lane_change",
}


def run_clip(video: Path) -> str | None:
    sysp, usr = render_prompt(DUR, FPS)
    usr = usr.strip() + ("\n\nAnswer the question using the following format:\n\n"
                         "<think>\nYour reasoning.\n</think>\n\n"
                         "Write your final answer immediately after the </think> tag.")
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    msgs = build_messages(sysp, usr, str(video), Path(PFX), PFX)
    comp = client.chat.completions.create(
        model=MODEL, messages=msgs, max_tokens=4096, temperature=0.0, top_p=1.0,
        extra_body={"mm_processor_kwargs": {"fps": FPS, "do_sample_frames": True}},
    )
    msg = comp.choices[0].message
    text = msg.content or ""
    dump = msg.model_dump() if hasattr(msg, "model_dump") else {}
    reasoning = dump.get("reasoning") or ""
    full = text if not reasoning else f"<think>\n{reasoning}\n</think>\n\n{text}"
    (OUTDIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUTDIR / "logs" / f"{video.stem}.log").write_text(full)
    parsed = extract_json_block(full) or {}
    return parsed.get("overall_behavior")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CLIPDIR.mkdir(parents=True, exist_ok=True)
    man = {c["id"]: c for c in json.load(open(config.CLIPS_DIR / "manifest_all.json"))["clips"]}
    human_raw = json.load(open("results/human_labels_old_taxonomy.json"))
    human = {k: HUMAN_MAP[v["behavior"]] for k, v in human_raw.items()
             if v.get("behavior") in HUMAN_MAP}

    preds: dict[str, str | None] = {}
    print(f"{'clip':28s} {'pred':16s} {'human':16s} ok")
    for cid in sorted(human):
        c = man[cid]
        route = c["scene"]
        t0 = c["start_timestamp_us"] / 1e6
        out = CLIPDIR / f"{cid}.mp4"
        if not out.exists():
            if not make_variant(route, t0, DUR, FPS, out, roi=ROI, scale_w=SCALE_W):
                print(f"{cid:28s} BUILD_FAILED"); continue
        try:
            p = run_clip(out)
        except Exception as e:  # noqa: BLE001
            print(f"{cid:28s} ERROR {e}"); p = None
        preds[cid] = p
        ok = "OK" if p == human[cid] else "xx"
        print(f"{cid:28s} {str(p):16s} {human[cid]:16s} {ok}", flush=True)

    # metrics
    ids = [c for c in human if preds.get(c)]
    correct = sum(1 for c in ids if preds[c] == human[c])
    tp = sum(1 for c in ids if human[c] == "lane_change" and preds[c] == "lane_change")
    fp = sum(1 for c in ids if human[c] != "lane_change" and preds[c] == "lane_change")
    fn = sum(1 for c in ids if human[c] == "lane_change" and preds[c] != "lane_change")
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    summary = {
        "config": f"ROI{ROI} zoom w={SCALE_W} @ {FPS}fps greedy",
        "n": len(ids), "accuracy": round(correct / len(ids), 4) if ids else None,
        "lane_change": {"tp": tp, "fp": fp, "fn": fn,
                        "precision": round(prec, 4) if prec else None,
                        "recall": round(rec, 4) if rec else None,
                        "f1": round(f1, 4) if f1 else None},
        "preds": preds,
    }
    (OUTDIR / "results.json").write_text(json.dumps(summary, indent=2))
    print("\n=== ROI8 RESULT ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "preds"}, indent=2))
    print("=== ROI8 DONE ===")


if __name__ == "__main__":
    main()
