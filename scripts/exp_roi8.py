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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from exp_variant import make_variant, render_prompt  # noqa: E402
from run_batch import build_messages, extract_json_block  # noqa: E402
from openai import OpenAI  # noqa: E402
import time  # noqa: E402

OUTDIR = config.RESULTS_DIR / "exp_roi8"
CLIPDIR = config.CLIPS_DIR / "baton_roi8"  # model-agnostic; reused across models
MODEL = "nvidia/Cosmos3-Super"
PFX = "/home/ykirpichev/sources/cosmos-reason-lane-test"
ROI = (0.36, 0.95)
SCALE_W = 1052
DUR, FPS = 12.0, 8.0
MAX_TOKENS = 4096

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
        model=MODEL, messages=msgs, max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0,
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


def score(gt: dict[str, str], preds: dict[str, str | None]) -> dict:
    """Accuracy + lane_change P/R/F1 over clips that have both a label and a pred."""
    ids = [c for c in gt if preds.get(c)]
    correct = sum(1 for c in ids if preds[c] == gt[c])
    tp = sum(1 for c in ids if gt[c] == "lane_change" and preds[c] == "lane_change")
    fp = sum(1 for c in ids if gt[c] != "lane_change" and preds[c] == "lane_change")
    fn = sum(1 for c in ids if gt[c] == "lane_change" and preds[c] != "lane_change")
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {
        "n": len(ids), "accuracy": round(correct / len(ids), 4) if ids else None,
        "lane_change": {"tp": tp, "fp": fp, "fn": fn,
                        "precision": round(prec, 4) if prec else None,
                        "recall": round(rec, 4) if rec else None,
                        "f1": round(f1, 4) if f1 else None},
    }


def main() -> None:
    global MODEL, OUTDIR, MAX_TOKENS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--output", type=Path, default=OUTDIR)
    ap.add_argument("--clips", choices=["human27", "all"], default="human27",
                    help="human27 = 27 human-labeled clips; all = full 159-clip "
                         "manifest scored against openpilot pseudo-labels")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="clips requested in parallel (vLLM batches them)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help="output token budget; verbose reasoners need more")
    args = ap.parse_args()
    MODEL = args.model
    OUTDIR = args.output
    MAX_TOKENS = args.max_tokens
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CLIPDIR.mkdir(parents=True, exist_ok=True)
    man = {c["id"]: c for c in json.load(open(config.CLIPS_DIR / "manifest_all.json"))["clips"]}
    human_raw = json.load(open("results/human_labels_old_taxonomy.json"))
    human = {k: HUMAN_MAP[v["behavior"]] for k, v in human_raw.items()
             if v.get("behavior") in HUMAN_MAP}

    if args.clips == "all":
        # Pseudo-labels come from the manifest's openpilot-derived `behavior` field.
        gt = {cid: HUMAN_MAP[c["behavior"]] for cid, c in man.items()
              if c.get("behavior") in HUMAN_MAP}
        gt_name = "openpilot pseudo-labels (159)"
    else:
        gt = human
        gt_name = "human (27)"

    def dump(preds: dict[str, str | None]) -> None:
        summary = {
            "config": f"ROI{ROI} zoom w={SCALE_W} @ {FPS}fps greedy",
            "model": MODEL,
            "label_set": gt_name,
            **score(gt, preds),
            "preds": preds,
        }
        # When running the full set, also break out the 27 human-labeled clips.
        if args.clips == "all":
            summary["human27"] = score(human, preds)
        (OUTDIR / "results.json").write_text(json.dumps(summary, indent=2))

    preds: dict[str, str | None] = {}
    print(f"scoring against {gt_name}; {len(gt)} clips")

    # 1) Build any missing ROI variants sequentially (ffmpeg is not thread-safe
    #    for concurrent writes; cached clips are skipped, so this is usually a no-op).
    runnable: list[str] = []
    for cid in sorted(gt):
        out = CLIPDIR / f"{cid}.mp4"
        if not out.exists():
            c = man[cid]
            try:
                ok_build = make_variant(c["scene"], c["start_timestamp_us"] / 1e6,
                                        DUR, FPS, out, roi=ROI, scale_w=SCALE_W)
            except Exception as e:  # noqa: BLE001  (e.g. source scene not in cache)
                print(f"{cid:32s} BUILD_ERROR {e}"); continue
            if not ok_build:
                print(f"{cid:32s} BUILD_FAILED"); continue
        runnable.append(cid)

    # 2) Run inference, optionally in parallel (vLLM batches the requests).
    lock = threading.Lock()
    done = 0

    def work(cid: str) -> tuple[str, str | None]:
        try:
            return cid, run_clip(CLIPDIR / f"{cid}.mp4")
        except Exception as e:  # noqa: BLE001
            print(f"{cid:32s} ERROR {e}"); return cid, None

    def record(cid: str, p: str | None) -> None:
        nonlocal done
        with lock:
            preds[cid] = p
            done += 1
            ok = "OK" if p == gt[cid] else "xx"
            print(f"[{done}/{len(runnable)}] {cid:32s} {str(p):16s} {gt[cid]:16s} {ok}",
                  flush=True)
            dump(preds)  # incremental: a later crash never loses earlier work

    if args.concurrency <= 1:
        for cid in runnable:
            record(*work(cid))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for fut in as_completed([ex.submit(work, cid) for cid in runnable]):
                record(*fut.result())

    dump(preds)
    final = json.load(open(OUTDIR / "results.json"))
    print("\n=== ROI8 RESULT ===")
    print(json.dumps({k: v for k, v in final.items() if k != "preds"}, indent=2))
    print("=== ROI8 DONE ===")


if __name__ == "__main__":
    main()
