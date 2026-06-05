"""Self-consistency probe: sample each clip N times and report per-sample
overall_behavior plus union/majority aggregation. Tests whether sampling the
model multiple times and unioning lane_change detections recovers the
variance-limited crossing misses.
"""
import sys, json, argparse, time
from pathlib import Path
from collections import Counter

sys.path.insert(0, "scripts")
import config
import yaml
from openai import OpenAI
from run_batch import build_messages, extract_json_block

ap = argparse.ArgumentParser()
ap.add_argument("--ids", nargs="+", required=True)
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--fps", type=float, default=8.0)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--manifest", default="clips/manifest_8fps.json")
ap.add_argument("--prompt", default=None, help="override prompt yaml")
ap.add_argument("--model", default="nvidia/Cosmos3-Super")
ap.add_argument("--out", default=None)
args = ap.parse_args()

REPO = str(Path.cwd())
man = json.load(open(args.manifest))
clips = {c["id"]: c for c in (man["clips"] if isinstance(man, dict) else man)}

prompt_path = Path(args.prompt) if args.prompt else config.PROMPT_FILE
tmpl = yaml.safe_load(prompt_path.read_text())
system = tmpl.get("system_prompt", "")
user = tmpl.get("user_prompt", "")

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

results = {}
for cid in args.ids:
    clip = clips[cid]
    video = clip["video"]
    samples = []
    for k in range(args.n):
        t0 = time.time()
        try:
            comp = client.chat.completions.create(
                model=args.model,
                messages=build_messages(system, user, video, Path(REPO), REPO),
                max_tokens=4096,
                temperature=args.temp,
                top_p=0.95,
                extra_body={"mm_processor_kwargs": {"fps": args.fps, "do_sample_frames": True}},
            )
            msg = comp.choices[0].message
            txt = msg.content or ""
            _d = msg.model_dump() if hasattr(msg, "model_dump") else {}
            reasoning = _d.get("reasoning") or ""
            full = txt if not reasoning else f"<think>\n{reasoning}\n</think>\n\n{txt}"
            obj = extract_json_block(full)
            beh = (obj or {}).get("overall_behavior", "PARSE_FAIL")
        except Exception as e:
            beh = f"ERR:{e}"
        samples.append(beh)
        print(f"  {cid} sample {k+1}/{args.n}: {beh}  ({time.time()-t0:.0f}s)", flush=True)
    cnt = Counter(samples)
    union_lc = any(s == "lane_change" for s in samples)
    maj_lc = cnt["lane_change"] >= (args.n // 2 + 1)
    results[cid] = {"samples": samples, "counts": dict(cnt), "union_lane_change": union_lc, "majority_lane_change": maj_lc}
    print(f"== {cid}: {dict(cnt)} | union_LC={union_lc} majority_LC={maj_lc}\n", flush=True)

out = args.out or f"/tmp/selfconsist_{int(time.time())}.json"
json.dump(results, open(out, "w"), indent=2)
print("WROTE", out)
print("SELFCONSIST_DONE")
