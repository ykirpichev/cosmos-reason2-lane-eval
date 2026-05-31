#!/usr/bin/env python3
"""Batch Cosmos Reason 2 inference over lane-behavior clips via vLLM OpenAI API."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REASONING_PROMPT = """Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=config.MANIFEST)
    p.add_argument("--prompt-template", type=Path, default=config.PROMPT_FILE)
    p.add_argument("--model", default=config.MODEL)
    # Must match the clip authoring rate (12 s @ 4 Hz). At fps=1 the server
    # downsamples to ~12 frames and misses short maneuvers (lane changes /
    # recoveries that span only a few seconds). See README "frame-rate" note.
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--output", type=Path, default=config.RESULTS_DIR)
    p.add_argument("--host", default=config.VLLM_HOST)
    p.add_argument("--port", type=int, default=config.VLLM_PORT)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--media-path-prefix",
        default=config.MEDIA_PATH_PREFIX,
        help="Path prefix visible to vLLM server (Docker mount)",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=config.MEDIA_ROOT,
        help="Local media root mapped to media-path-prefix",
    )
    return p.parse_args()


def media_url(path: str, project_root: Path, media_prefix: str) -> str:
    if "://" in path:
        return path
    abs_path = Path(path).resolve()
    try:
        rel = abs_path.relative_to(project_root.resolve())
        server_path = Path(media_prefix) / rel
        return f"file://{server_path}"
    except ValueError:
        return f"file://{abs_path}"


def build_messages(system: str, user: str, video: str, project_root: Path, media_prefix: str) -> list[dict]:
    user = user.strip()
    user = f"{user}\n\n{REASONING_PROMPT}"
    return [
        {"role": "system", "content": system.strip()},
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": media_url(video, project_root, media_prefix)}},
                {"type": "text", "text": user},
            ],
        },
    ]


def _balanced_json_objects(s: str) -> list[str]:
    """Return all top-level balanced ``{...}`` substrings, in order of appearance."""
    out: list[str] = []
    depth = 0
    start: int | None = None
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(s[start : i + 1])
                    start = None
    return out


def extract_json_block(text: str) -> dict | None:
    # Prefer JSON after the final </think>, else scan the whole text. Return the last
    # balanced object that parses and looks like our schema.
    tail = text.rsplit("</think>", 1)[-1]
    for src in (tail, text):
        for cand in reversed(_balanced_json_objects(src)):
            try:
                obj = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and (
                "overall_behavior" in obj or "events" in obj or "behavior" in obj
            ):
                return obj
    return None


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    tmpl = yaml.safe_load(args.prompt_template.read_text())
    system = tmpl.get("system_prompt", "")
    user_tmpl = tmpl.get("user_prompt", "")
    sampling = tmpl.get("sampling_params", {})

    client = OpenAI(api_key="EMPTY", base_url=f"http://{args.host}:{args.port}/v1")
    args.output.mkdir(parents=True, exist_ok=True)
    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"

    def running_accuracy(rs: list[dict]) -> tuple[int, int]:
        # Behavior-only accuracy against whatever reference label is on the clip.
        # The reference may be stale (old taxonomy); this is just a progress signal.
        c = t = 0
        for r in rs:
            if not r.get("parsed"):
                continue
            gtb = (r["ground_truth"] or "").split(" / ")[0].strip()
            if config.overall_behavior(r["parsed"]) == gtb:
                c += 1
            t += 1
        return c, t

    results: list[dict] = []
    total_clips = len(manifest["clips"])
    for ci, clip in enumerate(manifest["clips"], 1):
        clip_id = clip["id"]
        gt = clip["ground_truth_label"]
        user = user_tmpl
        video = clip["video"]
        log_path = log_dir / f"{clip_id}.log"

        print(f"Running inference [{ci}/{total_clips}]: {clip_id} ...")
        t0 = time.time()
        output_text = ""
        rc = 0
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=build_messages(system, user, video, args.project_root, args.media_path_prefix),
                max_tokens=args.max_tokens,
                temperature=sampling.get("temperature", 0.6),
                top_p=sampling.get("top_p", 0.95),
                extra_body={
                    "mm_processor_kwargs": {
                        "fps": args.fps,
                        "do_sample_frames": True,
                    }
                },
            )
            output_text = completion.choices[0].message.content or ""
        except Exception as exc:
            rc = 1
            output_text = f"ERROR: {exc}"

        elapsed = time.time() - t0
        log_path.write_text(output_text)
        parsed = extract_json_block(output_text) if rc == 0 else None
        results.append(
            {
                "id": clip_id,
                "ground_truth": gt,
                "video": video,
                "scene": clip.get("scene"),
                "return_code": rc,
                "elapsed_sec": round(elapsed, 1),
                "parsed": parsed,
                "log": str(log_path),
            }
        )
        # Persist incrementally so a long run can be monitored / resumed-safe.
        summary_path.write_text(json.dumps(results, indent=2))
        rc_correct, rc_total = running_accuracy(results)
        acc = f"{rc_correct}/{rc_total}" if rc_total else "0/0"
        print(f"  done in {elapsed:.1f}s (rc={rc}) | running acc {acc}")

    correct, total = running_accuracy(results)

    print(f"\nSummary: {summary_path}")
    if total:
        print(f"Behavior match vs reference label: {correct}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
