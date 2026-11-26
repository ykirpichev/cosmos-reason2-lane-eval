#!/usr/bin/env python3
"""Upscale lane clips by an integer factor (default 2x) and emit a parallel
manifest. This is the production fix for Cosmos missing subtle lane changes:
the source qcamera is only 526x330, so a maneuver (a lane line sliding under the
hood) falls below the model's effective spatial resolution. Upscaling the frames
raises the number of visual tokens per frame, which is what lets the model
resolve the crossing (verified: prompt video tokens 4.6k -> 12k at 2x, and missed
lane changes flip to correct). See docs/cosmos3_report.md.

Upscaling adds no new information; it just forces a larger token budget. The
cleaner per-request `min_pixels/max_pixels` knob does NOT work for video in this
vLLM/Cosmos build (tokens unchanged), so we re-encode instead.

Usage:
  python scripts/upscale_clips.py                 # 2x all clips -> manifest_2x.json
  python scripts/upscale_clips.py --factor 2 --ids lane_recovery__17 ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError as exc:  # pragma: no cover
    raise SystemExit("imageio-ffmpeg required: pip install imageio-ffmpeg") from exc


def upscale(src: Path, dst: Path, factor: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = [
        FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
        "-vf", f"scale=iw*{factor}:ih*{factor}:flags=lanczos",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=config.MANIFEST)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--suffix", default=None, help="dir/manifest suffix (default _{factor}x)")
    ap.add_argument("--ids", nargs="*", default=None, help="only upscale these clip ids")
    args = ap.parse_args()

    suffix = args.suffix or f"_{args.factor}x"
    manifest = json.loads(args.manifest.read_text())
    clips = manifest["clips"]
    if args.ids:
        idset = set(args.ids)
        clips = [c for c in clips if c["id"] in idset]

    out_clips: list[dict] = []
    for i, clip in enumerate(clips, 1):
        src = config.resolve_media(clip["video"])
        rel = Path(clip["video"])
        # clips/baton/foo.mp4 -> clips/baton_2x/foo.mp4
        new_rel = rel.parent.with_name(rel.parent.name + suffix) / rel.name
        dst = config.MEDIA_ROOT / new_rel
        if not src.exists():
            print(f"  ! [{i}/{len(clips)}] missing source {src}", flush=True)
            continue
        if not (dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime):
            upscale(src, dst, args.factor)
        c2 = dict(clip)
        c2["video"] = str(new_rel)
        c2["upscale_factor"] = args.factor
        out_clips.append(c2)
        print(f"  + [{i}/{len(clips)}] {clip['id']} -> {new_rel}", flush=True)

    out_manifest = dict(manifest)
    out_manifest["clips"] = out_clips
    out_manifest["upscale_factor"] = args.factor
    out_path = config.CLIPS_DIR / f"manifest{suffix}.json"
    out_path.write_text(json.dumps(out_manifest, indent=2))
    print(f"\nWrote {len(out_clips)} clips -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
