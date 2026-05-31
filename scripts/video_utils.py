"""Browser-compatible MP4 helpers (H.264 / yuv420p for HTML5 video)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is required for browser-safe video. "
            "Install with: pip install imageio-ffmpeg"
        ) from exc


def transcode_browser_mp4(src: Path, dst: Path) -> Path:
    """Transcode any MP4 to H.264 yuv420p with faststart for browser playback."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)
    return dst


def ensure_browser_mp4(video_path: Path, cache_dir: Path | None = None) -> Path:
    """Return a browser-playable H.264 copy, using cache_dir when provided."""
    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / video_path.name
        if cached.exists() and cached.stat().st_mtime >= video_path.stat().st_mtime:
            return cached
        return transcode_browser_mp4(video_path, cached)

    out = video_path.with_name(f"{video_path.stem}_h264{video_path.suffix}")
    if out.exists() and out.stat().st_mtime >= video_path.stat().st_mtime:
        return out
    return transcode_browser_mp4(video_path, out)
