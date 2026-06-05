"""Central paths and run configuration for the Cosmos lane-behavior eval.

The repository is *code only*. Every large or generated artifact (raw datasets,
extracted clips, predictions, transcoded videos, logs) lives under a single
cache directory so the project is reproducible and relocatable.

Control the cache location from anywhere (e.g. a Cursor cloud/local agent or a
shared volume) by setting ``LANE_CACHE_DIR``::

    export LANE_CACHE_DIR=/mnt/volume/lane-eval-cache

If unset, the cache defaults to ``<repo>/cache``. For backwards compatibility,
if a legacy top-level directory (``clips/``, ``results/``, ``data/``) already
exists and the cache equivalent does not, the legacy directory is used.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_env_cache = os.environ.get("LANE_CACHE_DIR")
CACHE_DIR = Path(_env_cache).expanduser().resolve() if _env_cache else REPO_ROOT / "cache"


def _dir(name: str, *legacy_names: str) -> Path:
    """Return ``CACHE_DIR/name``, falling back to a legacy top-level dir if present."""
    primary = CACHE_DIR / name
    if not primary.exists():
        for legacy in legacy_names:
            legacy_path = REPO_ROOT / legacy
            if legacy_path.exists():
                return legacy_path
    return primary


# --- data roots (all under the cache dir) -----------------------------------
DATASETS_DIR = _dir("datasets", "data")  # raw source datasets (BATON, etc.)
CLIPS_DIR = _dir("clips", "clips")        # extracted clips + manifest
RESULTS_DIR = _dir("results", "results")  # predictions, labels, logs
VIDEO_CACHE_DIR = RESULTS_DIR / "video_cache"  # browser-transcoded videos
LOG_DIR = RESULTS_DIR / "logs"            # full per-clip model responses

# --- key files ---------------------------------------------------------------
MANIFEST = CLIPS_DIR / "manifest_all.json"
SUMMARY = RESULTS_DIR / "summary.json"
HUMAN_LABELS = RESULTS_DIR / "human_labels.json"

# Media root: clip "video" paths in the manifest are stored relative to this and
# resolved against it. It is the cache dir (or repo root in the legacy layout),
# and is also the directory mounted into the vLLM server at MEDIA_PATH_PREFIX.
MEDIA_ROOT = CLIPS_DIR.parent

# --- code-tracked assets -----------------------------------------------------
PROMPTS_DIR = REPO_ROOT / "prompts"
PROMPT_FILE = PROMPTS_DIR / "lane_behavior.yaml"
# Prompt variant describing the 3-pane multi-camera mosaic (see CAMERA_LAYOUTS).
PROMPT_FILE_MOSAIC = PROMPTS_DIR / "lane_behavior_mosaic.yaml"

# --- raw dataset locations ---------------------------------------------------
# nuScenes devkit data root (expects v1.0-mini, samples/, sweeps/, maps/ with the
# unzipped map-expansion under maps/expansion). Override with NUSCENES_DATAROOT.
NUSCENES_DATAROOT = Path(
    os.environ.get("NUSCENES_DATAROOT", str(DATASETS_DIR / "nuscenes"))
).expanduser()
NUSCENES_VERSION = os.environ.get("NUSCENES_VERSION", "v1.0-mini")

# --- clip geometry (kept in sync with ingestion) -----------------------------
CLIP_SECONDS = 12.0
CLIP_FPS = 4
FRAMES_PER_CLIP = int(CLIP_SECONDS * CLIP_FPS)

# --- camera layouts ----------------------------------------------------------
# How the frames in a clip are composed. "front_only" is a single forward camera
# (BATON/openpilot); "front_mosaic3" is a 2-row mosaic with CAM_FRONT on top
# (full width, higher res) and CAM_FRONT_LEFT | CAM_FRONT_RIGHT below (lower res).
CAMERA_LAYOUTS = ["front_only", "front_mosaic3"]
DEFAULT_CAMERA_LAYOUT = "front_only"

# Mosaic canvas geometry (pixels). Front pane spans the full width on top; the two
# side panes split the width on the bottom row.
MOSAIC_WIDTH = 1280
MOSAIC_FRONT_HEIGHT = 720
MOSAIC_SIDE_HEIGHT = 360
MOSAIC_HEIGHT = MOSAIC_FRONT_HEIGHT + MOSAIC_SIDE_HEIGHT

# --- model / serving ---------------------------------------------------------
MODEL = os.environ.get("COSMOS_MODEL", "nvidia/Cosmos-Reason2-32B")
VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
# Path prefix the vLLM server sees for local media (Docker mount of the cache).
MEDIA_PATH_PREFIX = os.environ.get("VLLM_MEDIA_PATH_PREFIX", "/workspace")

# Behavior taxonomy (lateral) and ordering for "most significant" reduction.
BEHAVIORS = ["keep_within_lane", "lane_change", "lane_wandering"]
GEOMETRIES = ["straight", "curved"]
BEHAVIOR_SEVERITY = {"keep_within_lane": 1, "lane_wandering": 2, "lane_change": 2}


def ensure_dirs() -> None:
    """Create the cache directory tree if missing."""
    for d in (DATASETS_DIR, CLIPS_DIR, RESULTS_DIR, VIDEO_CACHE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resolve_media(rel_path: str) -> Path:
    """Resolve a manifest ``video`` path (stored relative to MEDIA_ROOT)."""
    p = Path(rel_path)
    return p if p.is_absolute() else MEDIA_ROOT / p


def prompt_file_for_layout(layout: str | None) -> Path:
    """Return the prompt template matching a clip's camera layout."""
    return PROMPT_FILE_MOSAIC if layout == "front_mosaic3" else PROMPT_FILE


def overall_behavior(parsed: dict) -> str | None:
    """Reduce a (possibly multi-event) prediction to one behavior label."""
    if not parsed:
        return None
    if parsed.get("overall_behavior"):
        return parsed["overall_behavior"]
    events = parsed.get("events") or []
    if events:
        return max(
            events, key=lambda e: BEHAVIOR_SEVERITY.get(e.get("behavior"), 0)
        ).get("behavior")
    return parsed.get("behavior")
