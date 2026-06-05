"""Compose multi-camera frames into a single mosaic for VLM lane analysis.

Layout (``front_mosaic3``)::

    +-----------------------------------+
    |             CAM_FRONT             |   top: full width, higher res
    +-----------------+-----------------+
    |  CAM_FRONT_LEFT | CAM_FRONT_RIGHT |   bottom: each half width, lower res
    +-----------------+-----------------+

The front pane keeps maximum detail for lane-line tracking; the two bottom panes
add the peripheral context that separates a true lane change from drift-and-return
wandering. Only camera identities are drawn (no behavior/offset leak), and the
clip timestamp is burned onto the front pane, matching the fair-eval overlay rule
used by the single-camera ingest scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

WIDTH = config.MOSAIC_WIDTH
FRONT_H = config.MOSAIC_FRONT_HEIGHT
SIDE_H = config.MOSAIC_SIDE_HEIGHT
HEIGHT = config.MOSAIC_HEIGHT

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def mosaic_size() -> tuple[int, int]:
    """Return the (width, height) of the composed mosaic frame."""
    return WIDTH, HEIGHT


def _fit(img: np.ndarray | None, w: int, h: int) -> np.ndarray:
    """Resize ``img`` to exactly ``w x h`` (letterboxed), or a black pane if missing."""
    pane = np.zeros((h, w, 3), dtype=np.uint8)
    if img is None or img.size == 0:
        return pane
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    pane[y0 : y0 + nh, x0 : x0 + nw] = resized
    return pane


def _caption(pane: np.ndarray, text: str) -> None:
    cv2.rectangle(pane, (0, 0), (len(text) * 9 + 8, 18), (0, 0, 0), -1)
    cv2.putText(pane, text, (4, 13), _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def compose_front_mosaic(
    front: np.ndarray | None,
    left: np.ndarray | None,
    right: np.ndarray | None,
    timestamp_s: float | None = None,
) -> np.ndarray:
    """Compose front/left/right frames into the 2-row mosaic canvas (BGR)."""
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    front_pane = _fit(front, WIDTH, FRONT_H)
    _caption(front_pane, "FRONT")
    if timestamp_s is not None:
        label = f"t={timestamp_s:04.1f}s"
        cv2.rectangle(front_pane, (0, FRONT_H - 18), (78, FRONT_H), (0, 0, 0), -1)
        cv2.putText(front_pane, label, (4, FRONT_H - 5), _FONT, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    canvas[0:FRONT_H, 0:WIDTH] = front_pane

    half = WIDTH // 2
    left_pane = _fit(left, half, SIDE_H)
    _caption(left_pane, "FRONT-LEFT")
    right_pane = _fit(right, WIDTH - half, SIDE_H)
    _caption(right_pane, "FRONT-RIGHT")
    canvas[FRONT_H:HEIGHT, 0:half] = left_pane
    canvas[FRONT_H:HEIGHT, half:WIDTH] = right_pane
    return canvas
