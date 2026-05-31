#!/usr/bin/env python3
"""Blind human-labeling app for lane-behavior clips.

Creates real ground truth by letting a human watch each 12 s clip and assign a
behavior + geometry, WITHOUT seeing the clip id, pseudo-label, or model
prediction (all of which would bias the labeler — the clip ids even contain the
pseudo-label). Clips are shown in a fixed shuffled order under neutral numbers.

Labels are saved to ``results/human_labels.json`` and the app scores the
pseudo-labels and the Cosmos predictions against the human labels on whatever
subset has been labeled so far.

Run:
    .venv/bin/streamlit run apps/label_clips.py
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import config  # noqa: E402
from video_utils import ensure_browser_mp4  # noqa: E402

MANIFEST = config.MANIFEST
SUMMARY = config.SUMMARY
LABELS_PATH = config.HUMAN_LABELS
VIDEO_CACHE = config.VIDEO_CACHE_DIR
SHUFFLE_SEED = 1234

BEHAVIORS = config.BEHAVIORS
GEOMETRIES = config.GEOMETRIES
BEHAVIOR_HELP = (
    "keep_within_lane: never crosses a line.  "
    "lane_change: crosses a line and ends in a different lane.  "
    "lane_wandering: crosses/rides a line but ends in the same lane."
)


def cosmos_behavior(pred: dict) -> str:
    """Overall behavior from a (possibly multi-event) Cosmos prediction."""
    return config.overall_behavior(pred) or "—"


@st.cache_data
def load_clips() -> list[dict]:
    clips = json.loads(MANIFEST.read_text())["clips"]
    order = list(range(len(clips)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    return [clips[i] for i in order]


@st.cache_data
def load_predictions() -> dict:
    if not SUMMARY.exists():
        return {}
    return {r["id"]: (r.get("parsed") or {}) for r in json.loads(SUMMARY.read_text())}


def load_labels() -> dict:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}


def save_labels(labels: dict) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(labels, indent=2))


@st.cache_data(show_spinner="Preparing video…")
def browser_video(path_str: str, mtime: float) -> str:
    return str(ensure_browser_mp4(Path(path_str), VIDEO_CACHE))


def score_section(clips: list[dict], labels: dict, preds: dict) -> None:
    done = {cid: v for cid, v in labels.items() if v.get("behavior")}
    st.subheader(f"Scoring on {len(done)} human-labeled clips")
    if not done:
        st.info("No labels yet.")
        return
    meta = {c["id"]: c for c in clips}
    rows = []
    pb = mb = mg = ng = 0
    for cid, hv in done.items():
        c = meta.get(cid, {})
        pseudo = c.get("pseudo_3class", c.get("target_label", ""))
        pred = preds.get(cid, {})
        hb = hv["behavior"]
        mcb = cosmos_behavior(pred)
        pb += pseudo == hb
        mb += mcb == hb
        if hv.get("geometry"):
            ng += 1
            mg += pred.get("road_geometry") == hv["geometry"]
        rows.append(
            {
                "clip": cid,
                "human": f"{hb} / {hv.get('geometry','—')}",
                "pseudo": pseudo,
                "cosmos": f"{mcb} / {pred.get('road_geometry','—')}",
            }
        )
    n = len(done)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pseudo vs human (behavior)", f"{pb}/{n}")
    c2.metric("Cosmos vs human (behavior)", f"{mb}/{n}")
    c3.metric("Cosmos vs human (geometry)", f"{mg}/{ng}" if ng else "—")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.download_button("Download human_labels.json", LABELS_PATH.read_text(), "human_labels.json")


def main() -> None:
    st.set_page_config(page_title="Lane Clip Labeler", layout="centered")
    clips = load_clips()
    preds = load_predictions()
    labels = load_labels()

    if "idx" not in st.session_state:
        # Start at the first unlabeled clip.
        st.session_state.idx = next((i for i, c in enumerate(clips) if c["id"] not in labels), 0)

    st.sidebar.title("Lane clip labeler")
    st.sidebar.caption("Blind labeling — clip id, pseudo-label and model prediction are hidden.")
    labeler = st.sidebar.text_input("Your name/initials", value=st.session_state.get("labeler", ""))
    st.session_state.labeler = labeler

    n_done = sum(1 for c in clips if c["id"] in labels and labels[c["id"]].get("behavior"))
    st.sidebar.progress(n_done / len(clips), text=f"{n_done}/{len(clips)} labeled")

    nav = st.sidebar.radio("Show", ["Unlabeled only", "All clips"], index=0)
    if st.sidebar.button("↻ Jump to next unlabeled"):
        nxt = next((i for i, c in enumerate(clips) if c["id"] not in labels), st.session_state.idx)
        st.session_state.idx = nxt

    st.sidebar.divider()
    show_scores = st.sidebar.toggle("Show scoring vs human labels", value=False)
    if show_scores:
        score_section(clips, labels, preds)
        return

    idx = max(0, min(st.session_state.idx, len(clips) - 1))
    clip = clips[idx]
    cid = clip["id"]

    st.markdown(f"### Clip {idx + 1} of {len(clips)}")
    video_path = config.resolve_media(clip["video"])
    if video_path.exists():
        try:
            st.video(browser_video(str(video_path.resolve()), video_path.stat().st_mtime))
        except Exception as exc:
            st.error(f"video error: {exc}")
    else:
        st.warning(f"missing video: {video_path}")

    existing = labels.get(cid, {})
    b_default = BEHAVIORS.index(existing["behavior"]) if existing.get("behavior") in BEHAVIORS else None
    g_default = GEOMETRIES.index(existing["geometry"]) if existing.get("geometry") in GEOMETRIES else None

    behavior = st.radio(
        "Overall ego lane behavior over the 12 s",
        BEHAVIORS,
        index=b_default,
        help=BEHAVIOR_HELP,
    )
    geometry = st.radio("Road geometry / context", GEOMETRIES, index=g_default, horizontal=True)
    cols = st.columns(2)
    unclear = cols[0].checkbox("Ambiguous / hard to tell", value=existing.get("unclear", False))
    notes = cols[1].text_input("Notes (optional)", value=existing.get("notes", ""))

    bcol1, bcol2, bcol3 = st.columns(3)
    if bcol1.button("◀ Prev", use_container_width=True):
        st.session_state.idx = max(0, idx - 1)
        st.rerun()
    if bcol2.button("Skip ▶", use_container_width=True):
        st.session_state.idx = min(len(clips) - 1, idx + 1)
        st.rerun()
    if bcol3.button("💾 Save & Next", type="primary", use_container_width=True):
        if behavior is None:
            st.warning("Pick a behavior first.")
        else:
            labels[cid] = {
                "behavior": behavior,
                "geometry": geometry,
                "unclear": unclear,
                "notes": notes,
                "labeler": labeler,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            save_labels(labels)
            if nav == "Unlabeled only":
                nxt = next((i for i, c in enumerate(clips) if c["id"] not in labels), idx + 1)
            else:
                nxt = idx + 1
            st.session_state.idx = min(len(clips) - 1, nxt)
            st.rerun()

    with st.expander("Reveal pseudo-label & model prediction (biasing — for adjudication only)"):
        st.write(f"**clip id:** `{cid}`")
        st.write(f"**pseudo-label:** {clip.get('target_label')}")
        st.write(f"**Cosmos prediction:** {preds.get(cid, {})}")
        st.write(f"**peak offset:** {clip['metrics'].get('lateral_peak_m')} m · "
                 f"end {clip['metrics'].get('lateral_drift_m')} m")


if __name__ == "__main__":
    main()
