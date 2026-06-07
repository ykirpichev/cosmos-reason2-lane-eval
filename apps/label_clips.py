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


def clip_dataset(clip: dict) -> str:
    """Best-effort source dataset for a clip (for filtering)."""
    if clip.get("dataset"):
        return clip["dataset"]
    cid = clip.get("id", "")
    if cid.startswith("nuscenes__"):
        return "nuscenes"
    if cid.startswith(("openlka__", "adasto__")):
        return "openpilot"
    return "baton"


def new_event(behavior: str = "keep_within_lane", time: float = 0.0) -> dict:
    """A fresh editable event row with a session-unique id (stable widget keys)."""
    st.session_state._eid = st.session_state.get("_eid", 0) + 1
    return {"eid": st.session_state._eid, "behavior": behavior, "time": float(time)}


def events_from_label(existing: dict) -> list[dict]:
    """Editable event rows from a saved label (supports the old single-behavior schema)."""
    evs = existing.get("events")
    if evs:
        return [new_event(e.get("behavior", "keep_within_lane"),
                          e.get("time_of_event_sec", 0.0)) for e in evs]
    if existing.get("behavior"):
        return [new_event(existing["behavior"], 0.0)]
    return [new_event()]


def overall_from_events(events: list[dict]) -> str | None:
    """Most significant behavior across edited events (matches config.overall_behavior)."""
    return config.overall_behavior({"events": [{"behavior": e["behavior"]} for e in events]})


def event_seq(events: list[dict] | None) -> str:
    """Compact 'behavior@t → behavior@t' summary of an event list."""
    if not events:
        return "—"
    return " → ".join(
        f"{e.get('behavior','?')}@{float(e.get('time_of_event_sec', 0.0)):.1f}s" for e in events
    )


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
    meta = {c["id"]: c for c in clips}
    done = {cid: v for cid, v in labels.items() if v.get("behavior") and cid in meta}
    st.subheader(f"Scoring on {len(done)} human-labeled clips")
    if not done:
        st.info("No labels yet for this selection.")
        return
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
                "human_events": event_seq(hv.get("events")),
                "pseudo": pseudo,
                "cosmos": f"{mcb} / {pred.get('road_geometry','—')}",
                "cosmos_events": event_seq(pred.get("events")),
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

    datasets = sorted({clip_dataset(c) for c in clips})
    if len(datasets) > 1:
        choice = st.sidebar.selectbox("Dataset", ["all", *datasets], index=0)
        if choice != "all":
            prev = st.session_state.get("dataset_filter")
            clips = [c for c in clips if clip_dataset(c) == choice]
            if prev != choice:
                st.session_state.dataset_filter = choice
                st.session_state.idx = 0
        else:
            st.session_state.dataset_filter = "all"
    if not clips:
        st.warning("No clips for this dataset filter.")
        return

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
    g_default = GEOMETRIES.index(existing["geometry"]) if existing.get("geometry") in GEOMETRIES else None

    # (Re)load the editable event list whenever we land on a different clip.
    if st.session_state.get("cur_cid") != cid:
        st.session_state.cur_cid = cid
        st.session_state.events = events_from_label(existing)

    st.markdown("**Time-ordered lane events** over the 12 s")
    st.caption(BEHAVIOR_HELP)
    events = st.session_state.events
    for i, ev in enumerate(events):
        c = st.columns([3, 2, 1])
        ev["behavior"] = c[0].selectbox(
            "Behavior", BEHAVIORS,
            index=BEHAVIORS.index(ev["behavior"]) if ev["behavior"] in BEHAVIORS else 0,
            key=f"beh_{cid}_{ev['eid']}",
        )
        ev["time"] = c[1].number_input(
            "Start (s)", min_value=0.0, max_value=float(config.CLIP_SECONDS),
            value=float(ev["time"]), step=0.5, key=f"time_{cid}_{ev['eid']}",
        )
        if c[2].button("✕", key=f"rm_{cid}_{ev['eid']}", help="Remove this event"):
            events.pop(i)
            st.rerun()

    addc1, addc2 = st.columns([1, 3])
    if addc1.button("➕ Add event", use_container_width=True):
        last_t = events[-1]["time"] if events else 0.0
        events.append(new_event(time=min(config.CLIP_SECONDS, last_t + 2.0)))
        st.rerun()
    overall = overall_from_events(events)
    addc2.markdown(f"→ **overall_behavior:** `{overall or '—'}`")

    geometry = st.radio("Road geometry / context", GEOMETRIES, index=g_default, horizontal=True)
    cols = st.columns(2)
    unclear = cols[0].checkbox("Ambiguous / hard to tell", value=existing.get("unclear", False))
    notes = cols[1].text_input("Notes (optional)", value=existing.get("notes", ""))

    bcol1, bcol2, bcol3 = st.columns(3)
    if bcol1.button("◀ Prev", use_container_width=True):
        st.session_state.idx = max(0, idx - 1)
        st.session_state.pop("cur_cid", None)
        st.rerun()
    if bcol2.button("Skip ▶", use_container_width=True):
        st.session_state.idx = min(len(clips) - 1, idx + 1)
        st.session_state.pop("cur_cid", None)
        st.rerun()
    if bcol3.button("💾 Save & Next", type="primary", use_container_width=True):
        if not events:
            st.warning("Add at least one event first.")
        else:
            ordered = sorted(events, key=lambda e: e["time"])
            labels[cid] = {
                "behavior": overall_from_events(ordered),  # overall (back-compat)
                "events": [
                    {"behavior": e["behavior"], "time_of_event_sec": round(float(e["time"]), 1)}
                    for e in ordered
                ],
                "geometry": geometry,
                "unclear": unclear,
                "notes": notes,
                "labeler": labeler,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            save_labels(labels)
            st.session_state.pop("cur_cid", None)
            if nav == "Unlabeled only":
                nxt = next((i for i, c in enumerate(clips) if c["id"] not in labels), idx + 1)
            else:
                nxt = idx + 1
            st.session_state.idx = min(len(clips) - 1, nxt)
            st.rerun()

    with st.expander("Reveal pseudo-label & model prediction (biasing — for adjudication only)"):
        st.write(f"**clip id:** `{cid}`  ·  **dataset:** {clip_dataset(clip)}  ·  "
                 f"**layout:** {clip.get('camera_layout', 'front_only')}")
        st.write(f"**pseudo-label:** {clip.get('pseudo_3class', clip.get('target_label'))}")
        st.write(f"**Cosmos prediction:** {preds.get(cid, {})}")
        metrics = clip.get("metrics", {}) or {}
        if metrics.get("lateral_peak_m") is not None:
            st.write(f"**peak offset:** {metrics.get('lateral_peak_m')} m · "
                     f"end {metrics.get('lateral_drift_m')} m")


if __name__ == "__main__":
    main()
