#!/usr/bin/env python3
"""Detailed disagreement review: video + human label + full Cosmos prediction + pseudo-label.

Lets you walk through the clips where labels disagree, watch the video, and read
exactly what Cosmos predicted (behavior/geometry/confidence/timing/description
and the raw model output), alongside the openpilot offset trace.

Run:
    .venv/bin/streamlit run apps/review_disagreements.py --server.port 8503
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import config  # noqa: E402
from video_utils import ensure_browser_mp4  # noqa: E402

MANIFEST = config.MANIFEST
SUMMARY = config.SUMMARY
LABELS_PATH = config.HUMAN_LABELS
LOG_DIR = config.LOG_DIR
VIDEO_CACHE = config.VIDEO_CACHE_DIR


@st.cache_data
def load_all():
    man = {c["id"]: c for c in json.loads(MANIFEST.read_text())["clips"]}
    preds = {r["id"]: r for r in json.loads(SUMMARY.read_text())} if SUMMARY.exists() else {}
    labels = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
    return man, preds, labels


@st.cache_data(show_spinner="Preparing video…")
def browser_video(path_str: str, mtime: float) -> str:
    return str(ensure_browser_mp4(Path(path_str), VIDEO_CACHE))


def behavior_of(label: str) -> str:
    return (label or "").split(" / ")[0]


def cosmos_behavior(pred: dict) -> str:
    return config.overall_behavior(pred) or "—"


def offset_plot(metrics: dict):
    d = metrics.get("signed_lateral_m", [])
    if not d:
        return None
    t = [i * 0.25 for i in range(len(d))]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.plot(t, d, color="#2563eb", marker="o", ms=3)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    for y in (1.75, -1.75):
        ax.axhline(y, color="#dc2626", lw=0.8, ls=":")
    for y in (0.35, -0.35):
        ax.axhline(y, color="#f59e0b", lw=0.7, ls=":")
    ax.set_ylabel("offset (m)")
    ax.set_xlabel("t (s)")
    ax.set_title("openpilot lane offset  (±1.75 m = line, ±0.35 m = centered)", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Disagreement Review", layout="wide")
    man, preds, labels = load_all()

    st.sidebar.title("Disagreement review")
    mode = st.sidebar.radio(
        "Show",
        [
            "Cosmos ≠ Human (behavior)",
            "Cosmos ≠ Pseudo (behavior)",
            "All human-labeled",
            "All clips",
        ],
    )

    labeled = {cid: v for cid, v in labels.items() if v.get("behavior")}

    def included(cid: str) -> bool:
        c = man.get(cid, {})
        pr = (preds.get(cid, {}).get("parsed") or {})
        mbeh = cosmos_behavior(pr)
        ps = c.get("pseudo_3class", c.get("target_label", ""))
        h = labels.get(cid)
        if mode == "All clips":
            return True
        if mode == "Cosmos ≠ Pseudo (behavior)":
            return mbeh != behavior_of(ps)
        if not (h and h.get("behavior")):
            return False
        if mode == "All human-labeled":
            return True
        if mode == "Cosmos ≠ Human (behavior)":
            return mbeh != h["behavior"]
        return False

    ids = [cid for cid in man if included(cid)]
    st.sidebar.caption(f"{len(ids)} clips match")
    if not ids:
        st.info("No clips match this filter (have you labeled any / run inference?).")
        return

    cid = st.sidebar.selectbox("Clip", ids)
    c = man[cid]
    pred_row = preds.get(cid, {})
    pr = pred_row.get("parsed") or {}
    h = labels.get(cid, {})

    st.markdown(f"## {cid}")
    left, right = st.columns([3, 2])

    with left:
        vp = config.resolve_media(c["video"])
        if vp.exists():
            try:
                st.video(browser_video(str(vp.resolve()), vp.stat().st_mtime))
            except Exception as exc:
                st.error(f"video error: {exc}")
        else:
            st.warning(f"missing video: {vp}")
        fig = offset_plot(c.get("metrics", {}))
        if fig:
            st.pyplot(fig, clear_figure=True)
        m = c.get("metrics", {})
        st.caption(
            f"peak offset {m.get('lateral_peak_m')} m · end {m.get('lateral_drift_m')} m · "
            f"mean speed {m.get('mean_speed_mps')} m/s · mean κ {m.get('mean_curvature')}"
        )

    with right:
        hb = h.get("behavior", "—") if h else "— (not labeled)"
        st.markdown("**Human label**")
        st.success(hb + ("  ·  _ambiguous_" if h.get("unclear") else "") + (f"\n\n_{h['notes']}_" if h.get("notes") else ""))

        st.markdown("**Cosmos prediction**")
        mcb = cosmos_behavior(pr)
        agree = h and mcb == h.get("behavior")
        box = st.success if agree else st.error
        box(f"overall: {mcb}  ·  geometry: {pr.get('road_geometry','—')}")
        events = pr.get("events") or []
        if events:
            st.markdown("_events_")
            st.dataframe(
                [
                    {
                        "t (s)": e.get("time_of_event_sec"),
                        "behavior": e.get("behavior"),
                        "conf": e.get("confidence"),
                        "description": e.get("description"),
                    }
                    for e in events
                ],
                use_container_width=True,
                hide_index=True,
            )
        st.markdown("**Pseudo-label (offset, 3-class)**")
        st.info(c.get("pseudo_3class", c.get("target_label", "—")))

    st.markdown("**Cosmos description**")
    descs = [e.get("description") for e in (pr.get("events") or []) if e.get("description")]
    st.write("  \n".join(f"- {d}" for d in descs) if descs else pr.get("description", "_(no parsed description)_"))

    with st.expander("Raw Cosmos output (full text)"):
        log = LOG_DIR / f"{cid}.log"
        st.code(log.read_text() if log.exists() else "(no log)", language="json")

    with st.expander("All metrics"):
        st.json(c.get("metrics", {}))


if __name__ == "__main__":
    main()
