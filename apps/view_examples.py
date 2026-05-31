#!/usr/bin/env python3
"""Dashboard for the lane-behavior eval: clips, predictions, labels, and metrics.

Reads the manifest, Cosmos predictions (multi-event schema), the offset-derived
pseudo-labels, and human labels (all via ``config``) and presents an overview
table with accuracy metrics plus a per-clip detail view (video + event timeline
+ offset trace + all three label sources).

Run:
    streamlit run apps/view_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import config  # noqa: E402
from video_utils import ensure_browser_mp4  # noqa: E402


@st.cache_data
def load_data() -> tuple[dict, dict, dict, dict]:
    manifest = json.loads(config.MANIFEST.read_text()) if config.MANIFEST.exists() else {"clips": []}
    clips = {c["id"]: c for c in manifest.get("clips", [])}
    preds = (
        {r["id"]: (r.get("parsed") or {}) for r in json.loads(config.SUMMARY.read_text())}
        if config.SUMMARY.exists()
        else {}
    )
    labels = json.loads(config.HUMAN_LABELS.read_text()) if config.HUMAN_LABELS.exists() else {}
    return manifest, clips, preds, labels


@st.cache_data(show_spinner="Preparing video…")
def browser_video(path_str: str, mtime: float) -> str:
    return str(ensure_browser_mp4(Path(path_str), config.VIDEO_CACHE_DIR))


def pseudo_of(clip: dict) -> str:
    return clip.get("pseudo_3class", clip.get("target_label", "—"))


def offset_plot(metrics: dict, events: list[dict]):
    sig = metrics.get("signed_lateral_m", [])
    if not sig:
        return None
    t = [i / config.CLIP_FPS for i in range(len(sig))]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(t, sig, color="#2563eb", marker="o", ms=3, label="lateral offset")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    for y in (1.6, -1.6):
        ax.axhline(y, color="#dc2626", lw=0.8, ls=":")
    for e in events or []:
        ts = e.get("time_of_event_sec")
        if isinstance(ts, (int, float)):
            ax.axvline(ts, color="#16a34a", lw=1.0, alpha=0.7)
            ax.text(ts, ax.get_ylim()[1], f" {e.get('behavior','')}", fontsize=7,
                    color="#16a34a", va="top", rotation=90)
    ax.set_ylabel("offset (m)")
    ax.set_xlabel("t (s)")
    ax.set_title("openpilot lane offset (+right; ±1.6 m ≈ lane line); green = Cosmos events", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def render_overview(clips: dict, preds: dict, labels: dict) -> None:
    rows = []
    cp = tp = ch = th = 0
    for cid, c in clips.items():
        pred = preds.get(cid, {})
        mcb = config.overall_behavior(pred) or "—"
        pseudo = pseudo_of(c)
        h = labels.get(cid, {})
        hb = h.get("behavior")
        if pseudo != "—":
            tp += 1
            cp += mcb == pseudo
        if hb:
            th += 1
            ch += mcb == hb
        rows.append(
            {
                "clip": cid,
                "cosmos": mcb,
                "geometry": pred.get("road_geometry", "—"),
                "events": len(pred.get("events") or []),
                "pseudo": pseudo,
                "human": hb or "—",
            }
        )
    c1, c2, c3 = st.columns(3)
    c1.metric("Clips", len(clips))
    c2.metric("Cosmos vs pseudo", f"{cp}/{tp}" if tp else "—")
    c3.metric("Cosmos vs human", f"{ch}/{th}" if th else "—")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_clip(c: dict, pred: dict, h: dict) -> None:
    left, right = st.columns([3, 2])
    with left:
        vp = config.resolve_media(c["video"])
        if vp.exists():
            try:
                st.video(browser_video(str(vp.resolve()), vp.stat().st_mtime))
            except Exception as exc:  # noqa: BLE001
                st.error(f"video error: {exc}")
        else:
            st.warning(f"missing video: {vp}")
        fig = offset_plot(c.get("metrics", {}), pred.get("events") or [])
        if fig:
            st.pyplot(fig, clear_figure=True)

    with right:
        mcb = config.overall_behavior(pred) or "—"
        st.markdown("**Cosmos**")
        st.success(f"overall: {mcb}  ·  geometry: {pred.get('road_geometry','—')}")
        st.markdown("**Pseudo-label (offset)**")
        st.info(pseudo_of(c))
        st.markdown("**Human label**")
        st.write(f"{h.get('behavior','— (not labeled)')} · {h.get('geometry','')}" if h else "— (not labeled)")

    events = pred.get("events") or []
    if events:
        st.markdown("**Event timeline**")
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

    with st.expander("Clip metrics"):
        st.json(c.get("metrics", {}))


def main() -> None:
    st.set_page_config(page_title="Lane Behavior Dashboard", layout="wide")
    manifest, clips, preds, labels = load_data()
    st.title("Lane behavior eval dashboard")
    st.caption(f"{len(clips)} clips · {manifest.get('label_source', '')}")
    if not clips:
        st.error(f"No manifest at {config.MANIFEST}. Run `python scripts/ingest_baton.py`.")
        st.stop()

    tab_overview, tab_detail = st.tabs(["Overview", "Clip detail"])
    with tab_overview:
        render_overview(clips, preds, labels)
    with tab_detail:
        behaviors = sorted({config.overall_behavior(preds.get(cid, {})) or "—" for cid in clips})
        flt = st.multiselect("Filter by Cosmos behavior", behaviors, default=[])
        ids = [
            cid
            for cid in sorted(clips)
            if not flt or (config.overall_behavior(preds.get(cid, {})) or "—") in flt
        ]
        sel = st.selectbox("Clip", ids, index=0 if ids else None)
        if sel:
            st.markdown(f"### {sel}")
            render_clip(clips[sel], preds.get(sel, {}), labels.get(sel, {}))


if __name__ == "__main__":
    main()
