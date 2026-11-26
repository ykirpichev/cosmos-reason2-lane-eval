#!/usr/bin/env python3
"""Detailed failure / disagreement review: video + prediction + pseudo + human.

Walk through the clips where a Cosmos run *fails* (predicted behavior disagrees
with the offset-derived pseudo-label or the human label), watch the video, and
read exactly what Cosmos predicted (behavior/geometry/confidence/timing/
description and the raw model output), alongside the openpilot offset trace.

Any run under ``results/`` that has a ``summary.json`` can be selected from the
sidebar (the default ``results/summary.json`` plus every ``results/<run>/``),
so you can review cosmos2, cosmos3, the upscaled runs, or any fps-sweep run.
Human labels load from ``human_labels.json`` or, if empty, the captured
``human_labels_old_taxonomy.json`` mapped to the current 3-class taxonomy.

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
LABELS_PATH = config.HUMAN_LABELS
LABELS_OLD_PATH = config.RESULTS_DIR / "human_labels_old_taxonomy.json"
VIDEO_CACHE = config.VIDEO_CACHE_DIR
TAXONOMY = set(config.BEHAVIORS)

HUMAN_MAP = {
    "lane_keeping": "keep_within_lane",
    "lane_recovery": "lane_wandering",
    "lane_violation_left": "lane_change",
    "lane_violation_right": "lane_change",
}


def discover_runs() -> dict[str, Path]:
    runs: dict[str, Path] = {}
    if config.SUMMARY.exists():
        runs["summary (cosmos2)"] = config.SUMMARY
    for sub in sorted(config.RESULTS_DIR.iterdir()):
        s = sub / "summary.json"
        if sub.is_dir() and s.exists():
            runs[sub.name] = s
    return runs


@st.cache_data
def load_manifest():
    return {c["id"]: c for c in json.loads(MANIFEST.read_text())["clips"]}


@st.cache_data
def load_preds(summary_str: str, mtime: float):
    return {r["id"]: r for r in json.loads(Path(summary_str).read_text())}


@st.cache_data
def load_labels(sig: tuple):
    raw: dict = {}
    src = "live"
    if LABELS_PATH.exists():
        raw = json.loads(LABELS_PATH.read_text()) or {}
    if not raw and LABELS_OLD_PATH.exists():
        raw = json.loads(LABELS_OLD_PATH.read_text()) or {}
        src = "old_taxonomy"
    out: dict = {}
    for cid, v in raw.items():
        b = v.get("behavior")
        out[cid] = {**v, "behavior_raw": b, "behavior": HUMAN_MAP.get(b, b), "source": src}
    return out


def _labels_sig() -> tuple:
    def mt(p):
        return p.stat().st_mtime if p.exists() else 0.0
    return (mt(LABELS_PATH), mt(LABELS_OLD_PATH))


@st.cache_data(show_spinner="Preparing video…")
def browser_video(path_str: str, mtime: float) -> str:
    return str(ensure_browser_mp4(Path(path_str), VIDEO_CACHE))


def behavior_of(label: str) -> str:
    return (label or "").split(" / ")[0]


def pseudo_label(clip: dict) -> str:
    return clip.get("pseudo_3class") or behavior_of(clip.get("target_label", "")) or "—"


def cosmos_behavior(pred: dict) -> str:
    return config.overall_behavior(pred) or "—"


def resolve_log(pred_row: dict, cid: str, summary_path: Path) -> Path:
    log = pred_row.get("log")
    if log:
        p = Path(log)
        for base in (config.RESULTS_DIR.parent, config.MEDIA_ROOT):
            cand = base / p if not p.is_absolute() else p
            if cand.exists():
                return cand
    return summary_path.parent / "logs" / f"{cid}.log"


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
    st.set_page_config(page_title="Failure / Disagreement Review", layout="wide")
    man = load_manifest()

    st.sidebar.title("Failure review")
    qp = st.query_params

    runs = discover_runs()
    if not runs:
        st.error("No results/*/summary.json found — have you run inference?")
        return
    run_names = list(runs)
    qp_run = qp.get("run")
    default_idx = run_names.index(qp_run) if qp_run in run_names else (
        run_names.index("cosmos3") if "cosmos3" in run_names else 0
    )
    run_name = st.sidebar.selectbox("Run", run_names, index=default_idx)
    summary_path = runs[run_name]
    preds = load_preds(str(summary_path), summary_path.stat().st_mtime)
    labels = load_labels(_labels_sig())
    n_labeled = sum(1 for v in labels.values() if v.get("behavior"))
    if labels:
        st.sidebar.caption(f"{n_labeled} human labels ({next(iter(labels.values())).get('source')})")

    modes = [
        "Cosmos ≠ Human (behavior)",
        "Cosmos ≠ Pseudo (failures)",
        "Out-of-taxonomy preds",
        "All human-labeled",
        "All clips",
    ]
    qp_mode = qp.get("mode")
    mode = st.sidebar.radio("Show", modes, index=modes.index(qp_mode) if qp_mode in modes else 0)

    def row_for(cid: str) -> dict:
        c = man.get(cid, {})
        pr = (preds.get(cid, {}).get("parsed") or {})
        return {
            "id": cid,
            "pred": cosmos_behavior(pr),
            "pseudo": pseudo_label(c),
            "human": (labels.get(cid) or {}).get("behavior", ""),
            "geom": pr.get("road_geometry", ""),
            "peak_m": c.get("metrics", {}).get("lateral_peak_m"),
            "drift_m": c.get("metrics", {}).get("lateral_drift_m"),
        }

    def included(cid: str) -> bool:
        if cid not in preds:
            return False
        r = row_for(cid)
        h = labels.get(cid)
        if mode == "All clips":
            return True
        if mode == "Cosmos ≠ Pseudo (failures)":
            return r["pred"] != "—" and r["pseudo"] != "—" and r["pred"] != r["pseudo"]
        if mode == "Out-of-taxonomy preds":
            return r["pred"] not in TAXONOMY and r["pred"] != "—"
        if not (h and h.get("behavior")):
            return False
        if mode == "All human-labeled":
            return True
        if mode == "Cosmos ≠ Human (behavior)":
            return r["pred"] != h["behavior"]
        return False

    ids = [cid for cid in preds if included(cid)]
    st.sidebar.caption(f"{len(ids)} / {len(preds)} clips match")
    if not ids:
        st.info("No clips match this filter.")
        return

    rows = [row_for(c) for c in ids]
    with st.expander(f"Overview — {len(ids)} clips ({run_name})", expanded=True):
        st.dataframe(
            [{"clip": r["id"], "pred": r["pred"], "pseudo": r["pseudo"],
              "human": r["human"], "geom": r["geom"],
              "peak_m": r["peak_m"], "drift_m": r["drift_m"]} for r in rows],
            use_container_width=True, hide_index=True,
        )

    qp_clip = qp.get("clip")
    clip_idx = ids.index(qp_clip) if qp_clip in ids else 0
    cid = st.sidebar.selectbox("Clip", ids, index=clip_idx)
    st.query_params.update({"run": run_name, "mode": mode, "clip": cid})

    c = man.get(cid, {})
    pred_row = preds.get(cid, {})
    pr = pred_row.get("parsed") or {}
    h = labels.get(cid, {})

    st.markdown(f"## {cid}  ·  _{run_name}_")
    left, right = st.columns([3, 2])

    with left:
        vid = pred_row.get("video") or c.get("video")
        vp = config.resolve_media(vid) if vid else None
        if vp and vp.exists():
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
        mcb = cosmos_behavior(pr)
        ps = pseudo_label(c)

        st.markdown("**Cosmos prediction**")
        box = st.success if mcb == ps else st.error
        box(f"overall: **{mcb}**  ·  geometry: {pr.get('road_geometry','—')}")

        st.markdown("**Pseudo-label (offset, 3-class)**")
        st.info(ps)

        if h and h.get("behavior"):
            st.markdown("**Human label**")
            hbox = st.success if mcb == h["behavior"] else st.error
            raw = h.get("behavior_raw")
            raw_txt = f"  ·  _raw: {raw}_" if raw and raw != h["behavior"] else ""
            hbox(
                f"**{h['behavior']}**{raw_txt}"
                + ("  ·  _ambiguous_" if h.get("unclear") else "")
                + (f"\n\n_{h['notes']}_" if h.get("notes") else "")
            )

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

    st.markdown("**Cosmos description**")
    descs = [e.get("description") for e in (pr.get("events") or []) if e.get("description")]
    st.write("  \n".join(f"- {d}" for d in descs) if descs else pr.get("description", "_(no parsed description)_"))

    with st.expander("Raw Cosmos output (full text)"):
        log = resolve_log(pred_row, cid, summary_path)
        st.code(log.read_text() if log.exists() else f"(no log at {log})", language="json")

    with st.expander("All metrics"):
        st.json(c.get("metrics", {}))


if __name__ == "__main__":
    main()
