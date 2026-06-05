"""Normalize stored prediction files so the artifacts are taxonomy-clean and
self-consistent with how we score them.

For every record in results/*/summary.json:
  * snap each event behavior to the 3-class taxonomy (right_turn -> keep, etc.),
    preserving the model's original string under ``behavior_raw``;
  * merge contiguous identical events (per the prompt's own rule);
  * recompute ``overall_behavior`` by precedence (lane_change > lane_wandering >
    keep_within_lane) from the cleaned timeline, preserving the model's free-text
    answer under ``overall_behavior_raw``.

Idempotent: re-running uses the preserved ``*_raw`` fields as the source of truth,
so it never double-wraps. Pass --check to report what WOULD change without writing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def normalize_events(events: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for ev in events:
        raw = ev.get("behavior_raw", ev.get("behavior"))
        norm = config.normalize_behavior(raw) or "keep_within_lane"
        new = dict(ev)
        new["behavior_raw"] = raw
        new["behavior"] = norm
        # merge contiguous identical behaviors (keep earliest)
        if cleaned and cleaned[-1]["behavior"] == norm:
            continue
        cleaned.append(new)
    return cleaned


def normalize_parsed(parsed: dict) -> tuple[dict, bool]:
    if not parsed:
        return parsed, False
    before = json.dumps(parsed, sort_keys=True)
    events = parsed.get("events") or []
    if events:
        parsed["events"] = normalize_events(events)
    raw_overall = parsed.get("overall_behavior_raw", parsed.get("overall_behavior"))
    parsed["overall_behavior_raw"] = raw_overall
    parsed["overall_behavior"] = config.overall_behavior(parsed)
    return parsed, json.dumps(parsed, sort_keys=True) != before


def process(path: Path, write: bool) -> tuple[int, int]:
    data = json.load(open(path))
    changed = 0
    for rec in data:
        parsed = rec.get("parsed")
        if not parsed:
            continue
        _, did = normalize_parsed(parsed)
        changed += int(did)
    if write and changed:
        json.dump(data, open(path, "w"), indent=2)
    return len(data), changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/*/summary.json")
    ap.add_argument("--check", action="store_true", help="dry run (no writes)")
    args = ap.parse_args()
    paths = sorted(Path(".").glob(args.glob))
    if not paths:
        print(f"no files match {args.glob}")
        return
    for p in paths:
        n, changed = process(p, write=not args.check)
        verb = "would change" if args.check else "normalized"
        print(f"{str(p):55s} {n:3d} recs, {verb} {changed}")


if __name__ == "__main__":
    main()
