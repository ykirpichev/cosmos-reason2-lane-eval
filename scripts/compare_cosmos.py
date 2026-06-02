"""Compare two model runs (e.g. Cosmos-Reason2 vs Cosmos 3) on the lane task.

Computes, for each model vs the human labels (old taxonomy mapped to 3-class):
accuracy, per-class confusion, and crossing (lane_change) precision/recall/F1.
Also reports head-to-head agreement and which clips each model flips on.

The console summary can be shown with different options:

* ``--format {plain,markdown,json}`` chooses how the summary is rendered
  (terse console lines, GitHub-flavoured markdown tables, or the raw report).
* ``--positive-class`` selects which behaviour counts as the positive class for
  precision/recall/F1 (default ``lane_change``).
* ``--sections`` selects which parts of the summary to show
  (any of ``metrics head2head crossings``; default: all).

Usage:
  python scripts/compare_cosmos.py \
    --a results/summary.json --a-name cosmos2 \
    --b results/cosmos3/summary.json --b-name cosmos3 \
    --human results/human_labels_old_taxonomy.json \
    --out results/cosmos_comparison.json \
    --format markdown --sections metrics crossings
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASSES = ["keep_within_lane", "lane_change", "lane_wandering"]
HUMAN_MAP = {
    "lane_keeping": "keep_within_lane",
    "lane_recovery": "lane_wandering",
    "lane_violation_left": "lane_change",
    "lane_violation_right": "lane_change",
}
SECTIONS = ["metrics", "head2head", "crossings"]


def load_preds(path: Path) -> dict[str, str | None]:
    data = json.load(open(path))
    out: dict[str, str | None] = {}
    for e in data:
        parsed = e.get("parsed") or {}
        out[e["id"]] = parsed.get("overall_behavior")
    return out


def load_human(path: Path) -> dict[str, str]:
    data = json.load(open(path))
    out: dict[str, str] = {}
    for cid, v in data.items():
        b = v.get("behavior")
        if b in HUMAN_MAP:
            out[cid] = HUMAN_MAP[b]
    return out


def metrics_vs_human(
    preds: dict[str, str | None], human: dict[str, str], positive: str
) -> dict:
    ids = [c for c in human if c in preds and preds[c] is not None]
    n = len(ids)
    correct = sum(1 for c in ids if preds[c] == human[c])
    # confusion[actual][pred]
    conf = {a: {p: 0 for p in CLASSES} for a in CLASSES}
    tp = fp = fn = 0
    for c in ids:
        a, p = human[c], preds[c]
        if a in conf and p in conf[a]:
            conf[a][p] += 1
        # positive class precision/recall (e.g. lane_change = "crossing")
        if a == positive and p == positive:
            tp += 1
        elif a != positive and p == positive:
            fp += 1
        elif a == positive and p != positive:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else None,
        "correct": correct,
        "confusion": conf,
        "crossing": {
            "positive_class": positive,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        },
    }


def build_report(args: argparse.Namespace) -> dict:
    A, B = load_preds(args.a), load_preds(args.b)
    human = load_human(args.human)
    positive = args.positive_class

    ma = metrics_vs_human(A, human, positive)
    mb = metrics_vs_human(B, human, positive)

    # head-to-head on all clips both models predicted
    both = sorted(c for c in A if c in B and A[c] and B[c])
    agree = sum(1 for c in both if A[c] == B[c])
    flips = [
        {"id": c, "human": human.get(c), args.a_name: A[c], args.b_name: B[c]}
        for c in both if A[c] != B[c]
    ]

    # what each model does on the human positives (recall detail)
    human_positives = sorted(c for c in human if human[c] == positive)
    crossing_detail = [
        {
            "id": c,
            args.a_name: A.get(c),
            args.b_name: B.get(c),
            f"{args.a_name}_caught": A.get(c) == positive,
            f"{args.b_name}_caught": B.get(c) == positive,
        }
        for c in human_positives
    ]

    return {
        "positive_class": positive,
        "models": {args.a_name: str(args.a), args.b_name: str(args.b)},
        "vs_human": {args.a_name: ma, args.b_name: mb},
        "head_to_head": {
            "n_both": len(both),
            "agree": agree,
            "agreement_rate": round(agree / len(both), 4) if both else None,
            "n_flips": len(flips),
            "flips": flips,
        },
        "human_crossings": crossing_detail,
    }


def render_plain(report: dict, a: str, b: str, sections: list[str]) -> str:
    pos = report["positive_class"]
    lines: list[str] = []

    def metric_line(name: str, m: dict) -> str:
        cr = m["crossing"]
        return (f"  {name:10s} n={m['n']:3d}  acc={m['accuracy']}  "
                f"{pos} P={cr['precision']} R={cr['recall']} F1={cr['f1']} "
                f"(tp={cr['tp']} fp={cr['fp']} fn={cr['fn']})")

    if "metrics" in sections:
        lines.append("vs human (old taxonomy -> 3-class):")
        lines.append(metric_line(a, report["vs_human"][a]))
        lines.append(metric_line(b, report["vs_human"][b]))

    if "head2head" in sections:
        h2h = report["head_to_head"]
        lines.append(f"head-to-head: agree {h2h['agree']}/{h2h['n_both']} "
                     f"({h2h['agreement_rate']}), flips={h2h['n_flips']}")

    if "crossings" in sections:
        detail = report["human_crossings"]
        na = sum(1 for d in detail if d[f"{a}_caught"])
        nb = sum(1 for d in detail if d[f"{b}_caught"])
        lines.append(f"human {pos} caught: {a}={na}/{len(detail)}  "
                     f"{b}={nb}/{len(detail)}")
    return "\n".join(lines)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(report: dict, a: str, b: str, sections: list[str]) -> str:
    pos = report["positive_class"]
    blocks: list[str] = []

    if "metrics" in sections:
        ma, mb = report["vs_human"][a], report["vs_human"][b]
        ca, cb = ma["crossing"], mb["crossing"]
        rows = [
            [f"accuracy (n={ma['n']})", str(ma["accuracy"]), str(mb["accuracy"])],
            [f"{pos} precision", str(ca["precision"]), str(cb["precision"])],
            [f"{pos} recall", str(ca["recall"]), str(cb["recall"])],
            [f"{pos} F1", str(ca["f1"]), str(cb["f1"])],
            ["false positives", str(ca["fp"]), str(cb["fp"])],
        ]
        blocks.append("## vs human (old taxonomy -> 3-class)\n\n"
                      + _md_table(["metric", a, b], rows))
        for name, m in ((a, ma), (b, mb)):
            conf = m["confusion"]
            crows = [[f"`{act}`"] + [str(conf[act][p]) for p in CLASSES]
                     for act in CLASSES]
            blocks.append(f"### {name} confusion (actual \\ pred)\n\n"
                          + _md_table(["actual \\ pred"] + CLASSES, crows))

    if "head2head" in sections:
        h2h = report["head_to_head"]
        lines = [f"## Head-to-head (clips both predicted)\n",
                 f"- Agreement: **{h2h['agree']}/{h2h['n_both']}** "
                 f"({h2h['agreement_rate']}), {h2h['n_flips']} flips."]
        if h2h["flips"]:
            frows = [[f"`{f['id']}`", str(f.get("human")),
                      str(f.get(a)), str(f.get(b))] for f in h2h["flips"]]
            lines.append("\n" + _md_table(["clip", "human", a, b], frows))
        blocks.append("\n".join(lines))

    if "crossings" in sections:
        detail = report["human_crossings"]
        na = sum(1 for d in detail if d[f"{a}_caught"])
        nb = sum(1 for d in detail if d[f"{b}_caught"])
        rows = [[f"`{d['id']}`", str(d.get(a)), str(d.get(b))] for d in detail]
        blocks.append(
            f"## Human {pos} ({len(detail)}) — who caught what\n\n"
            f"caught: **{a} {na}/{len(detail)}**, **{b} {nb}/{len(detail)}**\n\n"
            + _md_table(["clip", a, b], rows)
        )
    return "\n\n".join(blocks)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, default=Path("results/summary.json"))
    ap.add_argument("--a-name", default="cosmos2")
    ap.add_argument("--b", type=Path, default=Path("results/cosmos3/summary.json"))
    ap.add_argument("--b-name", default="cosmos3")
    ap.add_argument("--human", type=Path, default=Path("results/human_labels_old_taxonomy.json"))
    ap.add_argument("--out", default="results/cosmos_comparison.json",
                    help="path to write the full JSON report (use '' to skip)")
    ap.add_argument("--positive-class", default="lane_change", choices=CLASSES,
                    help="behaviour treated as positive for precision/recall/F1")
    ap.add_argument("--format", default="plain", choices=["plain", "markdown", "json"],
                    help="how to render the console summary")
    ap.add_argument("--sections", nargs="+", default=SECTIONS, choices=SECTIONS,
                    help="which sections of the summary to show")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(out_path, "w"), indent=2)

    if args.format == "json":
        print(json.dumps(report, indent=2))
    elif args.format == "markdown":
        print(render_markdown(report, args.a_name, args.b_name, args.sections))
    else:
        print(render_plain(report, args.a_name, args.b_name, args.sections))

    if out_path and args.format != "json":
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
