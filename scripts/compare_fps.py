import json

HMAP = {
    "lane_keeping": "keep_within_lane",
    "lane_recovery": "lane_wandering",
    "lane_violation_left": "lane_change",
    "lane_violation_right": "lane_change",
}
human = {
    k: HMAP[v["behavior"]]
    for k, v in json.load(open("results/human_labels_old_taxonomy.json")).items()
    if v["behavior"] in HMAP
}


def preds_from_summary(p):
    return {e["id"]: (e.get("parsed") or {}).get("overall_behavior") for e in json.load(open(p))}


RUNS = [
    ("cosmos2@4fps", "results/summary.json"),
    ("cosmos3@4fps", "results/cosmos3/summary.json"),
    ("cosmos3@8fps", "results/cosmos3_8fps/summary.json"),
    ("cosmos3@10fps", "results/cosmos3_10fps/summary.json"),
    ("cosmos3@20fps", "results/cosmos3_20fps/summary.json"),
]
preds = {name: preds_from_summary(p) for name, p in RUNS}


def stats(pred):
    ids = [c for c in human if pred.get(c)]
    corr = sum(pred[c] == human[c] for c in ids)
    tp = sum(1 for c in ids if human[c] == "lane_change" and pred[c] == "lane_change")
    fp = sum(1 for c in ids if human[c] != "lane_change" and pred[c] == "lane_change")
    fn = sum(1 for c in ids if human[c] == "lane_change" and pred[c] != "lane_change")
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    acc = corr / len(ids) if ids else 0
    return dict(n=len(ids), acc=acc, P=P, R=R, F=F, tp=tp, fp=fp, fn=fn)


print("=== vs human (27 clips) ===")
print(f"  {'run':16s} {'n':>3} {'acc':>6} {'P':>6} {'R':>6} {'F1':>6}  (tp/fp/fn)")
for name, _ in RUNS:
    s = stats(preds[name])
    print(f"  {name:16s} {s['n']:>3} {s['acc']:>6.3f} {s['P']:>6.3f} {s['R']:>6.3f} {s['F']:>6.3f}  ({s['tp']}/{s['fp']}/{s['fn']})")

print("\n=== 13 crossings across fps (Cosmos 3) ===")
hdr = ["4fps", "8fps", "10fps", "20fps"]
cols = ["cosmos3@4fps", "cosmos3@8fps", "cosmos3@10fps", "cosmos3@20fps"]
print(f"  {'clip':30s} " + " ".join(f"{h:>6}" for h in hdr) + "  c2")
for c in sorted(human):
    if human[c] != "lane_change":
        continue
    def mark(v):
        return "Y" if v == "lane_change" else ("." if v else "-")
    cells = [mark(preds[col].get(c)) for col in cols]
    print(f"  {c:30s} " + " ".join(f"{x:>6}" for x in cells) + f"  {mark(preds['cosmos2@4fps'].get(c))}")
