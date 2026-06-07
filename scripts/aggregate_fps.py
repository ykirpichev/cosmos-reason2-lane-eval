"""Aggregate the fps-sweep across two independent runs (sampling-variance check).

Reports per-run accuracy / crossing P,R,F1 for each fps, the mean across the two
runs, and per-clip run-to-run agreement (stability). Cosmos 2 @ 4 fps is the
single-run baseline for reference.
"""
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


def preds(path):
    try:
        data = json.load(open(path))
    except FileNotFoundError:
        return None
    return {e["id"]: (e.get("parsed") or {}).get("overall_behavior") for e in data}


# fps -> [run1 summary, run2 summary]
COSMOS3 = {
    4: ["results/cosmos3/summary.json", "results/cosmos3_4fps_r2/summary.json"],
    8: ["results/cosmos3_8fps/summary.json", "results/cosmos3_8fps_r2/summary.json"],
    10: ["results/cosmos3_10fps/summary.json", "results/cosmos3_10fps_r2/summary.json"],
    20: ["results/cosmos3_20fps/summary.json", "results/cosmos3_20fps_r2/summary.json"],
}
COSMOS2 = "results/summary.json"


def metrics(pred):
    ids = [c for c in human if pred.get(c)]
    corr = sum(pred[c] == human[c] for c in ids)
    tp = sum(1 for c in ids if human[c] == "lane_change" and pred[c] == "lane_change")
    fp = sum(1 for c in ids if human[c] != "lane_change" and pred[c] == "lane_change")
    fn = sum(1 for c in ids if human[c] == "lane_change" and pred[c] != "lane_change")
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(n=len(ids), acc=corr / len(ids) if ids else 0, P=P, R=R, F=F, tp=tp, fp=fp, fn=fn)


def fmt(s):
    return f"n={s['n']} acc={s['acc']:.3f} P={s['P']:.3f} R={s['R']:.3f} F1={s['F']:.3f} (tp{s['tp']}/fp{s['fp']}/fn{s['fn']})"


print("=== Cosmos 2 @ 4 fps (baseline, single run) ===")
c2 = preds(COSMOS2)
print("  ", fmt(metrics(c2)))

print("\n=== Cosmos 3 fps sweep: run1 / run2 / mean ===")
rows = []
for fps, (p1, p2) in COSMOS3.items():
    r1, r2 = preds(p1), preds(p2)
    if r1 is None or r2 is None:
        print(f"  {fps:>2} fps: MISSING ({'r1' if r1 is None else ''}{'r2' if r2 is None else ''})")
        continue
    m1, m2 = metrics(r1), metrics(r2)
    macc = (m1["acc"] + m2["acc"]) / 2
    mF = (m1["F"] + m2["F"]) / 2
    mR = (m1["R"] + m2["R"]) / 2
    # per-clip run-to-run agreement (stability)
    common = [c for c in human if r1.get(c) and r2.get(c)]
    agree = sum(1 for c in common if r1[c] == r2[c]) / len(common) if common else 0
    print(f"  {fps:>2} fps r1: {fmt(m1)}")
    print(f"  {fps:>2} fps r2: {fmt(m2)}")
    print(f"  {fps:>2} fps MEAN: acc={macc:.3f} R={mR:.3f} F1={mF:.3f} | run-to-run agreement={agree:.3f}")
    rows.append((fps, macc, mR, mF, agree))

print("\n=== summary (mean of 2 runs) ===")
print(f"  {'fps':>4} {'mean_acc':>9} {'mean_R':>7} {'mean_F1':>8} {'stability':>10}")
for fps, macc, mR, mF, agree in rows:
    print(f"  {fps:>4} {macc:>9.3f} {mR:>7.3f} {mF:>8.3f} {agree:>10.3f}")
if rows:
    best = max(rows, key=lambda r: (r[3], r[1]))  # by mean F1 then acc
    print(f"\n  BEST by mean F1: {best[0]} fps (F1={best[3]:.3f}, acc={best[1]:.3f})")

print("\n=== 13 crossings: detection across fps (r1|r2) ===")
print(f"  {'clip':30s}  4fps   8fps  10fps  20fps   c2")
for c in sorted(human):
    if human[c] != "lane_change":
        continue
    cells = []
    for fps in (4, 8, 10, 20):
        r1, r2 = preds(COSMOS3[fps][0]), preds(COSMOS3[fps][1])
        def m(v):
            return "Y" if v == "lane_change" else ("." if v else "-")
        cells.append(f"{m(r1.get(c) if r1 else None)}{m(r2.get(c) if r2 else None)}")
    c2m = "Y" if c2.get(c) == "lane_change" else "."
    print(f"  {c:30s}  " + "   ".join(f"{x:>4}" for x in cells) + f"    {c2m}")
