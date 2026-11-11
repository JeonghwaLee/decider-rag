"""Compose end-to-end single-query latency per policy from latency_e2e.py measurements (HotpotQA, cfg3).
Policies (routing decisions exactly as in analyze_cmp.py, thresholds from validation):
  all-edge, all-cloud
  rule            : retrieve + rerank + prune + complexity + (edge | cloud by the EC-RAG rule)
  router          : retrieve + rerank + prune + router + (edge | cloud), threshold at the rule's validation rate
  router+cmp/fix  : retrieve + rerank + prune + router + [cmp extraction if the question is detected];
                    guarded comparison answer -> done (no edge generation); else edge | cloud with the re-fit threshold
The router runs in parallel with nothing here (sequential sum), so these are upper bounds for a serial implementation.
Also reports the same totals with B1 microbatched pruning (mb4) using the val-measured ratio of mean prune times.
"""

from paths import RUNS  # noqa: E402
import json, math, sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from router_data import load  # noqa: E402
import analyze_cmp_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lat", required=True)
ap.add_argument("--split", default="test3")
ap.add_argument("--cand", default="f1_qk_s1234:pred_dq")
ap.add_argument("--out", required=True)
a = ap.parse_args()
P = RUNS
M = {(x["pool"], x["id"]): x for x in map(json.loads, open(a.lat))}
D0 = json.load(open(f"{P}/pilot4/scores_test2_qk.json"))["hotpotqa"]
vk = [tuple(k) for k in D0["val_keys"]]
vs = np.array(D0["val_scores"][a.cand])
r_rule = D0["val_rule_rate"]
t_plain = L.thr_rate(vs, r_rule)
vfix = np.array([k in L.cmp_f1("val", vk, "edge", True) for k in vk])
m = int(math.ceil(r_rule * len(vk) - 1e-9))
q = np.sort(vs[~vfix])[::-1]
t_fix = float((q[m - 1] + q[m]) / 2)
D = json.load(open(f"{P}/pilot4/scores_{a.split}_qk.json"))["hotpotqa"]
keys = [tuple(k) for k in D["keys"] if (k[1], k[2]) in M]
sc = dict(zip(map(tuple, D["keys"]), D["scores"][a.cand]))
R = {(r["split"], r["pool"], r["id"]): r for r in load("hotpotqa", a.split, with_text=False)}
fix = L.cmp_f1(a.split, keys, "edge", True)
b1 = json.load(open(f"{P}/pilot2/b1_microbatch_val.json"))
up = (
    lambda x, cx: x["upstream"]["retrieve"]
    + x["upstream"]["rerank"]
    + x["upstream"]["prune"]
    + (x["upstream"]["complexity"] if cx else 0)
)
rows = {n: [] for n in ["all-edge", "all-cloud", "rule", "router", "router+cmp/fix"]}
F = {n: [] for n in rows}
C = {n: [] for n in rows}
for k in keys:
    x = M[(k[1], k[2])]
    r = R[k]
    s = sc[k]
    rows["all-edge"].append(up(x, False) + x["edge"])
    F["all-edge"].append(r["f1_edge"])
    C["all-edge"].append(0)
    rows["all-cloud"].append(up(x, False) + x["cloud"])
    F["all-cloud"].append(r["f1_cloud"])
    C["all-cloud"].append(1)
    rc = not r["ecrag_edge"]
    rows["rule"].append(up(x, True) + (x["cloud"] if rc else x["edge"]))
    F["rule"].append(r["f1_cloud"] if rc else r["f1_edge"])
    C["rule"].append(rc)
    c1 = s > t_plain
    rows["router"].append(up(x, False) + x["router"] + (x["cloud"] if c1 else x["edge"]))
    F["router"].append(r["f1_cloud"] if c1 else r["f1_edge"])
    C["router"].append(c1)
    t = up(x, False) + x["router"] + x.get("cmp", 0.0)
    if k in fix:
        rows["router+cmp/fix"].append(t)
        F["router+cmp/fix"].append(fix[k])
        C["router+cmp/fix"].append(0)
    else:
        c2 = s > t_fix
        rows["router+cmp/fix"].append(t + (x["cloud"] if c2 else x["edge"]))
        F["router+cmp/fix"].append(r["f1_cloud"] if c2 else r["f1_edge"])
        C["router+cmp/fix"].append(c2)
comp = {
    c: np.array(
        [
            M[(k[1], k[2])].get(c, 0.0) if c in ("router", "edge", "cloud", "cmp") else M[(k[1], k[2])]["upstream"][c]
            for k in keys
        ]
    )
    for c in ["retrieve", "rerank", "prune", "complexity", "router", "cmp", "edge", "cloud"]
}
out = {
    "n": len(keys),
    "split": a.split,
    "components_s": {
        c: {
            "mean": float(v[v > 0].mean()) if (v > 0).any() else 0,
            "p50": float(np.median(v[v > 0])) if (v > 0).any() else 0,
            "p90": float(np.percentile(v[v > 0], 90)) if (v > 0).any() else 0,
            "n": int((v > 0).sum()),
        }
        for c, v in comp.items()
    },
    "policies": {},
}
print(f"n={len(keys)} ({a.split}); components (mean / p50 / p90 s, n>0):")
for c, o in out["components_s"].items():
    print(f"   {c:10s} {o['mean']:.3f} / {o['p50']:.3f} / {o['p90']:.3f}  (n={o['n']})")
for n in rows:
    v = np.array(rows[n])
    o = {
        "mean": float(v.mean()),
        "p50": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "p99": float(np.percentile(v, 99)),
        "cloud_rate": float(np.mean(C[n])),
        "F1": float(np.mean(F[n])),
    }
    out["policies"][n] = o
    print(
        f"   {n:15s} latency mean {o['mean']:.3f}s p50 {o['p50']:.3f} p90 {o['p90']:.3f} p99 {o['p99']:.3f} | cloud {o['cloud_rate']:.3f} F1 {o['F1']:.4f}"
    )
json.dump(out, open(a.out, "w"), indent=1)
