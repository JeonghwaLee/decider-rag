"""Pilot 3: choose the edge prompt and the cloud prompt independently (same chunks D, train pools 00-01).
For every (edge variant, cloud variant) pair: edge / cloud GLM quality and F1, oracle quality, share where only cloud is
CORRECT, mean edge output tokens, mean cloud output tokens and hypothetical cloud cost per query (DeepSeek-V3 list prices).
Selection rule (validation-free pilot on train; the chosen pair is then generated for val/test): maximize mean GLM quality
of a 50/50 mix (edge + cloud)/2 averaged over the two datasets, i.e. both sides count equally.
"""

from paths import RUNS  # noqa: E402
import json, os, glob, itertools
import numpy as np

A = f"{RUNS}/pilot2/answers"
S = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
V = ["orig", "typed", "extract", "min", "orig_think", "typed_think"]


def rows(ds, v):
    out = {}
    for p in sorted(glob.glob(f"{A}/{ds}/train/pool_0[01]/{v}.jsonl")):
        for l in open(p):
            x = json.loads(l)
            if x.get("j_edge") and x.get("j_cloud"):
                out[(p.split("/")[-2], x["id"])] = x
    return out


res = {}
for ds in ["hotpotqa", "qasper"]:
    R = {v: rows(ds, v) for v in V}
    R = {v: r for v, r in R.items() if len(r) >= 390}
    print(f"== {ds}: variants {[(v, len(r)) for v, r in R.items()]}")
    for ev, cv in itertools.product(R, R):
        if "think" in ev:  # edge thinking is not used (edge always no-think); think variants only change the cloud side
            continue
        keys = sorted(set(R[ev]) & set(R[cv]))
        je = np.array([S[R[ev][k]["j_edge"]] for k in keys])
        jc = np.array([S[R[cv][k]["j_cloud"]] for k in keys])
        fe = np.array([R[ev][k]["f1_edge"] for k in keys])
        fc = np.array([R[cv][k]["f1_cloud"] for k in keys])
        et = np.mean([R[ev][k]["edge_out_tokens"] for k in keys])
        ct = np.mean([R[cv][k]["cloud_out_tokens"] for k in keys])
        cost = np.mean(
            [0.27 * R[cv][k]["cloud_in_tokens"] / 1e6 + 1.1 * R[cv][k]["cloud_out_tokens"] / 1e6 for k in keys]
        )
        o = {
            "n": len(keys),
            "j_edge": je.mean(),
            "j_cloud": jc.mean(),
            "j_mix": (je.mean() + jc.mean()) / 2,
            "j_oracle": np.maximum(je, jc).mean(),
            "cloud_only": float(((jc == 1) & (je < 1)).mean()),
            "edge_only": float(((je == 1) & (jc < 1)).mean()),
            "f1_edge": fe.mean(),
            "f1_cloud": fc.mean(),
            "edge_tok": et,
            "cloud_tok": ct,
            "cloud_cost_per_q": cost,
        }
        res[f"{ds}|{ev}|{cv}"] = o
    for k, o in sorted(((k, o) for k, o in res.items() if k.startswith(ds)), key=lambda x: -x[1]["j_mix"])[:12]:
        print(
            f"  edge {k.split('|')[1]:8s} cloud {k.split('|')[2]:12s} | judge e {o['j_edge']:.3f} c {o['j_cloud']:.3f} mix {o['j_mix']:.3f} oracle {o['j_oracle']:.3f}"
            f" cloud-only {o['cloud_only']:.3f} | F1 e {o['f1_edge']:.3f} c {o['f1_cloud']:.3f} | tok e {o['edge_tok']:.0f} c {o['cloud_tok']:.0f} cost/q ${o['cloud_cost_per_q'] * 1e3:.3f}e-3"
        )
pairs = {k.split("|", 1)[1] for k in res}
score = {
    p: np.mean([res[f"{ds}|{p}"]["j_mix"] for ds in ["hotpotqa", "qasper"] if f"{ds}|{p}" in res])
    for p in pairs
    if all(f"{ds}|{p}" in res for ds in ["hotpotqa", "qasper"])
}
best = max(score, key=score.get)
print("SELECTED (edge|cloud):", best, round(score[best], 4))
json.dump(
    {"res": res, "score": score, "selected": best},
    open(f"{RUNS}/pilot2/pilot3_combo.json", "w"),
    indent=1,
    default=float,
)
