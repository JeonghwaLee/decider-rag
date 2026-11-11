"""Qasper paper-scoped abstention pilot (F1 primary, GLM auxiliary). Variants: orig, orig_abst, min, min_abst.
Groups by reference set: 'ans' (no Unanswerable reference), 'mixed', 'unans' (only Unanswerable).
Selection (fixed before looking): edge prompt and cloud prompt chosen independently by mean F1 on train pools 00-01;
val pools 00-01 are reported as confirmation."""

from paths import RUNS  # noqa: E402
import json, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from router_data import DATA  # noqa: E402

A = f"{RUNS}/pilot3/answers/qasper"
S = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
V = ["orig", "orig_abst", "min", "min_abst"]
says = lambda t: t.strip().lower().rstrip(".").startswith("unanswerable")
out = {}
for split in ["train", "val"]:
    R = {v: [] for v in V}
    for p in ["pool_00", "pool_01"]:
        refs = {
            json.loads(l)["id"]: json.loads(l)["answers"]
            for l in open(f"{DATA}/qasper/{split}/{p}/answers/answers.jsonl")
        }
        for v in V:
            for l in open(f"{A}/{split}/{p}/{v}.jsonl"):
                x = json.loads(l)
                u = [r.strip().lower() == "unanswerable" for r in refs[x["id"]]]
                x["grp"] = "unans" if all(u) else ("mixed" if any(u) else "ans")
                R[v].append(x)
    print(f"== {split}  groups:", {g: sum(x["grp"] == g for x in R["orig"]) for g in ["ans", "mixed", "unans"]})
    for side in ["edge", "cloud"]:
        for v in V:
            X = R[v]
            f = np.array([x[f"f1_{side}"] for x in X])
            g = np.array([x["grp"] for x in X])
            j = np.array([S.get(x.get(f"j_{side}"), np.nan) for x in X])
            ab = np.array([says(x[f"{side}_answer"]) for x in X])
            o = {
                "F1": f.mean(),
                "GLM": np.nanmean(j),
                "abst": ab.mean(),
                **{f"F1_{k}": f[g == k].mean() for k in ["ans", "mixed", "unans"]},
                "abst_ans": ab[g == "ans"].mean(),
                "abst_unans": ab[g == "unans"].mean(),
            }
            out[f"{split}|{side}|{v}"] = o
            print(
                f"  {side:5s} {v:9s} F1 {o['F1']:.4f} (ans {o['F1_ans']:.3f} mixed {o['F1_mixed']:.3f} unans {o['F1_unans']:.3f})"
                f" | abstain {o['abst']:.3f} on-ans {o['abst_ans']:.3f} on-unans {o['abst_unans']:.3f} | GLM {o['GLM']:.3f}"
            )
sel = {side: max(V, key=lambda v: out[f"train|{side}|{v}"]["F1"]) for side in ["edge", "cloud"]}
print(
    "SELECTED on train:",
    sel,
    "| val F1:",
    {s: round(out[f"val|{s}|{v}"]["F1"], 4) for s, v in sel.items()},
    "vs orig",
    {s: round(out[f"val|{s}|orig"]["F1"], 4) for s in sel},
)
json.dump({"res": out, "selected": sel}, open(f"{RUNS}/pilot5/abst_pilot.json", "w"), indent=1, default=float)
