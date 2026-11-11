"""Abstention by agreement (Qasper, paper-scoped). Base answer = the 'orig' answer of that side; replace it by 'Unanswerable'
when every abstention signal in the set abstains. Signal sets use answers from the same side and/or the other side
(edge_abst = edge orig_abst, ...). Selection on train pools 00-01 by F1, per side; val pools 00-01 confirm."""

from paths import RUNS  # noqa: E402
import json, sys, os, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from router_data import DATA, f1  # noqa: E402

A = f"{RUNS}/pilot3/answers/qasper"
says = lambda t: t.strip().lower().rstrip(".").startswith("unanswerable")
SIG = ["edge:orig_abst", "edge:min_abst", "cloud:orig_abst", "cloud:min_abst"]
res = {}
for split in ["train", "val"]:
    rows = {}
    for p in ["pool_00", "pool_01"]:
        refs = {
            json.loads(l)["id"]: json.loads(l)["answers"]
            for l in open(f"{DATA}/qasper/{split}/{p}/answers/answers.jsonl")
        }
        for v in ["orig", "orig_abst", "min_abst"]:
            for l in open(f"{A}/{split}/{p}/{v}.jsonl"):
                x = json.loads(l)
                r = rows.setdefault((p, x["id"]), {"refs": refs[x["id"]]})
                for side in ["edge", "cloud"]:
                    r[f"{side}:{v}"] = x[f"{side}_answer"]
                    r[f"f1_{side}:{v}"] = x[f"f1_{side}"]
    K = sorted(rows)
    un = np.array([f1("Unanswerable", rows[k]["refs"]) for k in K])
    grp = np.array(
        [
            (
                "unans"
                if all(r.strip().lower() == "unanswerable" for r in rows[k]["refs"])
                else "ans" if not any(r.strip().lower() == "unanswerable" for r in rows[k]["refs"]) else "mixed"
            )
            for k in K
        ]
    )
    for side in ["edge", "cloud"]:
        base = np.array([rows[k][f"f1_{side}:orig"] for k in K])
        for n in range(1, 5):
            for ss in itertools.combinations(SIG, n):
                ab = np.array([all(says(rows[k][s]) for s in ss) for k in K])
                y = np.where(ab, un, base)
                res[f"{split}|{side}|{'+'.join(ss)}"] = {
                    "F1": y.mean(),
                    "base": base.mean(),
                    "abst": ab.mean(),
                    "abst_ans": ab[grp == "ans"].mean(),
                    "abst_unans": ab[grp == "unans"].mean(),
                }
for side in ["edge", "cloud"]:
    keys = [k.split("|", 2)[2] for k in res if k.startswith(f"train|{side}|")]
    top = sorted(keys, key=lambda s: -res[f"train|{side}|{s}"]["F1"])[:6]
    print(
        f"== {side}  base orig F1 train {res[f'train|{side}|{top[0]}']['base']:.4f} val {res[f'val|{side}|{top[0]}']['base']:.4f}"
    )
    for s in top:
        t, v = res[f"train|{side}|{s}"], res[f"val|{side}|{s}"]
        print(
            f"   {s:62s} train F1 {t['F1']:.4f} | val F1 {v['F1']:.4f} abst {v['abst']:.3f} on-ans {v['abst_ans']:.3f} on-unans {v['abst_unans']:.3f}"
        )
    print("   SELECTED", top[0])
json.dump(res, open(f"{RUNS}/pilot5/abst_combo.json", "w"), indent=1, default=float)
