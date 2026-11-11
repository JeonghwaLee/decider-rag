"""Choose q for the adaptQ<q> load policy on the validation load run: the q with the highest mean F1-within-2s
(F1 if latency <= 2 s else 0) averaged over the validation rates; ties -> larger q. Usage: select_adapt.py <val load dir> <out json>."""

import glob, json, os, sys
import numpy as np

d, out = sys.argv[1], sys.argv[2]
S = {}
for f in glob.glob(f"{d}/*_r*.jsonl"):
    pol, rate = os.path.basename(f)[:-6].rsplit("_r", 1)
    X = [json.loads(l) for l in open(f)]
    S.setdefault(pol, {})[rate] = float(np.mean([x["f1"] if x["latency"] <= 2 else 0 for x in X]))
M = {p: float(np.mean(list(v.values()))) for p, v in S.items()}
best = max((p for p in M if p.startswith("adaptQ")), key=lambda p: (M[p], int(p[6:])))
json.dump({"per_policy_rate": S, "mean": M, "chosen": best}, open(out, "w"), indent=1)
print(json.dumps(M, indent=1), "chosen", best)
