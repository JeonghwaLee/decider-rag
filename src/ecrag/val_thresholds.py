"""Val thresholds (a, b) at m x rule->T val $ (answer cost only) for scorer patterns, as online_cmp/rankcmp (analyze_ent2.choose).
Usage: val_thresholds.py --val V --fallback E|N --fam E/N|E/N/T --mult 1.0 --scorers name=pattern ... -> json on stdout."""

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, choose  # noqa: E402
from analyze_en import load  # noqa: E402
from policy_probe import preds  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--val", required=True)
ap.add_argument("--fallback", default="N")
ap.add_argument("--fam", default="E/N/T")
ap.add_argument("--mult", type=float, default=1.0)
ap.add_argument("--scorers", nargs="+", required=True)
a = ap.parse_args()
va, vsp = load(a.val)
gv = arrays(va, a.fallback)
base = float(np.where(gv[5], gv[4], 0).mean())
out = {"base": base, "budget": base * a.mult}
for s in a.scorers:
    nm, p = s.split("=", 1)
    c = choose(preds(va, p, vsp), a.fam, base * a.mult, gv[:5])
    out[nm] = {"valF1": c[0], "a": c[1], "b": c[2]}
print(json.dumps(out))
