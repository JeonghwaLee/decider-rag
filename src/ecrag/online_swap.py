"""Diagnostic (not a predeclared comparison): online_frontier with scorer A's scores but scorer B's val frontier
(fractions per budget level), to separate ranking from the choice of fractions. Initial thresholds: A's val-score quantiles at
B's x1 val fractions. Usage: online_swap.py --val V --test T... --fallback N --fam E/N/T --a name=pat --b name=pat --out f"""

import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, choose  # noqa: E402
from analyze_en import load  # noqa: E402
from bootlib import boot  # noqa: E402
from policy_probe import preds, frontier, online_frontier  # noqa: E402


def thr_at(d, qc, qt):
    a = -np.inf if qc >= 1 else (np.inf if qc <= 0 else float(np.quantile(d["dEN"], 1 - qc)))
    cl = d["dEN"] > a
    b = np.inf if (qt <= 0 or cl.sum() == 0) else float(np.quantile(d["dNT"][cl], 1 - min(1.0, qt / max(qc, 1e-9))))
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--fallback", default="N")
    ap.add_argument("--fam", default="E/N/T")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--orders", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    va, vsp = load(a.val)
    gv = arrays(va, a.fallback)
    G5v = gv[:5]
    base = float(np.where(gv[5], gv[4], 0).mean())
    (na, pa), (nb, pb) = a.a.split("=", 1), a.b.split("=", 1)
    rows = []
    DA, DB = [], []
    for t in a.test:
        r, sp = load(t)
        rows += r
        DA.append(preds(r, pa, sp))
        DB.append(preds(r, pb, sp))
    cat = lambda L: {k: np.concatenate([x[k] for x in L]) for k in L[0]}
    DA, DB = cat(DA), cat(DB)
    G5 = arrays(rows, a.fallback)[:5]
    va_a, va_b = preds(va, pa, vsp), preds(va, pb, vsp)
    out = []
    for lab, dsc, dfr in [
        (f"{na} scores + {na} frontier", DA, va_a),
        (f"{na} scores + {nb} frontier", DA, va_b),
        (f"{nb} scores + {nb} frontier", DB, va_b),
        (f"{nb} scores + {na} frontier", DB, va_a),
    ]:
        M, Q = frontier(dfr, G5v, a.fam, base)
        j = int(np.argmin(np.abs(M - 1.0)))
        a0, b0 = thr_at(va_a if dsc is DA else va_b, *Q[j])
        R = [
            online_frontier(dsc, G5, a0, b0, M, Q, 1.0, base, np.random.default_rng(1000 + r)) for r in range(a.orders)
        ]
        F = np.mean([x[0] for x in R], 0)
        C = np.mean([x[1] for x in R], 0)
        out.append((lab, F))
        print(
            f"{lab:40s} F1 {F.mean():.4f} $ {C.mean():.4f} ({C.mean() / base:.2f}x) | x1 val fractions cloud {Q[j][0]:.3f} T {Q[j][1]:.3f}",
            flush=True,
        )
    for i, j in [(1, 2), (0, 3)]:
        w = boot(out[i][1] - out[j][1])
        print(f"{out[i][0]} - {out[j][0]}: {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}]", flush=True)


if __name__ == "__main__":
    main()
