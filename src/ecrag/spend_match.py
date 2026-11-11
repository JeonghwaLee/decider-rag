"""Realised-spend-matched comparison: ENTC spent 1.03x vs ENT 1.01x under the same online
controller on 2Wiki. Here --ref keeps its x1 controller run; the other scorer (--cand) is re-run with the controller target m x budget
for m on a grid (--grid), and the m whose realised mean spend is closest to but not above the ref's realised spend is taken (spend
only, F1 not used). Also the reverse (ref re-targeted up to the cand's x1 spend). Same val thresholds/frontiers, 5 arrival orders,
paired bootstrap. Usage: spend_match.py --val V --test T --fallback N --fam E/N/T --ref name=pattern --cand name=pattern --out f"""

import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, choose  # noqa: E402
from analyze_en import load  # noqa: E402
from bootlib import boot  # noqa: E402
from policy_probe import preds, frontier, online_frontier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--fallback", default="N")
    ap.add_argument("--fam", default="E/N/T")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--grid", nargs="+", type=float, default=[round(0.90 + 0.005 * i, 3) for i in range(31)])
    ap.add_argument("--orders", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    va, vsp = load(a.val)
    gv = arrays(va, a.fallback)
    G5v = gv[:5]
    base = float(np.where(gv[5], gv[4], 0).mean())
    rows, sps = [], []
    for t in a.test:
        r, sp = load(t)
        rows.append(r)
        sps.append(sp)
    G5 = arrays([x for r in rows for x in r], a.fallback)[:5]
    S = {}
    for s in [a.ref, a.cand]:
        nm, p = s.split("=", 1)
        dv = preds(va, p, vsp)
        dt = {
            k: np.concatenate([preds(r, p, sp)[k] for r, sp in zip(rows, sps)])
            for k in ["fE", "fN", "fT", "dEN", "dNT"]
        }
        S[nm] = (dv, dt)
    rn, cn = a.ref.split("=")[0], a.cand.split("=")[0]

    def run(nm, m):
        dv, dt = S[nm]
        bud = base * m
        c = choose(dv, a.fam, bud, G5v)
        M, Q = frontier(dv, G5v, a.fam, base)
        R = [
            online_frontier(dt, G5, c[1], c[2], M, Q, m, bud, np.random.default_rng(1000 + r)) for r in range(a.orders)
        ]
        return np.mean([x[0] for x in R], 0), np.mean([x[1] for x in R], 0)

    Fr, Cr = run(rn, 1.0)
    Fc, Cc = run(cn, 1.0)
    P(
        f"{'+'.join(a.test)} n {len(Fr)} | base $ {base:.4f} | x1: {rn} F1 {Fr.mean():.4f} $ {Cr.mean():.4f} ({Cr.mean() / base:.3f}x) ; {cn} F1 {Fc.mean():.4f} $ {Cc.mean():.4f} ({Cc.mean() / base:.3f}x)"
    )
    w = boot(Fc - Fr)
    P(f"  x1 (unmatched) {cn} - {rn} {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}]")
    for who, other, tgt in [(cn, rn, Cr.mean()), (rn, cn, Cc.mean())]:
        G = {m: run(who, m) for m in a.grid}
        ok = (
            [m for m in a.grid if G[m][1].mean() <= tgt]
            if who == cn
            else [
                m
                for m in a.grid + [round(1.0 + 0.005 * i, 3) for i in range(1, 21)]
                if (G.setdefault(m, run(who, m)) and G[m][1].mean() >= tgt)
            ]
        )
        if not ok:
            P(f"  {who}: no grid point reaches the target spend {tgt:.4f}")
            continue
        m = max(ok) if who == cn else min(ok)
        F, C = G[m]
        Fo = Fc if other == cn else Fr
        d = boot(F - Fr) if who == cn else boot(Fc - F)
        P(
            f"  matched: {who} at m={m} F1 {F.mean():.4f} $ {C.mean():.4f} (target {tgt:.4f}) | {cn} - {rn} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]"
        )
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
