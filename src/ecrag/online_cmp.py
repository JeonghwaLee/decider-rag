"""Deployable comparison under the same online budget controller.
Every scorer: own val thresholds (analyze_ent2.choose at m x rule->T val $) and own val frontier; policy_probe.online_frontier on the
concatenated test stream, R arrival orders shared by all scorers. Output: F1, realised $ (and / budget), paired bootstrap vs --ref.
Usage: online_cmp.py --val V --test T... --fallback E|N --fam E/N|E/N/T --ref name --scorers name=pattern ... --out file"""

import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, pol, choose, judge_f  # noqa: E402
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
    ap.add_argument("--scorers", nargs="+", required=True)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--orders", type=int, default=5)
    ap.add_argument(
        "--judge",
        default=None,
        help="auxiliary: score the TEST outcomes with judge labels (analyze_ent2.judge_f; glm53|dsv4|llama33); decisions, thresholds, frontiers and costs unchanged",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    SC = dict(s.split("=", 1) for s in a.scorers)
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    va, vsp = load(a.val)
    gv = arrays(va, a.fallback)
    G5v = gv[:5]
    base = float(np.where(gv[5], gv[4], 0).mean())
    bud = base * a.mult
    rows, D = [], {nm: {k: [] for k in ["fE", "fN", "fT", "dEN", "dNT"]} for nm in SC}
    for t in a.test:
        r, sp = load(t)
        rows += r
        for nm, p in SC.items():
            x = preds(r, p, sp)
            for k in D[nm]:
                D[nm][k].append(x[k])
    D = {nm: {k: np.concatenate(v) for k, v in d.items()} for nm, d in D.items()}
    G5 = arrays(rows, a.fallback)[:5]
    if a.judge:  # judge pass: only the quality of the realised test action changes
        JQ = [judge_f(t, load(t)[0], a.fallback, a.judge) for t in a.test]
        G5 = tuple(np.concatenate([j[i] for j in JQ]) for i in range(3)) + tuple(G5[3:])
    P(
        f"{'JUDGE ' + a.judge + ' (auxiliary) | ' if a.judge else ''}stream {'+'.join(a.test)} n {len(rows)} | fam {a.fam} fb {a.fallback} x{a.mult} budget $ {bud:.4f} | orders {a.orders}"
    )
    F, C = {}, {}
    for nm, p in SC.items():
        dv = preds(va, p, vsp)
        c = choose(dv, a.fam, bud, G5v)
        M, Q = frontier(dv, G5v, a.fam, base)
        R = [
            online_frontier(D[nm], G5, c[1], c[2], M, Q, a.mult, bud, np.random.default_rng(1000 + r))
            for r in range(a.orders)
        ]
        F[nm] = np.mean([x[0] for x in R], 0)
        C[nm] = np.mean([x[1] for x in R], 0)
        F0, C0 = pol(D[nm]["dEN"], D[nm]["dNT"], c[1], c[2], *G5)
        P(
            f"  {nm:12s} online F1 {F[nm].mean():.4f} $ {C[nm].mean():.4f} ({C[nm].mean() / bud:.2f}x budget; per order {' '.join(f'{x[1].mean() / bud:.2f}' for x in R)}) | fixed thr F1 {F0.mean():.4f} $ {C0.mean():.4f} ({C0.mean() / bud:.2f}x)"
        )
    for nm in SC:
        if nm != a.ref:
            w = boot(F[a.ref] - F[nm])
            P(
                f"  {a.ref} - {nm:12s} {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}] | $ diff {C[a.ref].mean() - C[nm].mean():+.4f}"
            )
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
