"""Selector comparison at IDENTICAL routing fractions: isolates "which questions" from "how many".
For each budget level m (x rule->T val $), the reference scorer's val-optimal thresholds (analyze_ent2.choose on val) give val routing
fractions q_cloud, q_T (E/N: q_T = 0). On each test split, EVERY scorer routes exactly those fractions by its own test-score
quantiles (top q_cloud by dEN to the cloud; among them, top q_T/q_cloud by dNT to T). Then F1 and $ are paired across scorers
(bootstrap of the F1 difference vs --ref). Also reported: each scorer with its OWN val thresholds (the primary comparison; spend varies).
Usage: rankcmp.py --val V --test T... --fallback E|N --fam E/N|E/N/T --ref name --scorers name=pattern[,pattern] ... --out file"""

import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, pol, choose  # noqa: E402
from analyze_en import load  # noqa: E402
from bootlib import boot  # noqa: E402
from policy_probe import preds  # noqa: E402


def route_frac(d, qc, qt, G5):
    n = len(d["dEN"])
    oc = np.argsort(-d["dEN"], kind="stable")
    cl = np.zeros(n, bool)
    cl[oc[: int(round(qc * n))]] = True
    th = np.zeros(n, bool)
    idx = np.where(cl)[0]
    nt = int(round(qt * n))
    if nt > 0:
        th[idx[np.argsort(-d["dNT"][idx], kind="stable")[:nt]]] = True
    fE, fN, fT, cN, cT = G5
    return np.where(th, fT, np.where(cl, fN, fE)), np.where(th, cT, np.where(cl, cN, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--fallback", default="E")
    ap.add_argument("--fam", default="E/N")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--scorers", nargs="+", required=True)
    ap.add_argument("--mults", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    SC = dict(s.split("=", 1) for s in a.scorers)
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    va, vsp = load(a.val)
    gv = arrays(va, a.fallback)
    G5v = gv[:5]
    base = float(np.where(gv[5], gv[4], 0).mean())
    Dv = {nm: preds(va, p, vsp) for nm, p in SC.items()}
    T = [(t, *load(t)) for t in a.test]
    Dt = {t: {nm: preds(rows, p, sp) for nm, p in SC.items()} for t, rows, sp in T}
    for m in a.mults:
        bud = base * m
        c = choose(Dv[a.ref], a.fam, bud, G5v)
        cl = Dv[a.ref]["dEN"] > c[1]
        qc = float(cl.mean())
        qt = float((cl & (Dv[a.ref]["dNT"] > c[2])).mean())
        TH = {nm: choose(Dv[nm], a.fam, bud, G5v) for nm in SC}
        P(
            f"== x{m} budget $ {bud:.4f} | fractions from {a.ref} on val: cloud {qc:.3f} T {qt:.3f} | val gate-free F1 (own thr): "
            + " ".join(f"{nm} {TH[nm][0]:.4f}" for nm in SC)
        )
        pooled = {nm: [] for nm in SC}
        pooled_own = {nm: [] for nm in SC}
        for t, rows, sp in T:
            g = arrays(rows, a.fallback)
            G5 = g[:5]
            R = {nm: route_frac(Dt[t][nm], qc, qt, G5) for nm in SC}
            O = {nm: pol(Dt[t][nm]["dEN"], Dt[t][nm]["dNT"], TH[nm][1], TH[nm][2], *G5) for nm in SC}
            for nm in SC:
                pooled[nm].append(R[nm][0])
                pooled_own[nm].append(O[nm][0])
            P(
                f"  {t}: same fractions | "
                + " | ".join(
                    f"{nm} F1 {R[nm][0].mean():.4f} $ {R[nm][1].mean():.4f}"
                    + ("" if nm == a.ref else " ({:+.4f} [{:+.4f},{:+.4f}])".format(*boot(R[a.ref][0] - R[nm][0])))
                    for nm in SC
                )
            )
            P(
                f"  {t}: own val thr   | "
                + " | ".join(f"{nm} F1 {O[nm][0].mean():.4f} $ {O[nm][1].mean():.4f}" for nm in SC)
            )
        if len(T) > 1:
            P(
                "  pooled tests, same fractions: "
                + " | ".join(
                    f"{a.ref} - {nm} "
                    + "{:+.4f} [{:+.4f},{:+.4f}]".format(
                        *boot(np.concatenate(pooled[a.ref]) - np.concatenate(pooled[nm]))
                    )
                    for nm in SC
                    if nm != a.ref
                )
            )
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
