"""Comparison at a TOTAL cloud budget that includes the
selector's own cloud tokens. Same online controller as online_cmp.py (policy_probe.online_frontier, own val frontier, 5 orders).
--scorers name=pattern[@tokfile_pattern@probes]: for a selector served on the cloud model, tokfile_pattern ({split} or {spec} = spec with ':' -> '_') points to its raw
pred file with prompt_tokens/completion_tokens per probe [edge, think, choice] and probes = indices actually called (e.g. 0 or 0+1 or 2).
Selector $ per query = sum over probes of (0.27*prompt + 1.1*completion)/1e3 (no cache discount). Scorers without '@' cost 0 cloud $.
Answer budget = total budget - mean val selector $; infeasible (<= 0) -> the selector is not called, all-E at $0. Output per total multiple m: F1, answer $, selector $,
total $ (/ total budget) and paired bootstrap vs --ref.
Usage: total_cost_cmp.py --val V --test T... --fallback E|N --fam E/N|E/N/T --ref name --mults 1 2 3 5 8 --scorers ... --out file"""

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, choose  # noqa: E402
from analyze_en import load  # noqa: E402
from bootlib import boot  # noqa: E402
from policy_probe import preds, frontier, online_frontier  # noqa: E402


def selcost(rows, tokpat, split, probes, spec):
    P = {(x["pool"], x["id"]): x for x in json.load(open(tokpat.format(split=split, spec=spec.replace(":", "_"))))}
    return np.array(
        [
            sum(
                0.27 * P[(r["pool"], r["id"])]["prompt_tokens"][j]
                + 1.1 * P[(r["pool"], r["id"])].get("completion_tokens", [1, 1, 1])[j]
                for j in probes
            )
            / 1e3
            for r in rows
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--fallback", default="N")
    ap.add_argument("--fam", default="E/N/T")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--scorers", nargs="+", required=True)
    ap.add_argument("--mults", nargs="+", type=float, default=[1, 2, 3, 5, 8])
    ap.add_argument("--orders", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    SC = {}
    for s in a.scorers:
        nm, rest = s.split("=", 1)
        parts = rest.split("@")
        SC[nm] = (
            parts[0],
            parts[1] if len(parts) > 1 else None,
            [int(j) for j in parts[2].split("+")] if len(parts) > 2 else [],
        )
    va, vsp = load(a.val)
    gv = arrays(va, a.fallback)
    G5v = gv[:5]
    base = float(np.where(gv[5], gv[4], 0).mean())
    rows, D, O = [], {nm: {k: [] for k in ["fE", "fN", "fT", "dEN", "dNT"]} for nm in SC}, {nm: [] for nm in SC}
    for t in a.test:
        r, sp = load(t)
        rows += r
        for nm, (p, tp, pr) in SC.items():
            x = preds(r, p, sp)
            for k in D[nm]:
                D[nm][k].append(x[k])
            O[nm].append(selcost(r, tp, sp, pr, t) if tp else np.zeros(len(r)))
    D = {nm: {k: np.concatenate(v) for k, v in d.items()} for nm, d in D.items()}
    O = {nm: np.concatenate(v) for nm, v in O.items()}
    G5 = arrays(rows, a.fallback)[:5]
    fE = G5[0]
    OV = {nm: (float(selcost(va, tp, vsp, pr, a.val).mean()) if tp else 0.0) for nm, (p, tp, pr) in SC.items()}
    P(
        f"stream {'+'.join(a.test)} n {len(rows)} | fam {a.fam} fb {a.fallback} | base (rule->T val $) {base:.4f} | all-E F1 {fE.mean():.4f}"
    )
    P(
        "  selector cloud $ per query (val mean / test mean): "
        + " | ".join(f"{nm} {OV[nm]:.4f}/{O[nm].mean():.4f} ({OV[nm] / base:.2f}x base)" for nm in SC)
    )
    for m in a.mults:
        tot = base * m
        F, C = {}, {}
        P(f"== total budget x{m} = $ {tot:.4f}")
        for nm, (p, tp, pr) in SC.items():
            ab = tot - OV[nm]
            if ab <= 1e-9:
                F[nm] = fE.copy()
                C[nm] = np.zeros(len(fE))
                note = "infeasible -> all-E, selector not called"
            else:
                dv = preds(va, p, vsp)
                c = choose(dv, a.fam, ab, G5v)
                M, Q = frontier(dv, G5v, a.fam, base)
                R = [
                    online_frontier(D[nm], G5, c[1], c[2], M, Q, ab / base, ab, np.random.default_rng(1000 + r))
                    for r in range(a.orders)
                ]
                F[nm] = np.mean([x[0] for x in R], 0)
                C[nm] = np.mean([x[1] for x in R], 0)
                note = f"answer budget $ {ab:.4f}"
            so = 0.0 if ab <= 1e-9 else O[nm].mean()
            T = C[nm].mean() + so
            P(
                f"  {nm:12s} F1 {F[nm].mean():.4f} | answer $ {C[nm].mean():.4f} + selector $ {so:.4f} = total $ {T:.4f} ({T / tot:.2f}x) | {note}"
            )
        for nm in SC:
            if nm != a.ref:
                w = boot(F[a.ref] - F[nm])
                P(f"  {a.ref} - {nm:12s} {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}]")
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
