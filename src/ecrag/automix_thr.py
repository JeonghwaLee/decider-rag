"""Pre-registered live operating point of the AutoMix-style cascade from VALIDATION only:
threshold maximizing val F1 with val cloud cost (incl. verification) <= x1 budget; if infeasible, the threshold maximizing AutoMix's
IBC lift on val. Prints and writes {"thr": escalate iff score < thr, ...}.
Usage: automix_thr.py --val SPEC --fallback E|N --cand E|N --ver_dir DIR --out json"""

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, pol  # noqa: E402
from analyze_en import load  # noqa: E402
from automix_cmp import ver, am_arrays, am_d, choose_am  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--fallback", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--ver_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    va, _ = load(a.val)
    gv = arrays(va, a.fallback)
    base = float(np.where(gv[5], gv[4], 0).mean())
    vs, vc, _, _ = ver(va, a.ver_dir, a.val)
    G5 = am_arrays(va, a.cand, vc)
    d = am_d(vs, a.cand)
    c, feas = choose_am(d, a.cand, base, G5)
    esc = (
        (d["dEN"] > c[1]) if a.cand == "E" else (d["dNT"] > c[2])
    )  # escalation is monotone in the score: snap thr between scores
    thr = float("inf") if esc.all() else (-float("inf") if not esc.any() else (vs[esc].max() + vs[~esc].min()) / 2)
    rule = "max val F1 within x1 budget"
    if (
        not feas
    ):  # IBC lift over the candidate -> escalation line (AutoMix Eq. 1), thresholds t on the official 1/8 grid
        fE, fN, fT, cN, cT = G5
        cand_f, up_f = (fE, fN) if a.cand == "E" else (fN, fT)
        cand_c = 0 * cN if a.cand == "E" else cN
        up_c = cN if a.cand == "E" else cT  # cascade G5: N branch = N+ver, T branch = N+ver+T
        Pc, Cc, Pu, Cu = cand_f.mean(), cand_c.mean(), up_f.mean(), up_c.mean()
        base_slope = (Pu - Pc) / (Cu - Cc)
        best = None
        for t in np.arange(0, 1.0001, 0.125):
            e = vs <= t + 1e-9
            Pm = np.where(e, up_f, cand_f).mean()
            Cm = np.where(e, up_c, cand_c).mean()
            if Cm - Cc > 1e-12:
                lift = ((Pm - Pc) / (Cm - Cc) - base_slope) / base_slope
                if best is None or lift > best[0]:
                    best = (lift, t)
        thr = best[1] + 1e-6
        rule = f"x1 INFEASIBLE on val; max IBC lift ({best[0]:+.3f}) at t={best[1]:.3f}"
    e = vs < thr
    fE, fN, fT, cN, cT = G5
    F, C = (np.where(e, fN, fE), np.where(e, cN, 0.0)) if a.cand == "E" else (np.where(e, fT, fN), np.where(e, cT, cN))
    o = {
        "thr": float(thr),
        "rule": rule,
        "val_f1": float(F.mean()),
        "val_cost": float(C.mean()),
        "budget_x1": base,
        "val_esc_rate": float(e.mean()),
        "feasible_x1": bool(feas),
    }
    print(json.dumps(o))
    json.dump(o, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
