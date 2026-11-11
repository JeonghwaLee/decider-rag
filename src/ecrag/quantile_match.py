"""Label-free threshold recalibration probe (POST HOC: designed after the 2Wiki test showed the decider under-spending
on a shifted test split). For a scorer and budget, the val thresholds (a, b) fix the val routing fractions q_c = P(dEN > a) and
q_t = P(dEN > a and dNT > b). On the test set, a' and b' are re-set from the TEST score quantiles only (no labels) so that the same
fractions are routed (a running-window controller would do the same online). Compared with the fixed val thresholds and with
rule->T, TF-IDF (also quantile-matched and fixed) on every dataset, to check it does not hurt where there is no shift.
Usage: quantile_match.py <analyze_ent2 out json> <train spec> <val spec> <test spec> <fallback> <scorer=pattern ...>."""

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, pol  # noqa: E402
from analyze_en import load  # noqa: E402
from bootlib import boot  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


def main():
    js, trs, vas, tes, fb = sys.argv[1:6]
    SC = dict(s.split("=", 1) for s in sys.argv[6:])
    TH = json.load(open(js))["thr"]
    trr = [r for s in trs.split(",") for r in load(s, "A")[0] if r["fT"] is not None]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in trr])
    X = lambda Rr: np.hstack([vec.transform([r["query"] for r in Rr]).toarray(), np.array([[r["k"] / 10] for r in Rr])])
    RG = {
        k: Ridge(alpha=3.0).fit(X(trr), [f(r) for r in trr])
        for k, f in [("dEN", lambda r: r["fN"] - r["fE"]), ("dNT", lambda r: r["fT"] - r["fN"])]
    }
    va, vsp = load(vas)
    te, tsp = load(tes)

    def sc(rows, split):
        S = {"tfidf": {k: m.predict(X(rows)) for k, m in RG.items()}}
        for nm, p in SC.items():
            pats = p.split(",")
            Pp = [{(x["pool"], x["id"]): x for x in json.load(open(q.format(split=split)))} for q in pats]
            S[nm] = {
                k: np.mean([np.array([P[(r["pool"], r["id"])][k] for r in rows]) for P in Pp], 0)
                for k in ["dEN", "dNT"]
            }
        return S

    Sv, St = sc(va, vsp), sc(te, tsp)
    g = arrays(te, fb)
    G5 = g[:5]
    rT = np.where(g[5], g[2], g[0])
    print(f"{tes} n {len(te)} fb {fb} | rule->T F1 {rT.mean():.4f} $ {np.where(g[5], g[4], 0).mean():.4f}")
    for key, (a, b) in TH.items():
        nm, fam, m = key.split("|")
        if nm not in Sv or fam != "E/N/T":
            continue
        dv, dt = Sv[nm], St[nm]
        qc = float((dv["dEN"] > a).mean())
        qt = float(((dv["dEN"] > a) & (dv["dNT"] > b)).mean())
        a2 = -np.inf if qc >= 1 else (np.inf if qc <= 0 else float(np.quantile(dt["dEN"], 1 - qc)))
        cl = dt["dEN"] > a2
        # b2: among test cloud rows, route the val share qt/qc to T
        b2 = np.inf if qt <= 0 or cl.sum() == 0 else float(np.quantile(dt["dNT"][cl], max(0.0, 1 - qt / max(qc, 1e-9))))
        F0, C0 = pol(dt["dEN"], dt["dNT"], a, b, *G5)
        F1_, C1 = pol(dt["dEN"], dt["dNT"], a2, b2, *G5)
        print(
            f"  {key:22s} fixed F1 {F0.mean():.4f} $ {C0.mean():.4f} | qmatch F1 {F1_.mean():.4f} $ {C1.mean():.4f} | qmatch-fixed {boot(F1_ - F0)[0]:+.4f} "
            f"[{boot(F1_ - F0)[1]:+.4f},{boot(F1_ - F0)[2]:+.4f}] | qmatch-ruleT {boot(F1_ - rT)[0]:+.4f} [{boot(F1_ - rT)[1]:+.4f},{boot(F1_ - rT)[2]:+.4f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
