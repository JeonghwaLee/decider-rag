"""Learned "value of running the decider" selector. E/N family, x1 thresholds frozen (analyze_ent2
json). Per question, the cheap TF-IDF action and the decider action are known only after the decider runs; the gain
g = F1(decider action) - F1(TF-IDF action). A Ridge on pre-decider features (TF-IDF question 1-2 grams, the TF-IDF score and its
distance to the threshold, k, query length) predicts g. Trained on hotpotqa test2 + test3 with OUT-OF-SAMPLE decider predictions
(the decider never saw them), the share cut-off is set on val, evaluated on test4 / test5. Compared with the exact expectation of
random selection at the same realised share and with TF-IDF-ambiguity selection. Usage: gain_selector.py --json J --scorer n=pattern."""

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays  # noqa: E402
from analyze_en import load  # noqa: E402
from bootlib import boot  # noqa: E402
from policy_probe import preds  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack, csr_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit", nargs="+", default=["legacy:test2", "legacy:test3"])
    ap.add_argument("--test", nargs="+", default=["pilot8:hotpotqa:test4", "pilot8:hotpotqa:test5"])
    ap.add_argument("--alpha", type=float, default=10.0)
    a = ap.parse_args()
    J = json.load(open(a.json))
    nm, pat = a.scorer.split("=", 1)
    ad = J["thr"][f"{nm}|E/N|x1.0"][0]
    at = J["thr"]["tfidf|E/N|x1.0"][0]
    trr = [r for r in load("legacy:train", "A")[0] if r["fT"] is not None]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in trr])
    X = lambda Rr: np.hstack([vec.transform([r["query"] for r in Rr]).toarray(), np.array([[r["k"] / 10] for r in Rr])])
    rg = Ridge(alpha=3.0).fit(X(trr), [r["fN"] - r["fE"] for r in trr])
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))

    def prep(spec):
        rows, sp = load(spec)
        g = arrays(rows, "E")
        fE, fN, cN = g[0], g[1], g[3]
        st = rg.predict(X(rows))
        sd = preds(rows, pat, sp)["dEN"]
        Ft = np.where(st > at, fN, fE)
        Fd = np.where(sd > ad, fN, fE)
        Z = hstack(
            [
                vec.transform([r["query"] for r in rows]),
                csr_matrix(
                    np.c_[
                        st,
                        np.abs(st - at),
                        (st > at).astype(float),
                        [r["k"] / 10 for r in rows],
                        [len(r["query"].split()) / 30 for r in rows],
                    ]
                ),
            ]
        ).tocsr()
        return dict(Z=Z, Ft=Ft, Fd=Fd, st=st, cN=cN, ct=np.where(st > at, cN, 0), cd=np.where(sd > ad, cN, 0))

    fit = [prep(s) for s in a.fit]
    from scipy.sparse import vstack

    Zf = vstack([d["Z"] for d in fit])
    gf = np.concatenate([d["Fd"] - d["Ft"] for d in fit])
    V = prep("legacy:val")
    bud = J["val_rule_T_cost"]
    dcf = np.concatenate(
        [d["cd"] - d["ct"] for d in fit]
    )  # cost-aware target: gain - lam * extra $; lam and the share chosen on val under the budget
    best = None
    for lam in [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        mm = Ridge(alpha=a.alpha).fit(Zf, gf - lam * dcf)
        pvv = mm.predict(V["Z"])
        for u in np.linspace(0.05, 1.0, 20):
            sel = pvv >= np.quantile(pvv, 1 - u)
            C = np.where(sel, V["cd"], V["ct"]).mean()
            F = np.where(sel, V["Fd"], V["Ft"]).mean()
            if C <= bud + 1e-12 and (best is None or F > best[0]):
                best = (F, lam, u)
    P(f"cost-aware choice on val (budget $ {bud:.4f}): lam {best[1]} share {best[2]:.2f} val F1 {best[0]:.4f}")
    m = Ridge(alpha=a.alpha).fit(Zf, gf - best[1] * dcf)
    pv = m.predict(V["Z"])
    USH = [0.25, 0.5, 0.75, round(float(best[2]), 2)]
    P(
        f"{nm} | fit {a.fit} n {len(gf)} mean gain {gf.mean():+.4f} | val corr(pred, gain) {np.corrcoef(pv, V['Fd'] - V['Ft'])[0, 1]:+.3f}"
    )
    # fixed decider share u (the load-relevant question): lam per u chosen on val under the budget, then test
    for u in [0.25, 0.5, 0.75]:
        bb = None
        for lam in [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
            mm = Ridge(alpha=a.alpha).fit(Zf, gf - lam * dcf)
            pvv = mm.predict(V["Z"])
            sel = pvv >= np.quantile(pvv, 1 - u)
            C = np.where(sel, V["cd"], V["ct"]).mean()
            F = np.where(sel, V["Fd"], V["Ft"]).mean()
            if C <= bud + 1e-12 and (bb is None or F > bb[0]):
                bb = (F, lam, mm, float(np.quantile(pvv, 1 - u)))
        if bb is None:
            P(f"  [budget] u {u}: no feasible lam on val")
            continue
        line = f"  [budget] u {u} lam {bb[1]} val F1 {bb[0]:.4f}"
        for spec in a.test:
            D = prep(spec)
            sel = bb[2].predict(D["Z"]) >= bb[3]
            pr = sel.mean()
            F = np.where(sel, D["Fd"], D["Ft"])
            C = np.where(sel, D["cd"], D["ct"])
            Fr = (1 - pr) * D["Ft"] + pr * D["Fd"]
            Cr = (1 - pr) * D["ct"] + pr * D["cd"]
            w = boot(F - Fr)
            line += f" | {spec.split(':')[-1]} share {pr:.2f} F1 {F.mean():.4f} $ {C.mean():.4f} vs random({Fr.mean():.4f} $ {Cr.mean():.4f}) {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}]"
        P(line)
    for spec in ["legacy:val"] + a.test:
        D = V if spec == "legacy:val" else prep(spec)
        pg = m.predict(D["Z"])
        gain = D["Fd"] - D["Ft"]
        P(
            f"== {spec} n {len(gain)} | tfidf {D['Ft'].mean():.4f} decider {D['Fd'].mean():.4f} | corr(pred, gain) {np.corrcoef(pg, gain)[0, 1]:+.3f}"
        )
        for u in USH:
            cut = float(np.quantile(pv, 1 - u))
            sel = pg >= cut
            pr = sel.mean()
            F = np.where(sel, D["Fd"], D["Ft"])
            C = np.where(sel, D["cd"], D["ct"])
            Fr = (1 - pr) * D["Ft"] + pr * D["Fd"]
            amb = np.abs(D["st"] - at) <= np.quantile(np.abs(D["st"] - at), pr)
            Fa = np.where(amb, D["Fd"], D["Ft"])
            w, w2 = boot(F - Fr), boot(F - D["Ft"])
            P(
                f"  u {u:<4} share {pr:.3f} | learned F1 {F.mean():.4f} $ {C.mean():.4f} | random {Fr.mean():.4f} | ambiguity {Fa.mean():.4f} | learned - random {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}] | learned - tfidf {w2[0]:+.4f} [{w2[1]:+.4f},{w2[2]:+.4f}]"
            )
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
