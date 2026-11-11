"""E/N/T test evaluation of the policies and thresholds fixed on validation before the test splits were opened."""

from paths import RUNS  # noqa: E402
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ent_data  # noqa: E402
from analyze_ent import fit_latency  # noqa: E402
from analyze_ent_cv import policy  # noqa: E402
from bootlib import boot  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

P7 = f"{RUNS}/pilot7"
co = fit_latency()


def arrays(rows):
    lat = (
        lambda r, act: co[act][0]
        + co[act][1] * r[f"{'N' if act == 'E' else act}_in"]
        + co[act][2] * (r["edge_out"] if act == "E" else r[f"{act}_out"])
    )
    f = tuple(np.array([r[k] for r in rows]) for k in ["fE", "fN", "fT"])
    dollar = (
        np.zeros(len(rows)),
        np.array([(0.27 * r["N_in"] + 1.1 * r["N_out"]) / 1e3 for r in rows]),
        np.array([(0.27 * r["T_in"] + 1.1 * r["T_out"]) / 1e3 for r in rows]),
    )
    latency = tuple(np.array([lat(r, x) for r in rows]) for x in "ENT")
    return f, {"dollar": dollar, "latency": latency}, np.array([not r["ecrag_edge"] for r in rows])


def scorers(tr, rows, split):
    S = {}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in tr])
    X = lambda R: np.hstack([vec.transform([r["query"] for r in R]).toarray(), np.array([[r["k"] / 10] for r in R])])
    S["tfidf"] = {
        k: Ridge(alpha=3.0).fit(X(tr), [f(r) for r in tr]).predict(X(rows))
        for k, f in [("dEN", lambda r: r["fN"] - r["fE"]), ("dNT", lambda r: r["fT"] - r["fN"])]
    }
    for fam in ["qd", "qk"]:
        seeds = []
        for s in [1234, 2345, 3456]:
            P = {(x["pool"], x["id"]): x for x in json.load(open(f"{P7}/router/ent_{fam}_s{s}/pred_{split}.json"))}
            pv = [P[(r["pool"], r["id"])] for r in rows]
            seeds.append({k: np.array([x[k] for x in pv]) for k in ["dEN", "dNT"]})
        S[f"ens3_{fam}"] = {k: np.mean([d[k] for d in seeds], 0) for k in ["dEN", "dNT"]}
    return S


def choose(d, fam, budget, f, c):
    qs = np.linspace(0, 1, 41)
    A = [-np.inf] if fam == "N/T" else list(np.r_[-np.inf, np.quantile(d["dEN"], qs), np.inf])
    best = None
    for x in A:
        for y in np.r_[-np.inf, np.quantile(d["dNT"], qs), np.inf]:
            F, C = policy(d, x, y, *f, *c)
            if C.mean() <= budget + 1e-12 and (best is None or F.mean() > best[0]):
                best = (F.mean(), x, y, (d["dEN"] > x).mean(), ((d["dEN"] > x) & (d["dNT"] > y)).mean())
    return best


tr, va = ent_data.load("train", "A"), ent_data.load("val", "B")
fv, cv_, rv = arrays(va)
Sv = scorers(tr, va, "val")
out = {}
plans = [("dollar", "E/N/T", ["ens3_qd", "tfidf", "ens3_qk"]), ("latency", "N/T", ["tfidf", "ens3_qd", "ens3_qk"])]
for split in ["test2", "test3"]:
    te = ent_data.load(split, "B")
    ft, ct, rt = arrays(te)
    St = scorers(tr, te, split)
    out[split] = {}
    for axis, fam, names in plans:
        budget = float(np.where(rv, cv_[axis][2], cv_[axis][0]).mean())
        ref = np.where(rt, ft[2], ft[0])
        refc = np.where(rt, ct[axis][2], ct[axis][0])
        print(
            f"== {split} {axis} {fam} | rule->T F1 {ref.mean():.4f} cost {refc.mean():.4f} | all-N F1 {ft[1].mean():.4f} cost {ct[axis][1].mean():.4f} | all-T {ft[2].mean():.4f} cost {ct[axis][2].mean():.4f} | all-E {ft[0].mean():.4f}"
        )
        Y = {}
        for nm in ["tfidf"] + [x for x in names if x != "tfidf"]:
            b = choose(Sv[nm], fam, budget, fv, cv_[axis])
            F, C = policy(St[nm], b[1], b[2], *ft, *ct[axis])
            Y[nm] = F
            o = {
                "deploy": {
                    "F1": float(F.mean()),
                    "cost": float(C.mean()),
                    "cloud": float((St[nm]["dEN"] > b[1]).mean()),
                    "think": float(((St[nm]["dEN"] > b[1]) & (St[nm]["dNT"] > b[2])).mean()),
                    "vs_rule": boot(F - ref),
                }
            }
            if nm != "tfidf" and "tfidf" in Y:
                o["deploy"]["vs_tfidf"] = boot(F - Y["tfidf"])
            out[split][f"{axis}|{fam}|{nm}"] = o
            d = o["deploy"]
            s = f"   {nm:9s} deploy F1 {d['F1']:.4f} cost {d['cost']:.4f} cloud {d['cloud']:.3f} think {d['think']:.3f} | vs rule {d['vs_rule'][0]:+.4f} [{d['vs_rule'][1]:+.4f},{d['vs_rule'][2]:+.4f}]"
            if "vs_tfidf" in d:
                s += f" | vs tfidf {d['vs_tfidf'][0]:+.4f} [{d['vs_tfidf'][1]:+.4f},{d['vs_tfidf'][2]:+.4f}]"
            print(s, flush=True)
json.dump(out, open(f"{P7}/ent_test.json", "w"), indent=1)
