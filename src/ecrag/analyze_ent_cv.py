"""E/N/T stage-2 gate (validation only). Cross-fitted thresholds: validation is split into 5 folds (5 repetitions with
different fold seeds); for each held-out fold, the two thresholds (a: cloud iff dEN > a; b: think iff dNT > b) are chosen on
the other folds to maximize F1 subject to mean cost <= budget, then applied to the held-out fold. The per-query outcome is
averaged over repetitions; paired bootstrap (bootlib, seed 0) compares scorers on the same queries.
Axes and families (fixed in advance):
  latency : family N/T and E/N/T, budget = EC-RAG rule->T mean latency on validation
  dollar  : family E/N/T, budget = EC-RAG rule->T cloud $ cost on validation
Scorers: tfidf (Ridge), decider seeds (ent_qk_s*), ens3 (mean of the q+k seeds' outputs; declared before looking),
q+D+k decider (ent_qd_s1234) when present."""

from paths import RUNS  # noqa: E402
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ent_data  # noqa: E402
from analyze_ent import fit_latency  # noqa: E402
from bootlib import boot  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

P7 = f"{RUNS}/pilot7"


def policy(d, a, b, fE, fN, fT, CE, CN, CT):
    cl = d["dEN"] > a
    th = cl & (d["dNT"] > b)
    return np.where(th, fT, np.where(cl, fN, fE)), np.where(th, CT, np.where(cl, CN, CE))


def cv(d, fam, budget, fE, fN, fT, CE, CN, CT, reps=5, K=5):
    n = len(fE)
    qs = np.linspace(0, 1, 41)
    A = [-np.inf] if fam == "N/T" else list(np.r_[-np.inf, np.quantile(d["dEN"], qs), np.inf])
    B = list(np.r_[-np.inf, np.quantile(d["dNT"], qs), np.inf])
    F = np.zeros((reps, n))
    C = np.zeros((reps, n))
    for rep in range(reps):
        fold = np.random.default_rng(100 + rep).permutation(n) % K
        for k in range(K):
            tr, te = fold != k, fold == k
            best = None
            for x in A:
                for y in B:
                    f, c = policy(
                        {key: v[tr] for key, v in d.items()}, x, y, fE[tr], fN[tr], fT[tr], CE[tr], CN[tr], CT[tr]
                    )
                    if c.mean() <= budget + 1e-12 and (best is None or f.mean() > best[0]):
                        best = (f.mean(), x, y)
            f, c = policy(
                {key: v[te] for key, v in d.items()}, best[1], best[2], fE[te], fN[te], fT[te], CE[te], CN[te], CT[te]
            )
            F[rep, te] = f
            C[rep, te] = c
    return F.mean(0), C.mean(0)


def main():
    tr, va = ent_data.load("train", "A"), ent_data.load("val", "B")
    co = fit_latency()
    lat = (
        lambda r, act: co[act][0]
        + co[act][1] * r[f"{'N' if act == 'E' else act}_in"]
        + co[act][2] * (r["edge_out"] if act == "E" else r[f"{act}_out"])
    )
    fE, fN, fT = (np.array([r[k] for r in va]) for k in ["fE", "fN", "fT"])
    rule = np.array([not r["ecrag_edge"] for r in va])
    cost = {
        "latency": tuple(np.array([lat(r, x) for r in va]) for x in "ENT"),
        "dollar": (
            np.zeros(len(va)),
            np.array([(0.27 * r["N_in"] + 1.1 * r["N_out"]) / 1e3 for r in va]),
            np.array([(0.27 * r["T_in"] + 1.1 * r["T_out"]) / 1e3 for r in va]),
        ),
    }
    S = {}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in tr])
    X = lambda R: np.hstack([vec.transform([r["query"] for r in R]).toarray(), np.array([[r["k"] / 10] for r in R])])
    S["tfidf"] = {
        k: Ridge(alpha=3.0).fit(X(tr), [f(r) for r in tr]).predict(X(va))
        for k, f in [("dEN", lambda r: r["fN"] - r["fE"]), ("dNT", lambda r: r["fT"] - r["fN"])]
    }
    for name in sorted(os.listdir(f"{P7}/router")):
        p = f"{P7}/router/{name}/pred_val.json"
        if os.path.exists(p):
            P = {(x["pool"], x["id"]): x for x in json.load(open(p))}
            pv = [P[(r["pool"], r["id"])] for r in va]
            S[name] = {"dEN": np.array([x["dEN"] for x in pv]), "dNT": np.array([x["dNT"] for x in pv])}
    seeds = [k for k in S if k.startswith("ent_qk_s")]
    if len(seeds) == 3:
        S["ens3_qk"] = {k: np.mean([S[s][k] for s in seeds], 0) for k in ["dEN", "dNT"]}
        # post-hoc (designed after the first cv run): decider ensemble for E/N, TF-IDF for N/T
        S["hybrid"] = {"dEN": S["ens3_qk"]["dEN"], "dNT": S["tfidf"]["dNT"]}
    qd = [k for k in S if k.startswith("ent_qd_s")]
    if len(qd) == 3:  # declared before the q+D seeds 2345/3456 finished
        S["ens3_qd"] = {k: np.mean([S[s][k] for s in qd], 0) for k in ["dEN", "dNT"]}
    out = {"scorers": list(S), "results": {}}
    for axis, fam in [("latency", "N/T"), ("latency", "E/N/T"), ("dollar", "E/N/T")]:
        CE, CN, CT = cost[axis]
        budget = float(np.where(rule, CT, CE).mean())
        ref = np.where(rule, fT, fE)
        allN = fN
        print(
            f"== {axis} {fam} budget (rule->T) {budget:.4f} | rule F1 {ref.mean():.4f} all-N F1 {allN.mean():.4f} cost {CN.mean():.4f}"
        )
        R = {}
        for sc, d in S.items():
            f, c = cv(d, fam, budget, fE, fN, fT, CE, CN, CT)
            R[sc] = f
            o = {
                "F1": float(f.mean()),
                "cost": float(c.mean()),
                "vs_rule": boot(f - ref),
                "vs_tfidf": boot(f - R["tfidf"]) if sc != "tfidf" else None,
            }
            if axis == "latency":
                o["vs_allN"] = boot(f - allN)
            out["results"][f"{axis}|{fam}|{sc}"] = o
            s = f"   {sc:14s} F1 {o['F1']:.4f} cost {o['cost']:.4f} | vs rule {o['vs_rule'][0]:+.4f} [{o['vs_rule'][1]:+.4f},{o['vs_rule'][2]:+.4f}]"
            if o["vs_tfidf"]:
                s += f" | vs tfidf {o['vs_tfidf'][0]:+.4f} [{o['vs_tfidf'][1]:+.4f},{o['vs_tfidf'][2]:+.4f}]"
            if axis == "latency":
                s += f" | vs all-N {o['vs_allN'][0]:+.4f} [{o['vs_allN'][1]:+.4f},{o['vs_allN'][2]:+.4f}]"
            print(s, flush=True)
    json.dump(out, open(os.environ.get("ENT_CV_OUT", f"{P7}/ent_val_cv.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
