"""Generic E/N evaluation of decider scorers. Cloud-$ axis, family E/N (cloud iff dEN > a).
Scorers (--scorers name=path_pattern[,path_pattern...]; several patterns = mean of their dEN, i.e. a seed ensemble). A pattern
contains {split} and points to a prediction json (train_router3 pred_<split>.json or train_router4 pred_<tag>.json); rows are
matched on (pool, id). Built in: tfidf (Ridge on the --train rows' question 1-2 grams + k) and ruleN_tuned (complexity rule,
lam and c tuned on validation).
Gate (validation, --val): 5-fold x 5 cross-fitted thresholds at budget x1 (rule->T validation $), paired bootstrap vs tfidf,
ruleN_tuned and rule->T. Deploy (each --test split): thresholds chosen on the full validation set at budgets x{0.5,...,2}, paired
bootstrap vs rule->T, rule->N, ruleN_tuned, tfidf (same budget) and --ref scorer. Output: --out .txt/.json."""

from paths import RUNS  # noqa: E402
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bootlib import boot  # noqa: E402
import ent_data, ent_data2  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

R = RUNS
MULTS = [0.5, 0.75, 1.0, 1.5, 2.0]
COSTAWARE = [float(x) for x in os.environ.get("COSTAWARE", "").split(",") if x]
LAMS = [round(0.5 + 0.05 * i, 2) for i in range(11)]


def load(spec, var="B"):
    p = spec.split(":")
    if p[0] == "legacy":
        rows = ent_data.load(p[1], var)
        C = {}
        for pool in {r["pool"] for r in rows}:
            for l in open(f"{R}/pilot1/trace/hotpotqa/{p[1]}/{pool}/trace.jsonl"):
                t = json.loads(l)
                C[(pool, t["id"])] = t["complexity"]
        for r in rows:
            r["complexity"] = C[(r["pool"], r["id"])]
        return rows, p[1]
    return ent_data2.load(f"{R}/{p[0]}", p[1], p[2], var), p[2]


def preds(pattern, split, rows):
    fn = pattern.format(split=split)
    P = {(x["pool"], x["id"]): x["dEN"] for x in json.load(open(fn))}
    return np.array([P[(r["pool"], r["id"])] for r in rows])


def arr(rows):
    fE, fN = np.array([r["fE"] for r in rows]), np.array([r["fN"] for r in rows])
    cN = np.array([(0.27 * r["N_in"] + 1.1 * r["N_out"]) / 1e3 for r in rows])
    has_t = all(r["fT"] is not None for r in rows)
    fT = np.array([r["fT"] for r in rows]) if has_t else None
    cT = np.array([(0.27 * r["T_in"] + 1.1 * r["T_out"]) / 1e3 for r in rows]) if has_t else None
    return (
        fE,
        fN,
        cN,
        fT,
        cT,
        np.array([not r["ecrag_edge"] for r in rows]),
        np.array([r["complexity"] for r in rows]),
        np.array([r["k"] for r in rows]),
    )


def choose(s, fE, fN, cN, budget):
    best = None
    for a in np.r_[-np.inf, np.quantile(s, np.linspace(0, 1, 81)), np.inf]:
        cl = s > a
        C = np.where(cl, cN, 0).mean()
        if C <= budget + 1e-12:
            F = np.where(cl, fN, fE).mean()
            if best is None or F > best[0]:
                best = (F, float(a))
    return best[1]


def rule_scores(cx, k):  # one score per lam; ruleN_tuned = best (lam, c) on validation
    return {lam: cx - (lam - (2 * lam - 1) * (k - 2) / 8) for lam in LAMS}


def choose_rule(cx, k, fE, fN, cN, budget):
    best = None
    for lam, s in rule_scores(cx, k).items():
        a = choose(s, fE, fN, cN, budget)
        F = np.where(s > a, fN, fE).mean()
        if best is None or F > best[0]:
            best = (F, lam, a)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="legacy:train")
    ap.add_argument("--val", default="legacy:val")
    ap.add_argument("--test", nargs="*", default=["legacy:test2", "legacy:test3"])
    ap.add_argument("--scorers", nargs="+", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--budget_rows", default=None, help="spec whose rule->T cost defines x1 (default: --val)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trr = [r for s in a.train.split(",") for r in load(s, "A")[0]]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in trr])
    X = lambda Rr: np.hstack([vec.transform([r["query"] for r in Rr]).toarray(), np.array([[r["k"] / 10] for r in Rr])])
    ridge = Ridge(alpha=3.0).fit(X(trr), [r["fN"] - r["fE"] for r in trr])
    SC = {}
    for s in a.scorers:
        nm, pats = s.split("=")
        SC[nm] = pats.split(",")

    med_out = float(np.median([r["N_out"] for r in trr]))

    def scores(rows, split):
        S = {"tfidf": ridge.predict(X(rows))}
        for nm, pats in SC.items():
            S[nm] = np.mean([preds(p, split, rows) for p in pats], 0)
        if COSTAWARE:  # benefit minus lam x predicted N cost; lam is chosen on validation (per scorer) below
            cp = np.array([(0.27 * r["N_in"] + 1.1 * med_out) / 1e3 for r in rows])
            for nm in list(S):
                for lam in COSTAWARE:
                    S[f"{nm}~ca{lam:g}"] = S[nm] - lam * cp
        return S

    va, vsp = load(a.val)
    fE, fN, cN, fT, cT, rv, cx, kk = arr(va)
    Sv = scores(va, vsp)
    base = float(np.where(rv, cT, 0).mean()) if fT is not None else float(np.where(rv, cN, 0).mean()) * 2.6
    out = {"val_rule_T_cost": base, "gate": {}, "deploy": {}}
    L = []
    P = lambda x: (print(x, flush=True), L.append(x))
    P(
        f"val n {len(va)} | rule->T $ {base:.4f} F1 {np.where(rv, fT, fE).mean() if fT is not None else float('nan'):.4f} | all-E {fE.mean():.4f} all-N {fN.mean():.4f} $ {cN.mean():.4f}"
    )
    # gate: cross-fitted at x1
    n = len(va)
    G = {}
    for nm in list(Sv) + ["ruleN_tuned"]:
        F = np.zeros((5, n))
        for rep in range(5):
            fold = np.random.default_rng(100 + rep).permutation(n) % 5
            for k in range(5):
                tr, te = fold != k, fold == k
                if nm == "ruleN_tuned":
                    lam, t = choose_rule(cx[tr], kk[tr], fE[tr], fN[tr], cN[tr], base)
                    s = rule_scores(cx[te], kk[te])[lam]
                else:
                    t = choose(Sv[nm][tr], fE[tr], fN[tr], cN[tr], base)
                    s = Sv[nm][te]
                F[rep, te] = np.where(s > t, fN[te], fE[te])
        G[nm] = F.mean(0)
    for nm, f in G.items():
        o = {"F1": float(f.mean()), "vs_tfidf": boot(f - G["tfidf"]), "vs_ruleN_tuned": boot(f - G["ruleN_tuned"])}
        if fT is not None:
            o["vs_ruleT"] = boot(f - np.where(rv, fT, fE))
        out["gate"][nm] = o
        P(
            f"  gate {nm:14s} F1 {o['F1']:.4f} | vs tfidf {o['vs_tfidf'][0]:+.4f} [{o['vs_tfidf'][1]:+.4f},{o['vs_tfidf'][2]:+.4f}] | vs ruleN_tuned "
            f"{o['vs_ruleN_tuned'][0]:+.4f} [{o['vs_ruleN_tuned'][1]:+.4f},{o['vs_ruleN_tuned'][2]:+.4f}]"
            + (
                f" | vs rule->T {o['vs_ruleT'][0]:+.4f} [{o['vs_ruleT'][1]:+.4f},{o['vs_ruleT'][2]:+.4f}]"
                if "vs_ruleT" in o
                else ""
            )
        )
    TH = {m: {nm: choose(Sv[nm], fE, fN, cN, base * m) for nm in Sv} for m in MULTS}
    RT = {m: choose_rule(cx, kk, fE, fN, cN, base * m) for m in MULTS}
    out["thresholds"] = {str(m): {**TH[m], "ruleN_tuned": list(RT[m])} for m in MULTS}
    for spec in a.test:
        te, sp = load(spec)
        fE2, fN2, cN2, fT2, cT2, rt, cx2, k2 = arr(te)
        St = scores(te, sp)
        o_sp = {}
        refT = np.where(rt, fT2, fE2) if fT2 is not None else None
        refN = np.where(rt, fN2, fE2)
        P(
            f"== {spec} n {len(te)} | rule->T F1 {refT.mean() if refT is not None else float('nan'):.4f} $ {np.where(rt, cT2, 0).mean() if cT2 is not None else float('nan'):.4f} | "
            f"rule->N {refN.mean():.4f} $ {np.where(rt, cN2, 0).mean():.4f} | all-E {fE2.mean():.4f} all-N {fN2.mean():.4f} $ {cN2.mean():.4f}"
        )
        for m in MULTS:
            lam, t = RT[m]
            cl = rule_scores(cx2, k2)[lam] > t
            Y = {"ruleN_tuned": (np.where(cl, fN2, fE2), np.where(cl, cN2, 0))}
            for nm in St:
                cl = St[nm] > TH[m][nm]
                Y[nm] = (np.where(cl, fN2, fE2), np.where(cl, cN2, 0))
            for nm, (F, C) in Y.items():
                o = {
                    "F1": float(F.mean()),
                    "cost": float(C.mean()),
                    "cloud": float((C > 0).mean()),
                    "vs_ruleN": boot(F - refN),
                }
                if refT is not None:
                    o["vs_ruleT"] = boot(F - refT)
                for rn in ["tfidf", "ruleN_tuned"] + ([a.ref] if a.ref else []):
                    if rn != nm:
                        o[f"vs_{rn}"] = boot(F - Y[rn][0])
                o_sp[f"{nm}|x{m}"] = o
                if m in (0.75, 1.0, 1.5) or nm not in ("tfidf", "ruleN_tuned"):
                    P(
                        f"  x{m:<4} {nm:14s} F1 {o['F1']:.4f} $ {o['cost']:.4f} cloud {o['cloud']:.3f} | "
                        + " | ".join(
                            f"{k[3:]} {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]"
                            for k, v in o.items()
                            if k.startswith("vs_")
                        )
                    )
        out["deploy"][spec] = o_sp
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
