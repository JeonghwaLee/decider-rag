"""Generic E/N/T evaluation for any data spec (legacy:<split> or <pilotN>:<ds>:<split>), cloud-$ axis.
Policy (two thresholds): cloud iff dEN > a; think iff cloud and dNT > b. Families: E/N/T, E/N (b = +inf), N/T (a = -inf).
Truncation fallback for T: 'E' (truncation B, the edge answer; cost of T only) or 'N' (call N after a truncated T: F1 of N,
cost of T + N). Scorers: --scorers name=pattern[,pattern] with {split} (mean of dEN/dNT over patterns); built in: tfidf (Ridge on
--train question 1-2 grams + k, both heads), and rules: ruleT (EC-RAG lam 0.8 -> T), ruleN, ruleT_tuned / ruleN_tuned (lam, c on val).
Gate: 5-fold x 5 cross-fitted thresholds on --val at each budget multiple (x rule->T val $); paired bootstrap vs tfidf and the tuned
rules. Deploy (each --test spec): thresholds from the full --val, paired bootstrap vs rule->T, tuned rules, tfidf (same family/budget).
--no_test prints only the gate. Env JUDGE=<name>: test scores are judge labels (thresholds still from val F1). Output: --out .txt/.json (thresholds included)."""

from paths import RUNS  # noqa: E402
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bootlib import boot  # noqa: E402
from analyze_en import load, LAMS  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

R = RUNS
QS = np.linspace(0, 1, 41)


def arrays(rows, fb):
    A = lambda k: np.array([r[k] for r in rows], float)
    fE, fN, fT = A("fE"), A("fN"), A("fT")
    cN = (0.27 * A("N_in") + 1.1 * A("N_out")) / 1e3
    cT = (0.27 * A("T_in") + 1.1 * A("T_out")) / 1e3
    tr = np.array([r["T_trunc"] for r in rows], bool)
    if fb == "N":
        fT = np.where(tr, fN, fT)
        cT = np.where(tr, cT + cN, cT)
    return fE, fN, fT, cN, cT, np.array([not r["ecrag_edge"] for r in rows]), A("complexity"), A("k")


SCJ = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


def judge_f(spec, rows, fb, jn):
    """Auxiliary (env JUDGE=<name>, e.g. glm53): per-row E/N/T judge scores from <root>/judge/<ds>/<split>/<pool>/judge_<jn>.jsonl; a truncated
    T takes the fallback action's label (E or N); missing labels -> mean of that action."""
    p = spec.split(":")
    base = f"{R}/{p[0]}/judge/{p[1]}/{p[2]}" if p[0] != "legacy" else f"{R}/pilot7/judge/hotpotqa/{p[1]}"
    L = {}
    for pool in {r["pool"] for r in rows}:
        for l in open(f"{base}/{pool}/judge_{jn}.jsonl"):
            x = json.loads(l)
            L[(pool, x["id"], x["who"])] = SCJ.get(x["label"], np.nan)
    g = lambda w: np.array([L.get((r["pool"], r["id"], w), np.nan) for r in rows])
    E, N, T = g("E"), g("N"), g("T")
    tr = np.array([r["T_trunc"] for r in rows], bool)
    T = np.where(tr, N if fb == "N" else E, T)
    return tuple(np.nan_to_num(x, nan=np.nanmean(x)) for x in (E, N, T))


def pol(dE, dT, a, b, fE, fN, fT, cN, cT):
    cl = dE > a
    th = cl & (dT > b)
    return np.where(th, fT, np.where(cl, fN, fE)), np.where(th, cT, np.where(cl, cN, 0.0))


def grid(d, fam):
    A = [-np.inf] if fam == "N/T" else list(np.r_[-np.inf, np.quantile(d["dEN"], QS), np.inf])
    B = [np.inf] if fam == "E/N" else list(np.r_[-np.inf, np.quantile(d["dNT"], QS), np.inf])
    return A, B


def choose(d, fam, budget, f, idx=None):
    fE, fN, fT, cN, cT = (x if idx is None else x[idx] for x in f)
    dd = d if idx is None else {k: v[idx] for k, v in d.items()}
    best = None
    A, B = grid(d, fam)
    for a in A:
        for b in B:
            F, C = pol(dd["dEN"], dd["dNT"], a, b, fE, fN, fT, cN, cT)
            if C.mean() <= budget + 1e-12 and (best is None or F.mean() > best[0]):
                best = (F.mean(), a, b)
    return best if best is not None else (np.nan, np.inf, np.inf)  # infeasible (e.g. N/T below all-N cost): all-E


def rule_d(cx, k, lam):
    m = cx - (lam - (2 * lam - 1) * (k - 2) / 8)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--scorers", nargs="*", default=[])
    ap.add_argument("--families", nargs="+", default=["E/N/T", "E/N", "N/T"])
    ap.add_argument("--mults", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    ap.add_argument("--fallback", default="E")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trr = [r for s in a.train.split(",") for r in load(s, "A")[0] if r["fT"] is not None]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in trr])
    X = lambda Rr: np.hstack([vec.transform([r["query"] for r in Rr]).toarray(), np.array([[r["k"] / 10] for r in Rr])])
    RG = {
        k: Ridge(alpha=3.0).fit(X(trr), [f(r) for r in trr])
        for k, f in [("dEN", lambda r: r["fN"] - r["fE"]), ("dNT", lambda r: r["fT"] - r["fN"])]
    }
    SC = {s.split("=")[0]: s.split("=")[1].split(",") for s in a.scorers}

    def scores(rows, split):
        S = {"tfidf": {k: m.predict(X(rows)) for k, m in RG.items()}}
        for nm, pats in SC.items():
            P = [{(x["pool"], x["id"]): x for x in json.load(open(p.format(split=split)))} for p in pats]
            S[nm] = {
                k: np.mean([np.array([q[(r["pool"], r["id"])][k] for r in rows]) for q in P], 0) for k in ["dEN", "dNT"]
            }
        return S

    va, vsp = load(a.val)
    f = arrays(va, a.fallback)
    fE, fN, fT, cN, cT, rv, cx, kk = f
    Sv = scores(va, vsp)
    F5 = (fE, fN, fT, cN, cT)
    base = float(np.where(rv, cT, 0).mean())
    out = {"val_rule_T_cost": base, "fallback": a.fallback, "gate": {}, "thr": {}, "deploy": {}}
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    refT = np.where(rv, fT, fE)
    P(
        f"val {a.val} n {len(va)} fallback {a.fallback} | rule->T F1 {refT.mean():.4f} $ {base:.4f} | all-E {fE.mean():.4f} all-N {fN.mean():.4f} $ {cN.mean():.4f} | all-T {fT.mean():.4f} $ {cT.mean():.4f}"
    )
    n = len(va)

    # rule families: (lam, c) on a 1-D score; ruleT_tuned: cloud=T; ruleN_tuned: cloud=N
    def rule_choose(idx, budget, act):
        best = None
        for lam in LAMS:
            s = rule_d(cx, kk, lam)[idx]
            for c in np.r_[-np.inf, np.quantile(s, QS), np.inf]:
                cl = s > c
                Fx = np.where(cl, (fN if act == "N" else fT)[idx], fE[idx])
                Cx = np.where(cl, (cN if act == "N" else cT)[idx], 0)
                if Cx.mean() <= budget + 1e-12 and (best is None or Fx.mean() > best[0]):
                    best = (Fx.mean(), lam, c)
        return best

    for m in a.mults:
        bud = base * m
        G = {}
        for nm in list(Sv) + ["ruleN_tuned", "ruleT_tuned"]:
            for fam in a.families if nm in Sv else ["-"]:
                Fo = np.zeros((5, n))
                for rep in range(5):
                    fold = np.random.default_rng(100 + rep).permutation(n) % 5
                    for k in range(5):
                        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                        if nm in Sv:
                            b = choose(Sv[nm], fam, bud, F5, tr)
                            Fo[rep, te] = pol(Sv[nm]["dEN"][te], Sv[nm]["dNT"][te], b[1], b[2], *(x[te] for x in F5))[0]
                        else:
                            act = nm[4]
                            b = rule_choose(tr, bud, act)
                            cl = rule_d(cx, kk, b[1])[te] > b[2]
                            Fo[rep, te] = np.where(cl, (fN if act == "N" else fT)[te], fE[te])
                G[f"{nm}|{fam}"] = Fo.mean(0)
        for k, v in G.items():
            o = {
                "F1": float(v.mean()),
                "vs_ruleT": boot(v - refT),
                "vs_tfidf": boot(v - G[f"tfidf|{k.split('|')[1]}"]) if k.split("|")[1] != "-" else None,
                "vs_ruleN_tuned": boot(v - G["ruleN_tuned|-"]),
                "vs_ruleT_tuned": boot(v - G["ruleT_tuned|-"]),
            }
            out["gate"][f"{k}|x{m}"] = o
            P(
                f"  gate x{m:<5} {k:20s} F1 {o['F1']:.4f} | "
                + " | ".join(
                    f"{q[3:]} {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}]" for q, w in o.items() if q.startswith("vs_") and w
                )
            )
        for nm in Sv:
            for fam in a.families:
                b = choose(Sv[nm], fam, bud, F5)
                out["thr"][f"{nm}|{fam}|x{m}"] = [float(b[1]), float(b[2])]
        for act in "NT":
            b = rule_choose(np.arange(n), bud, act)
            out["thr"][f"rule{act}_tuned|-|x{m}"] = [float(b[1]), float(b[2])]
    for spec in a.test:
        te, sp = load(spec)
        g = arrays(te, a.fallback)
        if os.environ.get("JUDGE"):
            g = judge_f(spec, te, a.fallback, os.environ["JUDGE"]) + g[3:]
        St = scores(te, sp)
        G5 = g[:5]
        rt, cx2, k2 = g[5], g[6], g[7]
        rT = np.where(rt, g[2], g[0])
        o_sp = {}
        P(
            f"== {spec} n {len(te)} | rule->T F1 {rT.mean():.4f} $ {np.where(rt, g[4], 0).mean():.4f} | rule->N {np.where(rt, g[1], g[0]).mean():.4f} | all-E {g[0].mean():.4f} "
            f"all-N {g[1].mean():.4f} $ {g[3].mean():.4f} | all-T {g[2].mean():.4f} $ {g[4].mean():.4f}"
        )
        for m in a.mults:
            Y = {}
            for key, (t1, t2) in out["thr"].items():
                if not key.endswith(f"|x{m}"):
                    continue
                nm, fam, _ = key.split("|")
                if nm.startswith("rule"):
                    act = nm[4]
                    cl = rule_d(cx2, k2, t1) > t2
                    Y[f"{nm}|-"] = (
                        np.where(cl, g[1] if act == "N" else g[2], g[0]),
                        np.where(cl, g[3] if act == "N" else g[4], 0),
                    )
                else:
                    Y[f"{nm}|{fam}"] = pol(St[nm]["dEN"], St[nm]["dNT"], t1, t2, *G5)
            for k, (Fq, Cq) in Y.items():
                o = {
                    "F1": float(Fq.mean()),
                    "cost": float(Cq.mean()),
                    "vs_ruleT": boot(Fq - rT),
                    "vs_ruleN_tuned": boot(Fq - Y["ruleN_tuned|-"][0]),
                    "vs_ruleT_tuned": boot(Fq - Y["ruleT_tuned|-"][0]),
                }
                fam = k.split("|")[1]
                if fam != "-" and not k.startswith("tfidf"):
                    o["vs_tfidf"] = boot(Fq - Y[f"tfidf|{fam}"][0])
                if a.ref and f"{a.ref}|{fam}" in Y and not k.startswith(a.ref):
                    o["vs_ref"] = boot(Fq - Y[f"{a.ref}|{fam}"][0])
                o_sp[f"{k}|x{m}"] = o
                P(
                    f"  x{m:<5} {k:20s} F1 {o['F1']:.4f} $ {o['cost']:.4f} | "
                    + " | ".join(
                        f"{q[3:]} {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}]" for q, w in o.items() if q.startswith("vs_")
                    )
                )
        out["deploy"][spec] = o_sp
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
