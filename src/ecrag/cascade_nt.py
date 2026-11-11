"""N->T cascade probe (validation only): call N first (cheap, ~0.13 s), then decide whether to also call T using
features of N's answer. Policy: every query gets N; T is called iff g(q, k, N answer) > b; final = T answer if T was called and
not truncated, else N answer (fallback N). Cost = cN + [T called] cT. Compared with the plain N/T selector (decision before N,
cost = cN or cT) at the same budget. g: Ridge on TF-IDF(question) + k + N-answer features (empty / 'unknown'-like / length /
N output tokens / answer-in-question overlap), trained on train, target fT - fN. Cross-fitted b on val (5x5), paired bootstrap.
Usage: cascade_nt.py <train spec> <val spec> <budget x rule->T>."""

from paths import RUNS  # noqa: E402
import re, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_en import load  # noqa: E402
from bootlib import boot  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import json

R = RUNS
UNK = re.compile(
    r"unknown|not (mentioned|provided|specified|stated|available|given)|no information|cannot be determined|unanswerable|none|n/a|not enough",
    re.I,
)


def n_answers(spec, rows):
    p = spec.split(":")
    out = {}
    if p[0] == "legacy":
        base = f"{R}/pilot7/answers/hotpotqa/{p[1]}"
    else:
        base = f"{R}/{p[0]}/answers/{p[1]}/{p[2]}"
    for pool in {r["pool"] for r in rows}:
        for l in open(f"{base}/{pool}/orig.jsonl"):
            x = json.loads(l)
            out[(pool, x["id"])] = x["cloud_answer"] or ""
    return [out[(r["pool"], r["id"])] for r in rows]


def feats(rows, ans):
    F = []
    for r, a in zip(rows, ans):
        w = a.split()
        qw = set(r["query"].lower().split())
        F.append(
            [
                len(a) == 0,
                bool(UNK.search(a)),
                min(len(w), 30) / 30,
                r["N_out"] / 50,
                np.mean([x.lower() in qw for x in w]) if w else 0,
                r["k"] / 10,
                a.lower() in ("yes", "no"),
                any(c.isdigit() for c in a),
            ]
        )
    return np.array(F, float)


def main():
    trs, vas, mult = sys.argv[1], sys.argv[2], float(sys.argv[3])
    tr = [r for r in load(trs, "A")[0] if r["fT"] is not None]
    va = load(vas, "B")[0]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in tr])
    Xq = lambda Rr: vec.transform([r["query"] for r in Rr]).toarray()
    Xtr = np.hstack([Xq(tr), feats(tr, n_answers(trs, tr))])
    Xva = np.hstack([Xq(va), feats(va, n_answers(vas, va))])
    Xtr0, Xva0 = np.hstack([Xq(tr), np.array([[r["k"] / 10] for r in tr])]), np.hstack(
        [Xq(va), np.array([[r["k"] / 10] for r in va])]
    )
    y = [r["fT"] - r["fN"] for r in tr]
    g = Ridge(alpha=3.0).fit(Xtr, y).predict(Xva)
    g0 = Ridge(alpha=3.0).fit(Xtr0, y).predict(Xva0)
    A = lambda k: np.array([r[k] for r in va], float)
    fN, fT, fE = A("fN"), A("fT"), A("fE")
    trunc = np.array([r["T_trunc"] for r in va])
    fTn = np.where(trunc, fN, fT)
    cN = (0.27 * A("N_in") + 1.1 * A("N_out")) / 1e3
    cT = (0.27 * A("T_in") + 1.1 * A("T_out")) / 1e3
    rt = np.array([not r["ecrag_edge"] for r in va])
    budget = mult * np.where(rt, cT, 0).mean()
    n = len(va)
    print(
        f"{vas} n {n} budget {budget:.4f} | all-N {fN.mean():.4f} all-T(fbN) {fTn.mean():.4f} | rule->T {np.where(rt, fT, fE).mean():.4f}"
    )

    def cv(s, cost_on):  # cost_on(think) -> per-query cost
        F = np.zeros((5, n))
        for rep in range(5):
            fold = np.random.default_rng(100 + rep).permutation(n) % 5
            for k in range(5):
                trm, te = fold != k, fold == k
                best = None
                for b in np.r_[-np.inf, np.quantile(s[trm], np.linspace(0, 1, 41)), np.inf]:
                    th = s[trm] > b
                    C = cost_on(th, trm).mean()
                    if C <= budget + 1e-12:
                        f = np.where(th, fTn[trm], fN[trm]).mean()
                        if best is None or f > best[0]:
                            best = (f, b)
                F[rep, te] = np.where(s[te] > best[1], fTn[te], fN[te])
        return F.mean(0)

    plain = cv(g0, lambda th, m: np.where(th, cT[m] + np.where(trunc[m], cN[m], 0), cN[m]))
    casc = cv(g, lambda th, m: cN[m] + np.where(th, cT[m], 0))
    print(f"  N/T plain (tfidf q, before N): F1 {plain.mean():.4f}")
    print(f"  N->T cascade (q + N answer):    F1 {casc.mean():.4f} | vs plain {boot(casc - plain)}")


if __name__ == "__main__":
    main()
