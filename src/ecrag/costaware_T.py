"""E4 validation pilot: per-question cost-aware T selection.
Policy: cloud iff dEN > a; T iff cloud and dNT - lam * dC > b, dC = cost(T, incl. fallback-N retry) - cost(N) in $/1k (proxy prices).
dC variants: 'oracle' = realised dC of that question (diagnostic upper bound); 'pred' = input part exact (0.27*(T_in - N_in)) + output part
predicted from TRAIN rows (Ridge on TF-IDF question 1-2 grams + k + passage-free features -> expected T extra $ incl. truncation retry);
'mean' = train mean dC (constant, equals lam shifting b: sanity check, must equal lam=0).
Budget = m x rule->T val $. Selection of (a, b, lam) by analyze_ent2.choose over a grid, cross-fitted on val (random halves, 10 repeats):
choose on half A, evaluate F1 and $ on half B, and vice versa. Also the in-sample optimum (choose and evaluate on all val).
Usage: costaware_T.py --train T --val V --dec pattern --fallback N --mults 0.5 1 2 --out file"""

import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, choose, pol  # noqa: E402
from analyze_en import load  # noqa: E402
from policy_probe import preds  # noqa: E402

LAMS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]


def dcost(rows, fb):
    fE, fN, fT, cN, cT = arrays(rows, fb)[:5]
    return cT - cN


def predictor(train, fb):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    tr = [r for r in train if r["fT"] is not None]
    y = dcost(tr, fb) - np.array([0.27 * (r["T_in"] - r["N_in"]) / 1e3 for r in tr])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in tr])
    X = lambda R: np.hstack(
        [
            vec.transform([r["query"] for r in R]).toarray(),
            np.array([[r["k"] / 10, len(r["query"]) / 100, r["N_in"] / 1e4] for r in R]),
        ]
    )
    m = Ridge(alpha=3.0).fit(X(tr), y)
    return lambda R: m.predict(X(R)) + np.array([0.27 * (r["T_in"] - r["N_in"]) / 1e3 for r in R]), float(
        dcost(tr, fb).mean()
    )


def best(d, dC, G5, idx, budget):
    out = None
    for lam in LAMS:
        dd = {"dEN": d["dEN"], "dNT": d["dNT"] - lam * dC}
        c = choose(dd, "E/N/T", budget, G5, idx)
        if not np.isnan(c[0]) and (out is None or c[0] > out[0]):
            out = (c[0], lam, c[1], c[2])
    return out


def evaluate(d, dC, G5, idx, lam, a, b):
    F, C = pol(d["dEN"][idx], d["dNT"][idx] - lam * dC[idx], a, b, *(x[idx] for x in G5))
    return F.mean(), C.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--dec", required=True)
    ap.add_argument("--fallback", default="N")
    ap.add_argument("--mults", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    tr = [r for s in a.train.split(",") for r in load(s, "A")[0]]
    va, vsp = load(a.val)
    gv = arrays(va, a.fallback)
    G5 = gv[:5]
    base = float(np.where(gv[5], gv[4], 0).mean())
    d = preds(va, a.dec, vsp)
    f, mean_dc = predictor(tr, a.fallback)
    DC = {"none": np.zeros(len(va)), "oracle": dcost(va, a.fallback), "pred": f(va), "mean": np.full(len(va), mean_dc)}
    P(
        f"val {a.val} n {len(va)} | dec {a.dec} | base $ {base:.4f} | dC oracle mean {DC['oracle'].mean():.3f} sd {DC['oracle'].std():.3f} | "
        f"pred corr {np.corrcoef(DC['pred'], DC['oracle'])[0, 1]:.3f}"
    )
    n = len(va)
    allidx = np.arange(n)
    for m in a.mults:
        bud = base * m
        P(f"== x{m} budget $ {bud:.4f}")
        for k, dC in DC.items():
            ins = best(d, dC, G5, allidx, bud)
            res = []
            for rep in range(a.reps):
                rng = np.random.default_rng(rep)
                perm = rng.permutation(n)
                A, B = perm[: n // 2], perm[n // 2 :]
                for X, Y in [(A, B), (B, A)]:
                    c = best(d, dC, G5, X, bud)
                    res.append((*evaluate(d, dC, G5, Y, c[1], c[2], c[3]), c[1]))
            res = np.array(res)
            P(
                f"  dC={k:6s} in-sample F1 {ins[0]:.4f} (lam {ins[1]}) | cross-fit F1 {res[:, 0].mean():.4f} sd {res[:, 0].std():.4f} $ {res[:, 1].mean():.4f} "
                f"({res[:, 1].mean() / bud:.2f}x) lam median {np.median(res[:, 2])}"
            )
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
