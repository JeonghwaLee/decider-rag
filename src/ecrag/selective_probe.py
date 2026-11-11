"""Per-question selector switch, offline. E/N family on stored answers. TF-IDF (cheap) decides
first; the decider is run only when the TF-IDF score is within delta of its threshold (ambiguous), else the TF-IDF action is kept.
delta is set on val so that a target share u of questions runs the decider (u = 0: TF-IDF only, u = 1: decider only); thresholds
of both selectors are the frozen val ones (analyze_ent2 json, E/N, x<mult>). Reports F1, $, decider share on val and tests, and the
paired bootstrap vs TF-IDF only and vs decider only. Usage: selective_probe.py --json J --val V --test T... --scorer name=pattern."""

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays  # noqa: E402
from analyze_en import load  # noqa: E402
from bootlib import boot  # noqa: E402
from policy_probe import preds  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--fallback", default="E")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    J = json.load(open(a.json))
    nm, pat = a.scorer.split("=", 1)
    ad = J["thr"][f"{nm}|E/N|x{a.mult}"][0]
    at = J["thr"][f"tfidf|E/N|x{a.mult}"][0]
    trr = [r for s in a.train.split(",") for r in load(s, "A")[0] if r["fT"] is not None]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in trr])
    X = lambda Rr: np.hstack([vec.transform([r["query"] for r in Rr]).toarray(), np.array([[r["k"] / 10] for r in Rr])])
    rg = Ridge(alpha=3.0).fit(X(trr), [r["fN"] - r["fE"] for r in trr])
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))

    def ev(rows, split):
        g = arrays(rows, a.fallback)
        fE, fN, cN = g[0], g[1], g[3]
        st = rg.predict(X(rows))
        sd = preds(rows, pat, split)["dEN"]
        return fE, fN, cN, st, sd

    va, vsp = load(a.val)
    V = ev(va, vsp)
    dist_v = np.abs(V[3] - at)
    U = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    D = {u: (-1.0 if u == 0 else (np.inf if u == 1 else float(np.quantile(dist_v, u)))) for u in U}
    P(
        f"{nm} E/N x{a.mult} | thr decider {ad:+.4f} tfidf {at:+.4f} | delta by val share: "
        + " ".join(f"{u}:{D[u]:.3f}" for u in U)
    )
    for lab, (rows, split) in [("val", (va, vsp))] + [(t, load(t)) for t in a.test]:
        fE, fN, cN, st, sd = V if lab == "val" else ev(rows, split)
        use = {}
        base_t = np.where(st > at, fN, fE)
        base_d = np.where(sd > ad, fN, fE)
        P(
            f"== {lab} n {len(fE)} | tfidf {base_t.mean():.4f} $ {np.where(st > at, cN, 0).mean():.4f} | decider {base_d.mean():.4f} $ {np.where(sd > ad, cN, 0).mean():.4f}"
        )
        for u in U:
            amb = np.abs(st - at) <= D[u]
            cl = np.where(amb, sd > ad, st > at)
            F = np.where(cl, fN, fE)
            C = np.where(cl, cN, 0)
            w1, w2 = boot(F - base_t), boot(F - base_d)
            # random baseline at the same realised share p: per-question expectation (1-p)*tfidf + p*decider (exact), and the
            # paired bootstrap of ambiguity-selection minus that expectation
            pr = amb.mean()
            Fr = (1 - pr) * base_t + pr * base_d
            Cr = (1 - pr) * np.where(st > at, cN, 0) + pr * np.where(sd > ad, cN, 0)
            w3 = boot(F - Fr)
            P(
                f"  u {u:<4} decider share {amb.mean():.3f} | F1 {F.mean():.4f} $ {C.mean():.4f} | vs tfidf {w1[0]:+.4f} [{w1[1]:+.4f},{w1[2]:+.4f}] | vs decider {w2[0]:+.4f} [{w2[1]:+.4f},{w2[2]:+.4f}]"
                f" | random same share F1 {Fr.mean():.4f} $ {Cr.mean():.4f}, ambiguity - random {w3[0]:+.4f} [{w3[1]:+.4f},{w3[2]:+.4f}]"
            )
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
