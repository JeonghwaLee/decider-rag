"""Offline confirmation on a held-out split (--split test4|test5; stored answers in runs/pilot8/answers; truncation B).
Every frozen grid point of pilot7/ent_policy.json (scorers s1234_qd, ens3_qd, tfidf; E/N and E/N/T; budgets x0.5-x2) and
pilot7/rule_tuned.json, thresholds applied unchanged, plus the secondary candidates big2e4 and old_big2 (E/N, thresholds from
pilot8/en_eval_big2.json, chosen on val). Decider predictions: pilot8/router/pred_ent_qd_s*/pred_pilot8_hotpotqa_<split>.json.
Paired bootstrap (bootlib). Output: pilot8/confirm_<split>_offline*.{txt,json}."""

from paths import RUNS  # noqa: E402
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ent_data, ent_data2  # noqa: E402
from analyze_ent_cv import policy  # noqa: E402
from bootlib import boot  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

R = RUNS


SCJ = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


SPLIT = os.environ.get("SPLIT", "test5")


def judge_arrays(te, jn):
    """E/N/T judge scores per row (truncated T -> E label, as truncation B)."""
    L = {}
    for pool in {r["pool"] for r in te}:
        for l in open(f"{R}/pilot8/judge/hotpotqa/{SPLIT}/{pool}/judge_{jn}.jsonl"):
            x = json.loads(l)
            L[(pool, x["id"], x["who"])] = SCJ.get(x["label"], np.nan)
    E = np.array([L.get((r["pool"], r["id"], "E"), np.nan) for r in te])
    N = np.array([L.get((r["pool"], r["id"], "N"), np.nan) for r in te])
    T = np.array([E[i] if r["T_trunc"] else L.get((r["pool"], r["id"], "T"), np.nan) for i, r in enumerate(te)])
    return E, N, T


def main():
    PJ = json.load(open(f"{R}/pilot7/ent_policy.json"))["policies"]
    RT = json.load(open(f"{R}/pilot7/rule_tuned.json"))["policies"]
    te = ent_data2.load(f"{R}/pilot8", "hotpotqa", SPLIT, "B")
    tr = ent_data.load("train", "A")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in tr])
    X = lambda Rr: np.hstack([vec.transform([r["query"] for r in Rr]).toarray(), np.array([[r["k"] / 10] for r in Rr])])
    S = {
        "tfidf": {
            k: Ridge(alpha=3.0).fit(X(tr), [f(r) for r in tr]).predict(X(te))
            for k, f in [("dEN", lambda r: r["fN"] - r["fE"]), ("dNT", lambda r: r["fT"] - r["fN"])]
        }
    }
    seeds = {}
    rd = lambda fn: (lambda P: {k: np.array([P[(r["pool"], r["id"])][k] for r in te]) for k in ["dEN", "dNT"]})(
        {(x["pool"], x["id"]): x for x in json.load(open(fn))}
    )
    for s in [1234, 2345, 3456]:
        fn = f"{R}/pilot8/router/pred_ent_qd_s{s}/pred_pilot8_hotpotqa_{SPLIT}.json"
        if os.path.exists(fn):
            seeds[s] = rd(fn)
    S["s1234_qd"] = seeds[1234]
    if len(seeds) == 3:
        S["ens3_qd"] = {k: np.mean([d[k] for d in seeds.values()], 0) for k in ["dEN", "dNT"]}
    else:
        PJ = {k: v for k, v in PJ.items() if v["scorer"] != "ens3_qd"}
    S["big2e4"] = rd(f"{R}/pilot8/router/big_ent_s2345_e4/pred_pilot8_hotpotqa_{SPLIT}.json")
    S["old_big2"] = {k: (S["s1234_qd"][k] + S["big2e4"][k]) / 2 for k in ["dEN", "dNT"]}
    TH2 = json.load(open(f"{R}/pilot8/en_eval_big2.json"))["thresholds"]
    for nm in ["big2e4", "old_big2"]:
        for m in ["0.5", "0.75", "1.0", "1.5", "2.0"]:
            PJ[f"{nm}|E/N|x{m}"] = {"scorer": nm, "a_dEN": TH2[m][nm], "b_dNT": float("inf")}
    f = tuple(np.array([r[k] for r in te]) for k in ["fE", "fN", "fT"])
    c = (
        np.zeros(len(te)),
        np.array([(0.27 * r["N_in"] + 1.1 * r["N_out"]) / 1e3 for r in te]),
        np.array([(0.27 * r["T_in"] + 1.1 * r["T_out"]) / 1e3 for r in te]),
    )
    rt = np.array([not r["ecrag_edge"] for r in te])
    cx = np.array([r["complexity"] for r in te])
    kk = np.array([r["k"] for r in te])
    JUDGE = os.environ.get("JUDGE")
    if JUDGE:  # auxiliary: same frozen policies, judge scores instead of F1
        f = judge_arrays(te, JUDGE)
    refT, refN = np.where(rt, f[2], f[0]), np.where(rt, f[1], f[0])
    out = {}
    L = []
    P_ = lambda x: (print(x, flush=True), L.append(x))
    P_(
        f"{SPLIT} n {len(te)} | rule->T F1 {refT.mean():.4f} $ {np.where(rt, c[2], 0).mean():.4f} | rule->N F1 {refN.mean():.4f} $ {np.where(rt, c[1], 0).mean():.4f} | "
        f"all-E {f[0].mean():.4f} all-N {f[1].mean():.4f} $ {c[1].mean():.4f} all-T {f[2].mean():.4f} $ {c[2].mean():.4f} | T truncated {np.mean([r['T_trunc'] for r in te]):.3f}"
    )
    Y = {}
    for k, v in RT.items():
        lam = v["lam"]
        act = 1 if k.startswith("ruleN") else 2
        cl = cx - (lam - (2 * lam - 1) * (kk - 2) / 8) > v["c"]
        Y[k] = (np.where(cl, f[act], f[0]), np.where(cl, c[act], 0))
    for k, v in PJ.items():
        Y[k] = policy(S[v["scorer"]], v["a_dEN"], v["b_dNT"], *f, *c)
    for k, (F, C) in Y.items():
        F = np.nan_to_num(F, nan=np.nanmean(F)) if JUDGE else F
        o = {
            "F1": float(F.mean()),
            "cost": float(C.mean()),
            "cloud": float((C > 0).mean()),
            "vs_ruleT": boot(F - refT),
            "vs_ruleN": boot(F - refN),
            "vs_allN": boot(F - f[1]),
        }
        m = k.split("|")[-1]
        for ref in [f"tfidf|E/N|{m}", f"ruleN_tuned|{m}", f"s1234_qd|E/N|{m}"]:
            if ref in Y and ref != k:
                o[f"vs_{ref}"] = boot(F - Y[ref][0])
        out[k] = o
        P_(
            f"  {k:24s} F1 {o['F1']:.4f} $ {o['cost']:.4f} cloud {o['cloud']:.3f} | "
            + " | ".join(f"{a[3:]} {b[0]:+.4f} [{b[1]:+.4f},{b[2]:+.4f}]" for a, b in o.items() if a.startswith("vs_"))
        )
    tag = (f"_judge_{JUDGE}" if JUDGE else "") + ("" if SPLIT == "test5" else "_v2")
    json.dump(out, open(f"{R}/pilot8/confirm_{SPLIT}_offline{tag}.json", "w"), indent=1)
    open(f"{R}/pilot8/confirm_{SPLIT}_offline{tag}.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
