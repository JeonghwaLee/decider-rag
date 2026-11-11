"""Freeze the E/N/T (and E/N) deployment policies chosen on validation, and add the comparisons missing from ent_test.txt.
Thresholds are always chosen on the FULL validation set (same procedure as
analyze_ent_test.py: 41 quantiles per score, maximize F1 s.t. mean cloud $ <= budget). Nothing is chosen on test.
Policies frozen (cloud-$ axis):
  ens3_qd  E/N/T   (= the test-plan primary policy; reproduces ent_test.txt)
  ens3_qd  E/N     (T forbidden)
  s1234    E/N/T and E/N  (single q+D+k seed for deployment; chosen by validation cross-fitted F1, ent_val_cv_qd3.txt)
  tfidf    E/N/T and E/N  (cheap selector with the same actions)
Budgets: rule->T validation $ x {0.5, 0.75, 1, 1.5, 2} (a cost-F1 curve fixed on validation; x1 is the headline).
Fixed comparators added: rule->N (EC-RAG rule, cloud = N), all-N.
Outputs: runs/pilot7/ent_policy.json (deploy config: thresholds, models, normalization, fallback) and
ent_test_supp.{txt,json} (supplementary test evaluation of the frozen policies; test2/test3 were already opened once)."""

from paths import RUNS  # noqa: E402
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ent_data  # noqa: E402
from analyze_ent_cv import policy  # noqa: E402
from bootlib import boot  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

P7 = f"{RUNS}/pilot7"
SEEDS = [1234, 2345, 3456]
SINGLE = 1234


def arrays(rows):
    f = tuple(np.array([r[k] for r in rows]) for k in ["fE", "fN", "fT"])
    c = (
        np.zeros(len(rows)),
        np.array([(0.27 * r["N_in"] + 1.1 * r["N_out"]) / 1e3 for r in rows]),
        np.array([(0.27 * r["T_in"] + 1.1 * r["T_out"]) / 1e3 for r in rows]),
    )
    return f, c, np.array([not r["ecrag_edge"] for r in rows])


def scorers(tr, rows, split):
    S = {}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in tr])
    X = lambda R: np.hstack([vec.transform([r["query"] for r in R]).toarray(), np.array([[r["k"] / 10] for r in R])])
    S["tfidf"] = {
        k: Ridge(alpha=3.0).fit(X(tr), [f(r) for r in tr]).predict(X(rows))
        for k, f in [("dEN", lambda r: r["fN"] - r["fE"]), ("dNT", lambda r: r["fT"] - r["fN"])]
    }
    seeds = {}
    for s in SEEDS:
        P = {(x["pool"], x["id"]): x for x in json.load(open(f"{P7}/router/ent_qd_s{s}/pred_{split}.json"))}
        pv = [P[(r["pool"], r["id"])] for r in rows]
        seeds[s] = {k: np.array([x[k] for x in pv]) for k in ["dEN", "dNT"]}
    S["ens3_qd"] = {k: np.mean([d[k] for d in seeds.values()], 0) for k in ["dEN", "dNT"]}
    S[f"s{SINGLE}_qd"] = seeds[SINGLE]
    return S


def choose(d, fam, budget, f, c):
    qs = np.linspace(0, 1, 41)
    A = list(np.r_[-np.inf, np.quantile(d["dEN"], qs), np.inf])
    B = [np.inf] if fam == "E/N" else list(np.r_[-np.inf, np.quantile(d["dNT"], qs), np.inf])
    best = None
    for x in A:
        for y in B:
            F, C = policy(d, x, y, *f, *c)
            if C.mean() <= budget + 1e-12 and (best is None or F.mean() > best[0]):
                best = (float(F.mean()), float(x), float(y), float(C.mean()))
    return best


def main():
    tr, va = ent_data.load("train", "A"), ent_data.load("val", "B")
    fv, cv, rv = arrays(va)
    Sv = scorers(tr, va, "val")
    base = float(np.where(rv, cv[2], cv[0]).mean())
    pol = {
        "budget_unit": "cloud $ per query x1000 (0.27/M in, 1.1/M out, edge = 0)",
        "rule_T_val_cost": base,
        "policies": {},
    }
    for nm in Sv:
        for fam in ["E/N/T", "E/N"]:
            for m in [0.5, 0.75, 1.0, 1.5, 2.0]:
                b = choose(Sv[nm], fam, base * m, fv, cv)
                pol["policies"][f"{nm}|{fam}|x{m}"] = {
                    "scorer": nm,
                    "family": fam,
                    "budget_mult": m,
                    "budget": base * m,
                    "a_dEN": b[1],
                    "b_dNT": b[2],
                    "val_F1": b[0],
                    "val_cost": b[3],
                }
    pol["deploy"] = {
        "edge": {"model": "Qwen/Qwen3-4B (vLLM :8195)", "prompt": "extract", "normalize": "normalize_edge.py"},
        "N": {"model": "Qwen3.8-Flash-Next FP8 (:8193)", "prompt": "orig", "thinking": False, "max_tokens": 256},
        "T": {
            "model": "Qwen3.8-Flash-Next FP8 (:8193)",
            "prompt": "orig_think",
            "thinking": True,
            "max_tokens": 6144,
            "truncation_fallback": "B: finish=length or no </think> -> normalized edge answer",
        },
        "decider": {
            "input": "q+D+k (pruned chunks, build_text max 6144 tokens)",
            "heads": ["fE", "fN", "fT", "dEN", "dNT"],
            "ens3_qd": [f"runs/pilot7/router/ent_qd_s{s}/best.pt" for s in SEEDS],
            f"s{SINGLE}_qd": f"runs/pilot7/router/ent_qd_s{SINGLE}/best.pt",
            "single_seed_selection": "highest validation cross-fitted F1 on dollar|E/N/T (ent_val_cv_qd3.txt: s1234 0.7630)",
        },
        "rule": "cloud iff dEN > a_dEN; think iff cloud and dNT > b_dNT; ens3 = mean of the 3 seeds' dEN/dNT",
    }
    json.dump(pol, open(f"{P7}/ent_policy.json", "w"), indent=1)
    out, lines = {}, []
    P = lambda s: (print(s, flush=True), lines.append(s))
    P(f"validation rule->T $ {base:.4f}/1k; thresholds from ent_policy.json (full validation)")
    for k, v in pol["policies"].items():
        if v["budget_mult"] == 1.0:
            P(f"  val {k:22s} a {v['a_dEN']:+.4f} b {v['b_dNT']:+.4f} F1 {v['val_F1']:.4f} cost {v['val_cost']:.4f}")
    for split in ["test2", "test3"]:
        te = ent_data.load(split, "B")
        ft, ct, rt = arrays(te)
        St = scorers(tr, te, split)
        out[split] = {}
        refT = np.where(rt, ft[2], ft[0])
        refN = np.where(rt, ft[1], ft[0])
        P(
            f"== {split} n {len(te)} | rule->T F1 {refT.mean():.4f} $ {np.where(rt, ct[2], ct[0]).mean():.4f} | rule->N F1 {refN.mean():.4f} "
            f"$ {np.where(rt, ct[1], ct[0]).mean():.4f} | all-N F1 {ft[1].mean():.4f} $ {ct[1].mean():.4f} | all-E {ft[0].mean():.4f}"
        )
        P(f"   rule->N vs rule->T {boot(refN - refT)}")
        Y = {}
        for k, v in pol["policies"].items():
            d = St[v["scorer"]]
            F, C = policy(d, v["a_dEN"], v["b_dNT"], *ft, *ct)
            Y[k] = F
            cl = d["dEN"] > v["a_dEN"]
            th = cl & (d["dNT"] > v["b_dNT"])
            o = {
                "F1": float(F.mean()),
                "cost": float(C.mean()),
                "budget": v["budget"],
                "nE": int((~cl).sum()),
                "nN": int((cl & ~th).sum()),
                "nT": int(th.sum()),
                "vs_ruleT": boot(F - refT),
                "vs_ruleN": boot(F - refN),
                "vs_allN": boot(F - ft[1]),
            }
            tk = f"tfidf|{v['family']}|x{v['budget_mult']}"
            if v["scorer"] != "tfidf":
                o["vs_tfidf_same"] = boot(F - Y[tk]) if tk in Y else None
            out[split][k] = o
        for k, o in out[split].items():
            s = (
                f"   {k:24s} F1 {o['F1']:.4f} $ {o['cost']:.4f} (budget {o['budget']:.4f}) E/N/T {o['nE']}/{o['nN']}/{o['nT']} | "
                f"vs rule->T {o['vs_ruleT'][0]:+.4f} [{o['vs_ruleT'][1]:+.4f},{o['vs_ruleT'][2]:+.4f}] | vs rule->N {o['vs_ruleN'][0]:+.4f} "
                f"[{o['vs_ruleN'][1]:+.4f},{o['vs_ruleN'][2]:+.4f}] | vs all-N {o['vs_allN'][0]:+.4f} [{o['vs_allN'][1]:+.4f},{o['vs_allN'][2]:+.4f}]"
            )
            if o.get("vs_tfidf_same"):
                s += f" | vs tfidf(same fam,budget) {o['vs_tfidf_same'][0]:+.4f} [{o['vs_tfidf_same'][1]:+.4f},{o['vs_tfidf_same'][2]:+.4f}]"
            P(s)
        # equal-cost check vs tfidf, without test labels in the selection: the decider policy from the validation budget
        # grid whose TEST cost is closest to (and not above) tfidf x1's test cost (costs only, labels are not used)
        for fam in ["E/N/T", "E/N"]:
            tc = out[split][f"tfidf|{fam}|x1.0"]["cost"]
            for nm in ["ens3_qd", f"s{SINGLE}_qd"]:
                c = [(o["cost"], k) for k, o in out[split].items() if k.startswith(f"{nm}|{fam}|") and o["cost"] <= tc]
                if c:
                    k = max(c)[1]
                    o = out[split][k]
                    P(
                        f"   cost<=tfidf x1 ({tc:.4f}): {k} F1 {o['F1']:.4f} $ {o['cost']:.4f} vs tfidf x1 {boot(Y[k] - Y[f'tfidf|{fam}|x1.0'])}"
                    )
    json.dump(out, open(f"{P7}/ent_test_supp.json", "w"), indent=1)
    open(f"{P7}/ent_test_supp.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
