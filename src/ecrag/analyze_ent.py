"""E/N/T decider analysis on VALIDATION ONLY (test splits stay closed until the gain is confirmed).
Scorers (each predicts dEN = F1_N - F1_E and dNT = F1_T - F1_N):
  decider   train_router3.py predictions (pred_val.json); also its direct heads (fN - fE, fT - fN)
  tfidf     question TF-IDF (1-2 grams) + k -> Ridge, trained on HotpotQA train (a quick baseline)
Policies on validation: two thresholds (cloud iff dEN > a; think iff cloud and dNT > b), swept on quantile grids.
  E/N/T (both), E/N (b = +inf), N/T (a = -inf), E/T (cloud always thinks; a only, scored by dEN + dNT)
  fixed: all-E, all-N, all-T, EC-RAG rule -> T, EC-RAG rule -> N, oracle (per query best F1)
Cost per query = latency proxy fitted on runs/pilot7/latency_actions_val_pool_00.jsonl (least squares on input and
output tokens, per action); cloud $ cost (0.27/M in, 1.1/M out) is reported too. Output: Pareto fronts (F1 vs mean latency)
and, at fixed latency budgets (rule's, all-N's, all-T's), the best F1 per policy family on validation."""

from paths import RUNS  # noqa: E402
import json, os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ent_data  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

P7 = f"{RUNS}/pilot7"


def fit_latency():
    X = [json.loads(l) for l in open(f"{P7}/latency_actions_val_pool_00.jsonl")]
    co = {}
    for a in "ENT":
        A = np.array([[1, x[a]["in"], x[a]["out"]] for x in X])
        y = np.array([x[a]["t"] for x in X])
        co[a] = np.linalg.lstsq(A, y, rcond=None)[0].tolist()
        co[a + "_r2"] = float(1 - ((A @ np.array(co[a]) - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        co[a + "_mean"] = float(y.mean())
    return co


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--decider", nargs="*", default=[f"{P7}/router/ent_qk_s1234"])
    ap.add_argument("--out", default=f"{P7}/ent_val.json")
    ap.add_argument(
        "--cost", default="latency", choices=["latency", "dollar"], help="dollar: cloud token cost (edge = 0)"
    )
    a = ap.parse_args()
    tr, va = ent_data.load("train", "A"), ent_data.load("val", "B")
    co = fit_latency()
    print(
        "latency fit (s = c0 + c1*in + c2*out):",
        {k: (np.round(v, 6).tolist() if isinstance(v, list) else round(v, 3)) for k, v in co.items()},
    )
    lat = (
        lambda r, act: co[act][0]
        + co[act][1] * (r[f"{act}_in"] if act != "E" else r["N_in"])
        + co[act][2] * (r[f"{act}_out"] if act != "E" else r["edge_out"])
    )
    fE, fN, fT = (np.array([r[k] for r in va]) for k in ["fE", "fN", "fT"])
    LE, LN, LT = (np.array([lat(r, x) for r in va]) for x in "ENT")
    if a.cost == "dollar":  # cloud $ per 1k queries at the assumed price (0.27/M input, 1.1/M output); edge is free
        LE = np.zeros(len(va))
        LN = np.array([(0.27 * r["N_in"] + 1.1 * r["N_out"]) / 1e3 for r in va])
        LT = np.array([(0.27 * r["T_in"] + 1.1 * r["T_out"]) / 1e3 for r in va])
    rule = np.array([not r["ecrag_edge"] for r in va])
    S = {}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in tr])
    feat = lambda R: np.hstack([vec.transform([r["query"] for r in R]).toarray(), np.array([[r["k"] / 10] for r in R])])
    for key, f in [("dEN", lambda r: r["fN"] - r["fE"]), ("dNT", lambda r: r["fT"] - r["fN"])]:
        m = Ridge(alpha=3.0).fit(feat(tr), [f(r) for r in tr])
        S.setdefault("tfidf", {})[key] = m.predict(feat(va))
    for d in a.decider:
        name = os.path.basename(d)
        P = {(p["pool"], p["id"]): p for p in json.load(open(f"{d}/pred_val.json"))}
        pv = [P[(r["pool"], r["id"])] for r in va]
        S[name] = {"dEN": np.array([p["dEN"] for p in pv]), "dNT": np.array([p["dNT"] for p in pv])}
        S[name + ":direct"] = {
            "dEN": np.array([p["fN"] - p["fE"] for p in pv]),
            "dNT": np.array([p["fT"] - p["fN"] for p in pv]),
        }
    fixed = {
        "all-E": (fE.mean(), LE.mean()),
        "all-N": (fN.mean(), LN.mean()),
        "all-T": (fT.mean(), LT.mean()),
        "rule->T": (np.where(rule, fT, fE).mean(), np.where(rule, LT, LE).mean()),
        "rule->N": (np.where(rule, fN, fE).mean(), np.where(rule, LN, LE).mean()),
        "oracle": (np.maximum.reduce([fE, fN, fT]).mean(), None),
    }
    print(
        "fixed (F1, mean latency s):",
        {k: (round(v[0], 4), None if v[1] is None else round(v[1], 3)) for k, v in fixed.items()},
    )
    qs = np.linspace(0, 1, 41)
    res = {"latency_fit": co, "fixed": fixed, "families": {}}
    budgets = {
        "rule->T": fixed["rule->T"][1],
        "rule->N": fixed["rule->N"][1],
        "all-N": fixed["all-N"][1],
        "all-T": fixed["all-T"][1],
    }
    for sc, d in S.items():
        th_a = np.r_[-np.inf, np.quantile(d["dEN"], qs), np.inf]
        th_b = np.r_[-np.inf, np.quantile(d["dNT"], qs), np.inf]
        fam = {
            "E/N/T": [(x, y) for x in th_a for y in th_b],
            "E/N": [(x, np.inf) for x in th_a],
            "N/T": [(-np.inf, y) for y in th_b],
        }
        sET = d["dEN"] + d["dNT"]
        th_c = np.r_[-np.inf, np.quantile(sET, qs), np.inf]
        for fn, grid in list(fam.items()) + [("E/T", None)]:
            pts = []
            if fn == "E/T":
                for c in th_c:
                    cl = sET > c
                    pts.append(
                        (
                            float(np.where(cl, fT, fE).mean()),
                            float(np.where(cl, LT, LE).mean()),
                            float(cl.mean()),
                            0.0,
                            c,
                            None,
                        )
                    )
            else:
                for x, y in grid:
                    cl = d["dEN"] > x
                    th = cl & (d["dNT"] > y)
                    F = np.where(th, fT, np.where(cl, fN, fE))
                    L = np.where(th, LT, np.where(cl, LN, LE))
                    pts.append((float(F.mean()), float(L.mean()), float(cl.mean()), float(th.mean()), x, y))
            pts.sort(key=lambda p: p[1])
            front, best = [], -1
            for p in pts:
                if p[0] > best:
                    front.append(p)
                    best = p[0]
            at = {
                b: max([p for p in pts if p[1] <= v + 1e-9], key=lambda p: p[0], default=None)
                for b, v in budgets.items()
            }
            res["families"][f"{sc}|{fn}"] = {"front": front, "at_budget": at}
    print("\nbest validation F1 within latency budget (F1 / mean latency / cloud rate / think rate):")
    for b, v in budgets.items():
        print(f" budget = {b} ({v:.3f}s): ref F1 {fixed[b][0]:.4f}")
        for k, o in res["families"].items():
            p = o["at_budget"][b]
            if p:
                print(f"   {k:28s} F1 {p[0]:.4f}  lat {p[1]:.3f}  cloud {p[2]:.3f}  think {p[3]:.3f}")
    json.dump(
        res,
        open(a.out, "w"),
        default=lambda x: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x),
    )


if __name__ == "__main__":
    main()
