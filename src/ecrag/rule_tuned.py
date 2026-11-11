"""Tuned EC-RAG rule baselines: the EC-RAG complexity rule with its threshold re-tuned on
validation so that it may use more (or less) cloud, with cloud = N (rule->N tuned) or cloud = T (rule->T tuned).
Rule score: cloud iff complexity > lam - (2 lam - 1)(k - 2)/8 + c. Grid lam in {0.5, 0.55, ..., 1.0}, c over 41 quantiles of
the validation margin (plus +-inf). Chosen on the FULL validation set: max F1 s.t. mean cloud $ <= budget, budgets = rule->T
validation $ x {0.5, 0.75, 1, 1.5, 2} (same grid as ent_policy.json). Complexity scores: pilot-1 stage-1 trace (same k/chunks).
Compared on test2/test3 with the frozen s1234_qd E/N and tfidf E/N policies of ent_policy.json at the same budget multiple.
Outputs: runs/pilot7/rule_tuned.{txt,json} (thresholds included, for the confirmation run)."""

from paths import RUNS  # noqa: E402
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ent_data  # noqa: E402
from freeze_ent_policy import arrays, scorers  # noqa: E402
from analyze_ent_cv import policy  # noqa: E402
from bootlib import boot  # noqa: E402

R = RUNS
P7 = f"{R}/pilot7"
LAMS = [round(0.5 + 0.05 * i, 2) for i in range(11)]
MULTS = [0.5, 0.75, 1.0, 1.5, 2.0]


def comp(rows):
    C = {}
    for sp in {r["split"] for r in rows}:
        for pool in {r["pool"] for r in rows if r["split"] == sp}:
            for l in open(f"{R}/pilot1/trace/hotpotqa/{sp}/{pool}/trace.jsonl"):
                t = json.loads(l)
                C[(sp, pool, t["id"])] = t["complexity"]
    return np.array([C[(r["split"], r["pool"], r["id"])] for r in rows]), np.array([r["k"] for r in rows])


def margin(cx, k, lam):
    return cx - (lam - (2 * lam - 1) * (k - 2) / 8)


def main():
    va = ent_data.load("val", "B")
    fv, cv, rv = arrays(va)
    cxv, kv = comp(va)
    base = float(np.where(rv, cv[2], cv[0]).mean())
    # sanity: lam 0.8, c 0 reproduces the stored EC-RAG routing
    assert np.mean((margin(cxv, kv, 0.8) >= 0) == rv) > 0.99, np.mean((margin(cxv, kv, 0.8) >= 0) == rv)
    PJ = json.load(open(f"{P7}/ent_policy.json"))["policies"]
    tr = ent_data.load("train", "A")
    out = {"rule_T_val_cost": base, "policies": {}, "test": {}}
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    for act, ai in [("N", 1), ("T", 2)]:
        for m in MULTS:
            best = None
            for lam in LAMS:
                mg = margin(cxv, kv, lam)
                for c in np.r_[-np.inf, np.quantile(mg, np.linspace(0, 1, 41)), np.inf]:
                    cl = mg > c
                    F = np.where(cl, fv[ai], fv[0])
                    C = np.where(cl, cv[ai], cv[0])
                    if C.mean() <= base * m + 1e-12 and (best is None or F.mean() > best[0]):
                        best = (float(F.mean()), lam, float(c), float(C.mean()), float(cl.mean()))
            out["policies"][f"rule{act}_tuned|x{m}"] = {
                "lam": best[1],
                "c": best[2],
                "val_F1": best[0],
                "val_cost": best[3],
                "val_cloud": best[4],
            }
            P(
                f"val rule->{act} tuned x{m}: lam {best[1]} c {best[2]:+.4f} F1 {best[0]:.4f} $ {best[3]:.4f} cloud {best[4]:.3f}"
            )
    for sp in ["test2", "test3"]:
        te = ent_data.load(sp, "B")
        ft, ct, rt = arrays(te)
        cxt, kt = comp(te)
        St = scorers(tr, te, sp)
        out["test"][sp] = {}
        P(f"== {sp}")
        for m in MULTS:
            Y = {}
            for nm in ["s1234_qd", "tfidf"]:
                v = PJ[f"{nm}|E/N|x{m}"]
                Y[nm], C = policy(St[nm], v["a_dEN"], v["b_dNT"], *ft, *ct)
                Y[nm + "_cost"] = C
            for act, ai in [("N", 1), ("T", 2)]:
                v = out["policies"][f"rule{act}_tuned|x{m}"]
                cl = margin(cxt, kt, v["lam"]) > v["c"]
                F = np.where(cl, ft[ai], ft[0])
                C = np.where(cl, ct[ai], ct[0])
                o = {
                    "F1": float(F.mean()),
                    "cost": float(C.mean()),
                    "cloud": float(cl.mean()),
                    "s1234_minus": boot(Y["s1234_qd"] - F),
                    "tfidf_minus": boot(Y["tfidf"] - F),
                }
                out["test"][sp][f"rule{act}_tuned|x{m}"] = o
                P(
                    f"  x{m} rule->{act} tuned F1 {o['F1']:.4f} $ {o['cost']:.4f} cloud {o['cloud']:.3f} | s1234 E/N F1 {Y['s1234_qd'].mean():.4f} $ {Y['s1234_qd_cost'].mean():.4f} "
                    f"minus rule {o['s1234_minus'][0]:+.4f} [{o['s1234_minus'][1]:+.4f},{o['s1234_minus'][2]:+.4f}] | tfidf E/N {Y['tfidf'].mean():.4f} minus rule "
                    f"{o['tfidf_minus'][0]:+.4f} [{o['tfidf_minus'][1]:+.4f},{o['tfidf_minus'][2]:+.4f}]"
                )
    json.dump(out, open(f"{P7}/rule_tuned.json", "w"), indent=1)
    open(f"{P7}/rule_tuned.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
