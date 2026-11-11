"""Replay analysis for the EC-RAG router pilot on fixed edge/cloud outputs (extends analyze.py).

1. Stacker
   - Configuration (inputs x depth x n_estimators) is chosen by grouped cross-validation INSIDE validation.
     The groups are the 4 val pools.
   - The final stacker is then fit on all of validation. Its validation scores, used for thresholds and selection, are the
     out-of-fold ones.
   - Caveat: decider validation predictions come from the epoch chosen on validation loss.
2. Pre-registration
   - With --select, the primary policy is chosen on validation only: highest mean normalized area over the two datasets,
     restricted to policies that do not use test information. It is written to PRIMARY.json.
   - Runs with --split test2 read PRIMARY.json and report that policy first.
   - "rankavg" was devised after looking at the LongBench test. It is reported and flagged post_hoc_test_informed, and is
     not eligible for selection.
3. Deployment evaluation
   - For each policy, the threshold that gives a target cloud rate on validation is applied unchanged to the eval split.
     The realized cloud rate, F1 and cost are reported.
   - Targets: the EC-RAG rule's validation rate per dataset, plus 10/20/30/50%.
   - Ranking evaluation (top-r% within the eval set) is kept separately.
4. Qasper answerability: references that are "Unanswerable" vs answers that say so (edge, cloud, routed).
Score convention: higher = send to cloud. Mean F1 is always over all queries of a dataset.
"""

from paths import ECRAG_SRC, RUNS  # noqa: E402
import argparse, json, math, os, sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from router_data import load, DATA  # noqa: E402
from analyze import feats  # noqa: E402

sys.path.insert(0, ECRAG_SRC)
from router import complexity_threshold  # noqa: E402

DSS = ["hotpotqa", "qasper"]
RATES = [0.1, 0.2, 0.3, 0.5]
UNANS = (
    "unanswerable",
    "the context does not",
    "the provided context does not",
    "not mentioned",
    "no information",
    "not specified",
)


def rows_for(ds, splits):
    return [r for s in splits for r in load(ds, s, with_text=False)]


def topk_mask(s, r):
    n = len(s)
    m = int(math.ceil(r * n - 1e-9))
    c = np.zeros(n, bool)
    c[np.lexsort((np.arange(n), -np.asarray(s)))[:m]] = True
    return c


def norm_area(s, fe, fc):
    rs = np.linspace(0, 0.5, 51)
    dq = fc - fe
    f = lambda sc: np.trapezoid([np.where(topk_mask(sc, r), fc, fe).mean() for r in rs], rs)
    a_r = np.trapezoid([fe.mean() + r * dq.mean() for r in rs], rs)
    a_o = f(dq)
    return float((f(s) - a_r) / (a_o - a_r)) if a_o > a_r else float("nan")


def thr_for_rate(sval, r):
    """Score threshold t such that 'score > t' sends about r of validation to the cloud."""
    if r <= 0:
        return np.inf
    q = np.sort(sval)[::-1]
    m = int(math.ceil(r * len(q) - 1e-9))
    return float((q[m - 1] + q[m]) / 2) if m < len(q) else -np.inf


def run_preds(run, split, rows):
    key = {(r["ds"], r["split"], r["pool"], r["id"]): i for i, r in enumerate(rows)}
    out = {f: np.full(len(rows), np.nan) for f in ["pred_dq", "pred_logit"]}
    for s in {r["split"] for r in rows}:
        p = f"{run}/pred_{s}.json"
        if not os.path.exists(p):
            continue
        for x in json.load(open(p)):
            i = key.get((x["ds"], s, x["pool"], x["id"]))
            if i is not None:
                for f in out:
                    out[f][i] = x[f]
    return out


def ecdf(ref):
    ref = np.sort(ref)
    return lambda v: np.searchsorted(ref, v, side="right") / len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--split", required=True, help="eval split(s), '+'-joined to pool")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--select", action="store_true", help="choose the primary policy on validation and write PRIMARY.json"
    )
    ap.add_argument(
        "--dump_scores",
        default=None,
        help="write per-query policy scores (eval split) and validation thresholds to this json",
    )
    ap.add_argument("--primary", default=f"{RUNS}/pilot1/PRIMARY.json")
    a = ap.parse_args()
    splits = a.split.split("+")
    tr = [r for ds in DSS for r in load(ds, "train", with_text=False)]
    gbm = GradientBoostingRegressor(
        random_state=1234, n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8
    ).fit(feats(tr), [r["dq"] for r in tr])
    V = {ds: load(ds, "val", with_text=False) for ds in DSS}
    Vall = V["hotpotqa"] + V["qasper"]
    E = {ds: rows_for(ds, splits) for ds in DSS}

    # ---- policy scores on validation (sv) and eval (se), per dataset ----
    sv, se, meta = {ds: {} for ds in DSS}, {ds: {} for ds in DSS}, {}

    def add(name, fv, fe_, **m):
        meta[name] = m
        for ds in DSS:
            sv[ds][name] = fv(ds)
            se[ds][name] = fe_(ds)

    marg = lambda rows: np.array([r["complexity"] - complexity_threshold(r["k"], 2, 10) for r in rows])
    add("ecrag_margin", lambda ds: marg(V[ds]), lambda ds: marg(E[ds]))
    add(
        "complexity",
        lambda ds: np.array([r["complexity"] for r in V[ds]]),
        lambda ds: np.array([r["complexity"] for r in E[ds]]),
    )
    add("feature_gbm", lambda ds: gbm.predict(feats(V[ds])), lambda ds: gbm.predict(feats(E[ds])))
    for run in a.runs:
        name = os.path.basename(run.rstrip("/"))
        pv = {ds: run_preds(run, "val", V[ds]) for ds in DSS}
        pe = {ds: run_preds(run, a.split, E[ds]) for ds in DSS}
        if any(np.isnan(pe[ds]["pred_dq"]).any() or np.isnan(pv[ds]["pred_dq"]).any() for ds in DSS):
            print("skip (missing predictions)", name)
            continue
        for f in ["pred_dq", "pred_logit"]:
            add(f"{name}:{f}", lambda ds, f=f: pv[ds][f], lambda ds, f=f: pe[ds][f])
        # stacker: grouped CV inside validation picks the configuration; the final fit uses all of validation
        Xf = lambda rows, p: np.hstack([feats(rows), np.stack([p["pred_dq"], p["pred_logit"]], 1)])
        Xd = lambda rows, p: np.stack([p["pred_dq"], p["pred_logit"]], 1)
        Xall = {
            "feat+dec": np.vstack([Xf(V[ds], pv[ds]) for ds in DSS]),
            "dec": np.vstack([Xd(V[ds], pv[ds]) for ds in DSS]),
        }
        y = np.array([r["dq"] for r in Vall])
        g = np.array([f"{r['ds']}/{r['pool']}" for r in Vall])
        nH = len(V["hotpotqa"])
        best = None
        for inp in Xall:
            for depth in [1, 2, 3]:
                for ne in [100, 300]:
                    oof = np.zeros(len(y))
                    for tri, tei in GroupKFold(n_splits=4).split(Xall[inp], y, g):
                        m = GradientBoostingRegressor(
                            random_state=1234, n_estimators=ne, max_depth=depth, learning_rate=0.05, subsample=0.8
                        ).fit(Xall[inp][tri], y[tri])
                        oof[tei] = m.predict(Xall[inp][tei])
                    sc = np.mean(
                        [
                            norm_area(
                                o, np.array([r["f1_edge"] for r in V[ds]]), np.array([r["f1_cloud"] for r in V[ds]])
                            )
                            for ds, o in [("hotpotqa", oof[:nH]), ("qasper", oof[nH:])]
                        ]
                    )
                    if best is None or sc > best[0]:
                        best = (sc, inp, depth, ne, oof.copy())
        sc, inp, depth, ne, oof = best
        final = GradientBoostingRegressor(
            random_state=1234, n_estimators=ne, max_depth=depth, learning_rate=0.05, subsample=0.8
        ).fit(Xall[inp], y)
        mk = Xf if inp == "feat+dec" else Xd
        add(
            f"stack:{name}",
            lambda ds, o=oof: o[:nH] if ds == "hotpotqa" else o[nH:],
            lambda ds, mk=mk: final.predict(mk(E[ds], pe[ds])),
            stack_cfg={"inputs": inp, "max_depth": depth, "n_estimators": ne, "cv_val_area": sc},
            val_scores="out-of-fold",
        )

        # rank-average via validation ECDFs (deployable per query); post hoc: devised after looking at LongBench test
        def ra_v(ds):
            fg, fd = ecdf(sv[ds]["feature_gbm"]), ecdf(pv[ds]["pred_dq"])
            return (fg(sv[ds]["feature_gbm"]) + fd(pv[ds]["pred_dq"])) / 2

        def ra_e(ds):
            fg, fd = ecdf(sv[ds]["feature_gbm"]), ecdf(pv[ds]["pred_dq"])
            return (fg(se[ds]["feature_gbm"]) + fd(pe[ds]["pred_dq"])) / 2

        add(f"rankavg:{name}", ra_v, ra_e, post_hoc_test_informed=True)

    # ---- pre-registered primary selection (validation only) ----
    val_area = {
        n: float(
            np.mean(
                [
                    norm_area(
                        sv[ds][n], np.array([r["f1_edge"] for r in V[ds]]), np.array([r["f1_cloud"] for r in V[ds]])
                    )
                    for ds in DSS
                ]
            )
        )
        for n in meta
    }
    if a.select:
        elig = {n: v for n, v in val_area.items() if not meta[n].get("post_hoc_test_informed")}
        prim = max(elig, key=elig.get)
        json.dump(
            {
                "primary": prim,
                "criterion": "max mean normalized area (r in [0,0.5]) over hotpotqa+qasper on validation",
                "val_area": val_area,
                "meta": meta,
            },
            open(a.primary, "w"),
            indent=1,
            default=str,
        )
        print("PRIMARY (validation-selected):", prim, round(elig[prim], 3))
    primary = json.load(open(a.primary))["primary"] if os.path.exists(a.primary) else None

    res = {"split": a.split, "primary": primary, "val_area": val_area, "meta": meta}
    for ds in DSS:
        rows = E[ds]
        fe = np.array([r["f1_edge"] for r in rows])
        fc = np.array([r["f1_cloud"] for r in rows])
        cost = np.array([r["cost"] for r in rows])
        dq = fc - fe
        rule = np.array([not r["ecrag_edge"] for r in rows])
        r_rule = rule.mean()
        vrule = np.mean([not r["ecrag_edge"] for r in V[ds]])
        o = {
            "n": len(rows),
            "edge_only": fe.mean(),
            "cloud_only": fc.mean(),
            "cloud_only_cost": cost.sum(),
            "oracle_at_rule_rate": np.where(topk_mask(dq, r_rule), fc, fe).mean(),
            "ecrag_rule": {"cloud_rate": r_rule, "f1": np.where(rule, fc, fe).mean(), "cost": cost[rule].sum()},
            "random_at_rule_rate": fe.mean() + r_rule * dq.mean(),
            "policies": {},
        }
        rng = np.random.default_rng(1234)
        idx = rng.integers(0, len(rows), (10000, len(rows)))
        base = np.where(rule, fc, fe)
        for n in meta:
            s = se[ds][n]
            d = {"rank": {}, "deploy": {}}
            d["rank"]["area_norm"] = norm_area(s, fe, fc)
            for r in RATES + [r_rule]:
                c = topk_mask(s, r)
                d["rank"][f"f1@{r:.3f}"] = np.where(c, fc, fe).mean()
            c = topk_mask(s, r_rule)
            diff = np.where(c, fc, fe) - base
            bs = diff[idx].mean(1)
            d["rank"]["vs_rule_at_rule_rate"] = {
                "diff": diff.mean(),
                "ci95": [np.percentile(bs, 2.5), np.percentile(bs, 97.5)],
            }
            for lab, r in [("val_rule_rate", vrule)] + [(f"{int(x * 100)}%", x) for x in RATES]:
                t = thr_for_rate(sv[ds][n], r)
                c = s > t
                d["deploy"][lab] = {
                    "target": r,
                    "realized_rate": c.mean(),
                    "f1": np.where(c, fc, fe).mean(),
                    "cost": cost[c].sum(),
                }
            o["policies"][n] = d
        if ds == "qasper":
            refs = {}
            for s_ in splits:
                for p in os.listdir(f"{DATA}/{ds}/{s_}"):
                    for l in open(f"{DATA}/{ds}/{s_}/{p}/answers/answers.jsonl"):
                        x = json.loads(l)
                        refs[(s_, p, x["id"])] = x["answers"]
            un_ref = np.array(
                [
                    any(a_.strip().lower() == "unanswerable" for a_ in refs[(r["split"], r["pool"], r["id"])])
                    for r in rows
                ]
            )
            says = lambda t: t.strip().lower().startswith(UNANS)
            ue = np.array([says(r["edge_answer"]) for r in rows])
            uc = np.array([says(r["cloud_answer"]) for r in rows])

            def pr(pred):
                tp = (pred & un_ref).sum()
                return {
                    "precision": tp / max(1, pred.sum()),
                    "recall": tp / max(1, un_ref.sum()),
                    "n_pred": int(pred.sum()),
                }

            o["answerability"] = {
                "n_unanswerable_ref": int(un_ref.sum()),
                "edge": pr(ue),
                "cloud": pr(uc),
                "rule_routed": pr(np.where(rule, uc, ue)),
                "f1_on_answerable": {"edge": fe[~un_ref].mean(), "cloud": fc[~un_ref].mean()},
                "f1_on_unanswerable": {
                    "edge": fe[un_ref].mean() if un_ref.any() else None,
                    "cloud": fc[un_ref].mean() if un_ref.any() else None,
                },
            }
        res[ds] = o
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    if a.dump_scores:  # per-query scores for downstream (e.g. judge-based) evaluation; thresholds come from validation
        dump = {
            ds: {
                "keys": [[r["split"], r["pool"], r["id"]] for r in E[ds]],
                "ecrag_cloud": [bool(not r["ecrag_edge"]) for r in E[ds]],
                "val_rule_rate": float(np.mean([not r["ecrag_edge"] for r in V[ds]])),
                "scores": {n: [float(x) for x in se[ds][n]] for n in meta},
                "val_keys": [[r["split"], r["pool"], r["id"]] for r in V[ds]],
                "val_scores": {n: [float(x) for x in sv[ds][n]] for n in meta},  # out-of-fold for stackers
                "val_thr_at_val_rule": {
                    n: thr_for_rate(sv[ds][n], float(np.mean([not r["ecrag_edge"] for r in V[ds]]))) for n in meta
                },
            }
            for ds in DSS
        }
        json.dump({"split": a.split, "primary": primary, **dump}, open(a.dump_scores, "w"), default=float)
    # ---- printout ----
    order = ([primary] if primary in meta else []) + [n for n in meta if n != primary]
    for ds in DSS:
        o = res[ds]
        print(
            f"== {ds} [{a.split}] n={o['n']} edge {o['edge_only']:.4f} cloud {o['cloud_only']:.4f} | rule rate {o['ecrag_rule']['cloud_rate']:.3f} F1 {o['ecrag_rule']['f1']:.4f}"
            f" random {o['random_at_rule_rate']:.4f} oracle {o['oracle_at_rule_rate']:.4f}"
        )
        for n in order:
            d = o["policies"][n]
            v = d["rank"]["vs_rule_at_rule_rate"]
            dep = d["deploy"]["val_rule_rate"]
            tag = (" [PRIMARY]" if n == primary else "") + (
                " [post-hoc]" if meta[n].get("post_hoc_test_informed") else ""
            )
            print(
                f"  {n:34s} area {d['rank']['area_norm']:+.3f} | @rule {d['rank'][f'f1@{o['ecrag_rule']['cloud_rate']:.3f}']:.4f} vsRule {v['diff']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
                f" | deploy(val-thr @val-rule {dep['target']:.2f}): rate {dep['realized_rate']:.3f} F1 {dep['f1']:.4f}{tag}"
            )
        if "answerability" in o:
            an = o["answerability"]
            print(
                f"  answerability: unanswerable refs {an['n_unanswerable_ref']} | edge P/R {an['edge']['precision']:.2f}/{an['edge']['recall']:.2f} cloud P/R {an['cloud']['precision']:.2f}/{an['cloud']['recall']:.2f}"
                f" | F1 answerable edge {an['f1_on_answerable']['edge']:.4f} cloud {an['f1_on_answerable']['cloud']:.4f}"
            )


if __name__ == "__main__":
    main()
