"""Budget-curve analysis for the EC-RAG router pilot (replay on fixed edge/cloud outputs).

A policy is a score s (higher = send to cloud). At cloud rate r, the top ceil(r*n) queries by s go to the cloud.
Mean F1 is computed over ALL queries; the edge F1 / cloud F1 subsets are not compared across methods.
Policies:
- edge-only, cloud-only.
- EC-RAG rule (actual lambda=0.8 decisions) as a point.
- EC-RAG margin: s = complexity - threshold(k); the curve sweeps the threshold offset, i.e. validation-calibratable.
- complexity only.
- random: exact expectation mean(F1_e) + r*mean(dq).
- oracle: s = dq.
- feature GBM: k, complexity, rerank stats, token counts. Trained on train, regression on dq.
- decider runs: pred_dq / pred_logit from train_router.py.
Reported per dataset (test):
- F1 at the EC-RAG rule's cloud rate.
- F1 at 10/20/30/50% cloud.
- Normalized area over r in [0, 0.5] (0 = random, 1 = oracle).
- Hypothetical cost (DeepSeek-V3 list prices on Qwen3.5 tokens) at the same rates.
- Paired bootstrap of (decider - EC-RAG margin) F1 at the EC-RAG rate.
"""

from paths import ECRAG_SRC  # noqa: E402
import argparse, json, math, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from router_data import load  # noqa: E402

sys.path.insert(0, ECRAG_SRC)
from router import complexity_threshold  # noqa: E402


def at_rate(s, fe, fc, cost, r, rng=None):
    n = len(s)
    m = int(math.ceil(r * n - 1e-9))
    order = np.lexsort((rng.random(n) if rng is not None else np.arange(n), -np.asarray(s)))
    cloud = np.zeros(n, bool)
    cloud[order[:m]] = True
    return float(np.where(cloud, fc, fe).mean()), float(cost[cloud].sum()), cloud


def area(s, fe, fc, cost, rmax=0.5, steps=51):
    rs = np.linspace(0, rmax, steps)
    return float(np.trapezoid([at_rate(s, fe, fc, cost, r)[0] for r in rs], rs))


def feats(rows):
    X = []
    for r in rows:
        rr = r["rerank"] + [r["rerank"][-1]] * (10 - len(r["rerank"])) if r["rerank"] else [0] * 10
        X.append(
            [
                r["k"],
                r["complexity"],
                r["complexity"] - complexity_threshold(r["k"], 2, 10),
                np.mean(rr[: r["k"]]),
                rr[0],
                rr[min(r["k"], 9)],
                r["edge_in"],
                len(r["query"].split()),
                r["ds"] == "qasper",
            ]
        )
    return np.asarray(X, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="train_router.py output dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", help="split name, or several joined by + (e.g. test+test2) to pool them")
    ap.add_argument(
        "--stack",
        action="store_true",
        help="add stacked policies: GBM on features + decider scores, fit on val predictions",
    )
    a = ap.parse_args()
    from sklearn.ensemble import GradientBoostingRegressor

    tr = [r for ds in ["hotpotqa", "qasper"] for r in load(ds, "train", with_text=False)]
    gbm = (
        GradientBoostingRegressor(
            random_state=1234, n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8
        ).fit(feats(tr), [r["dq"] for r in tr])
        if tr
        else None
    )
    res = {}
    for ds in ["hotpotqa", "qasper"]:
        splits = a.split.split("+")
        rows = [r for s in splits for r in load(ds, s, with_text=False)]
        if not rows:
            continue
        fe = np.array([r["f1_edge"] for r in rows])
        fc = np.array([r["f1_cloud"] for r in rows])
        cost = np.array([r["cost"] for r in rows])
        dq = fc - fe
        pol = {
            "oracle": dq,
            "ecrag_margin": np.array([r["complexity"] - complexity_threshold(r["k"], 2, 10) for r in rows]),
            "complexity": np.array([r["complexity"] for r in rows]),
        }
        if gbm is not None:
            pol["feature_gbm"] = gbm.predict(feats(rows))
        key = {(r["split"], r["pool"], r["id"]): i for i, r in enumerate(rows)}
        for run in a.runs:
            name = os.path.basename(run.rstrip("/"))
            vals = {fld: np.full(len(rows), np.nan) for fld in ["pred_dq", "pred_logit"]}
            for s in splits:
                if not os.path.exists(f"{run}/pred_{s}.json"):
                    continue
                for x in json.load(open(f"{run}/pred_{s}.json")):
                    i = key.get((s, x["pool"], x["id"])) if x["ds"] == ds else None
                    if i is not None:
                        for fld in vals:
                            vals[fld][i] = x[fld]
            for fld, v in vals.items():
                if not np.isnan(v).any():
                    pol[f"{name}:{fld}"] = v
            if a.stack and not np.isnan(vals["pred_dq"]).any():
                # stacker: fit on validation (decider predictions there are out-of-sample; val was used only for epoch choice)
                va = [r for d2 in ["hotpotqa", "qasper"] for r in load(d2, "val", with_text=False)]
                pv = {(x["ds"], x["pool"], x["id"]): x for x in json.load(open(f"{run}/pred_val.json"))}
                Xv = np.hstack(
                    [
                        feats(va),
                        [
                            [
                                pv[(r["ds"], r["pool"], r["id"])]["pred_dq"],
                                pv[(r["ds"], r["pool"], r["id"])]["pred_logit"],
                            ]
                            for r in va
                        ],
                    ]
                )
                st = GradientBoostingRegressor(
                    random_state=1234, n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8
                ).fit(Xv, [r["dq"] for r in va])
                pol[f"stack:{name}"] = st.predict(
                    np.hstack([feats(rows), np.stack([vals["pred_dq"], vals["pred_logit"]], 1)])
                )
                if (
                    "feature_gbm" in pol
                ):  # fit-free ensemble: mean of within-set ranks of the feature GBM and the decider
                    rk = lambda v: np.argsort(np.argsort(v)) / max(1, len(v) - 1)
                    pol[f"rankavg:{name}"] = (rk(pol["feature_gbm"]) + rk(vals["pred_dq"])) / 2
        rule_cloud = np.array([not r["ecrag_edge"] for r in rows])
        r_rule = rule_cloud.mean()
        out = {
            "n": len(rows),
            "edge_only": fe.mean(),
            "cloud_only": fc.mean(),
            "cloud_only_cost": cost.sum(),
            "share_cloud_better": float((dq > 0).mean()),
            "share_edge_better": float((dq < 0).mean()),
            "share_tie": float((dq == 0).mean()),
            "both_zero": float(((fe == 0) & (fc == 0)).mean()),
            "ecrag_rule": {
                "cloud_rate": r_rule,
                "f1": float(np.where(rule_cloud, fc, fe).mean()),
                "cost": float(cost[rule_cloud].sum()),
            },
            "random_at_rule_rate": float(fe.mean() + r_rule * dq.mean()),
        }
        a_rand = float(
            np.trapezoid([fe.mean() + r * dq.mean() for r in np.linspace(0, 0.5, 51)], np.linspace(0, 0.5, 51))
        )
        a_orc = area(dq, fe, fc, cost)
        for name, s in pol.items():
            d = {"f1_at_rule_rate": at_rate(s, fe, fc, cost, r_rule)[0]}
            for r in [0.1, 0.2, 0.3, 0.5]:
                f, c, _ = at_rate(s, fe, fc, cost, r)
                d[f"f1@{int(r * 100)}"] = f
                d[f"cost@{int(r * 100)}"] = c
            d["area_norm"] = (area(s, fe, fc, cost) - a_rand) / (a_orc - a_rand) if a_orc > a_rand else float("nan")
            out[name] = d
        # paired bootstrap: each policy vs EC-RAG margin at the rule's cloud rate
        rng = np.random.default_rng(1234)
        _, _, c_base = at_rate(pol["ecrag_margin"], fe, fc, cost, r_rule)
        for name, s in pol.items():
            if name in ("oracle", "ecrag_margin"):
                continue
            _, _, c_m = at_rate(s, fe, fc, cost, r_rule)
            diff = np.where(c_m, fc, fe) - np.where(c_base, fc, fe)
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
            out[name]["vs_ecrag_margin_at_rule_rate"] = {
                "diff": float(diff.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            }
        res[ds] = out
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    for ds, o in res.items():
        print(
            f"== {ds} n={o['n']} edge-only {o['edge_only']:.4f} cloud-only {o['cloud_only']:.4f} rule: rate {o['ecrag_rule']['cloud_rate']:.3f} F1 {o['ecrag_rule']['f1']:.4f}"
            f" random@rule {o['random_at_rule_rate']:.4f} | cloud>edge {o['share_cloud_better']:.2f} edge>cloud {o['share_edge_better']:.2f} tie {o['share_tie']:.2f}"
        )
        for name in [k for k in o if isinstance(o[k], dict) and "area_norm" in o[k]]:
            d = o[name]
            v = d.get("vs_ecrag_margin_at_rule_rate")
            print(
                f"  {name:38s} @rule {d['f1_at_rule_rate']:.4f}  @10 {d['f1@10']:.4f} @20 {d['f1@20']:.4f} @30 {d['f1@30']:.4f} @50 {d['f1@50']:.4f}"
                f"  area {d['area_norm']:+.3f}"
                + (f"  vsRule {v['diff']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" if v else "")
            )


if __name__ == "__main__":
    main()
