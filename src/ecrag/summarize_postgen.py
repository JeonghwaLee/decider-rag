"""Strict, full-cohort summary of learned pre-generation vs learned post-generation. Sign convention: pre-generation minus learned post-generation."""

import argparse, json, hashlib, sys
from pathlib import Path
import numpy as np
from bootlib import boot
from analyze_e2e4 import read


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    root = Path(a.root)
    allres = {}
    lines = [
        "# P0: learned pre-generation vs learned post-generation",
        "",
        "All comparisons below are pre-generation minus learned post-generation. Frozen validation policies; reused tests; one interleaved live campaign. Costs are cloud-token $/1,000 requests, without cache discounts or local hardware pricing.",
        "",
    ]
    for ds, n, calib, cp in [
        ("hotpot", 1600, "hotpot_calib", "ruleN_tuned"),
        ("musique", 400, "musique_calib", "ruleT_tuned"),
    ]:
        raw = read(root / ds / "live")
        assert set(raw) == {"pre", "post"}, set(raw)
        keys = sorted(raw["pre"])
        assert set(keys) == set(raw["post"]) and len(keys) == n, (ds, len(keys), len(raw["post"]))
        # Reuse the exact deadline from the original non-test calibration cohort.
        cd = read(root.parent / "e2e4" / calib)
        if cp not in cd:
            raise KeyError((cp, list(cd)))
        D = float(np.percentile([x["wall"] for x in cd[cp].values()], 90))
        metrics = {}
        vec = {}
        for arm in ["pre", "post"]:
            x = [raw[arm][k] for k in keys]
            f = np.array([r["f1"] for r in x])
            w = np.array([r["wall"] for r in x])
            c = np.array([r["total_cost"] for r in x])
            dq = np.where(w <= D, f, 0.0)
            stages = {
                t: float(np.mean([r["time"][t] for r in x if t in r["time"]]))
                for t in sorted(set.union(*(set(r["time"]) for r in x)))
            }
            metrics[arm] = {
                "n": n,
                "f1": float(f.mean()),
                "cost_per_1k": float(c.mean()),
                "wall_mean": float(w.mean()),
                "wall_p95": float(np.quantile(w, 0.95)),
                "dq_f1": float(dq.mean()),
                "met_fraction": float(np.mean(w <= D)),
                "branches": {
                    b: sum(r["branch"] == code for r in x) for b, code in [("E", "E"), ("C", "N"), ("R", "T")]
                },
                "retry_n": sum(r.get("retry") is not None for r in x),
                "stage_conditional_mean_s": stages,
                "n_input_tokens": float(
                    np.mean([r["ver_input_tokens"] if arm == "post" else r["decider_parts"]["n_tok"] for r in x])
                ),
            }
            vec[arm] = {"f1": f, "cost_per_1k": c, "wall_mean": w, "dq_f1": dq}
        ci = {k: boot(vec["pre"][k] - vec["post"][k]) for k in vec["pre"]}
        blocks = np.array([raw["pre"][k]["block"] for k in keys])
        block_ci = {k: boot(vec["pre"][k] - vec["post"][k], cluster=blocks) for k in ["wall_mean", "dq_f1"]}
        curves = []
        for multiple in [0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0]:
            ff = {arm: np.where(vec[arm]["wall_mean"] <= multiple * D, vec[arm]["f1"], 0.0) for arm in vec}
            curves.append(
                {
                    "multiple": multiple,
                    "deadline": multiple * D,
                    "pre": float(ff["pre"].mean()),
                    "post": float(ff["post"].mean()),
                    "pre_minus_post": boot(ff["pre"] - ff["post"]),
                }
            )
        up = {arm: np.array([raw[arm][k]["branch"] != ("E" if ds == "hotpot" else "N") for k in keys]) for arm in vec}
        dis = {}
        for name, mask in [
            ("both_keep", ~up["pre"] & ~up["post"]),
            ("both_escalate", up["pre"] & up["post"]),
            ("pre_only", up["pre"] & ~up["post"]),
            ("post_only", ~up["pre"] & up["post"]),
        ]:
            dis[name] = {
                "n": int(mask.sum()),
                **({arm: float(vec[arm]["f1"][mask].mean()) for arm in vec} if mask.any() else {}),
            }
        report = {
            "deadline": D,
            "metrics": metrics,
            "pre_minus_post": ci,
            "block_bootstrap": block_ci,
            "deadline_curve": curves,
            "disagreement": dis,
            "k_agreement": float(np.mean([raw["pre"][k]["k"] == raw["post"][k]["k"] for k in keys])),
        }
        allres[ds] = report
        (root / ds / "LIVE_SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
        lines += [
            f"## {ds} ({n} queries; D={D:.6f} s)",
            "",
            "| Arm | F1 | Cloud $/1k | Mean wall s | p95 s | DQ-F1 | E/C/R |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for arm, m in metrics.items():
            lines.append(
                f"| {arm} | {m['f1']:.4f} | {m['cost_per_1k']:.4f} | {m['wall_mean']:.4f} | {m['wall_p95']:.4f} | {m['dq_f1']:.4f} | {'/'.join(str(v) for v in m['branches'].values())} |"
            )
        lines += ["", "Paired question bootstrap (10,000 draws), pre minus post:", ""] + [
            f"- {k}: {v[0]:+.5f} [{v[1]:+.5f}, {v[2]:+.5f}]" for k, v in ci.items()
        ]
        lines += (
            ["", "Block-bootstrap sensitivity for timing:", ""]
            + [f"- {k}: {v[0]:+.5f} [{v[1]:+.5f}, {v[2]:+.5f}]" for k, v in block_ci.items()]
            + [""]
        )
    (root / "LIVE_SUMMARY.json").write_text(json.dumps(allres, indent=2) + "\n")
    (root / "FINAL_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
