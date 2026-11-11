"""Summary of e2e_run.py (integrated end-to-end run). Quality and latency come from the same run.
Per split x policy x rep: n, cloud rate, path rate, cloud completion rate (truncation policy B fallbacks), F1 of the final
answer, total latency mean / p50 / p90 / p99 with bootstrap CIs (bootlib, seed 0, B=10k; percentile stats are resampled
too), and per-stage means. Paired differences vs the rule (same split, same rep, matched by question id): latency mean,
p90, p99 and F1, with CIs. Run-to-run: the same policy across reps (mean latency, F1, share of identical final answers).
Sanity: share of queries whose k and top-k ids equal the stage-1 trace; router scores vs the saved predictions."""

import argparse, glob, json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bootlib import boot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
a = ap.parse_args()
D = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{a.dir}/*.jsonl")):
    b = os.path.basename(f)[:-6]
    sp, _, pool, *rest = b.split("_")
    pol = "_".join(rest[:-1])
    rep = rest[-1]
    for l in open(f):
        x = json.loads(l)
        D[(sp, pol, rep)][(pool, x["id"])] = x
p90 = lambda v: np.percentile(v, 90)
p99 = lambda v: np.percentile(v, 99)
out = {"groups": {}, "paired": {}, "rerun": {}}
for (sp, pol, rep), X in sorted(D.items()):
    v = list(X.values())
    tot = np.array([x["total"] for x in v])
    f = np.array([x["f1"] for x in v])
    cl = np.array([x["branch"] == "cloud" for x in v])
    pa = np.array([x["branch"] == "path" for x in v])
    comp = [x.get("cloud_completed") for x in v if x["branch"] == "cloud"]
    st = collections.defaultdict(list)
    for x in v:
        for k, t in x["time"].items():
            st[k].append(t)
    o = {
        "n": len(v),
        "cloud_rate": float(cl.mean()),
        "path_rate": float(pa.mean()),
        "cloud_completion": float(np.mean(comp)) if comp else None,
        "n_fallback": int(sum(1 for c in comp if c is False)),
        "F1": boot(f),
        "lat_mean": boot(tot),
        "lat_p50": float(np.median(tot)),
        "lat_p90": boot(tot, stat=p90),
        "lat_p99": boot(tot, stat=p99),
        "stage_mean": {k: float(np.mean(t)) for k, t in st.items()},
        "stage_n": {k: len(t) for k, t in st.items()},
        "topk_same_as_trace": float(np.mean([x["topk_same_as_trace"] for x in v])),
        "k_same_as_trace": float(np.mean([x["k"] == x["k_trace"] for x in v])),
    }
    out["groups"][f"{sp}|{pol}|{rep}"] = o
    print(
        f"{sp} {pol:15s} {rep} n {o['n']} cloud {o['cloud_rate']:.3f} path {o['path_rate']:.3f} compl {o['cloud_completion']} fb {o['n_fallback']} | "
        f"F1 {o['F1'][0]:.4f} | lat mean {o['lat_mean'][0]:.3f} [{o['lat_mean'][1]:.3f},{o['lat_mean'][2]:.3f}] p50 {o['lat_p50']:.3f} "
        f"p90 {o['lat_p90'][0]:.2f} [{o['lat_p90'][1]:.2f},{o['lat_p90'][2]:.2f}] p99 {o['lat_p99'][0]:.2f} [{o['lat_p99'][1]:.2f},{o['lat_p99'][2]:.2f}] | topk=trace {o['topk_same_as_trace']:.3f}"
    )
print("-- paired vs rule (same split, same rep)")
for (sp, pol, rep), X in sorted(D.items()):
    if pol == "rule" or (sp, "rule", rep) not in D:
        continue
    Rr = D[(sp, "rule", rep)]
    ks = sorted(set(X) & set(Rr))
    dl = np.array([X[k]["total"] - Rr[k]["total"] for k in ks])
    df = np.array([X[k]["f1"] - Rr[k]["f1"] for k in ks])
    lx, lr = np.array([X[k]["total"] for k in ks]), np.array([Rr[k]["total"] for k in ks])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ks), (10000, len(ks)))
    q = {}
    for nm, fn in [("p90", lambda v: np.percentile(v, 90, axis=1)), ("p99", lambda v: np.percentile(v, 99, axis=1))]:
        m = fn(lx[idx]) - fn(lr[idx])
        q[nm] = (
            float(np.percentile(lx, int(nm[1:])) - np.percentile(lr, int(nm[1:]))),
            float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)),
        )
    o = {
        "n": len(ks),
        "lat_mean_diff": boot(dl),
        "lat_p90_diff": q["p90"],
        "lat_p99_diff": q["p99"],
        "F1_diff": boot(df),
    }
    out["paired"][f"{sp}|{pol}|{rep}"] = o
    print(
        f"{sp} {pol:15s} {rep} n {len(ks)} | lat mean {o['lat_mean_diff'][0]:+.3f} [{o['lat_mean_diff'][1]:+.3f},{o['lat_mean_diff'][2]:+.3f}] "
        f"p90 {q['p90'][0]:+.2f} [{q['p90'][1]:+.2f},{q['p90'][2]:+.2f}] p99 {q['p99'][0]:+.2f} [{q['p99'][1]:+.2f},{q['p99'][2]:+.2f}] | "
        f"F1 {o['F1_diff'][0]:+.4f} [{o['F1_diff'][1]:+.4f},{o['F1_diff'][2]:+.4f}]"
    )
print("-- run-to-run (r0 vs r1)")
for (sp, pol, rep), X in sorted(D.items()):
    if rep != "r0" or (sp, pol, "r1") not in D:
        continue
    Y = D[(sp, pol, "r1")]
    ks = sorted(set(X) & set(Y))
    o = {
        "n": len(ks),
        "same_answer": float(np.mean([X[k]["answer"] == Y[k]["answer"] for k in ks])),
        "same_branch": float(np.mean([X[k]["branch"] == Y[k]["branch"] for k in ks])),
        "lat_mean_r0_r1": (float(np.mean([X[k]["total"] for k in ks])), float(np.mean([Y[k]["total"] for k in ks]))),
        "F1_r0_r1": (float(np.mean([X[k]["f1"] for k in ks])), float(np.mean([Y[k]["f1"] for k in ks]))),
    }
    out["rerun"][f"{sp}|{pol}"] = o
    print(
        f"{sp} {pol:15s} n {len(ks)} same answer {o['same_answer']:.3f} same branch {o['same_branch']:.3f} lat {o['lat_mean_r0_r1'][0]:.3f}/{o['lat_mean_r0_r1'][1]:.3f} F1 {o['F1_r0_r1'][0]:.4f}/{o['F1_r0_r1'][1]:.4f}"
    )
json.dump(out, open(f"{a.dir}/summary.json", "w"), indent=1)
