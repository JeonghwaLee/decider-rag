"""Summary of e2e_ent2.py runs (v2: wall-clock). Per policy (and rep): n, E/N/T counts, T fallbacks, F1, cloud $ per 1k, wall latency
mean/p50/p90/p99 (bootlib CIs), stage means. Paired vs --primary (same rep, question): F1, $, wall mean, p90, p99.
Also |wall - sum of stages| as an instrumentation check. Usage: analyze_e2e2.py --dir <run dir> [--primary s1234_EN]."""

import argparse, glob, json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bootlib import boot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--primary", default="s1234_EN")
a = ap.parse_args()
D = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{a.dir}/*_r[0-9].jsonl")):
    for l in open(f):
        x = json.loads(l)
        sp_pool = "_".join(os.path.basename(f).split("_")[:3])
        D[(x["policy"], x.get("rep", 0))][(sp_pool, x["id"])] = x
p90 = lambda v: np.percentile(v, 90)
p99 = lambda v: np.percentile(v, 99)
fmt = lambda t, d=4: f"{t[0]:+.{d}f} [{t[1]:+.{d}f},{t[2]:+.{d}f}]"
out = {"groups": {}, "paired": {}}
L = []
P = lambda s: (print(s, flush=True), L.append(s))
for (pol, rep), X in sorted(D.items()):
    v = list(X.values())
    w = np.array([x["wall"] for x in v])
    f = np.array([x["f1"] for x in v])
    c = np.array([x["cost"] for x in v])
    br = collections.Counter(x["branch"] for x in v)
    fb = sum(1 for x in v if x["branch"] == "T" and not x.get("cloud_completed"))
    st = collections.defaultdict(list)
    for x in v:
        for k, t in x["time"].items():
            st[k].append(t)
    o = {
        "n": len(v),
        "branch": dict(br),
        "fallback": fb,
        "F1": boot(f),
        "cost": boot(c),
        "wall_mean": boot(w),
        "wall_p50": float(np.median(w)),
        "wall_p90": boot(w, stat=p90),
        "wall_p99": boot(w, stat=p99),
        "stage_mean": {k: float(np.mean(t)) for k, t in st.items()},
        "wall_minus_stages_max_ms": float(np.max(np.abs(w - np.array([x["total"] for x in v]))) * 1e3),
        "topk_same_as_trace": float(np.mean([bool(x["topk_same_as_trace"]) for x in v])),
    }
    out["groups"][f"{pol}|{rep}"] = o
    P(
        f"{pol:12s} r{rep} n {o['n']} E/N/T {br.get('E', 0)}/{br.get('N', 0)}/{br.get('T', 0)} fb {fb} | F1 {o['F1'][0]:.4f} | $ {o['cost'][0]:.4f} | wall mean "
        f"{o['wall_mean'][0]:.3f} [{o['wall_mean'][1]:.3f},{o['wall_mean'][2]:.3f}] p50 {o['wall_p50']:.3f} p90 {o['wall_p90'][0]:.2f} p99 {o['wall_p99'][0]:.2f} | "
        f"|wall-stages| max {o['wall_minus_stages_max_ms']:.1f} ms | topk=trace {o['topk_same_as_trace']:.3f}"
    )
P("-- stage means (s)")
for k, o in out["groups"].items():
    P(f"{k:16s} " + " ".join(f"{s} {t:.3f}" for s, t in o["stage_mean"].items()))
P(f"-- paired: {a.primary} minus other (same rep, question)")
for (pol, rep), X in sorted(D.items()):
    if pol == a.primary or (a.primary, rep) not in D:
        continue
    Y = D[(a.primary, rep)]
    ks = sorted(set(X) & set(Y))
    wy, wx = np.array([Y[k]["wall"] for k in ks]), np.array([X[k]["wall"] for k in ks])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ks), (10000, len(ks)))
    q = {}
    for pc in [90, 99]:
        m = np.percentile(wy[idx], pc, axis=1) - np.percentile(wx[idx], pc, axis=1)
        q[pc] = (
            float(np.percentile(wy, pc) - np.percentile(wx, pc)),
            float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)),
        )
    o = {
        "n": len(ks),
        "F1": boot([Y[k]["f1"] - X[k]["f1"] for k in ks]),
        "cost": boot([Y[k]["cost"] - X[k]["cost"] for k in ks]),
        "wall_mean": boot(wy - wx),
        "wall_p90": q[90],
        "wall_p99": q[99],
    }
    out["paired"][f"{pol}|{rep}"] = o
    P(
        f"vs {pol:12s} r{rep} n {o['n']} | F1 {fmt(o['F1'])} | $ {fmt(o['cost'])} | wall mean {fmt(o['wall_mean'], 3)} p90 {fmt(o['wall_p90'], 2)} p99 {fmt(o['wall_p99'], 2)}"
    )
json.dump(out, open(f"{a.dir}/summary.json", "w"), indent=1)
open(f"{a.dir}/summary.txt", "w").write("\n".join(L) + "\n")
