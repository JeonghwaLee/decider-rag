"""Summary of e2e_ent.py (integrated run of the frozen E/N/T and E/N policies). Quality, cloud $ and latency from the same run.
Per split x policy x rep: n, E/N/T rates, T fallbacks, F1, cloud $ per 1k queries, latency mean/p50/p90/p99 (bootlib CIs),
stage means (incl. decider). Paired (same split, rep, question): each policy vs rule (rule->T) and vs ruleN, and
s1234_EN vs ens3_ENT: F1, $, latency mean, p90, p99. Run-to-run agreement. Sanity: top-k = stage-1 trace; live decider scores
vs the offline predictions; live branch vs the branch the offline scores give; offline (stored-answer) F1 of the same
policy on the same questions."""

import argparse, glob, json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bootlib import boot  # noqa: E402
import ent_data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
a = ap.parse_args()
cfg = json.load(open(f"{a.dir}/config.json"))
thr = cfg["thresholds"]
D = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{a.dir}/*_r[0-9].jsonl")):
    b = os.path.basename(f)[:-6]
    sp, _, pool, *rest = b.split("_")
    pol = "_".join(rest[:-1])
    rep = rest[-1]
    for l in open(f):
        x = json.loads(l)
        D[(sp, pol, rep)][(f"pool_{pool}", x["id"])] = x
OFF = {sp: {(r["pool"], r["id"]): r for r in ent_data.load(sp, "B")} for sp in {k[0] for k in D}}
p90 = lambda v: np.percentile(v, 90)
p99 = lambda v: np.percentile(v, 99)
out = {"groups": {}, "paired": {}, "rerun": {}}
L = []
P = lambda s: (print(s, flush=True), L.append(s))
for (sp, pol, rep), X in sorted(D.items()):
    v = list(X.values())
    tot = np.array([x["total"] for x in v])
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
        "n_fallback": fb,
        "F1": boot(f),
        "cost": boot(c),
        "lat_mean": boot(tot),
        "lat_p50": float(np.median(tot)),
        "lat_p90": boot(tot, stat=p90),
        "lat_p99": boot(tot, stat=p99),
        "stage_mean": {k: float(np.mean(t)) for k, t in st.items()},
        "topk_same_as_trace": float(np.mean([x["topk_same_as_trace"] for x in v])),
    }
    if pol in thr:  # live vs offline decider
        ta, tb = thr[pol]
        agree, dif, offF = [], [], []
        for (pool, i), x in X.items():
            off = [x["offline"][s] for s in x["offline"]]
            if None in off:
                continue
            oe, on = np.mean([z[0] for z in off]), np.mean([z[1] for z in off])
            dif.append(abs(oe - x["dEN"]))
            ob = "T" if (oe > ta and on > tb) else ("N" if oe > ta else "E")
            agree.append(ob == x["branch"])
            r = OFF[sp][(pool, i)]
            offF.append({"E": r["fE"], "N": r["fN"], "T": r["fT"]}[ob])
        o.update(
            {
                "branch_eq_offline": float(np.mean(agree)),
                "dEN_absdiff_mean": float(np.mean(dif)),
                "offline_F1_same_q": float(np.mean(offF)),
                "decider_peak_gib": float(np.max([x["decider_peak_gib"] for x in v])),
                "decider_in_tokens_mean": float(np.mean([x["decider_in_tokens"] for x in v])),
            }
        )
    else:
        offF = []
        for (pool, i), x in X.items():
            r = OFF[sp][(pool, i)]
            offF.append(r["fE"] if r["ecrag_edge"] else (r["fT"] if pol == "rule" else r["fN"]))
        o["offline_F1_same_q"] = float(np.mean(offF))
    out["groups"][f"{sp}|{pol}|{rep}"] = o
    s = (
        f"{sp} {pol:9s} {rep} n {o['n']} E/N/T {br.get('E', 0)}/{br.get('N', 0)}/{br.get('T', 0)} fb {fb} | F1 {o['F1'][0]:.4f} (offline {o['offline_F1_same_q']:.4f}) "
        f"| $ {o['cost'][0]:.4f} | lat mean {o['lat_mean'][0]:.3f} [{o['lat_mean'][1]:.3f},{o['lat_mean'][2]:.3f}] p50 {o['lat_p50']:.3f} "
        f"p90 {o['lat_p90'][0]:.2f} p99 {o['lat_p99'][0]:.2f} [{o['lat_p99'][1]:.2f},{o['lat_p99'][2]:.2f}] | topk=trace {o['topk_same_as_trace']:.3f}"
    )
    if pol in thr:
        s += f" | branch=offline {o['branch_eq_offline']:.3f} |dEN diff| {o['dEN_absdiff_mean']:.4f} decider {o['stage_mean']['decider']*1e3:.0f} ms peak {o['decider_peak_gib']:.1f} GiB"
    P(s)
P("-- stage means (s)")
for k, o in out["groups"].items():
    P(f"{k:24s} " + " ".join(f"{s} {t:.3f}" for s, t in o["stage_mean"].items()))


def paired(X, Rr):
    ks = sorted(set(X) & set(Rr))
    lx, lr = np.array([X[k]["total"] for k in ks]), np.array([Rr[k]["total"] for k in ks])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ks), (10000, len(ks)))
    q = {}
    for nm in ["p90", "p99"]:
        pc = int(nm[1:])
        m = np.percentile(lx[idx], pc, axis=1) - np.percentile(lr[idx], pc, axis=1)
        q[nm] = (
            float(np.percentile(lx, pc) - np.percentile(lr, pc)),
            float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)),
        )
    return {
        "n": len(ks),
        "F1": boot([X[k]["f1"] - Rr[k]["f1"] for k in ks]),
        "cost": boot([X[k]["cost"] - Rr[k]["cost"] for k in ks]),
        "lat_mean": boot(lx - lr),
        "lat_p90": q["p90"],
        "lat_p99": q["p99"],
    }


fmt = lambda t, d=4: f"{t[0]:+.{d}f} [{t[1]:+.{d}f},{t[2]:+.{d}f}]"
P("-- paired (same split, rep, question)")
for (sp, pol, rep), X in sorted(D.items()):
    for ref in ["rule", "ruleN"] + (["ens3_ENT"] if pol == "s1234_EN" else []):
        if pol == ref or (sp, ref, rep) not in D or (ref == "ruleN" and pol == "rule"):
            continue
        o = paired(X, D[(sp, ref, rep)])
        out["paired"][f"{sp}|{pol}|vs_{ref}|{rep}"] = o
        P(
            f"{sp} {pol:9s} vs {ref:8s} {rep} n {o['n']} | F1 {fmt(o['F1'])} | $ {fmt(o['cost'])} | lat mean {fmt(o['lat_mean'], 3)} p90 {fmt(o['lat_p90'], 2)} p99 {fmt(o['lat_p99'], 2)}"
        )
P("-- run-to-run (r0 vs r1)")
for (sp, pol, rep), X in sorted(D.items()):
    if rep != "r0" or (sp, pol, "r1") not in D:
        continue
    Y = D[(sp, pol, "r1")]
    ks = sorted(set(X) & set(Y))
    o = {
        "n": len(ks),
        "same_answer": float(np.mean([X[k]["answer"] == Y[k]["answer"] for k in ks])),
        "same_branch": float(np.mean([X[k]["branch"] == Y[k]["branch"] for k in ks])),
    }
    out["rerun"][f"{sp}|{pol}"] = o
    P(f"{sp} {pol:9s} n {len(ks)} same answer {o['same_answer']:.3f} same branch {o['same_branch']:.3f}")
json.dump(out, open(f"{a.dir}/summary.json", "w"), indent=1)
open(f"{a.dir}/summary.txt", "w").write("\n".join(L) + "\n")
