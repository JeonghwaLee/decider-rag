"""Summary of load_ent.py (open-loop). Per policy x arrival rate: n, cloud rate, F1, cloud $ per 1k, latency (arrival -> answer)
mean / p50 / p90 / p99, local-queue wait mean, makespan; goodput within deadlines d in {1, 2, 5, 10} s:
  F1-goodput = sum of F1 over requests answered within d, per second of the run (makespan = last arrival time);
  correct-goodput = number of requests with F1 = 1 answered within d, per second.
Paired bootstrap (same requests) of s1234_EN minus each other policy at the same rate: F1-within-d per request (F1 if latency <= d
else 0) and latency mean. Usage: analyze_load.py --dir <load dir>."""

import argparse, glob, json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bootlib import boot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--primary", default="s1234_EN")
a = ap.parse_args()
D = {}
for f in sorted(glob.glob(f"{a.dir}/*_r*.jsonl")):
    b = os.path.basename(f)[:-6]
    pol, rate = b.rsplit("_r", 1)
    D[(pol, float(rate))] = {x["i"]: x for x in map(json.loads, open(f))}
DL = [1, 2, 5, 10]
out = {}
L = []
P = lambda s: (print(s, flush=True), L.append(s))
for rate in sorted({r for _, r in D}):
    P(f"== rate {rate} req/s")
    for pol in sorted({p for p, r in D if r == rate}):
        X = D[(pol, rate)]
        v = list(X.values())
        lat = np.array([x["latency"] for x in v])
        f = np.array([x["f1"] for x in v])
        dur = max(x["t_arr"] for x in v)
        wait = np.array([x["t_local_start"] - x["t_arr"] for x in v])
        o = {
            "n": len(v),
            "cloud": float(np.mean([x["branch"] != "E" for x in v])),
            "F1": float(f.mean()),
            "cost": float(np.mean([x["cost"] for x in v])),
            "lat_mean": float(lat.mean()),
            "lat_p50": float(np.median(lat)),
            "lat_p90": float(np.percentile(lat, 90)),
            "lat_p99": float(np.percentile(lat, 99)),
            "wait_mean": float(wait.mean()),
            "makespan": float(max(x["t_done"] for x in v)),
            "f1_goodput": {d: float(f[lat <= d].sum() / dur) for d in DL},
            "correct_goodput": {d: float(((f == 1) & (lat <= d)).sum() / dur) for d in DL},
            "f1_within": {d: float(np.where(lat <= d, f, 0).mean()) for d in DL},
        }
        out[f"{pol}|{rate}"] = o
        P(
            f"  {pol:10s} n {o['n']} cloud {o['cloud']:.3f} F1 {o['F1']:.4f} $ {o['cost']:.4f} | lat mean {o['lat_mean']:.2f} p50 {o['lat_p50']:.2f} p90 {o['lat_p90']:.2f} "
            f"p99 {o['lat_p99']:.2f} wait {o['wait_mean']:.2f} | F1-within "
            + " ".join(f"{d}s {o['f1_within'][d]:.3f}" for d in DL)
            + " | F1-goodput/s "
            + " ".join(f"{d}s {o['f1_goodput'][d]:.3f}" for d in DL)
        )
    if (a.primary, rate) in D:
        Y = D[(a.primary, rate)]
        for pol in sorted({p for p, r in D if r == rate and p != a.primary}):
            X = D[(pol, rate)]
            ks = sorted(set(X) & set(Y))
            s = f"  {a.primary} - {pol:10s}: lat mean {boot([Y[k]['latency'] - X[k]['latency'] for k in ks])[0]:+.2f}"
            for d in DL:
                b = boot(
                    [
                        (Y[k]["f1"] if Y[k]["latency"] <= d else 0) - (X[k]["f1"] if X[k]["latency"] <= d else 0)
                        for k in ks
                    ]
                )
                s += f" | F1-within {d}s {b[0]:+.3f} [{b[1]:+.3f},{b[2]:+.3f}]"
            P(s)
json.dump(out, open(f"{a.dir}/summary.json", "w"), indent=1, default=str)
open(f"{a.dir}/summary.txt", "w").write("\n".join(L) + "\n")
