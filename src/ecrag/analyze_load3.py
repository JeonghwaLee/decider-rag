"""Summary of the load test (load_ent3.py). Per load level (rate) and policy, pooled over the arrival seeds: F1, total cloud $ (answer +
selector, $/1k-query units as E2, no cache discount), latency mean/p50/p95 (arrival -> answer), DQ-F1 at the E2 deadline D (F1 if
latency <= D else 0; D = p90 single-user wall of the calibration policy on non-test data, as E2), throughput (completed / makespan)
and goodput = sum of in-deadline F1 per second of makespan. Paired bootstrap ref - policy over (seed, question) for F1, $, latency
and DQ-F1. Also the selector stage time (t_sel_end - t_local_end) mean/p95 and the local queue wait (t_local_start - t_arr) mean.
Usage: analyze_load3.py --dir D --ref big2e4_vllm --deadline 0.722 --out f.txt"""

import argparse, glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bootlib import boot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--deadline", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    R = {}
    for fn in glob.glob(f"{a.dir}/*_r*_s*.jsonl"):
        m = re.match(r"(.+)_r([0-9.]+)_s(\d+)\.jsonl$", os.path.basename(fn))
        pol, rate, seed = m.group(1), float(m.group(2)), int(m.group(3))
        R.setdefault(rate, {}).setdefault(pol, {})[seed] = [json.loads(l) for l in open(fn)]
    P(f"{a.dir} | deadline D {a.deadline} s | ref {a.ref}")
    for rate in sorted(R):
        pols = sorted(R[rate])
        seeds = sorted(set.intersection(*[set(R[rate][p]) for p in pols]))
        P(f"== rate {rate:g}/s | seeds {seeds} | policies {len(pols)}")
        V = {}
        for p in pols:
            rows = [(s, r) for s in seeds for r in R[rate][p][s]]
            key = {(s, r["run"], r["id"]): r for s, r in rows}
            V[p] = key
            lat = np.array([r["latency"] for _, r in rows])
            f = np.array([r["f1"] for _, r in rows])
            c = np.array([r["total_cost"] for _, r in rows])
            dq = np.where(lat <= a.deadline, f, 0.0)
            ms = [max(r["t_done"] for r in R[rate][p][s]) for s in seeds]
            n = [len(R[rate][p][s]) for s in seeds]
            sel = np.array([r["t_sel_end"] - r["t_local_end"] for _, r in rows if "t_sel_end" in r])
            qw = np.array([r["t_local_start"] - r["t_arr"] for _, r in rows])
            br = [r.get("branch") for _, r in rows]
            err = sum("error" in r for _, r in rows)
            P(
                f"  {p:14s} F1 {f.mean():.4f} | $ {c.mean():.4f} | lat mean {lat.mean():6.2f} p50 {np.median(lat):6.2f} p95 {np.percentile(lat, 95):6.2f} | DQ-F1 {dq.mean():.4f} (met {np.mean(lat <= a.deadline):.3f})"
                f" | thr {np.mean([x / y for x, y in zip(n, ms)]):.2f}/s goodput {dq.sum() / sum(ms):.2f}/s | sel ms mean {1e3 * sel.mean():.0f} p95 {1e3 * np.percentile(sel, 95):.0f}"
                f" | local wait {qw.mean():.2f}s | E/N/T {br.count('E')}/{br.count('N')}/{br.count('T')} err {err}"
            )
        ks = sorted(set.intersection(*[set(V[p]) for p in pols]))
        for p in pols:
            if p == a.ref:
                continue
            g = lambda pp, fn: np.array([fn(V[pp][k]) for k in ks])
            cells = []
            for nm, fn in [
                ("F1", lambda r: r["f1"]),
                ("$", lambda r: r["total_cost"]),
                ("lat", lambda r: r["latency"]),
                ("DQ-F1", lambda r: r["f1"] if r["latency"] <= a.deadline else 0.0),
            ]:
                w = boot(g(a.ref, fn) - g(p, fn))
                cells.append(f"{nm} {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}]")
            P(f"    {a.ref} - {p:14s} " + " | ".join(cells))
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
