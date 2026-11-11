"""Load v3 re-analysis with repeat-aware CIs. Same inputs as analyze_load3.py (which treats every
(seed, question) as an independent sample).
  F1, $      per question: mean over the arrival seeds of (ref - policy), then question-level paired bootstrap (bootlib.boot).
             The seeds repeat the same questions, so the unit is the question, not the (seed, question) pair.
  latency,   queueing makes neighbouring arrivals dependent. Per seed, the paired difference (ref - policy) at each arrival position
  DQ-F1      (same question, same arrival time for every policy within a seed) is resampled with a circular moving-block bootstrap
             (block --block arrivals, default 50; --block2 sensitivity), blocks drawn within each seed; the pooled estimate is the
             equal-weight mean of the seeds. Per-seed means are printed first. Two seeds do not estimate run-to-run variability of
             the serving environment; the CI covers question/arrival sampling only.
Usage: analyze_load3b.py --dir D --ref big2e4_vllm --deadline 0.722 --out f.txt"""

import argparse, glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bootlib import boot  # noqa: E402


def block_boot(diffs, L, B=4000, seed=0):
    """diffs: list (per seed) of arrays ordered by arrival position. Circular moving-block bootstrap within each seed."""
    rng = np.random.default_rng(seed)
    est = np.mean([d.mean() for d in diffs])
    out = np.empty(B)
    for b in range(B):
        ms = []
        for d in diffs:
            n = len(d)
            nb = int(np.ceil(n / L))
            st = rng.integers(0, n, nb)
            idx = ((st[:, None] + np.arange(L)[None, :]) % n).ravel()[:n]
            ms.append(d[idx].mean())
        out[b] = np.mean(ms)
    return est, np.percentile(out, 2.5), np.percentile(out, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--deadline", type=float, required=True)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--block2", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    R = {}
    for fn in glob.glob(f"{a.dir}/*_r*_s*.jsonl"):
        m = re.match(r"(.+)_r([0-9.]+)_s(\d+)\.jsonl$", os.path.basename(fn))
        pol, rate, seed = m.group(1), float(m.group(2)), int(m.group(3))
        R.setdefault(rate, {}).setdefault(pol, {})[seed] = {r["pos"]: r for r in map(json.loads, open(fn))}
    P(
        f"{a.dir} | deadline D {a.deadline} s | ref {a.ref} | F1/$: question-level bootstrap of seed-averaged differences | "
        f"lat/DQ-F1: per-seed + moving-block bootstrap (block {a.block}; sensitivity {a.block2})"
    )
    dq = lambda r: r["f1"] if r["latency"] <= a.deadline else 0.0
    for rate in sorted(R):
        if a.ref not in R[rate]:
            continue
        pols = sorted(R[rate])
        seeds = sorted(set.intersection(*[set(R[rate][p]) for p in pols]))
        P(f"== rate {rate:g}/s | seeds {seeds}")
        for p in pols:
            per = []
            for s in seeds:
                X = list(R[rate][p][s].values())
                lat = np.array([r["latency"] for r in X])
                ms = max(r["t_done"] for r in X)
                per.append(
                    f"s{s}: F1 {np.mean([r['f1'] for r in X]):.4f} p95 {np.percentile(lat, 95):5.2f} DQ-F1 {np.mean([dq(r) for r in X]):.4f} thr {len(X) / ms:.2f}/s"
                )
            P(f"  {p:14s} " + " | ".join(per))
        for p in pols:
            if p == a.ref:
                continue
            cells = []
            for nm, fn in [("F1", lambda r: r["f1"]), ("$", lambda r: r["total_cost"])]:
                D = {}
                for s in seeds:
                    A, Bp = R[rate][a.ref][s], R[rate][p][s]
                    for pos in set(A) & set(Bp):
                        assert (A[pos]["run"], A[pos]["id"]) == (Bp[pos]["run"], Bp[pos]["id"])
                        D.setdefault((A[pos]["run"], A[pos]["id"]), []).append(fn(A[pos]) - fn(Bp[pos]))
                w = boot(np.array([np.mean(v) for v in D.values()]))
                cells.append(f"{nm} {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}] (nq {len(D)})")
            for nm, fn in [("lat", lambda r: r["latency"]), ("DQ-F1", dq)]:
                ds = []
                for s in seeds:
                    A, Bp = R[rate][a.ref][s], R[rate][p][s]
                    ks = sorted(set(A) & set(Bp))
                    ds.append(np.array([fn(A[k]) - fn(Bp[k]) for k in ks]))
                w = block_boot(ds, a.block)
                w2 = block_boot(ds, a.block2)
                cells.append(
                    f"{nm} {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}] (b{a.block2} [{w2[1]:+.4f},{w2[2]:+.4f}]; per seed "
                    + " ".join(f"{d.mean():+.4f}" for d in ds)
                    + ")"
                )
            P(f"    {a.ref} - {p:14s} " + " | ".join(cells))
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
