#!/usr/bin/env python3
"""Per-(policy, rate, seed) stage times of a load_ent3 run directory.
wait = local-stage queueing (t_local_start - t_arr); local = retrieval/rerank/prune (+ rule complexity); sel = selection call;
gen = answer generation incl. recovery; also error count (failed requests are scored F1 0 / cost 0 by load_ent3 and excluded here).
Usage: stage_times.py --dir runs/load3/hotpot_place --out ..._summary_stages.txt"""
import argparse, glob, json, os, re, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--dir")
ap.add_argument("--out")
a = ap.parse_args()
L = []
for f in sorted(
    glob.glob(os.path.join(a.dir, "*_r*_s*.jsonl")), key=lambda f: (float(re.search(r"_r([\d.]+)_s", f).group(1)), f)
):
    pol, rate, seed = re.match(r"(.+)_r([\d.]+)_s(\d+)\.jsonl", os.path.basename(f)).groups()
    R = [json.loads(x) for x in open(f)]
    ok = [x for x in R if "error" not in x]
    m = lambda k: st.mean(k(x) for x in ok)
    L.append(
        f"rate {rate} seed {seed} {pol:15s} n {len(R)} err {len(R) - len(ok)} | lat {m(lambda x: x['latency']):.3f} wait {m(lambda x: x['t_local_start'] - x['t_arr']):.3f} "
        f"local {m(lambda x: x['t_local_end'] - x['t_local_start']):.3f} sel_ms {1000 * m(lambda x: x['t_sel_end'] - x['t_local_end']):.1f} gen {m(lambda x: x['t_done'] - x['t_sel_end']):.3f}"
    )
open(a.out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
