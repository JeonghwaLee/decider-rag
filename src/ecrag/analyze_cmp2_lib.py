"""(helpers copied from analyze_cmp2.py)"""

from paths import RUNS  # noqa: E402
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from router_data import load, f1, DATA, TRACE  # noqa: E402
from bootlib import boot  # noqa: E402
import analyze_cmp_lib as L  # noqa: E402
from cmp_guard2 import guard2, guard2b  # noqa: E402

A = f"{RUNS}/pilot2/answers/hotpotqa"


def thr_rate(s, r):
    q = np.sort(s)[::-1]
    m = int(math.ceil(r * len(q) - 1e-9))
    return float((q[m - 1] + q[m]) / 2)


def cmp_answers(sp, keys, guard):
    out, why = {}, {}
    for p in sorted({k[1] for k in keys}):
        refs = {
            json.loads(l)["id"]: json.loads(l)["answers"]
            for l in open(f"{DATA}/hotpotqa/{sp}/{p}/answers/answers.jsonl")
        }
        d = f"{TRACE}/hotpotqa/{sp}/{p}"
        ch = {json.loads(l)["node_id"]: json.loads(l)["text"] for l in open(f"{d}/chunks.jsonl")}
        tr = {json.loads(l)["id"]: json.loads(l) for l in open(f"{d}/trace.jsonl")}
        for l in open(f"{A}/{sp}/{p}/cmp_edge.jsonl"):
            x = json.loads(l)
            if x["answer"] is None:
                continue
            t = tr[x["id"]]
            cs = [ch[i] for i in t["topk_ids"][: t["k"]]]
            ok = {
                "none": True,
                "v1": L.guard_ok(x, t["query"], "\n".join(cs)),
                "v2": guard2(x, t["query"], "\n".join(cs))[0],
                "v2b": guard2b(x, t["query"], cs)[0],
            }[guard]
            why[(sp, p, x["id"])] = ok
            if ok:
                out[(sp, p, x["id"])] = f1(x["answer"], refs[x["id"]])
    return out, why
