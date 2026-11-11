"""Qasper analyses. Primary metric F1; question-level and paper-cluster bootstrap CIs.

(1) Scope vs engine/prompt decomposition of the EC-RAG rule's F1, same questions:
    P1   pilot-1 pooled retrieval, stage-1 HF edge answer, stage-2 cloud answer (EC-RAG setup)
    P2   pooled retrieval, regenerated 'orig' answers with the cfg3 engines (vLLM edge, Qwen3.8 chat cloud)
    S3   paper-scoped retrieval (target paper given), 'orig' answers (= cfg3 Qasper)
    rule branches come from the respective trace (pooled for P1/P2, scoped for S3).
    P2 - P1 = engine/prompt formatting, S3 - P2 = retrieval scope.
(2) Edge prompt at policy level (val only; decision rule fixed: adopt only if the val question-level CI and the paper-cluster
    CI both exclude 0): cfg3 Qasper with the edge answer from 'orig_abst' instead of 'orig', same EC-RAG branches; also all-edge.
    By reference group: ans / mixed / unans.
"""

from paths import RUNS  # noqa: E402
import json, os, sys, glob
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from router_data import f1, DATA  # noqa: E402
from bootlib import boot, paper_ids  # noqa: E402

R = RUNS


def jl(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


def group(refs):
    u = [r.strip().lower() == "unanswerable" for r in refs]
    return "unans" if all(u) else ("mixed" if any(u) else "ans")


out = {}
for sp in ["val", "test2", "test3"]:
    rows = []
    for pd in sorted(glob.glob(f"{R}/pilot3/trace/qasper/{sp}/pool_*")):
        pool = pd.split("/")[-1]
        refs = {k: v["answers"] for k, v in jl(f"{DATA}/qasper/{sp}/{pool}/answers/answers.jsonl").items()}
        t1 = jl(f"{R}/pilot1/trace/qasper/{sp}/{pool}/trace.jsonl")
        c1 = jl(f"{R}/pilot1/trace/qasper/{sp}/{pool}/cloud_q38flashfp8.jsonl")
        t3 = jl(f"{pd}/trace.jsonl")
        a3 = jl(f"{R}/pilot3/answers/qasper/{sp}/{pool}/orig.jsonl")
        p2f = f"{R}/pilot6/answers_pooled/qasper/{sp}/{pool}/orig.jsonl"
        a2 = jl(p2f) if os.path.exists(p2f) else {}
        ab = f"{R}/pilot3/answers/qasper/{sp}/{pool}/orig_abst.jsonl"
        aa = jl(ab) if os.path.exists(ab) else {}
        for i in t3:
            if i not in t1 or i not in c1 or i not in a3:
                continue
            r = {
                "key": (sp, pool, i),
                "grp": group(refs[i]),
                "P1": (
                    f1(c1[i]["cloud_answer"], refs[i])
                    if not t1[i]["ecrag_route_edge"]
                    else f1(t1[i]["edge_answer"], refs[i])
                ),
                "S3": a3[i]["f1_cloud"] if not t3[i]["ecrag_route_edge"] else a3[i]["f1_edge"],
                "S3_alledge": a3[i]["f1_edge"],
                "rule_scoped": not t3[i]["ecrag_route_edge"],
            }
            if i in a2:
                r["P2"] = a2[i]["f1_cloud"] if not t1[i]["ecrag_route_edge"] else a2[i]["f1_edge"]
            if i in aa:
                r["S3_abst"] = a3[i]["f1_cloud"] if not t3[i]["ecrag_route_edge"] else aa[i]["f1_edge"]
                r["S3_alledge_abst"] = aa[i]["f1_edge"]
            rows.append(r)
    keys = [r["key"] for r in rows]
    cl = paper_ids(sp, keys)
    o = {"n": len(rows), "n_papers": int(len(set(cl)))}
    print(
        f"== {sp} n={len(rows)} papers={o['n_papers']} groups",
        {g: sum(r["grp"] == g for r in rows) for g in ["ans", "mixed", "unans"]},
    )
    has = lambda f: all(f in r for r in rows)
    for f in ["P1", "P2", "S3", "S3_abst"]:
        if has(f):
            v = np.array([r[f] for r in rows])
            o[f] = float(v.mean())
            print(f"   rule F1 {f:8s} {v.mean():.4f}")
    for a_, b_ in [("P2", "P1"), ("S3", "P2"), ("S3", "P1"), ("S3_abst", "S3"), ("S3_alledge_abst", "S3_alledge")]:
        if has(a_) and has(b_):
            d = np.array([r[a_] - r[b_] for r in rows])
            q, c = boot(d), boot(d, cl)
            o[f"{a_}-{b_}"] = {"q": q, "cluster": c}
            print(
                f"   {a_:16s} - {b_:11s} {q[0]:+.4f}  q-CI [{q[1]:+.4f},{q[2]:+.4f}]  paper-CI [{c[1]:+.4f},{c[2]:+.4f}]"
            )
            for g in ["ans", "mixed", "unans"]:
                m = np.array([r["grp"] == g for r in rows])
                if m.any():
                    o[f"{a_}-{b_}"][g] = float(d[m].mean())
            print(
                "      by group:",
                {g: round(o[f"{a_}-{b_}"].get(g, float("nan")), 4) for g in ["ans", "mixed", "unans"]},
            )
    out[sp] = o
json.dump(out, open(f"{R}/pilot6/qasper2.json", "w"), indent=1)
