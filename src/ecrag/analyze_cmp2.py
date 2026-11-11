"""HotpotQA fixed-policy analysis with stored paired CIs. Same policies as analyze_cmp.py, plus:
  --guard {none,v1,v2,v2b}  date-path guard version (cmp_guard2.py; v1 = analyze_cmp_lib.guard_ok)
  trace root / cloud tag from ECRAG_TRACE / ECRAG_CLOUD_TAG (e.g. pilot6 truncation policies A/B)
  --dump  analyze2 --dump_scores files of the router run set (validation scores, test scores)
Saves per-query paired arrays (keys, branch, F1 per policy) and every difference CI (bootlib.boot, seed 0, B=10k) so the
numbers are reproducible from the output file alone. Comparisons: each policy vs rule, and router+cmp/fix vs rule+cmp."""

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


ap = argparse.ArgumentParser()
ap.add_argument("--cand", required=True)
ap.add_argument("--dump", required=True)
ap.add_argument("--guard", default="v2b", choices=["none", "v1", "v2", "v2b"])
ap.add_argument("--out", required=True)
a = ap.parse_args()
D0 = json.load(open(a.dump.format(split="test2")))["hotpotqa"]
vk = [tuple(k) for k in D0["val_keys"]]
vs = np.array(D0["val_scores"][a.cand])
r_rule = D0["val_rule_rate"]
t_plain = thr_rate(vs, r_rule)
vfix = np.array([k in cmp_answers("val", vk, a.guard)[0] for k in vk])
m = int(math.ceil(r_rule * len(vk) - 1e-9))
q = np.sort(vs[~vfix])[::-1]
t_fix = float((q[m - 1] + q[m]) / 2)
res = {"cand": a.cand, "guard": a.guard, "trace": TRACE, "val_rule_rate": r_rule, "t_plain": t_plain, "t_fix": t_fix}
print(f"cand {a.cand} guard {a.guard} trace {TRACE.split('/')[-1]}")
for sp in ["val", "test2", "test3"]:
    if sp == "val":
        keys, s = vk, vs
    else:
        D = json.load(open(a.dump.format(split=sp)))["hotpotqa"]
        keys = [tuple(k) for k in D["keys"]]
        s = np.array(D["scores"][a.cand])
    R = {(r["split"], r["pool"], r["id"]): r for r in load("hotpotqa", sp, with_text=False)}
    ce, why = cmp_answers(sp, keys, a.guard)
    fe = np.array([R[k]["f1_edge"] for k in keys])
    fc = np.array([R[k]["f1_cloud"] for k in keys])
    fe2 = np.array([ce.get(k, R[k]["f1_edge"]) for k in keys])
    fix = np.array([k in ce for k in keys])
    cost = np.array([R[k]["cost"] for k in keys])
    rule = np.array([not R[k]["ecrag_edge"] for k in keys])
    base = np.where(rule, fc, fe)
    pol = {
        "rule": (rule, fe),
        "rule+cmp": (rule, fe2),
        "router": (s > t_plain, fe),
        "router+cmp": (s > t_plain, fe2),
        "router+cmp/fix": ((s > t_fix) & ~fix, fe2),
    }
    o = {
        "n": len(keys),
        "n_cmp_answered": int(sum(why.values()) if a.guard == "none" else len(why)),
        "n_cmp_used": int(fix.sum()),
        "path_F1_on_used": float(np.mean(list(ce.values()))) if ce else None,
        "edge_F1_on_used": float(fe[fix].mean()) if fix.any() else None,
        "keys": [list(k) for k in keys],
        "policies": {},
    }
    Y = {}
    for n, (cl, e) in pol.items():
        y = np.where(cl, fc, e)
        Y[n] = y
        o["policies"][n] = {
            "rate": float(cl.mean()),
            "n_cloud": int(cl.sum()),
            "F1": float(y.mean()),
            "cost_ratio": float(cost[cl].sum() / cost[rule].sum()),
            "diff_vs_rule": boot(y - base),
            "cloud": cl.astype(int).tolist(),
            "f1": y.tolist(),
        }
    o["router+cmp/fix_vs_rule+cmp"] = boot(Y["router+cmp/fix"] - Y["rule+cmp"])
    res[sp] = o
    print(
        f"== {sp} n {len(keys)} path used {fix.sum()} (edge F1 {o['edge_F1_on_used']} -> path {o['path_F1_on_used']})"
    )
    for n, p in o["policies"].items():
        d = p["diff_vs_rule"]
        print(
            f"   {n:15s} rate {p['rate']:.4f} ({p['n_cloud']}) cost x{p['cost_ratio']:.2f} F1 {p['F1']:.4f} d {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]"
        )
    d = o["router+cmp/fix_vs_rule+cmp"]
    print(f"   router+cmp/fix - rule+cmp: {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]")
json.dump(res, open(a.out, "w"))
