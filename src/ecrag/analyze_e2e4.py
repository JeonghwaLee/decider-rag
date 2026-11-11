"""Summary of an e2e_ent4 run. Per policy: n, branch mix, F1, answer $, selector $, total $,
cloud prefix-cache hits (selector / answer tokens), total $ with a cached-input price (--cache_price x input price, applied only to
tokens vLLM reports as prefix-cache hits), wall mean/p50/p95, deadline-qualified F1 (F1 if wall <= D else 0; D = p90 wall of the
--calib_policy in --calib dir, fixed before test). Paired bootstrap (questions) of --ref minus each policy for F1, total $, wall, DQ-F1.
Usage: analyze_e2e4.py --dir D --calib C --calib_policy ruleN_tuned --ref big2e4_EN [--cache_price 0.1]"""

import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bootlib import boot  # noqa: E402


def read(d):
    P = {}
    for f in glob.glob(f"{d}/*_r0.jsonl"):
        for l in open(f):
            x = json.loads(l)
            P.setdefault(x["policy"], {})[(os.path.basename(f).rsplit("_" + x["policy"], 1)[0], x["id"])] = x
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--calib_policy", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cache_price", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L = []
    P_ = lambda s: (print(s, flush=True), L.append(s))
    C = read(a.calib)[a.calib_policy]
    D = float(np.percentile([x["wall"] for x in C.values()], 90))
    P = read(a.dir)
    keys = sorted(set.intersection(*[set(v) for v in P.values()]))
    cp = a.cache_price
    P_(
        f"{a.dir} | common questions {len(keys)} | deadline D = p90 wall of {a.calib_policy} on {a.calib} = {D:.3f} s | cached input price {cp} x"
    )
    A = {}
    for nm, R in sorted(P.items()):
        X = [R[k] for k in keys]
        g = lambda f: np.array([f(x) for x in X], float)
        f1, w = g(lambda x: x["f1"]), g(lambda x: x["wall"])
        ans, sel = g(lambda x: x["cost"]), g(lambda x: x.get("sel_cost", 0.0))
        sc, ac = g(lambda x: x.get("sel_cached", 0.0)), g(lambda x: x.get("ans_cached", 0.0))
        tot_c = ans + sel - 0.27 * (1 - cp) * (sc + ac) / 1e3
        dq = np.where(w <= D, f1, 0.0)
        br = {b: sum(x["branch"] == b for x in X) for b in "ENT"}
        sp = g(lambda x: sum(x.get("sel_prompt_tokens", [0])))
        A[nm] = {"f1": f1, "tot": ans + sel, "totc": tot_c, "wall": w, "dq": dq}
        P_(
            f"  {nm:14s} E/N/T {br['E']}/{br['N']}/{br['T']} | F1 {f1.mean():.4f} | $ answer {ans.mean():.4f} + selector {sel.mean():.4f} = {ans.mean() + sel.mean():.4f}"
            f" | cache hits sel {sc.mean():.0f} ans {ac.mean():.0f} tok (sel prompt {sp.mean():.0f}) -> total $ w/ cache {tot_c.mean():.4f}"
            f" | wall mean {w.mean():.3f} p50 {np.median(w):.3f} p95 {np.percentile(w, 95):.3f} | DQ-F1 {dq.mean():.4f} (met {np.mean(w <= D):.3f})"
        )
    for nm in sorted(A):
        if nm == a.ref:
            continue
        r, o = A[a.ref], A[nm]
        s = lambda k: boot(r[k] - o[k])
        f, t, tc, w, dq = s("f1"), s("tot"), s("totc"), s("wall"), s("dq")
        P_(
            f"  {a.ref} - {nm:14s} F1 {f[0]:+.4f} [{f[1]:+.4f},{f[2]:+.4f}] | total $ {t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}] (w/ cache {tc[0]:+.4f})"
            f" | wall {w[0]:+.3f} [{w[1]:+.3f},{w[2]:+.3f}] | DQ-F1 {dq[0]:+.4f} [{dq[1]:+.4f},{dq[2]:+.4f}]"
        )
    if a.out:
        open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
