"""E5 workload shift. One checkpoint per scorer, no retraining; the same online controller
(policy_probe.online_frontier, W=100, eta=0.05) for every scorer. Val = HotpotQA + MuSiQue + 2Wiki val (fallback N, E/N/T family,
budget = m x rule->T val $ of the pooled val); each scorer's own val frontier/thresholds. Test stream: --n questions per dataset
(HotpotQA test4, MuSiQue test, 2Wiki test), either in segments H -> M -> W (random order within a segment) or fully mixed.
Per scorer: F1, $/budget, per-segment $/budget, and after each shift the max rolling-W spend / budget and the number of requests
until the rolling-W spend is back within +-10% of the budget. Scores: --scorers name=dir (pred_<spec with _>.json in dir, keys dEN/dNT;
for LLM selectors 'dir' is the raw selector dir, probe 'two' = dEN/dNT). 5 repetitions with seeds 1000+r; paired bootstrap vs --ref."""

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, choose  # noqa: E402
from analyze_en import load  # noqa: E402
from bootlib import boot  # noqa: E402
from policy_probe import frontier, online_frontier  # noqa: E402

VAL = ["legacy:val", "pilot9:musique:val", "pilot10:2wiki:val"]
TEST = ["pilot8:hotpotqa:test4", "pilot9:musique:test", "pilot10:2wiki:test"]


class Fixed:
    def __init__(self, order):
        self.o = order

    def permutation(self, n):
        return self.o


def scores(d, specs_rows):
    """d = dir or dir#p1,p2,p3: per-dataset probe (two -> dEN/dNT, choice -> c_dEN/c_dNT), in VAL/TEST dataset order"""
    d, pr = (d.split("#") + ["two,two,two"])[:2]
    pr = pr.split(",")
    out = {"dEN": [], "dNT": []}
    for i, (spec, rows) in enumerate(specs_rows):
        P = {(x["pool"], x["id"]): x for x in json.load(open(f"{d}/pred_{spec.replace(':', '_')}.json"))}
        for k in out:
            kk = ("c_" + k) if pr[i] == "choice" else k
            out[k].append(np.array([P[(r["pool"], r["id"])][kk] for r in rows]))
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorers", nargs="+", required=True)
    ap.add_argument("--ref", default="decider")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--W", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    SC = dict(s.split("=", 1) for s in a.scorers)
    vr = [(s, load(s)[0]) for s in VAL]
    va = [r for _, rs in vr for r in rs]
    gv = arrays(va, "N")
    G5v = gv[:5]
    base = float(np.where(gv[5], gv[4], 0).mean())
    bud = base * a.mult
    tr = []
    for s in TEST:
        rs = load(s)[0]
        rng = np.random.default_rng(7)
        tr.append((s, [rs[i] for i in sorted(rng.choice(len(rs), min(a.n, len(rs)), replace=False))]))
    rows = [r for _, rs in tr for r in rs]
    G5 = arrays(rows, "N")[:5]
    seg = np.concatenate([np.full(len(rs), i) for i, (_, rs) in enumerate(tr)])
    n = len(rows)
    P(
        f"val {'+'.join(VAL)} n {len(va)} | test {'+'.join(TEST)} n {n} | budget x{a.mult} $ {bud:.4f} | per-dataset all-N $ "
        + " ".join(f"{G5[3][seg == i].mean():.3f}" for i in range(3))
    )
    res = {}
    for nm, d in SC.items():
        dv = scores(d, vr)
        dt = scores(d, tr)
        c = choose(dv, "E/N/T", bud, G5v)
        M, Q = frontier(dv, G5v, "E/N/T", base)
        res[nm] = {}
        for mode in ["segment", "mixed"]:
            FF, CC, info = [], [], []
            for r in range(a.reps):
                rng = np.random.default_rng(1000 + r)
                order = (
                    np.concatenate([rng.permutation(np.where(seg == i)[0]) for i in range(3)])
                    if mode == "segment"
                    else rng.permutation(n)
                )
                F, C = online_frontier(dt, G5, c[1], c[2], M, Q, a.mult, bud, Fixed(order))
                FF.append(F)
                CC.append(C)
                cs = C[order]
                roll = np.convolve(cs, np.ones(a.W) / a.W, "valid")  # roll[j] = mean of requests j..j+W-1
                if mode == "segment":
                    sh = []
                    for b in [a.n, 2 * a.n]:
                        first = float(cs[b : b + a.W].mean() / bud)  # spend of the first W requests after the shift
                        ends = roll[b - a.W + 1 : b + a.n - a.W + 1]  # rolling windows ending at b .. b + n - 1
                        rec = next(
                            (
                                j
                                for j in range(len(ends))
                                if j > 0 and ends[j] <= 1.10 * bud and ends[j - 1] > 1.10 * bud
                            ),
                            0 if ends[0] <= 1.10 * bud else None,
                        )
                        mx = float(ends.max() / bud)
                        sh.append((first, rec, mx))
                    info.append(sh)
            F = np.mean(FF, 0)
            C = np.mean(CC, 0)
            res[nm][mode] = F
            line = (
                f"  {nm:14s} {mode:7s} F1 {F.mean():.4f} $ {C.mean() / bud:.2f}x | per segment $/bud "
                + " ".join(f"{C[seg == i].mean() / bud:.2f}" for i in range(3))
                + " | F1 "
                + " ".join(f"{F[seg == i].mean():.3f}" for i in range(3))
            )
            if mode == "segment":
                line += (
                    " | shift H->M, M->W: first-W $/bud "
                    + ", ".join(f"{np.mean([x[k][0] for x in info]):.2f}" for k in range(2))
                    + " ; max roll $/bud "
                    + ", ".join(f"{np.mean([x[k][2] for x in info]):.2f}" for k in range(2))
                    + " ; requests until roll <= 1.1x "
                    + ", ".join(str([x[k][1] for x in info]) for k in range(2))
                )
            P(line)
    for mode in ["segment", "mixed"]:
        for nm in SC:
            if nm != a.ref:
                w = boot(res[a.ref][mode] - res[nm][mode])
                P(f"  {mode:7s} {a.ref} - {nm:14s} {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}]")
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
