"""Cost-aware T with the ENTC decider's own cost heads, decomposed.
Policy (as costaware_T.py): cloud iff dEN > a; T iff cloud and dNT - lam * dC > b. Fallback N: a truncated T is retried with N; the
evaluation uses the retried answer's F1 and the T + N cost (analyze_ent2.arrays). Expected extra cost of T over N per question
($/1k, proxy prices):
  dC = 0.27 (T_in - N_in)/1e3                         (exact, known before the call)
     + 1.1 (E[T_out] - Nbar_out)/1e3                   (output length)
     + p_trunc * (0.27 N_in + 1.1 Nbar_out)/1e3        (expected retry)
Variants (dEN/dNT always the ENTC model's own scores, so only dC changes):
  none   lam = 0 only
  trunc  p_trunc = ENTC trunc head; E[T_out] = p_trunc * Tmax + (1 - p_trunc) * Tbar_ok (train-free: Tbar_ok = mean T_out of
         non-truncated rows in the OTHER cross-fit half / in-sample for the in-sample line)
  len    E[T_out] = expm1(9 * lout) * s (s = smearing factor mean(T_out)/mean(expm1(9 lout)) fitted on the other half); p_trunc = base
         rate of the other half
  both   E[T_out] from lout (as len), p_trunc from the trunc head
  oracle realised dC (diagnostic upper bound, as costaware_T.py)
Nbar_out = mean N_out of the fitting half. Selection (a, b, lam) by analyze_ent2.choose over costaware_T.LAMS, cross-fitted on val
halves (10 repeats x 2), budget m x rule->T val $. Also prints the lout calibration (corr, mean predicted vs actual T_out) and the
trunc head AUC on val.
The trunc head output is clipped to [0,1] (raw regression outputs can be negative, which would give a negative expected retry cost).
Usage: costaware_T2.py --val V --dec <ENTC pred pattern with {split}> --fallback N --mults 0.5 1 2 --out file"""

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, choose, pol  # noqa: E402
from analyze_en import load  # noqa: E402
from policy_probe import preds  # noqa: E402
from costaware_T import LAMS, dcost  # noqa: E402

TMAX = 6144


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--dec", required=True)
    ap.add_argument("--fallback", default="N")
    ap.add_argument("--mults", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    va, vsp = load(a.val)
    gv = arrays(va, a.fallback)
    G5 = gv[:5]
    base = float(np.where(gv[5], gv[4], 0).mean())
    d = preds(va, a.dec, vsp)
    X = {(x["pool"], x["id"]): x for x in json.load(open(a.dec.format(split=vsp)))}
    pt_raw = np.array([X[(r["pool"], r["id"])]["trunc"] for r in va])
    pt = np.clip(pt_raw, 0.0, 1.0)  # regression head -> probability in [0,1]
    lo = np.array([X[(r["pool"], r["id"])]["lout"] for r in va])
    A = lambda k: np.array([r[k] for r in va], float)
    Tin, Nin, Tout, Nout = A("T_in"), A("N_in"), A("T_out"), A("N_out")
    tr = np.array([r["T_trunc"] for r in va], bool)
    raw = np.expm1(9 * lo)
    orc = dcost(va, a.fallback)
    n = len(va)
    P(
        f"val {a.val} n {n} | dec {a.dec} | base $ {base:.4f} | trunc rate {tr.mean():.3f} AUC(trunc head) {auc(tr, pt_raw):.3f} (trunc pred <0: {int((pt_raw < 0).sum())}, >1: {int((pt_raw > 1).sum())}, min {pt_raw.min():.4f}; clipped to [0,1]) n_trunc {int(tr.sum())} | lout: corr(log) "
        f"{np.corrcoef(lo, np.log1p(Tout) / 9)[0, 1]:.3f} corr(tokens) {np.corrcoef(raw, Tout)[0, 1]:.3f} mean pred expm1 {raw.mean():.0f} vs actual {Tout.mean():.0f} "
        f"(smearing {Tout.mean() / raw.mean():.2f}) | oracle dC mean {orc.mean():.3f} sd {orc.std():.3f}"
    )

    def dC(kind, fit):
        nb = Nout[fit].mean()
        s = Tout[fit].mean() / raw[fit].mean()
        p0 = tr[fit].mean()
        tok_ok = Tout[fit][~tr[fit]].mean()
        p = pt if kind in ("trunc", "both") else np.full(n, p0)
        eT = raw * s if kind in ("len", "both") else p * TMAX + (1 - p) * tok_ok
        out = 0.27 * (Tin - Nin) / 1e3 + 1.1 * (eT - nb) / 1e3
        if a.fallback == "N":
            out = out + p * (0.27 * Nin + 1.1 * nb) / 1e3
        return out

    for k in ["trunc", "len", "both"]:
        c = dC(k, np.arange(n))
        P(f"  dC={k:6s} corr with oracle dC {np.corrcoef(c, orc)[0, 1]:.3f} mean {c.mean():.3f}")

    def best(dCv, idx, bud, lams):
        out = None
        for lam in lams:
            c = choose({"dEN": d["dEN"], "dNT": d["dNT"] - lam * dCv}, "E/N/T", bud, G5, idx)
            if not np.isnan(c[0]) and (out is None or c[0] > out[0]):
                out = (c[0], lam, c[1], c[2])
        return out

    for m in a.mults:
        bud = base * m
        P(f"== x{m} budget $ {bud:.4f}")
        for k in ["none", "trunc", "len", "both", "oracle"]:
            lams = [0.0] if k == "none" else LAMS
            res = []
            for rep in range(a.reps):
                perm = np.random.default_rng(rep).permutation(n)
                H1, H2 = perm[: n // 2], perm[n // 2 :]
                for Fi, Ev in [(H1, H2), (H2, H1)]:
                    dv = orc if k == "oracle" else (np.zeros(n) if k == "none" else dC(k, Fi))
                    c = best(dv, Fi, bud, lams)
                    F, C = pol(d["dEN"][Ev], d["dNT"][Ev] - c[1] * dv[Ev], c[2], c[3], *(x[Ev] for x in G5))
                    res.append((F.mean(), C.mean(), c[1]))
            res = np.array(res)
            P(
                f"  dC={k:6s} cross-fit F1 {res[:, 0].mean():.4f} sd {res[:, 0].std():.4f} $ {res[:, 1].mean():.4f} ({res[:, 1].mean() / bud:.2f}x) lam median {np.median(res[:, 2])}"
                f" | diff vs none computed below"
            )
            if k == "none":
                ref = res[:, 0]
            else:
                dd = res[:, 0] - ref
                P(
                    f"      paired (same halves) F1 - none: mean {dd.mean():+.4f} min {dd.min():+.4f} max {dd.max():+.4f} positive {np.mean(dd > 0):.2f}"
                )
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
