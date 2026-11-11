"""Offline comparison with an AutoMix-style self-verification cascade. Same stored answers, budget, validation
calibration and equal-spend controller as online_cmp.py (policy_probe.online_frontier, 5 shared arrival orders), but the cascade's own
cost arrays: every executed call is charged in cloud tokens.
  --cand E (HotpotQA, E/N): edge answer -> edge self-verification (cloud $0; local time only) -> N if score <= t. Escalated: cost N.
  --cand N (MuSiQue, N/T): N answer -> cloud self-verification -> T if score <= t. Cost N + verification always, + T if escalated.
           A truncated T returns the N answer already produced (no second N call; e2e/online_cmp instead charge a new N call).
Escalation score for the shared machinery: 1 - verification score (higher = escalate). If even 'never escalate' exceeds the budget on
validation, the operating point is reported as INFEASIBLE (no substitution by E). Other scorers (Decider, LLM selectors) exactly as
online_cmp (arrays with --fallback, selector tokens not charged; their selector cost is printed separately where recorded).
Usage: automix_cmp.py --val V --test T... --fallback E|N --fam E/N|E/N/T --cand E|N --ver_dir DIR --ref decider --scorers name=pattern ... --mults 0.5 1 1.5 --out file"""

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, pol, choose  # noqa: E402
from analyze_en import load  # noqa: E402
from bootlib import boot  # noqa: E402
from policy_probe import preds, frontier, online_frontier  # noqa: E402


def ver(rows, vdir, spec):
    V = {}
    fn = f"{vdir}/{spec.replace(':', '_')}.jsonl"
    for l in open(fn):
        x = json.loads(l)
        V[(x["pool"], x["id"])] = x
    miss = [r for r in rows if (r["pool"], r["id"]) not in V]
    assert not miss, f"{len(miss)} rows without verification in {fn}"
    g = lambda k: np.array([V[(r["pool"], r["id"])][k] for r in rows], float)
    return g("score"), g("ver_cost"), g("ver_sec"), g("n_valid")


def am_arrays(rows, cand, vcost):
    """G5 of the cascade; T falls back to the existing N answer when truncated."""
    fE, fN, fT, cN, cT = arrays(rows, "E")[:5]
    tr = np.array([bool(r["T_trunc"]) for r in rows])
    if cand == "E":
        return fE, fN, fT, cN, cT  # E/N only; verification is local
    return fE, fN, np.where(tr, fN, fT), cN + vcost, cN + vcost + cT


def am_d(score, cand):
    s = 1.0 - score
    return {"dEN": s, "dNT": np.full(len(s), -np.inf)} if cand == "E" else {"dEN": np.full(len(s), np.inf), "dNT": s}


def choose_am(d, cand, bud, G5):
    if cand == "E":
        return choose(d, "E/N", bud, G5), True
    best = None
    for b in np.r_[-np.inf, np.quantile(d["dNT"], np.linspace(0, 1, 41)), np.inf]:
        F, C = pol(d["dEN"], d["dNT"], -np.inf, b, *G5)
        if C.mean() <= bud + 1e-12 and (best is None or F.mean() > best[0]):
            best = (F.mean(), -np.inf, b)
    if best is None:  # infeasible: never escalate is the cheapest cascade plan
        F, C = pol(d["dEN"], d["dNT"], -np.inf, np.inf, *G5)
        return (F.mean(), -np.inf, np.inf), False
    return best, True


def frontier_am(d, cand, G5, base):
    if cand == "E":
        return frontier(d, G5, "E/N", base)
    M = np.round(np.arange(0.1, 3.01, 0.05), 2)
    Q = []
    for m in M:
        c, _ = choose_am(d, cand, base * m, G5)
        Q.append((1.0, float((d["dNT"] > c[2]).mean())))
    return M, np.array(Q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--fallback", default="N")
    ap.add_argument("--fam", default="E/N/T")
    ap.add_argument("--cand", choices=["E", "N"], required=True)
    ap.add_argument("--ver_dir", required=True)
    ap.add_argument("--ref", default="decider")
    ap.add_argument("--scorers", nargs="+", required=True)
    ap.add_argument("--mults", nargs="+", type=float, default=[1.0])
    ap.add_argument("--orders", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    SC = dict(s.split("=", 1) for s in a.scorers)
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    J = {"mults": {}}
    va, vsp = load(a.val)
    gv = arrays(va, a.fallback)
    G5v = gv[:5]
    base = float(np.where(gv[5], gv[4], 0).mean())
    vs, vc, vt, vn = ver(va, a.ver_dir, a.val)
    G5va = am_arrays(va, a.cand, vc)
    dva = am_d(vs, a.cand)
    rows, D = [], {nm: {k: [] for k in ["fE", "fN", "fT", "dEN", "dNT"]} for nm in SC}
    TS = []
    for t in a.test:
        r, sp = load(t)
        rows += r
        TS.append(ver(r, a.ver_dir, t))
        for nm, p in SC.items():
            x = preds(r, p, sp)
            for k in D[nm]:
                D[nm][k].append(x[k])
    D = {nm: {k: np.concatenate(v) for k, v in d.items()} for nm, d in D.items()}
    G5 = arrays(rows, a.fallback)[:5]
    ts, tc, tt, tn = (np.concatenate([x[i] for x in TS]) for i in range(4))
    G5a = am_arrays(rows, a.cand, tc)
    da = am_d(ts, a.cand)
    fE, fN, fT, cN, cT = arrays(rows, "E")[:5]
    tr = np.array([bool(r["T_trunc"]) for r in rows])
    P(
        f"stream {'+'.join(a.test)} n {len(rows)} | val {a.val} n {len(va)} | base $ {base:.4f} | cand {a.cand} | orders {a.orders}"
    )
    P(
        f"  verification: mean score val {vs.mean():.3f} test {ts.mean():.3f} | parsable samples/8 test {tn.mean():.2f} (0 parsable: {(tn == 0).mean():.3f})"
        f" | cloud $ per 1k {tc.mean():.4f} | mean sec {tt.mean():.3f} p95 {np.quantile(tt, .95):.3f}"
    )
    for m in a.mults:
        bud = base * m
        F, C, X = {}, {}, {}
        P(f"== x{m} budget $ {bud:.4f}")
        for nm, p in SC.items():
            dv = preds(va, p, vsp)
            c = choose(dv, a.fam, bud, G5v)
            M, Q = frontier(dv, G5v, a.fam, base)
            R = [
                online_frontier(D[nm], G5, c[1], c[2], M, Q, m, bud, np.random.default_rng(1000 + r))
                for r in range(a.orders)
            ]
            F[nm] = np.mean([x[0] for x in R], 0)
            C[nm] = np.mean([x[1] for x in R], 0)
            F0, C0 = pol(D[nm]["dEN"], D[nm]["dNT"], c[1], c[2], *G5)
            X[nm] = dict(thr=[c[1], c[2]], fixed_f1=F0.mean(), fixed_cost=C0.mean())
            P(
                f"  {nm:14s} online F1 {F[nm].mean():.4f} $ {C[nm].mean():.4f} ({C[nm].mean() / bud:.2f}x) | fixed thr F1 {F0.mean():.4f} $ {C0.mean():.4f} ({C0.mean() / bud:.2f}x)"
            )
        c, feas = choose_am(dva, a.cand, bud, G5va)
        M, Q = frontier_am(dva, a.cand, G5va, base)
        F0, C0 = pol(da["dEN"], da["dNT"], c[1], c[2], *G5a)
        R = [
            online_frontier(da, G5a, c[1], c[2], M, Q, m, bud, np.random.default_rng(1000 + r)) for r in range(a.orders)
        ]
        F["automix"] = np.mean([x[0] for x in R], 0)
        C["automix"] = np.mean([x[1] for x in R], 0)
        thr = 1 - (c[1] if a.cand == "E" else c[2])  # escalate iff verification score < thr  (score <= official t)
        esc = (da["dEN"] > c[1]) if a.cand == "E" else (da["dNT"] > c[2])
        X["automix"] = dict(
            feasible=feas,
            thr_escalate_if_score_below=float(thr),
            fixed_f1=F0.mean(),
            fixed_cost=C0.mean(),
            esc_rate=float(esc.mean()),
        )
        P(
            f"  {'automix':14s} online F1 {F['automix'].mean():.4f} $ {C['automix'].mean():.4f} ({C['automix'].mean() / bud:.2f}x) | fixed thr F1 {F0.mean():.4f} $ {C0.mean():.4f}"
            f" ({C0.mean() / bud:.2f}x) | val {'feasible' if feas else 'INFEASIBLE (never-escalate already over budget)'} | escalate iff score < {thr:.3f} (rate {esc.mean():.3f})"
        )
        # diagnostics at the fixed validation threshold
        cand_f, up_f = (fE, fN) if a.cand == "E" else (fN, np.where(tr, fN, fT))
        keep = ~esc
        P(
            f"  automix diag: kept a candidate that escalation would improve {np.mean(keep & (up_f > cand_f)):.3f} | escalated without gain {np.mean(esc & (up_f <= cand_f)):.3f}"
            f" | escalated with gain {np.mean(esc & (up_f > cand_f)):.3f} | escalated because all samples were unparsable {np.mean(esc & (tn == 0)):.3f} (all-unparsable {np.mean(tn == 0):.3f})"
        )
        if a.cand == "N":
            P(
                f"  automix cost split per 1k: N {cN.mean():.4f} | verification {tc.mean():.4f} | T on escalation {np.where(esc, cT, 0).mean():.4f}"
            )
        for nm in list(SC) + ["automix"]:
            if nm != a.ref:
                w = boot(F[a.ref] - F[nm])
                P(
                    f"  {a.ref} - {nm:14s} {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}] | $ diff {C[a.ref].mean() - C[nm].mean():+.4f}"
                )
        J["mults"][str(m)] = {
            "budget": bud,
            "online": {k: [float(F[k].mean()), float(C[k].mean())] for k in F},
            "fixed": X,
            "diff": {nm: [float(v) for v in boot(F[a.ref] - F[nm])] for nm in F if nm != a.ref},
        }
    # test frontier of the cascade over all verification thresholds (descriptive; thresholds are not chosen on test)
    fr = []
    for t in np.r_[np.arange(0, 1.0001, 0.125)]:
        e = ts <= t + 1e-9
        Fx, Cx = (
            (np.where(e, fN, fE), np.where(e, cN, 0.0))
            if a.cand == "E"
            else (np.where(e, np.where(tr, fN, fT), fN), cN + tc + np.where(e, cT, 0.0))
        )
        fr.append((float(t), float(Fx.mean()), float(Cx.mean()), float(e.mean())))
    P(
        "  cascade test curve (escalate iff score <= t): "
        + " ".join(f"t={t:.3f}:F1 {f:.4f} $ {c:.4f} esc {e:.2f}" for t, f, c, e in fr)
    )
    J["curve"] = fr
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump(J, open(a.out.replace(".txt", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
