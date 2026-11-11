"""Policy probes on stored answers. No training; scorer predictions and val thresholds are
read from existing files. For one scorer, family E/N/T and budget multiple m (budget = m x rule->T val $, from the analyze_ent2 json):

1. diag  - is a test miss a ranking problem or a threshold problem? Spearman of predicted vs realised dEN / dNT (val, test), and
           the test F1 of thresholds chosen on the TEST itself at the same $ budget (oracle, upper bound of any recalibration).
2. fixed - the val thresholds (as deployed).
3. online budget controller (deployable; uses only past requests' scores and realised $, never labels). Requests arrive in a
   random order (R orders). The controller keeps target routing fractions (q_c cloud, q_t T) initialised from val, and a pressure
   s (start 1). Thresholds for request i = quantiles of the scores of the last W requests at 1 - s*q_c (dEN) and, among those
   above it, 1 - s*q_t/q_c (dNT); before W requests the val thresholds are used. After each request, s is updated from the
   running spend of the last W requests vs the budget: s <- clip(s * exp(eta * (budget - spend) / budget), s_min, s_max), so
   it spends more when under budget and less when over, with a bounded step.
3b. online frontier (deployable): as 3, but the controller moves along the val budget frontier (the val-optimal routing
   fractions at each budget level), so a cut removes T before N. See online_frontier().
4. argmax  - direct E/N/T choice: action = argmax_a (f_a_hat - lam * c_a_bar + beta_a) with the decider's quality heads f_E/f_N/f_T
           and the mean val $ of each action; (lam, beta_T) chosen on val under the budget (cross-fitted gate on val, then test).
Paired bootstrap vs rule->T and vs fixed. Usage:
policy_probe.py --json <analyze_ent2 json> --val <spec> --test <spec...> --fallback N --scorer name=pattern --mult 1.0 --out <file>"""

import argparse, json, os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ent2 import arrays, pol, choose  # noqa: E402
from analyze_en import load  # noqa: E402
from bootlib import boot  # noqa: E402


TF = {}


def preds(rows, pat, split):
    """mean over comma-separated patterns (seed ensembles); pat 'tfidf:<train spec>' = analyze_ent2's TF-IDF Ridge (dEN, dNT only)."""
    if pat.startswith("tfidf:"):
        if pat not in TF:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import Ridge

            trr = [r for s in pat[6:].split(",") for r in load(s, "A")[0] if r["fT"] is not None]
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in trr])
            X = lambda Rr: np.hstack(
                [vec.transform([r["query"] for r in Rr]).toarray(), np.array([[r["k"] / 10] for r in Rr])]
            )
            TF[pat] = (
                X,
                {
                    k: Ridge(alpha=3.0).fit(X(trr), [f(r) for r in trr])
                    for k, f in [("dEN", lambda r: r["fN"] - r["fE"]), ("dNT", lambda r: r["fT"] - r["fN"])]
                },
            )
        X, RG = TF[pat]
        Z = X(rows)
        o = {k: m.predict(Z) for k, m in RG.items()}
        o.update({k: np.zeros(len(rows)) for k in ["fE", "fN", "fT"]})
        return o
    PP = [{(x["pool"], x["id"]): x for x in json.load(open(q.format(split=split)))} for q in pat.split(",")]
    return {
        k: np.mean([np.array([P[(r["pool"], r["id"])][k] for r in rows]) for P in PP], 0)
        for k in ["fE", "fN", "fT", "dEN", "dNT"]
    }


def online(d, G5, a0, b0, qc, qt, bud, rng, W=100, eta=0.05, smin=0.25, smax=4.0):
    n = len(d["dEN"])
    order = rng.permutation(n)
    F = np.zeros(n)
    C = np.zeros(n)
    s = 1.0
    hist = []
    costs = []
    for t, i in enumerate(order):
        if len(hist) < W:
            a, b = a0, b0
        else:
            H = np.array(hist[-W:])
            fc = min(1.0, s * qc)
            a = -np.inf if fc >= 1 else float(np.quantile(H[:, 0], 1 - fc))
            cl = H[:, 0] > a
            ft = min(1.0, s * qt / max(qc, 1e-9))
            b = np.inf if (qt <= 0 or cl.sum() == 0) else (-np.inf if ft >= 1 else float(np.quantile(H[cl, 1], 1 - ft)))
        Fi, Ci = pol(d["dEN"][i : i + 1], d["dNT"][i : i + 1], a, b, *(x[i : i + 1] for x in G5))
        F[i], C[i] = Fi[0], Ci[0]
        hist.append((d["dEN"][i], d["dNT"][i]))
        costs.append(C[i])
        sp = float(np.mean(costs[-W:]))
        if len(costs) >= 20:
            s = float(np.clip(s * np.exp(eta * (bud - sp) / bud), smin, smax))
    return F, C


def frontier(dv, G5v, fam, base):
    """val routing fractions (q_cloud, q_T) of the val-optimal thresholds at budget levels m x base."""
    M = np.round(np.arange(0.1, 3.01, 0.05), 2)
    Q = []
    for m in M:
        c = choose(dv, fam, base * m, G5v)
        cl = dv["dEN"] > c[1]
        Q.append((float(cl.mean()), float((cl & (dv["dNT"] > c[2])).mean())))
    return M, np.array(Q)


def online_frontier(d, G5, a0, b0, M, Q, m0, bud, rng, W=100, eta=0.05):
    """Controller on the val budget frontier: state u = log budget level (start log m0); request i uses the val-optimal routing
    fractions of level exp(u), turned into thresholds by quantiles of the last W requests' scores (val thresholds before W).
    After each request u += eta * (budget - spend of the last W) / budget, clipped to the frontier range."""
    n = len(d["dEN"])
    order = rng.permutation(n)
    F = np.zeros(n)
    C = np.zeros(n)
    u = np.log(m0)
    hist = []
    costs = []
    for i in order:
        if len(hist) < W:
            a, b = a0, b0
        else:
            j = int(np.argmin(np.abs(M - np.exp(u))))
            qc, qt = Q[j]
            H = np.array(hist[-W:])
            a = -np.inf if qc >= 1 else (np.inf if qc <= 0 else float(np.quantile(H[:, 0], 1 - qc)))
            cl = H[:, 0] > a
            ft = qt / max(qc, 1e-9)
            b = np.inf if (qt <= 0 or cl.sum() == 0) else (-np.inf if ft >= 1 else float(np.quantile(H[cl, 1], 1 - ft)))
        Fi, Ci = pol(d["dEN"][i : i + 1], d["dNT"][i : i + 1], a, b, *(x[i : i + 1] for x in G5))
        F[i], C[i] = Fi[0], Ci[0]
        hist.append((d["dEN"][i], d["dNT"][i]))
        costs.append(C[i])
        if len(costs) >= 20:
            u = float(np.clip(u + eta * (bud - np.mean(costs[-W:])) / bud, np.log(M[0]), np.log(M[-1])))
    return F, C


def online_async(d, G5, a0, b0, M, Q, m0, bud, rng, walls, cbar, rate=2.0, W=100, eta=0.05, reserve=True):
    """Frontier controller under concurrency: Poisson arrivals at `rate`; each request's duration is drawn from the
    measured wall-time distribution of its action (MuSiQue integrated run, `walls`); its $ is known only at completion. Spend
    estimate over the last W DISPATCHED requests: reserve=False counts only completed ones (in-flight requests are invisible);
    reserve=True also counts in-flight requests at the val mean $ of their action (reservation), replaced by the realised $ on
    completion (settlement). Returns per-question F1 and $ and the max running overspend."""
    import heapq

    n = len(d["dEN"])
    order = rng.permutation(n)
    t = np.cumsum(rng.exponential(1 / rate, n))
    F = np.zeros(n)
    C = np.zeros(n)
    u = np.log(m0)
    hist = []
    disp = []
    done = {}
    pend = []
    act_c = {"E": 0.0, "N": cbar[1], "T": cbar[2]}
    worst = 0.0
    brs = {}
    cnt = {"E": 0, "N": 0, "T": 0}
    for j, i in enumerate(order):
        now = t[j]
        while pend and pend[0][0] <= now:
            _, q = heapq.heappop(pend)
            done[q] = True
            if (
                reserve == "learn"
            ):  # settlement also updates the per-action reservation price (running mean of realised $)
                bq = brs[q]
                cnt[bq] += 1
                act_c[bq] += (C[q] - act_c[bq]) / (cnt[bq] + 1)
        if len(hist) < W:
            a, b = a0, b0
        else:
            k = int(np.argmin(np.abs(M - np.exp(u))))
            qc, qt = Q[k]
            H = np.array(hist[-W:])
            a = -np.inf if qc >= 1 else (np.inf if qc <= 0 else float(np.quantile(H[:, 0], 1 - qc)))
            cl = H[:, 0] > a
            ft = qt / max(qc, 1e-9)
            b = np.inf if (qt <= 0 or cl.sum() == 0) else (-np.inf if ft >= 1 else float(np.quantile(H[cl, 1], 1 - ft)))
        e = d["dEN"][i]
        tt = d["dNT"][i]
        br = "T" if (e > a and tt > b) else ("N" if e > a else "E")
        Fi, Ci = pol(d["dEN"][i : i + 1], d["dNT"][i : i + 1], a, b, *(x[i : i + 1] for x in G5))
        F[i], C[i] = Fi[0], Ci[0]
        heapq.heappush(pend, (now + float(rng.choice(walls[br])), i))
        hist.append((e, tt))
        disp.append((i, br))
        brs[i] = br
        win = disp[-W:]
        sp = (
            np.mean([C[q] if q in done else (act_c[b_] if reserve else 0.0) for q, b_ in win])
            if reserve
            else (np.mean([C[q] for q, _ in win if q in done]) if any(q in done for q, _ in win) else 0.0)
        )
        if len(disp) >= 20:
            u = float(np.clip(u + eta * (bud - sp) / bud, np.log(M[0]), np.log(M[-1])))
        if j >= W:
            worst = max(worst, float(np.mean([C[q] for q, _ in win])) / bud - 1)
    return F, C, worst


def argmax_pol(d, G5, cbar, lam, beta):
    fE, fN, fT, cN, cT = G5
    S = np.stack([d["fE"] - lam * cbar[0], d["fN"] - lam * cbar[1], d["fT"] - lam * cbar[2] + beta], 1)
    act = S.argmax(1)
    return np.choose(act, [fE, fN, fT]), np.choose(act, [np.zeros_like(cN), cN, cT])


def argmax_choose(d, G5, cbar, bud, idx):
    best = None
    sub = lambda x: x[idx]
    dd = {k: v[idx] for k, v in d.items()}
    g = [x[idx] for x in G5]
    for lam in [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 12, 20, 50]:
        for beta in np.linspace(-0.3, 0.3, 13):
            F, C = argmax_pol(dd, g, cbar, lam, beta)
            if C.mean() <= bud + 1e-12 and (best is None or F.mean() > best[0]):
                best = (F.mean(), lam, beta)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--fallback", default="N")
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--fam", default="E/N/T")
    ap.add_argument("--orders", type=int, default=5)
    ap.add_argument("--walls", default=None)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    J = json.load(open(a.json))
    nm, pat = a.scorer.split("=", 1)
    bud = J["val_rule_T_cost"] * a.mult
    a0, b0 = J["thr"][f"{nm}|{a.fam}|x{a.mult}"]
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    va, vsp = load(a.val)
    gv = arrays(va, a.fallback)
    G5v = gv[:5]
    dv = preds(va, pat, vsp)
    qc = float((dv["dEN"] > a0).mean())
    qt = float(((dv["dEN"] > a0) & (dv["dNT"] > b0)).mean())
    cbar = [0.0, float(G5v[3].mean()), float(G5v[4].mean())]
    P(
        f"{nm} {a.fam} x{a.mult} fb {a.fallback} | budget $ {bud:.4f} | val thr a {a0:+.4f} b {b0:+.4f} | val q_cloud {qc:.3f} q_T {qt:.3f} | mean $ N {cbar[1]:.4f} T {cbar[2]:.4f}"
    )
    rv = lambda d, g: (spearmanr(d["dEN"], g[1] - g[0])[0], spearmanr(d["dNT"], g[2] - g[1])[0])
    P(f"  val  spearman dEN {rv(dv, G5v)[0]:+.3f} dNT {rv(dv, G5v)[1]:+.3f}")
    # argmax gate on val (5x5 cross-fitted, same folds as analyze_ent2) vs two-threshold gate
    n = len(va)
    Fa = np.zeros((5, n))
    Ft = np.zeros((5, n))
    for rep in range(5):
        fold = np.random.default_rng(100 + rep).permutation(n) % 5
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            b = argmax_choose(dv, G5v, cbar, bud, tr)
            Fa[rep, te] = argmax_pol({kk: v[te] for kk, v in dv.items()}, [x[te] for x in G5v], cbar, b[1], b[2])[0]
            c = choose(dv, a.fam, bud, G5v, tr)
            Ft[rep, te] = pol(dv["dEN"][te], dv["dNT"][te], c[1], c[2], *(x[te] for x in G5v))[0]
    w = boot(Fa.mean(0) - Ft.mean(0))
    P(
        f"  val gate: two-threshold {Ft.mean():.4f} | argmax {Fa.mean():.4f} | argmax - two-threshold {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}]"
    )
    am = argmax_choose(dv, G5v, cbar, bud, np.arange(n))
    P(f"  argmax full-val choice lam {am[1]} beta_T {am[2]:+.3f}")
    FM, FQ = frontier(dv, G5v, a.fam, J["val_rule_T_cost"])
    for spec in a.test:
        te, tsp = load(spec)
        g = arrays(te, a.fallback)
        G5 = g[:5]
        dt = preds(te, pat, tsp)
        rT = np.where(g[5], g[2], g[0])
        cRT = np.where(g[5], g[4], 0)
        P(
            f"== {spec} n {len(te)} | rule->T F1 {rT.mean():.4f} $ {cRT.mean():.4f} | spearman dEN {rv(dt, G5)[0]:+.3f} dNT {rv(dt, G5)[1]:+.3f}"
        )
        F0, C0 = pol(dt["dEN"], dt["dNT"], a0, b0, *G5)
        o = choose(dt, a.fam, bud, G5)
        Fo, Co = pol(dt["dEN"], dt["dNT"], o[1], o[2], *G5)
        Fm, Cm = argmax_pol(dt, G5, cbar, am[1], am[2])
        R = [online(dt, G5, a0, b0, qc, qt, bud, np.random.default_rng(1000 + r)) for r in range(a.orders)]
        Fc = np.mean([x[0] for x in R], 0)
        Cc = np.mean([x[1] for x in R], 0)
        R2 = [
            online_frontier(dt, G5, a0, b0, FM, FQ, a.mult, bud, np.random.default_rng(1000 + r))
            for r in range(a.orders)
        ]
        Ff = np.mean([x[0] for x in R2], 0)
        Cf = np.mean([x[1] for x in R2], 0)
        for lab, Fx, Cx in [
            ("fixed (val thr)", F0, C0),
            ("online controller", Fc, Cc),
            ("online frontier", Ff, Cf),
            ("argmax E/N/T", Fm, Cm),
            ("oracle thr on test", Fo, Co),
        ]:
            w1, w2 = boot(Fx - rT), boot(Fx - F0)
            P(
                f"  {lab:20s} F1 {Fx.mean():.4f} $ {Cx.mean():.4f} | vs rule->T {w1[0]:+.4f} [{w1[1]:+.4f},{w1[2]:+.4f}] | vs fixed {w2[0]:+.4f} [{w2[1]:+.4f},{w2[2]:+.4f}]"
            )
        P(
            f"  online per order F1 {' '.join(f'{x[0].mean():.4f}' for x in R)} | $ {' '.join(f'{x[1].mean():.4f}' for x in R)}"
        )
        if a.walls:
            Wl = json.load(open(a.walls))
            for res in (False, True, "learn"):
                R3 = [
                    online_async(
                        dt,
                        G5,
                        a0,
                        b0,
                        FM,
                        FQ,
                        a.mult,
                        bud,
                        np.random.default_rng(2000 + r),
                        Wl,
                        cbar,
                        rate=a.rate,
                        reserve=res,
                    )
                    for r in range(a.orders)
                ]
                F3 = np.mean([x[0] for x in R3], 0)
                C3 = [x[1].mean() for x in R3]
                w1, w2 = boot(F3 - rT), boot(F3 - F0)
                P(
                    f"  async {({False: 'completed-only', True: 'reserve', 'learn': 'reserve+learn'})[res]:14s} rate {a.rate} F1 {F3.mean():.4f} $ {np.mean(C3):.4f} (per order {' '.join(f'{c:.4f}' for c in C3)}; "
                    f"max over budget {max(C3) / bud - 1:+.1%}; worst 100-request window {max(x[2] for x in R3):+.1%}) | vs rule->T {w1[0]:+.4f} [{w1[1]:+.4f},{w1[2]:+.4f}] | vs fixed {w2[0]:+.4f} [{w2[1]:+.4f},{w2[2]:+.4f}]"
                )
        P(
            f"  frontier per order F1 {' '.join(f'{x[0].mean():.4f}' for x in R2)} | $ {' '.join(f'{x[1].mean():.4f}' for x in R2)}"
        )
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
