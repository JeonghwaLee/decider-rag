"""Log-only analyses of the live AutoMix-style cascade runs (no new generation).
(1) DQ-F1 vs deadline over a wide grid (x0.6 .. x6 D; D fixed before test), Decider and cascade and any other policy;
(2) stage times of the cascade (candidate, verification, escalation) and of the Decider;
(3) cascade failure types from its own logged candidate: accepted a candidate with F1 < 0.5 ('wrong accept'), escalated a candidate
    with F1 >= 0.5 whose escalated answer was not better ('discarded good, no gain'); share of escalations from all-unparsable samples;
(4) Decider vs cascade on the same questions: branch agreement, and F1 of each where they disagree (paired bootstrap).
Usage: analyze_automix_live.py --dirs D [D ...] --ds hotpotqa|musique --ref big2e4_vllm --am automix --D 0.722 --out file"""

import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bootlib import boot  # noqa: E402
from router_data import f1, DATA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--am", required=True)
    ap.add_argument("--D", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    R = {}
    for d in a.dirs:
        for fn in glob.glob(f"{d}/*_r0.jsonl"):
            b = os.path.basename(fn)
            spl, _, pool = b.split("_")[:3]
            pool = "pool_" + pool
            for l in open(fn):
                x = json.loads(l)
                R.setdefault(x["policy"], {})[(spl, pool, x["id"])] = x
    keys = sorted(set.intersection(*[set(v) for v in R.values()]))
    pols = sorted(R)
    refs = {}
    for spl, pool, _ in keys:
        if (spl, pool) not in refs:
            refs[(spl, pool)] = {
                json.loads(l)["id"]: json.loads(l)["answers"]
                for l in open(f"{DATA}/{a.ds}/{spl}/{pool}/answers/answers.jsonl")
            }
    P(f"{' + '.join(a.dirs)} | common questions {len(keys)} | policies {pols} | D = {a.D} s")
    A = lambda p, k: np.array([R[p][q][k] for q in keys], float)
    W = {p: A(p, "wall") for p in pols}
    F = {p: A(p, "f1") for p in pols}
    M = [0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    P("(1) DQ-F1 vs deadline: " + " ".join(f"{m}xD={m * a.D:.2f}s" for m in M))
    for p in pols:
        P(
            f"  {p:16s} "
            + " ".join(f"{np.where(W[p] <= m * a.D, F[p], 0).mean():.4f}" for m in M)
            + f" | F1 {F[p].mean():.4f} wall mean {W[p].mean():.3f} p50 {np.median(W[p]):.3f} p95 {np.quantile(W[p], .95):.3f}"
        )
    for p in pols:
        if p != a.ref:
            P(
                f"  {a.ref} - {p}: "
                + " ".join(
                    "{:+.4f}[{:+.4f}]".format(
                        *boot(np.where(W[a.ref] <= m * a.D, F[a.ref], 0) - np.where(W[p] <= m * a.D, F[p], 0))[:2]
                    )
                    for m in M
                )
            )
    P("(2) stage times, mean ms over requests that ran the stage (n):")
    for p in (a.ref, a.am):
        st = {}
        for q in keys:
            for s, t in R[p][q]["time"].items():
                st.setdefault(s, []).append(t)
        P(f"  {p:16s} " + " ".join(f"{s} {1e3 * np.mean(v):.1f}(n={len(v)})" for s, v in st.items()))
    X = [R[a.am][q] for q in keys]
    esc = np.array([x["escalated"] for x in X])
    cf = np.array([f1(x["cand_answer"] or "", refs[q[:2]][q[2]]) for x, q in zip(X, keys)])
    fin = F[a.am]
    nv = np.array([x["ver_n_valid"] for x in X])
    vk = X[0].get("ver_k", 8)
    vo = np.array([x["ver_out"] for x in X])
    vc = np.array([x["ver_cost"] for x in X])
    P(
        f"(3) cascade (k={vk}): escalation rate {esc.mean():.3f} | verification output tokens/request {vo.mean():.0f} | verification cloud $/1k {vc.mean():.4f}"
        f" | all-unparsable {np.mean(nv == 0):.3f} (escalations from it {np.mean(esc & (nv == 0)):.3f})"
    )
    P(
        f"  accepted candidate with F1<0.5 (wrong accept) {np.mean(~esc & (cf < 0.5)):.3f} | accepted with F1>=0.5 {np.mean(~esc & (cf >= 0.5)):.3f}"
        f" | escalated, candidate F1>=0.5 and no gain (discarded good) {np.mean(esc & (cf >= 0.5) & (fin <= cf)):.3f} | escalated with gain {np.mean(esc & (fin > cf)):.3f}"
        f" | escalated, candidate F1<0.5, no gain {np.mean(esc & (cf < 0.5) & (fin <= cf)):.3f}"
    )
    dec_up = np.array([R[a.ref][q]["branch"] != ("E" if a.ds == "hotpotqa" else "N") for q in keys])
    for nm, m in [
        ("both keep", ~dec_up & ~esc),
        ("both escalate", dec_up & esc),
        ("Decider escalates, cascade keeps", dec_up & ~esc),
        ("Decider keeps, cascade escalates", ~dec_up & esc),
    ]:
        if m.sum() > 0:
            w = boot(F[a.ref][m] - F[a.am][m])
            P(
                f"(4) {nm:34s} n {m.sum():5d} ({m.mean():.3f}) | F1 {a.ref} {F[a.ref][m].mean():.4f} vs cascade {F[a.am][m].mean():.4f} | diff {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}]"
                f" | wall {W[a.ref][m].mean():.3f} vs {W[a.am][m].mean():.3f}"
            )
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
