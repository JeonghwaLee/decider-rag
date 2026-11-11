"""E2 judge scores (auxiliary; labels from judge_final.py). Per policy: mean token F1 and mean judge score
(CORRECT 1 / PARTIAL .5 / INCORRECT 0) for each judge; paired bootstrap (bootlib.boot) ref - policy on the same questions. Also
the agreement of each judge with F1>=0.5 and Cohen's kappa between judges (binary CORRECT vs not) on the E2 rows.
Usage: analyze_judge_e2.py --name musique_test --ref joint_vllm --judges glm53 dsv4 llama33 --out f.txt"""

from paths import RUNS  # noqa: E402
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bootlib import boot  # noqa: E402

R = RUNS
SC = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
E2 = {"hotpot_test6": f"{R}/e2e5/hotpot_test6", "musique_test": f"{R}/e2e5/musique_test"}


def kappa(a, b):
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    F = {}
    for fn in glob.glob(f"{E2[a.name]}/*_r0.jsonl"):
        b = os.path.basename(fn)
        for l in open(fn):
            x = json.loads(l)
            F[(b, x["id"])] = (x["policy"], x["f1"])
    J = {}
    for jn in a.judges:
        for l in open(f"{R}/judge_final/e2_{a.name}_{jn}.jsonl"):
            x = json.loads(l)
            J.setdefault((x["file"], x["id"]), {})[jn] = SC.get(x["label"], np.nan)
    pols = sorted({v[0] for v in F.values()})
    M = {}
    for (b, i), (p, f) in F.items():
        q = (b.split("_" + p + "_r0")[0], i)  # question key (split_pool, id) shared by all policies
        M.setdefault(p, {})[q] = [f] + [J[(b, i)].get(jn, np.nan) for jn in a.judges]
    qs = sorted(set.intersection(*[set(M[p]) for p in pols]))
    P(f"{a.name}: {len(qs)} common questions, judges {a.judges} (auxiliary; F1 is primary)")
    A = {p: np.array([M[p][q] for q in qs], float) for p in pols}
    P(f"  {'policy':16s} {'F1':>7s} " + " ".join(f"{jn:>8s}" for jn in a.judges) + "   unlabeled")
    for p in sorted(pols, key=lambda p: -A[p][:, 0].mean()):
        P(
            f"  {p:16s} {A[p][:, 0].mean():7.4f} "
            + " ".join(f"{np.nanmean(A[p][:, j + 1]):8.4f}" for j in range(len(a.judges)))
            + f"   {int(np.isnan(A[p][:, 1:]).sum())}"
        )
    P(f"  paired bootstrap {a.ref} - policy (mean [95% CI]):")
    for p in pols:
        if p == a.ref:
            continue
        cells = []
        for j, nm in enumerate(["F1"] + a.judges):
            d = A[a.ref][:, j] - A[p][:, j]
            d = d[~np.isnan(d)]
            w = boot(d)
            cells.append(f"{nm} {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}]")
        P(f"    vs {p:16s} " + " | ".join(cells))
    allr = np.vstack([A[p] for p in pols])
    f1b = (allr[:, 0] >= 0.5).astype(float)
    for j, jn in enumerate(a.judges):
        c = allr[:, j + 1]
        ok = ~np.isnan(c)
        jb = (c[ok] >= 1.0).astype(float)
        P(f"  {jn}: agreement with F1>=0.5 {np.mean(jb == f1b[ok]):.3f} | kappa {kappa(jb, f1b[ok]):.3f}")
    for x in range(len(a.judges)):
        for y in range(x + 1, len(a.judges)):
            cx, cy = allr[:, x + 1], allr[:, y + 1]
            ok = ~np.isnan(cx) & ~np.isnan(cy)
            P(
                f"  kappa(CORRECT) {a.judges[x]} vs {a.judges[y]}: {kappa((cx[ok] >= 1).astype(float), (cy[ok] >= 1).astype(float)):.3f}"
            )
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
