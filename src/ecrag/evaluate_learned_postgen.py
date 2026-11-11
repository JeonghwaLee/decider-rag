"""Validation-only threshold freezing and paired replay, with candidate/recovery costs.
No generation and no wall-time estimates. Internal E/N/T keys map to paper E/C/R.
"""

import argparse, json, hashlib
from pathlib import Path
import numpy as np
from analyze_en import load
from analyze_ent2 import arrays
from bootlib import boot


def pair_arrays(rows, ds, post):
    a = arrays(rows, "E" if ds == "hotpot" else "N")
    e, c, r, cc, cr = a[:5]
    if ds == "hotpot":
        return e, c, np.zeros(len(rows)), cc
    if post:
        # Existing C candidate is always paid; never buy a duplicate C recovery.
        cr = np.array([(0.27 * x["T_in"] + 1.1 * x["T_out"]) / 1e3 for x in rows]) + cc
    return c, r, cc, cr


def scores(path, rows, ds):
    p = json.loads(Path(path).read_text())
    m = {(x["pool"], x["id"]): x for x in p}
    assert len(p) == len(m), "duplicate prediction keys"
    s = np.array([m[(x["pool"], x["id"])]["dEN" if ds == "hotpot" else "dNT"] for x in rows], float)
    assert np.isfinite(s).all()
    return s


def choose(s, g, budget):
    best = None
    for t in np.r_[-np.inf, np.quantile(s, np.linspace(0, 1, 41)), np.inf]:
        f, c = execute(s, g, t)
        if c.mean() <= budget + 1e-12:
            key = (float(f.mean()), -float(c.mean()))
            if best is None or key > best[0]:
                best = (key, float(t), float(np.mean(s > t)))
    return None if best is None else {"threshold": best[1], "fraction": best[2], "f1": best[0][0], "cost": -best[0][1]}


def execute(s, g, t):
    f0, f1, c0, c1 = g
    up = s > t
    return np.where(up, f1, f0), np.where(up, c1, c0)


def online(s, g, policy, budget, seed):
    # Same past-only 1D specialization for both E/C or C/R scorers. Infeasible
    # frontier levels are excluded: a compulsory C candidate cannot become E.
    fr = policy["frontier"]
    levels = np.array([v["multiple"] for v in fr])
    u = 0.0
    hist = []
    cost = []
    f = np.zeros(len(s))
    c = np.zeros(len(s))
    f0, f1, c0, c1 = g
    for i in np.random.default_rng(seed).permutation(len(s)):
        if len(hist) < 100:
            t = policy["primary"]["threshold"]
        else:
            q = fr[int(np.argmin(abs(levels - np.exp(u))))]["fraction"]
            t = -np.inf if q >= 1 else (np.inf if q <= 0 else float(np.quantile(hist[-100:], 1 - q)))
        up = s[i] > t
        f[i] = f1[i] if up else f0[i]
        c[i] = c1[i] if up else c0[i]
        hist.append(s[i])
        cost.append(c[i])
        if len(cost) >= 20:
            u = float(
                np.clip(u + 0.05 * (budget - np.mean(cost[-100:])) / budget, np.log(levels.min()), np.log(levels.max()))
            )
    return f, c


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["calibrate", "test"], required=True)
    p.add_argument("--ds", choices=["hotpot", "musique"], required=True)
    p.add_argument("--val")
    p.add_argument("--test")
    p.add_argument("--pre-pred", required=True)
    p.add_argument("--post-pred", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fn = out / "PRIMARY.json"
    if a.phase == "calibrate":
        if fn.exists():
            raise RuntimeError("PRIMARY already frozen; use a new run directory")
        rows, _ = load(a.val)
        raw = arrays(rows, "E" if a.ds == "hotpot" else "N")
        budget = float(np.where(raw[5], raw[4], 0).mean())
        report = {
            "ds": a.ds,
            "val": a.val,
            "budget": budget,
            "prediction_hashes": {},
            "policies": {},
            "scope": "Validation-only; same backbone/targets; fixed E/C or C/R profiles.",
        }
        for name, path in [("pre", a.pre_pred), ("post", a.post_pred)]:
            s = scores(path, rows, a.ds)
            g = pair_arrays(rows, a.ds, name == "post")
            primary = choose(s, g, budget)
            if primary is None:
                raise RuntimeError(name + " has no feasible x1 policy; report infeasibility, do not silently choose E")
            fr = []
            for m in np.round(np.arange(0.1, 3.01, 0.05), 2):
                v = choose(s, g, budget * m)
                if v is not None:
                    fr.append({"multiple": float(m), **v})
            report["policies"][name] = {"primary": primary, "frontier": fr}
            report["prediction_hashes"][name] = sha(path)
        fn.write_text(json.dumps(report, indent=2) + "\n")
        print("Frozen", fn)
        return
    frozen = json.loads(fn.read_text())
    assert frozen["ds"] == a.ds
    rows, _ = load(a.test)
    results = {}
    vectors = {}
    for name, path in [("pre", a.pre_pred), ("post", a.post_pred)]:
        s = scores(path, rows, a.ds)
        g = pair_arrays(rows, a.ds, name == "post")
        pv = frozen["policies"][name]
        f, c = execute(s, g, pv["primary"]["threshold"])
        runs = [online(s, g, pv, frozen["budget"], 100 + i) for i in range(5)]
        fo = np.mean([r[0] for r in runs], axis=0)
        co = np.mean([r[1] for r in runs], axis=0)
        vectors[name] = {"fixed_f1": f, "fixed_cost": c, "online_f1": fo, "online_cost": co}
        results[name] = {k: float(v.mean()) for k, v in vectors[name].items()}
        results[name]["online_budget_ratio"] = float(co.mean() / frozen["budget"])
    differences = {k: list(map(float, boot(vectors["post"][k] - vectors["pre"][k]))) for k in vectors["pre"]}
    report = {
        "test": a.test,
        "n": len(rows),
        "primary_sha256": sha(fn),
        "budget": frozen["budget"],
        "metrics": results,
        "post_minus_pre": differences,
        "note": "Offline replay on reused questions. Cloud-token cost includes compulsory C candidate and R, with reuse after truncation. Online means past-only budget replay, not live latency. Report realized spend ratios; do not declare exact equal-spend if ratios differ materially.",
    }
    (out / "REPLAY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
