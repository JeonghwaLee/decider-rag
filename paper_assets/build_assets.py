"""Rebuild v5 assets from the isolated, hashed snapshot. No experiment calls."""

from pathlib import Path
import os, json, re, hashlib

ROOT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(ROOT / "tools/mpl-cache")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DATA = ROOT / "data/automix"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
    }
)
COL = {"ours": "#176b58", "am": "#bc6330", "low": "#b88a23", "shared": "#5274a3", "llm": "#8a689d"}
SPECS = {
    "H": (["hotpot_test6"], "big2e4_vllm", "automix", 1600, 0.722),
    "M": (["musique_test_a", "musique_test_b"], "joint_vllm", "automix", 400, 23.871),
    "H4": (["hotpot_k4"], "big2e4_vllm", "automix_k4", 1600, 0.722),
    "M1": (["musique_k1"], "joint_vllm", "automix_k1", 400, 23.871),
}
RAW = {}
OUT = {}
CHECKS = []


def check(name, ok):
    CHECKS.append({"name": name, "pass": bool(ok)})
    if not ok:
        raise AssertionError(name)


def boot(a):
    a = np.asarray(a)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(20):
        vals.append(a[rng.integers(0, len(a), (500, len(a)))].mean(1))
    return [float(a.mean()), *map(float, np.quantile(np.concatenate(vals), [0.025, 0.975]))]


for ds, (dirs, ref, am, n, D) in SPECS.items():
    rows = {}
    for d in dirs:
        for p in sorted((DATA / d).glob("*_r0.jsonl")):
            pre = p.stem.split("_")
            split, pool = pre[0], "pool_" + pre[2]
            for l in p.read_text().splitlines():
                x = json.loads(l)
                key = (split, pool, x["id"])
                pol = x["policy"]
                check(f"{ds}/{pol}/{key}: unique", key not in rows.setdefault(pol, {}))
                rows[pol][key] = x
    keys = sorted(set.intersection(*(set(r) for r in rows.values())))
    check(ds + ": complete paired sample", len(keys) == n and all(len(v) == n for v in rows.values()))
    R = {p: [v[k] for k in keys] for p, v in rows.items()}
    RAW[ds] = R
    A = {p: {k: np.array([r[k] for r in rr], float) for k in ["f1", "wall", "total_cost"]} for p, rr in R.items()}
    stats = {}
    for p, rr in R.items():
        v = A[p]
        check(ds + "/" + p + ": finite", all(np.isfinite(a).all() for a in v.values()))
        stats[p] = {
            "n": n,
            "f1": float(v["f1"].mean()),
            "cost": float(v["total_cost"].mean()),
            "wall": float(v["wall"].mean()),
            "p95": float(np.quantile(v["wall"], 0.95)),
            "dq": float(np.mean(v["f1"] * (v["wall"] <= D))),
            "branch": {b: sum(r["branch"] == b for r in rr) for b in "ENT"},
            "stages": {k: sum(r["time"].get(k, 0) for r in rr) / n for k in sorted({k for r in rr for k in r["time"]})},
        }
        if p.startswith("automix"):
            stats[p].update(
                unparsed=sum(r["ver_n_valid"] == 0 for r in rr) / n, ver_cost=sum(r["ver_cost"] for r in rr) / n
            )
    diffs = {
        p: {
            "f1": boot(A[ref]["f1"] - v["f1"]),
            "cost": boot(A[ref]["total_cost"] - v["total_cost"]),
            "wall": boot(A[ref]["wall"] - v["wall"]),
            "dq": boot(A[ref]["f1"] * (A[ref]["wall"] <= D) - v["f1"] * (v["wall"] <= D)),
        }
        for p, v in A.items()
        if p != ref
    }
    OUT[ds] = {"ref": ref, "am": am, "D": D, "stats": stats, "diffs": diffs}
    # Reconcile independent raw-log recomputation with reported summary, not rounded FINAL_SUMMARY.
    summary = {"H": "hotpot_live", "M": "musique_live", "H4": "hotpot_k4", "M1": "musique_k1"}[ds]
    txt = (DATA / (summary + "_summary.txt")).read_text()
    for p, s in stats.items():
        m = re.search(r"^  " + re.escape(p) + r"\s+E/N/T .*?F1 ([0-9.]+)", txt, re.M)
        check(ds + "/" + p + ": reported F1", bool(m) and abs(s["f1"] - float(m[1])) < 0.000051)
        if p != ref:
            m = re.search(
                re.escape(ref) + r" - " + re.escape(p) + r"\s+F1 ([+\-0-9.]+) \[([+\-0-9.]+),([+\-0-9.]+)\]", txt
            )
            check(
                ds + "/" + p + ": paired CI",
                bool(m) and max(abs(a - float(b)) for a, b in zip(diffs[p]["f1"], m.groups())) < 0.000051,
            )
(ROOT / "data/metrics_v5.json").write_text(json.dumps(OUT, indent=2) + "\n")
# Tables
names = {
    "big2e4_vllm": r"\textbf{Decider (Ours)}",
    "joint_vllm": r"\textbf{Decider (Ours)}",
    "automix": r"AutoMix-style ($s=8$)",
    "automix_k4": r"AutoMix-style ($s=4$)",
    "automix_k1": r"AutoMix-style ($s=1$)",
    "qwen38_zs": "Qwen3.8 zero-shot",
    "qwen38_shared": "Qwen3.8 shared-prefix",
    "qwen38_think": "Qwen3.8 thinking",
}


def ci(v):
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"


l = [
    r"\begin{tabular}{@{}lrrrrrrl@{}}",
    r"\toprule",
    r"Policy & F1$\uparrow$ & Cloud \$/1k$\downarrow$ & Mean (s)$\downarrow$ & p95 (s)$\downarrow$ & DQ-F1$\uparrow$ & E/C/R & Ours$-$X F1 [95\% CI]\\",
    r"\midrule",
]
for ds, title in [("H", "HotpotQA, 1,600 requests; D=0.722 s"), ("M", "MuSiQue, 400 requests; D=23.871 s")]:
    o = OUT[ds]
    l.append(r"\multicolumn{8}{@{}l}{\textit{" + title + r"}}\\")
    PS = [o["ref"], o["am"], "qwen38_shared", "qwen38_zs" if ds == "H" else "qwen38_think"]
    FM = {
        "f1": ("{:.3f}", max),
        "cost": ("{:.3f}", min),
        "wall": ("{:.2f}", min),
        "p95": ("{:.2f}", min),
        "dq": ("{:.3f}", max),
    }  # bold = best displayed value per workload
    BEST = {k: f(float(fmt.format(o["stats"][q][k])) for q in PS) for k, (fmt, f) in FM.items()}
    bb = lambda q, k: (lambda t: r"\textbf{" + t + "}" if float(t) == BEST[k] else t)(FM[k][0].format(o["stats"][q][k]))
    for p in PS:
        s = o["stats"][p]
        bc = "/".join(str(s["branch"][k]) for k in "ENT")
        dv = "---" if p == o["ref"] else ci(o["diffs"][p]["f1"])
        l.append(
            f"{names[p]} & {bb(p,'f1')} & {bb(p,'cost')} & {bb(p,'wall')} & {bb(p,'p95')} & {bb(p,'dq')} & {bc} & {dv}"
            + r"\\"
        )
    l.append(r"\midrule")
l[-1] = r"\bottomrule"
l += [r"\end{tabular}"]
(TAB / "pipeline.tex").write_text("\n".join(l) + "\n")
l = [
    r"\begin{tabular}{@{}llrrrrrl@{}}",
    r"\toprule",
    r"Workload & Policy & F1$\uparrow$ & Cloud \$/1k$\downarrow$ & Mean (s)$\downarrow$ & Verify (ms)$\downarrow$ & DQ-F1$\uparrow$ & Ours$-$X F1 [95\% CI]\\",
    r"\midrule",
]
for ds, title in [("H4", "HotpotQA"), ("M1", "MuSiQue")]:
    o = OUT[ds]
    PS = [o["ref"], o["am"]]
    FM = {
        "f1": ("{:.3f}", max),
        "cost": ("{:.3f}", min),
        "wall": ("{:.2f}", min),
        "dq": ("{:.3f}", max),
    }  # bold = better displayed value per workload
    BEST = {k: f(float(fmt.format(o["stats"][q][k])) for q in PS) for k, (fmt, f) in FM.items()}
    bb = lambda q, k: (lambda t: r"\textbf{" + t + "}" if float(t) == BEST[k] else t)(FM[k][0].format(o["stats"][q][k]))
    for p in PS:
        s = o["stats"][p]
        v = "---" if p == o["ref"] else f"{s['stages']['verify']*1000:.0f}"
        d = "---" if p == o["ref"] else ci(o["diffs"][p]["f1"])
        l.append(
            f"{title} & {names[p]} & {bb(p,'f1')} & {bb(p,'cost')} & {bb(p,'wall')} & {v} & {bb(p,'dq')} & {d}" + r"\\"
        )
l += [r"\bottomrule", r"\end{tabular}"]
(TAB / "samples.tex").write_text("\n".join(l) + "\n")
# Figure 1: measured points (not an interpolated frontier) + deadline curves.
fig, ax = plt.subplots(2, 2, figsize=(7.15, 5.2))
fig.subplots_adjust(left=0.085, right=0.98, bottom=0.18, top=0.94, wspace=0.31, hspace=0.42)
for i, (ds, title) in enumerate([("H", "HotpotQA"), ("M", "MuSiQue")]):
    o = OUT[ds]
    ref = o["ref"]
    P = [ref, "automix", "qwen38_shared", "qwen38_zs" if ds == "H" else "qwen38_think"]
    labels = ["Decider (Ours)", "AutoMix-style, s=8", "LLM shared-prefix", "LLM zero-shot / thinking"]
    styles = [(COL["ours"], "*"), (COL["am"], "s"), (COL["shared"], "o"), (COL["llm"], "^")]
    for p, lab, (col, mk) in zip(P, labels, styles):
        s = o["stats"][p]
        ax[i, 0].scatter(s["cost"], s["f1"], s=90 if p == ref else 36, c=col, marker=mk, zorder=5, label=lab)
        rr = RAW[ds][p]
        w = np.array([r["wall"] for r in rr])
        f = np.array([r["f1"] for r in rr])
        xx = np.linspace(0, 4 if ds == "H" else 2.5, 220)
        yy = [np.mean(f * (w <= t * o["D"])) for t in xx]
        ax[i, 1].plot(xx, yy, color=col, lw=2 if p == ref else 1.5, ls="-" if p == ref else "--", label=lab)
    ax[i, 0].set(
        title=f"({chr(97+i*2)}) {title}: quality and cost",
        xlabel="Cloud-token cost ($ / 1k requests)",
        ylabel="Answer F1",
    )
    ax[i, 0].set_ylim((0.59, 0.65) if ds == "H" else (0.42, 0.515))
    ax[i, 0].set_xlim((0.13, 0.85) if ds == "H" else (1.6, 5.9))
    ax[i, 1].set(
        title=f"({chr(98+i*2)}) {title}: deadline sensitivity",
        xlabel=f'Deadline / D  (D = {o["D"]:.3f} s)',
        ylabel="Deadline-qualified F1",
        ylim=(0, 0.69 if ds == "H" else 0.55),
    )
    ax[i, 1].axvline(1, color="#777777", lw=0.9, ls=":")
    ax[i, 1].text(1.03, 0.08, "D", color="#666666")
    from matplotlib.lines import Line2D  # legend shows both the point marker (a,c) and the line style (b,d)

    h = [
        Line2D(
            [0],
            [0],
            color=col,
            marker=mk,
            ls="-" if k == 0 else "--",
            lw=2 if k == 0 else 1.5,
            markersize=9 if k == 0 else 6,
        )
        for k, (col, mk) in enumerate(styles)
    ]
    lb = labels
fig.legend(
    h, lb, ncol=2, loc="lower center", bbox_to_anchor=(0.53, 0.005), frameon=False, columnspacing=1.8, handlelength=3.2
)
fig.savefig(FIG / "fig01_tradeoffs.pdf")
fig.savefig(FIG / "fig01_tradeoffs.png", dpi=170)
plt.close(fig)
# Figure 2: original vector pictograms, shared by standalone and full asset builds.
from draw_figure2 import render as render_figure2

for name, ok in render_figure2(FIG):
    check(name, ok)
# Figure 3: measured stage contributions across all requests (conditional stage means are not summed).
fig, axs = plt.subplots(1, 2, figsize=(7.15, 2.7))
fig.subplots_adjust(left=0.17, right=0.95, bottom=0.30, top=0.86, wspace=0.75)
cats = [
    ("Local retrieval", ["retrieve", "rerank", "prune"], "#9ba8b5"),
    ("Candidate / final answer", ["edge", "cloud", "cloud_T", "cloud_retry", "edge_fallback"], "#5274a3"),
    ("LLM selection", ["selector"], "#8a689d"),
    ("Self-verification", ["verify"], "#bc6330"),
    ("Decider", ["decider"], "#176b58"),
]
for ax, (ds, title) in zip(axs, [("H", "HotpotQA"), ("M", "MuSiQue")]):
    o = OUT[ds]
    P = [o["ref"], "automix", "qwen38_shared", "qwen38_zs" if ds == "H" else "qwen38_think"]
    left = np.zeros(4)
    for label, ks, col in cats:
        vals = np.array([sum(o["stats"][p]["stages"].get(k, 0) for k in ks) for p in P])
        ax.barh(np.arange(4), vals, left=left, color=col, label=label, height=0.6)
        left += vals
    ax.set_yticks(
        np.arange(4), ["Decider (Ours)", "Cascade s=8", "LLM shared", "LLM zs" if ds == "H" else "LLM thinking"]
    )
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Time (s / request); lower is better")
    ax.grid(axis="y", visible=False)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.53, 0), frameon=False, fontsize=7.5)
fig.savefig(FIG / "fig03_stages.pdf")
fig.savefig(FIG / "fig03_stages.png", dpi=170)
plt.close(fig)
# Compact legacy baseline comparison: same earlier cohort as judges/load, not fused into new run.
s = (ROOT / "tables/legacy/e2.tex").read_text()
lines = []
for l in s.splitlines():
    if l.startswith(("TF-IDF", "EC-RAG rule", "All-N")):
        a = l.rstrip("\\").split(" & ")
        lines.append((a[0], a[2], a[3], a[4]))
t = [
    r"\begin{tabular}{@{}llrl@{}}",
    r"\toprule",
    r"Workload & Baseline & F1$\uparrow$ & Ours$-$X [95\% CI]\\",
    r"\midrule",
]
for i, (name, f, d, c) in enumerate(lines):
    g = [float(x[1]) for x in (lines[:3] if i < 3 else lines[3:])]  # bold = highest baseline F1 within each workload
    t.append(
        ("HotpotQA" if i < 3 else "MuSiQue")
        + " & "
        + name.replace("All-N", "All-C")
        + " & "
        + (r"\textbf{" + f + "}" if float(f) == max(g) else f)
        + " & "
        + d
        + r"\\"
    )
t += [r"\bottomrule", r"\end{tabular}"]
(TAB / "legacy_baselines.tex").write_text("\n".join(t) + "\n")
# Validate frozen bytes after read and export checks.
m = json.loads((ROOT / "data/manifest.json").read_text())
for p, h in m.items():
    check("frozen " + p, hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h)
(ROOT / "data/asset_checks.json").write_text(
    json.dumps({"passed": sum(c["pass"] for c in CHECKS), "total": len(CHECKS), "checks": CHECKS}, indent=2) + "\n"
)
print("Built 3 vector figures, 3 tables, metrics; checks", len(CHECKS), "passed")
