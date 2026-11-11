"""Original vector icon schematic (current Figure 1; legacy asset name fig02); no external image assets."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tools/mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, PathPatch
from matplotlib.path import Path as MPath


def render(out=None):
    out = Path(out or ROOT / "figures")
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.15, 4.85))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 100), ylim=(0, 70))
    ax.axis("off")
    green = "#176b58"
    blue = "#526b91"
    orange = "#ad6136"
    ink = "#263640"
    muted = "#5c6b76"
    checks = []
    texts = []

    def line(xs, ys, c=ink, lw=1, **kw):
        ax.plot(xs, ys, color=c, lw=lw, solid_capstyle="round", solid_joinstyle="round", **kw)

    def rr(x, y, w, h, fc="white", ec=ink, lw=0.9, r=0.6, z=3):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec, lw=lw, zorder=z
        )
        ax.add_patch(p)
        return p

    def arrow(x, y, X, Y, c=muted, dash=False):
        ax.add_patch(
            FancyArrowPatch(
                (x, y),
                (X, Y),
                arrowstyle="-|>",
                mutation_scale=8,
                lw=1,
                color=c,
                linestyle=(0, (3, 2)) if dash else "-",
                zorder=2,
            )
        )

    def icon(kind, x, y, c):
        # Each illustration occupies a 6 x 6 local coordinate square.
        def L(a, b, **kw):
            line([x + v for v in a], [y + v for v in b], c, **kw)

        def R(a, b, w, h, fc="white", lw=0.9):
            return rr(x + a, y + b, w, h, fc, c, lw, 0.35, 5)

        def C(a, b, r, fc="white"):
            ax.add_patch(Circle((x + a, y + b), r, fc=fc, ec=c, lw=0.9, zorder=6))

        def doc(a=1, b=0.5):
            R(a, b, 3.8, 4.7)
            L([a + 0.7, a + 2.9], [b + 3.2, b + 3.2], zorder=6)
            L([a + 0.7, a + 2.9], [b + 2.3, b + 2.3], zorder=6)
            L([a + 0.7, a + 2.0], [b + 1.4, b + 1.4], zorder=6)

        if kind == "evidence":
            R(2, 1.2, 3.5, 4.5, fc="#eef3f5")
            doc(0.6, 0.3)
            R(0, 3.7, 2.6, 2.1)
            ax.text(x + 1.3, y + 4.75, "?", ha="center", va="center", fontsize=8.5, color=c, weight="bold", zorder=7)
        elif kind in ("model", "decider"):
            R(1, 1, 4, 4, fc="#f5faf8" if kind == "decider" else "white")
            for p in (1.8, 3, 4.2):
                L([p, p], [0.35, 1], zorder=6)
                L([p, p], [5, 5.65], zorder=6)
                L([0.35, 1], [p, p], zorder=6)
                L([5, 5.65], [p, p], zorder=6)
            for a, b in [(1.8, 2), (1.8, 4)]:
                for d in (2, 4):
                    L([a, 4.2], [b, d], lw=0.7, zorder=6)
            for a, b in [(1.8, 2), (1.8, 4), (4.2, 2), (4.2, 4)]:
                C(a, b, 0.35, fc=c)
        elif kind == "answer":
            doc(0.6, 0.4)
            ax.add_patch(
                Polygon(
                    [(x + 3.2, y + 0.3), (x + 3.4, y + 1.4), (x + 5.6, y + 3.6), (x + 6.2, y + 3), (x + 4, y + 0.8)],
                    closed=True,
                    fc="#f7e3d3",
                    ec=c,
                    lw=0.9,
                    zorder=7,
                )
            )
        elif kind == "verify":
            verts = [
                (x + 3, y + 0.2),
                (x + 0.7, y + 1.5),
                (x + 0.7, y + 4.6),
                (x + 3, y + 5.7),
                (x + 5.3, y + 4.6),
                (x + 5.3, y + 1.5),
                (x + 3, y + 0.2),
            ]
            ax.add_patch(
                PathPatch(
                    MPath(
                        verts,
                        [
                            MPath.MOVETO,
                            MPath.CURVE3,
                            MPath.CURVE3,
                            MPath.LINETO,
                            MPath.LINETO,
                            MPath.CURVE3,
                            MPath.CURVE3,
                        ],
                    ),
                    fc="#fff8f2",
                    ec=c,
                    lw=1.1,
                    zorder=5,
                )
            )
            L([1.6, 2.6, 4.5], [3, 2.1, 4.1], lw=1.5, zorder=6)
        elif kind in ("plans", "escalate"):
            ends = (1, 3, 5) if kind == "plans" else (1.3, 4.7)
            for e in ends:
                L([1, 2.5, 2.5, 4.5], [3, 3, e, e], lw=0.9, zorder=5)
                C(4.8, e, 0.63)
            C(0.8, 3, 0.63, fc=c)
        elif kind == "traces":
            R(0.4, 0.7, 4.9, 4.9)
            L([1.3, 1.3], [1.4, 4.8], zorder=6)
            for yy, ww in [(2, 1.3), (3, 2.4), (4, 3.3)]:
                L([1.4, 1.4 + ww], [yy, yy], lw=2, zorder=6)
        elif kind == "learn":
            for a, b, d, e in [
                (1, 1, 3, 2),
                (1, 4.8, 3, 2),
                (1, 1, 3, 4.5),
                (1, 4.8, 3, 4.5),
                (3, 2, 5.4, 3.2),
                (3, 4.5, 5.4, 3.2),
            ]:
                L([a, d], [b, e], lw=0.8, zorder=5)
            for a, b in [(1, 1), (1, 4.8), (3, 2), (3, 4.5), (5.4, 3.2)]:
                C(a, b, 0.58, fc="white")
        elif kind == "calibrate":
            for xx, yy in [(1, 3.9), (3, 2), (5, 4.6)]:
                L([xx, xx], [0.7, 5.6], lw=1, zorder=5)
                R(xx - 0.7, yy - 0.45, 1.4, 0.9, fc="#e5f2ed")

    def card(x, y, w, title, sub, kind, c, fc="white"):
        p = rr(x, y, w, 9.1, fc, "#c3ccd2", 0.75, 0.85)
        rr(x + 0.7, y + 0.9, 7.3, 7.3, fc, fc, 0, 0.8)
        icon(kind, x + 1.25, y + 1.4, c)
        ts = [
            ax.text(x + 9, y + 5.85, title, ha="left", va="center", fontsize=8, weight="bold", color=ink, zorder=7),
            ax.text(
                x + 9, y + 2.85, sub, ha="left", va="center", fontsize=7.1, color=muted, linespacing=1.25, zorder=7
            ),
        ]
        texts.extend((p, t) for t in ts)

    # Three restrained lanes align the comparable stages.
    ax.text(2, 67.8, "ONLINE", fontsize=9, weight="bold", color=ink)
    ax.text(17, 67.8, "Same question and retrieved evidence", fontsize=8.5, color=muted)
    lanes = [
        (49.8, blue, "Prompted LLM selector", "Read evidence, then choose a plan"),
        (34.6, orange, "Self-verification cascade", "Generate a candidate, then check it"),
        (19.4, green, "Decider (Ours)", "Predict execution gains before generation"),
    ]
    for y, c, title, sub in lanes:
        rr(
            0.8,
            y - 1.1,
            98.4,
            13.1,
            {"#526b91": "#f4f6fa", "#ad6136": "#fcf7f2", green: "#f0f7f4"}[c],
            "none",
            0,
            1.3,
            0,
        )
        ax.text(2, y + 10.7, title, fontsize=8.7, weight="bold", color=c)
        ax.text(98, y + 10.7, sub, ha="right", fontsize=7.3, color=muted)
    card(2, 49.8, 26, "Input", "Question + evidence\n+ passage count k", "evidence", blue)
    card(35, 49.8, 29, "LLM judgment", "Select E / C / R", "model", blue)
    card(71, 49.8, 27, "Execute plan", "Selected E / C / R", "plans", blue)
    card(2, 34.6, 26, "Candidate answer", "Generate first", "answer", orange)
    card(35, 34.6, 29, "Self-verification", "Question, evidence\nand candidate answer", "verify", orange)
    card(71, 34.6, 27, "Keep or escalate", "Execute if escalated", "escalate", orange)
    card(2, 19.4, 26, "Input", "Question + evidence\n+ passage count k", "evidence", green)
    card(35, 19.4, 29, "One 4B pass", "Predict gains\nwith learned heads", "decider", green, "#f9fdfb")
    card(71, 19.4, 27, "Calibrated choice", "Select E / C / R,\nthen execute", "plans", green, "#f9fdfb")
    for y, c, _, __ in lanes:
        arrow(28.4, y + 4.55, 34.6, y + 4.55, c)
        arrow(64.4, y + 4.55, 70.6, y + 4.55, c)
    ax.text(2, 14.5, "OFFLINE", fontsize=8.7, weight="bold", color=ink)
    ax.text(
        18, 14.5, "Execution outcomes train the planner; validation calibrates the policy", fontsize=7.4, color=muted
    )
    card(2, 1, 26, "Outcome traces", "Execute E / C / R\non common evidence", "traces", green)
    card(35, 1, 29, "Fit the planner", "Quality + gain targets;\nLoRA + 5-output head", "learn", green)
    card(71, 1, 27, "Calibrate policy", "Validation profiles\nand budget thresholds", "calibrate", green)
    arrow(28.4, 5.55, 34.6, 5.55)
    arrow(64.4, 5.55, 70.6, 5.55)
    # Dashed deployment connectors travel through the whitespace around the heading.
    for xx in (49.5, 84.5):
        line([xx, xx], [10.25, 12.6], green, 0.85, linestyle=(0, (3, 2)), zorder=1)
        arrow(xx, 16.3, xx, 19.1, green, True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for p, t in texts:
        b = p.get_window_extent(renderer)
        a = t.get_window_extent(renderer)
        ok = b.x0 + 1 <= a.x0 and b.x1 - 1 >= a.x1 and b.y0 + 1 <= a.y0 and b.y1 - 1 >= a.y1
        checks.append(("diagram text within box: " + t.get_text().replace("\n", " / "), ok))
        if not ok:
            raise AssertionError(checks[-1][0])
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out / f"fig02_paths.{ext}", dpi=200)
    plt.close(fig)
    return checks


if __name__ == "__main__":
    print("Figure 2 vector export:", len(render()), "text-bound checks passed")
