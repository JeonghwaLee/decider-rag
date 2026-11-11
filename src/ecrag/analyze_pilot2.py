"""Analyze pilot 2 (prompt variants on train pools). For each dataset x variant:
- edge/cloud mean F1 and mean GLM judge score
- abstention rate ("Unanswerable" reply), over-abstention on questions with no Unanswerable reference, recall on
  all-Unanswerable questions (Qasper)
- metric agreement: Spearman(F1, judge) per answer and per query dq; mean |F1 dq| among judge ties
- routing headroom under the judge: share cloud-only-correct / edge-only-correct, oracle mean
- engine check (orig only): share of vLLM edge answers identical to the stage-1 HF trace answers
Selection rule (fixed before looking): one prompt for both datasets, highest mean judge score over edge and cloud
(averaged over datasets), subject to over-abstention <= 5% on answerable questions in each dataset.
"""

from paths import RUNS  # noqa: E402
import json, os, sys, glob
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from router_data import DATA, TRACE  # noqa: E402

A = f"{RUNS}/pilot2/answers"
S = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
says_un = lambda t: t.strip().lower().rstrip(".").startswith("unanswerable")

res = {}
for ds in ["hotpotqa", "qasper"]:
    pools = sorted(glob.glob(f"{A}/{ds}/*/*"))
    for v in ["orig", "orig_abst", "min", "min_abst", "typed", "extract", "orig_think", "typed_think"]:
        R = []
        for pd in pools:
            p = os.path.relpath(pd, A)
            if not os.path.exists(f"{pd}/{v}.jsonl"):
                continue
            refs = {json.loads(l)["id"]: json.loads(l)["answers"] for l in open(f"{DATA}/{p}/answers/answers.jsonl")}
            tr = {json.loads(l)["id"]: json.loads(l)["edge_answer"] for l in open(f"{TRACE}/{p}/trace.jsonl")}
            for l in open(f"{pd}/{v}.jsonl"):
                x = json.loads(l)
                if x.get("j_edge") is None or x.get("j_cloud") is None:
                    continue
                u = [r.strip().lower() == "unanswerable" for r in refs[x["id"]]]
                x["unref"] = "all" if all(u) else ("mixed" if any(u) else "none")
                x["trace_edge"] = tr[x["id"]]
                R.append(x)
        if not R:
            continue
        fe = np.array([x["f1_edge"] for x in R])
        fc = np.array([x["f1_cloud"] for x in R])
        je = np.array([S[x["j_edge"]] for x in R])
        jc = np.array([S[x["j_cloud"]] for x in R])
        none = np.array([x["unref"] == "none" for x in R])
        allu = np.array([x["unref"] == "all" for x in R])
        ae = np.array([says_un(x["edge_answer"]) for x in R])
        ac = np.array([says_un(x["cloud_answer"]) for x in R])
        tie = je == jc
        o = {
            "n": len(R),
            "f1_edge": fe.mean(),
            "f1_cloud": fc.mean(),
            "j_edge": je.mean(),
            "j_cloud": jc.mean(),
            "j_avg": (je.mean() + jc.mean()) / 2,
            "j_oracle": np.maximum(je, jc).mean(),
            "cloud_only": float(((jc == 1) & (je < 1)).mean()),
            "edge_only": float(((je == 1) & (jc < 1)).mean()),
            "abst_edge": ae.mean(),
            "abst_cloud": ac.mean(),
            "overabst_edge": ae[none].mean() if none.any() else 0.0,
            "overabst_cloud": ac[none].mean() if none.any() else 0.0,
            "n_all_unans": int(allu.sum()),
            "recall_unans_edge": ae[allu].mean() if allu.any() else None,
            "recall_unans_cloud": ac[allu].mean() if allu.any() else None,
            "rho_answer": spearmanr(np.r_[fe, fc], np.r_[je, jc])[0],
            "rho_dq": spearmanr(fc - fe, jc - je)[0],
            "tie_rate": tie.mean(),
            "absF1dq_given_tie": np.abs(fc - fe)[tie].mean(),
            "edge_words_med": float(np.median([len(x["edge_answer"].split()) for x in R])),
            "cloud_words_med": float(np.median([len(x["cloud_answer"].split()) for x in R])),
        }
        if v == "orig":
            o["engine_same_as_trace"] = float(np.mean([x["edge_answer"].strip() == x["trace_edge"].strip() for x in R]))
        res[f"{ds}/{v}"] = o

for k, o in res.items():
    print(f"== {k} n={o['n']}")
    print(
        f"  F1 e/c {o['f1_edge']:.3f}/{o['f1_cloud']:.3f} | judge e/c {o['j_edge']:.3f}/{o['j_cloud']:.3f} avg {o['j_avg']:.3f} oracle {o['j_oracle']:.3f}"
        f" | cloud-only {o['cloud_only']:.3f} edge-only {o['edge_only']:.3f}"
    )
    print(
        f"  abst e/c {o['abst_edge']:.3f}/{o['abst_cloud']:.3f} overabst e/c {o['overabst_edge']:.3f}/{o['overabst_cloud']:.3f}"
        f" | all-unans n={o['n_all_unans']} recall e/c {o['recall_unans_edge']}/{o['recall_unans_cloud']}"
    )
    print(
        f"  rho answer {o['rho_answer']:.3f} rho dq {o['rho_dq']:.3f} tie {o['tie_rate']:.3f} |F1dq| at tie {o['absF1dq_given_tie']:.3f}"
        f" | words med e/c {o['edge_words_med']}/{o['cloud_words_med']}"
        + (f" | engine same {o['engine_same_as_trace']:.3f}" if "engine_same_as_trace" in o else "")
    )
# selection
vs = sorted({k.split("/")[1] for k in res})
ok = {
    v: all(
        res.get(f"{ds}/{v}", {}).get("overabst_edge", 1) <= 0.05
        and res.get(f"{ds}/{v}", {}).get("overabst_cloud", 1) <= 0.05
        for ds in ["hotpotqa", "qasper"]
    )
    for v in vs
}
score = {v: np.mean([res[f"{ds}/{v}"]["j_avg"] for ds in ["hotpotqa", "qasper"] if f"{ds}/{v}" in res]) for v in vs}
print("eligible:", ok, "score:", {v: round(s, 4) for v, s in score.items()})
el = [v for v in vs if ok[v]]
print("SELECTED:", max(el, key=score.get) if el else None)
json.dump(
    {"res": res, "score": score, "eligible": ok},
    open(f"{RUNS}/pilot2/pilot3_prompt.json", "w"),
    indent=1,
    default=float,
)
