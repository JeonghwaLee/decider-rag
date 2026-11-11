"""Generic E/N/T action data for the pilot-8/9 roots (stage1_trace.py traces + regen_answers.py answers), same definitions as
ent_data.py: E = edge 'extract' answer + normalize_edge.py; N = cloud 'orig' thinking off (max 256); T = cloud 'orig_think'
thinking on (max 6144). T truncated = finish 'length' or no '</think>' in the raw tail; var 'A' scores a truncated T as 0,
var 'B' falls back to the E answer. T fields are None when orig_think.jsonl is absent (e.g. train2).
load(root, ds, split, var, with_text) -> rows with the keys of ent_data.load (+ 'ds', 'complexity')."""

from paths import RUNS  # noqa: E402
import json, os, glob, sys

sys.path.insert(0, os.path.dirname(__file__))
from router_data import f1, DATA  # noqa: E402
from normalize_edge import normalize  # noqa: E402

R = RUNS
ROOT = {("hotpotqa", "test4"): f"{R}/pilot8", ("hotpotqa", "train2"): f"{R}/pilot8"}


def rd(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else None


def load(root, ds, split, var="B", with_text=False):
    rows = []
    for d in sorted(glob.glob(f"{root}/trace/{ds}/{split}/pool_*")):
        pool = d.split("/")[-1]
        A = f"{root}/answers/{ds}/{split}/{pool}"
        E, N, T = rd(f"{A}/extract.jsonl"), rd(f"{A}/orig.jsonl"), rd(f"{A}/orig_think.jsonl")
        if E is None or N is None or not os.path.exists(f"{d}/DONE"):
            continue
        refs = {
            json.loads(l)["id"]: json.loads(l)["answers"]
            for l in open(f"{DATA}/{ds}/{split}/{pool}/answers/answers.jsonl")
        }
        ch = {json.loads(l)["node_id"]: json.loads(l)["text"] for l in open(f"{d}/chunks.jsonl")} if with_text else {}
        for l in open(f"{d}/trace.jsonl"):
            t = json.loads(l)
            i = t["id"]
            if i not in E or i not in N or (T is not None and i not in T):
                continue
            ea = normalize(t["query"], E[i]["edge_answer"])[0]
            fE = f1(ea, refs[i])
            r = {
                "ds": ds,
                "split": split,
                "pool": pool,
                "id": i,
                "query": t["query"],
                "k": t["k"],
                "ecrag_edge": t["ecrag_route_edge"],
                "complexity": t["complexity"],
                "fE": fE,
                "fN": f1(N[i]["cloud_answer"], refs[i]),
                "N_in": N[i]["cloud_in_tokens"],
                "N_out": N[i]["cloud_out_tokens"],
                "edge_out": E[i]["edge_out_tokens"],
                "fT": None,
                "T_in": None,
                "T_out": None,
                "T_trunc": None,
                "chunks": [ch[c] for c in t["topk_ids"][: t["k"]]] if with_text else None,
            }
            if T is not None:
                x = T[i]
                tr = x["cloud_finish"] == "length" or "</think>" not in (x.get("cloud_raw") or "")
                r.update(
                    {
                        "T_in": x["cloud_in_tokens"],
                        "T_out": x["cloud_out_tokens"],
                        "T_trunc": tr,
                        "fT": (0.0 if var == "A" else fE) if tr else f1(x["cloud_answer"], refs[i]),
                    }
                )
            rows.append(r)
    return rows
