"""E/N/T action data for HotpotQA (cfg3n). Per query (split, pool, id): query, k, F1 of
  E = normalized edge answer (trace_cfg3n{A|B}), N = cloud thinking off, same 'orig' instruction as T
      (runs/pilot7/answers/hotpotqa/<split>/<pool>/orig.jsonl), T = cloud thinking on (cloud_cfg3n.jsonl;
      truncation policy A for labels, B for evaluation), plus cloud input/output tokens of N and T and T's truncation flag."""

from paths import RUNS  # noqa: E402
import json, os, glob, sys

sys.path.insert(0, os.path.dirname(__file__))
from router_data import f1, DATA  # noqa: E402

R = RUNS


def load(split, var="B", with_text=False):
    rows = []
    for d in sorted(glob.glob(f"{R}/pilot6/trace_cfg3n{var}/hotpotqa/{split}/pool_*")):
        pool = d.split("/")[-1]
        nf = f"{R}/pilot7/answers/hotpotqa/{split}/{pool}/orig.jsonl"
        if not os.path.exists(nf):
            continue
        refs = {
            json.loads(l)["id"]: json.loads(l)["answers"]
            for l in open(f"{DATA}/hotpotqa/{split}/{pool}/answers/answers.jsonl")
        }
        N = {json.loads(l)["id"]: json.loads(l) for l in open(nf)}
        T = {json.loads(l)["id"]: json.loads(l) for l in open(f"{d}/cloud_cfg3n.jsonl")}
        ch = {json.loads(l)["node_id"]: json.loads(l)["text"] for l in open(f"{d}/chunks.jsonl")} if with_text else {}
        for l in open(f"{d}/trace.jsonl"):
            t = json.loads(l)
            i = t["id"]
            if i not in N or i not in T:
                continue
            rows.append(
                {
                    "split": split,
                    "pool": pool,
                    "id": i,
                    "query": t["query"],
                    "k": t["k"],
                    "ecrag_edge": t["ecrag_route_edge"],
                    "fE": f1(t["edge_answer"], refs[i]),
                    "fN": N[i]["f1_cloud"],
                    "fT": f1(T[i]["cloud_answer"], refs[i]),
                    "N_in": N[i]["cloud_in_tokens"],
                    "N_out": N[i]["cloud_out_tokens"],
                    "T_in": T[i]["cloud_in_tokens"],
                    "T_out": T[i]["cloud_out_tokens"],
                    "T_trunc": bool(T[i].get("truncated")),
                    "edge_out": t.get("edge_out_tokens", 0),
                    "chunks": [ch[c] for c in t["topk_ids"][: t["k"]]] if with_text else None,
                }
            )
    return rows
