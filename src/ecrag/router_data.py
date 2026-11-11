"""Join stage-1 traces and stage-2 cloud answers into one router table per (dataset, split).

Per query:
- F1_edge, F1_cloud: EC-RAG cal_f1 (max over references).
- dq = F1_cloud - F1_edge.
- Hypothetical cloud cost on Qwen3.5 token counts at DeepSeek-V3 list prices ($0.27/M in, $1.1/M out; EC-RAG utils.calc_cost).
- Router inputs: query, pruned chunk texts, k, complexity, rerank scores, token counts, EC-RAG route.
"""

from paths import DATA, ECRAG_SRC, RUNS  # noqa: E402
import json, os, sys

sys.path.insert(0, ECRAG_SRC)
from cal_f1 import normalize_answer, f1_score  # noqa: E402

DATA = DATA
TRACE = os.environ.get(
    "ECRAG_TRACE", f"{RUNS}/pilot1/trace"
)  # override for other trace roots (e.g. pilot3 paper-scoped)


def f1(pred, refs):
    p = normalize_answer(pred.replace("**", "").replace("\n", " "))
    return max(f1_score(p, normalize_answer(r)) for r in refs)


def load(ds, split, with_text=True, cloud_tag=None):
    cloud_tag = os.environ.get("ECRAG_CLOUD_TAG", "") if cloud_tag is None else cloud_tag
    rows = []
    base = f"{TRACE}/{ds}/{split}"
    for pool in sorted(os.listdir(base)):
        d = f"{base}/{pool}"
        if not os.path.exists(f"{d}/DONE") or not os.path.exists(f"{d}/cloud{cloud_tag}.jsonl"):
            continue
        refs = {
            json.loads(l)["id"]: json.loads(l)["answers"]
            for l in open(f"{DATA}/{ds}/{split}/{pool}/answers/answers.jsonl")
        }
        cloud = {}
        for l in open(f"{d}/cloud{cloud_tag}.jsonl"):
            c = json.loads(l)
            if "error" not in c:
                cloud[c["id"]] = c
        chunks = {}
        if with_text:
            for l in open(f"{d}/chunks.jsonl"):
                c = json.loads(l)
                chunks[c["node_id"]] = c["text"]
        for l in open(f"{d}/trace.jsonl"):
            t = json.loads(l)
            if t["id"] not in cloud:
                continue
            c = cloud[t["id"]]
            fe, fc = f1(t["edge_answer"], refs[t["id"]]), f1(c["cloud_answer"], refs[t["id"]])
            rows.append(
                {
                    "ds": ds,
                    "split": split,
                    "pool": pool,
                    "id": t["id"],
                    "query": t["query"],
                    "chunks": [chunks[i] for i in t["topk_ids"][: t["k"]]] if with_text else None,
                    "k": t["k"],
                    "complexity": t["complexity"],
                    "rerank": t["topk_scores"],
                    "judgments": t["judgments"],
                    "ecrag_edge": t["ecrag_route_edge"],
                    "f1_edge": fe,
                    "f1_cloud": fc,
                    "dq": fc - fe,
                    "edge_in": t["edge_in_tokens"],
                    "cloud_in": c["cloud_in_tokens"],
                    "cloud_out": c["cloud_out_tokens"],
                    "cost": 0.27 * c["cloud_in_tokens"] / 1e6 + 1.1 * c["cloud_out_tokens"] / 1e6,
                    "edge_answer": t["edge_answer"],
                    "cloud_answer": c["cloud_answer"],
                }
            )
    return rows
