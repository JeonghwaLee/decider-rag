"""Stage 1 with paper-scoped retrieval for Qasper.

Qasper questions are asked about one given paper ("what is the original model they refer to?"), so the paper is part of
the query input in the real task; the pooled 200-paper corpus of EC-RAG's setup drops that information. Here retrieval is
restricted to the chunks of the question's paper. Everything after retrieval (rerank with early stopping, dynamic pruning,
complexity score, EC-RAG route, greedy edge answer) is identical to stage1_trace.py, so the EC-RAG rule and every router
get the same information.

Retrieval within the paper (same components and parameters as EC-RAG's fusion retriever):
- vector: cosine between the query embedding (bge-small-en-v1.5) and the stored chunk embeddings, top 40
- BM25: llama-index BM25Retriever over the paper's chunks (English stemmer), top 40
- reciprocal-rank fusion (k=60), as QueryFusionRetriever(mode="reciprocal_rerank")
The chunk index is reused from pilot 1 (_index/<ds>_<split>_<pool>), so chunking is unchanged. Paper of a chunk: node ids
are <doc uuid>_<i>; the uuid is mapped to documents/doc_<n>.txt by matching the first chunk against the file text.
Output: runs/pilot3/trace/<ds>/<split>/<pool>/{chunks.jsonl, trace.jsonl, DONE}
"""

from paths import DATA, RUNS  # noqa: E402
import argparse, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import stage1_trace as S1  # noqa: E402  (sets sys.path for the EC-RAG modules)
from llama_index.core import Settings  # noqa: E402
from llama_index.core.storage.docstore import SimpleDocumentStore  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from llama_index.retrievers.bm25 import BM25Retriever  # noqa: E402
from llama_index.core.schema import NodeWithScore  # noqa: E402
import Stemmer  # noqa: E402
from slm_inference import slm  # noqa: E402
from reranker import local_reranker  # noqa: E402
from router import complexity_threshold  # noqa: E402

IDX = f"{RUNS}/pilot1/trace/_index"
nrm = lambda s: re.sub(r"\s+", " ", s).strip()


def paper_map(pool_dir, docs):
    first = {}
    for nid, n in docs.items():
        u, i = nid.rsplit("_", 1)
        if i == "0":
            first[u] = nrm(n.text)[:300]
    files = {}
    for f in os.listdir(f"{pool_dir}/documents"):
        files[int(f[4:-4])] = nrm(open(f"{pool_dir}/documents/{f}", encoding="utf-8").read())
    m = {}
    for u, t in first.items():
        hit = [d for d, txt in files.items() if txt.startswith(t[:200])]
        if len(hit) == 1:
            m[u] = hit[0]
    return m


_CACHE = {}


def scoped_retrieve(q, nodes, emb, top=40, k_rrf=60):
    key = (id(nodes), len(nodes))
    if key not in _CACHE:  # per-paper embedding matrix and BM25 index, built once per pool run
        E = np.stack([emb[n.node_id] for n in nodes])
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        _CACHE[key] = (
            E,
            BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=min(top, len(nodes)),
                stemmer=Stemmer.Stemmer("english"),
                language="english",
            ),
        )
    E, bm = _CACHE[key]
    qe = np.array(Settings.embed_model.get_query_embedding(q))
    qe /= np.linalg.norm(qe)
    vec = [nodes[i] for i in np.argsort(-(E @ qe))[:top]]
    bmr = [x.node for x in bm.retrieve(q)]
    score = {}
    for lst in (vec, bmr):
        for r, n in enumerate(lst):
            score[n.node_id] = score.get(n.node_id, 0.0) + 1.0 / (k_rrf + r)
    byid = {n.node_id: n for n in nodes}
    return [NodeWithScore(node=byid[i], score=s) for i, s in sorted(score.items(), key=lambda x: -x[1])]


def run_pool(ds, split, pool, a):
    pool_dir = f"{a.data}/{ds}/{split}/{pool}"
    od = f"{a.out}/{ds}/{split}/{pool}"
    os.makedirs(od, exist_ok=True)
    if os.path.exists(f"{od}/DONE"):
        return
    store = f"{IDX}/{ds}_{split}_{pool}"
    S1.build_index(pool_dir, store)  # no-op when the pilot-1 index exists
    docs = SimpleDocumentStore.from_persist_path(store + "_docstore.pkl").docs
    emb = {
        k: np.array(v) for k, v in json.load(open(f"{store}_vec/default__vector_store.json"))["embedding_dict"].items()
    }
    pm = paper_map(pool_dir, docs)
    by_doc = {}
    _CACHE.clear()
    for nid, n in docs.items():
        d = pm.get(nid.rsplit("_", 1)[0])
        if d is not None:
            by_doc.setdefault(d, []).append(n)
    with open(f"{od}/chunks.jsonl", "w") as f:
        for nid, n in docs.items():
            f.write(
                json.dumps({"node_id": nid, "doc": pm.get(nid.rsplit("_", 1)[0]), "text": n.text}, ensure_ascii=False)
                + "\n"
            )
    qs = [json.loads(l) for l in open(f"{pool_dir}/questions/questions.jsonl")]
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(f"{pool_dir}/meta.jsonl")}
    tp = f"{od}/trace.jsonl"
    done = {json.loads(l)["id"] for l in open(tp)} if os.path.exists(tp) else set()
    n_unmapped = 0
    with open(tp, "a") as f:
        for q in qs:
            if q["id"] in done:
                continue
            cand = by_doc.get(meta[q["id"]]["doc"])
            if not cand:
                n_unmapped += 1
                continue
            t0 = time.perf_counter()
            nodes = scoped_retrieve(q["query"], cand, emb)
            t1 = time.perf_counter()
            top, rscores, n_scored = S1.rerank_es(q["query"], nodes, a.max_k)
            t2 = time.perf_counter()
            judgments = []
            k = min(a.max_k, len(top))
            if k > a.min_k:
                i = a.min_k
                while i < k:
                    rel = slm.judge_relevance(top[i], q["query"], False, None)
                    judgments.append(bool(rel))
                    if not rel:
                        break
                    i += 1
                k = i
            t3 = time.perf_counter()
            cscore = slm.judge_complexity(q["query"])
            t4 = time.perf_counter()
            ecrag_edge = cscore < complexity_threshold(k, a.min_k, a.max_k, lam=0.8)
            ans, ntok_in, ntok_out = S1.edge_generate(q["query"], top[:k])
            t5 = time.perf_counter()
            rec = {
                "id": q["id"],
                "query": q["query"],
                "n_retrieved": len(nodes),
                "n_rerank_scored": n_scored,
                "scope_doc": meta[q["id"]]["doc"],
                "scope_n_chunks": len(cand),
                "topk_ids": [n.node.node_id for n in top],
                "topk_scores": rscores,
                "judgments": judgments,
                "k": k,
                "complexity": cscore,
                "ecrag_route_edge": bool(ecrag_edge),
                "edge_answer": ans,
                "edge_in_tokens": ntok_in,
                "edge_out_tokens": ntok_out,
                "time": {
                    "retrieve": t1 - t0,
                    "rerank": t2 - t1,
                    "prune": t3 - t2,
                    "complexity": t4 - t3,
                    "edge_gen": t5 - t4,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    print(pool_dir, "unmapped questions", n_unmapped, "mapped papers", len(by_doc), flush=True)
    open(f"{od}/DONE", "w").write(time.strftime("%FT%TZ", time.gmtime()) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=f"{RUNS}/pilot3/trace")
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--min_k", type=int, default=2)
    ap.add_argument("--max_k", type=int, default=10)
    a = ap.parse_args()
    Settings.embed_model = HuggingFaceEmbedding(model_name=S1.EMB)
    Settings.llm = None
    slm.init(S1.SLM)
    from FlagEmbedding import FlagReranker

    local_reranker.reranker = FlagReranker(S1.RERANK, use_fp16=True, devices=["cuda:0"])
    local_reranker.args = a
    for p in a.pools:
        ds, split, pool = p.split("/")
        print(time.strftime("%FT%TZ", time.gmtime()), "start", p, flush=True)
        run_pool(ds, split, pool, a)


if __name__ == "__main__":
    main()
