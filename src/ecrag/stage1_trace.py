"""Stage 1 of the EC-RAG router pilot: run the official EC-RAG edge pipeline per pool and record a replayable trace.

Steps per pool:
- Index with EC-RAG's own chunking.py logic (SentenceSplitter 512/20, bge-small-en-v1.5, BM25). No chunk KV cache:
  the pilot measures routing quality, not latency.
- Retrieve (vector 40 + BM25 40, RRF), rerank with early stopping (bge-reranker-v2-m3, max_k=10), dynamic pruning with
  Qwen3-4B Yes/No (min_k=2, max_k=10), and the Easy/Hard complexity score.
- The EC-RAG adaptive route (lambda=0.8) is recomputed from the score and k.
- Edge answer: Qwen3-4B on the pruned chunks, same prompt as EC-RAG, max_new_tokens=50, but GREEDY.
  The official code samples (Qwen3 default generation_config); the pilot needs fixed outputs for replay.
Everything except greedy decoding and the disabled KV cache uses the official modules
(third_party/ec_rag/src, commit 1379d14).
Output: <out>/<ds>/<split>/<pool>/{chunks.jsonl, trace.jsonl}
"""

from paths import DATA, ECRAG_SRC, HF_HUB, RUNS  # noqa: E402
import argparse, hashlib, heapq, json, os, sys, time
from itertools import islice

SRC = ECRAG_SRC
sys.path.insert(0, SRC)
import torch  # noqa: E402
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex  # noqa: E402
from llama_index.core.node_parser import SentenceSplitter  # noqa: E402
from llama_index.core.storage.docstore import SimpleDocumentStore  # noqa: E402
from llama_index.core.vector_stores import SimpleVectorStore  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402

from chunking import get_nodes_from_documents  # noqa: E402
from slm_inference import slm, query_prompt  # noqa: E402
from reranker import local_reranker  # noqa: E402
from retriever import Retriever  # noqa: E402
from router import complexity_threshold  # noqa: E402

SLM = "Qwen/Qwen3-4B"
HUB = HF_HUB
EMB = f"{HUB}/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
RERANK = f"{HUB}/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def build_index(pool_dir, store):
    if os.path.exists(store + "_docstore.pkl") and os.path.exists(store + "_vec"):
        return
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    docs = SimpleDirectoryReader(f"{pool_dir}/documents").load_data()
    nodes = get_nodes_from_documents(docs, splitter)
    ds = SimpleDocumentStore()
    ds.add_documents(nodes)
    index = VectorStoreIndex(nodes=nodes, vector_store=SimpleVectorStore())
    os.makedirs(os.path.dirname(store), exist_ok=True)
    ds.persist(store + "_docstore.pkl")
    index.storage_context.persist(persist_dir=store + "_vec/")


def rerank_es(query, nodes, top_k):
    """Same logic as RerankerWrapper.rerank_nodes_with_early_stopping, but also returns scores."""
    heap, processed, scored = [], 0, 0
    it = iter(nodes)
    while True:
        batch = list(islice(it, top_k))
        if not batch:
            break
        scores = local_reranker.reranker.compute_score([(query, n.text) for n in batch])
        if not isinstance(scores, list):
            scores = [scores]
        scored += len(batch)
        if len(heap) == top_k and max(scores) <= heap[0][0]:
            break
        for s, n in zip(scores, batch):
            if len(heap) < top_k:
                heapq.heappush(heap, (s, -processed, n))
            elif s > heap[0][0]:
                heapq.heappushpop(heap, (s, -processed, n))
            processed += 1
    top = sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)[:top_k]
    return [n for _, _, n in top], [float(s) for s, _, _ in top], scored


def edge_generate(query, nodes):
    prompt = query_prompt([n.text for n in nodes], query, slm.model_type)
    inputs = slm.tokenizer(prompt, return_tensors="pt").to(slm.device)
    with torch.no_grad():
        out = slm.model.generate(**inputs, max_new_tokens=50, do_sample=False, temperature=None, top_p=None, top_k=None)
    gen = out[0][inputs.input_ids.shape[1] :]
    return slm.tokenizer.decode(gen, skip_special_tokens=True).strip(), int(inputs.input_ids.shape[1]), int(len(gen))


def run_pool(ds, split, pool, a):
    pool_dir = f"{a.data}/{ds}/{split}/{pool}"
    od = f"{a.out}/{ds}/{split}/{pool}"
    os.makedirs(od, exist_ok=True)
    if os.path.exists(f"{od}/DONE"):
        return
    store = f"{a.out}/_index/{ds}_{split}_{pool}"
    build_index(pool_dir, store)
    rargs = argparse.Namespace(
        docstore=store, similarity_top_k=40, bm25_similarity_top_k=40, use_kvcache=False, preload_kvcache=False
    )
    retr = Retriever(rargs)
    with open(f"{od}/chunks.jsonl", "w") as f:
        for nid, n in retr.docstore.docs.items():
            f.write(json.dumps({"node_id": nid, "doc": n.ref_doc_id, "text": n.text}, ensure_ascii=False) + "\n")
    qs = [json.loads(l) for l in open(f"{pool_dir}/questions/questions.jsonl")]
    done = set()
    tp = f"{od}/trace.jsonl"
    if os.path.exists(tp):
        done = {json.loads(l)["id"] for l in open(tp)}
    with open(tp, "a") as f:
        for q in qs:
            if q["id"] in done:
                continue
            t0 = time.perf_counter()
            nodes = retr.fusion_retrieve(q["query"])
            t1 = time.perf_counter()
            top, rscores, n_scored = rerank_es(q["query"], nodes, a.max_k)
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
            ans, ntok_in, ntok_out = edge_generate(q["query"], top[:k])
            t5 = time.perf_counter()
            rec = {
                "id": q["id"],
                "query": q["query"],
                "n_retrieved": len(nodes),
                "n_rerank_scored": n_scored,
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
    open(f"{od}/DONE", "w").write(time.strftime("%FT%TZ", time.gmtime()) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=f"{RUNS}/pilot1/trace")
    ap.add_argument("--pools", nargs="+", required=True, help="ds/split/pool entries")
    ap.add_argument("--min_k", type=int, default=2)
    ap.add_argument("--max_k", type=int, default=10)
    a = ap.parse_args()
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMB)  # same model, local snapshot path (offline)
    Settings.llm = None
    slm.init(SLM)
    from FlagEmbedding import FlagReranker

    local_reranker.reranker = FlagReranker(
        RERANK, use_fp16=True, devices=["cuda:0"]
    )  # = RerankerWrapper.init with a local path
    local_reranker.args = a
    for p in a.pools:
        ds, split, pool = p.split("/")
        print(time.strftime("%FT%TZ", time.gmtime()), "start", p, flush=True)
        run_pool(ds, split, pool, a)
        print(time.strftime("%FT%TZ", time.gmtime()), "done", p, flush=True)


if __name__ == "__main__":
    main()
