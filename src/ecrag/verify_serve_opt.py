"""Agreement + timing of serve_opt against the reference local stages. On --runs (split/pool, pilot8 index):
retrieve -> rerank -> prune (sequential reference vs prune_batched) -> build_text (reference vs fast) -> decider (unmerged vs
merged). Reports k agreement (and vs the stage-1 trace), text equality, dEN max diff, E/N decision agreement at the frozen
threshold, and mean ms per stage. Output json --out."""

from paths import RUNS  # noqa: E402
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e2e_run import (
    rerank_es,
    sync,
    Settings,
    HuggingFaceEmbedding,
    slm,
    local_reranker,
    Retriever,
    EMB,
    RERANK,
)  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from train_router import build_text  # noqa: E402
from e2e_ent2 import load_decider  # noqa: E402
from serve_opt import build_text_fast, prune_batched, prune_fwd  # noqa: E402
from router_data import DATA  # noqa: E402

R = RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--idx", default=f"{R}/pilot8/trace/_index")
    ap.add_argument("--trace", default=f"{R}/pilot8/trace/hotpotqa")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMB)
    Settings.llm = None
    slm.init("Qwen/Qwen3-4B")
    print("attn", slm.model.config._attn_implementation, flush=True)
    from FlagEmbedding import FlagReranker

    local_reranker.reranker = FlagReranker(RERANK, use_fp16=True, devices=["cuda:0"])
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    tok.padding_side = "right"
    dec = load_decider(f"{R}/pilot7/router/ent_qd_s1234")
    thr = json.load(open(f"{R}/pilot7/ent_policy.json"))["policies"]["s1234_qd|E/N|x1.0"]["a_dEN"]
    recs = []
    for sp_pool in a.runs:
        sp, pool = sp_pool.split("/")
        ret = Retriever(
            argparse.Namespace(
                docstore=f"{a.idx}/hotpotqa_{sp}_{pool}",
                similarity_top_k=40,
                bm25_similarity_top_k=40,
                use_kvcache=False,
                preload_kvcache=False,
            )
        )
        tr = {json.loads(l)["id"]: json.loads(l) for l in open(f"{a.trace}/{sp}/{pool}/trace.jsonl")}
        qs = [json.loads(l) for l in open(f"{DATA}/hotpotqa/{sp}/{pool}/questions/questions.jsonl")]
        for x in qs:
            q = x["query"]
            T = {}
            t0 = time.perf_counter()
            top = rerank_es(q, ret.fusion_retrieve(q), 10)
            sync()
            T["retr_rerank"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            k = min(10, len(top))
            if k > 2:
                i = 2
                while i < k:
                    if not slm.judge_relevance(top[i], q, False, None):
                        break
                    i += 1
                k = i
            sync()
            T["prune_seq"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            kb = prune_batched(slm, top, q, 10)
            sync()
            T["prune_batch"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            kf = prune_fwd(slm, top, q, 10)
            sync()
            T["prune_fwd"] = time.perf_counter() - t0
            r = {"query": q, "k": k, "chunks": [n.node.text for n in top[:k]]}
            t0 = time.perf_counter()
            s_ref = build_text(r, tok, 6144, use_chunks=True)
            enc = tok([s_ref], return_tensors="pt").to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                d_ref = float(dec(enc.input_ids, enc.attention_mask)[0, 3])
            sync()
            T["dec_ref"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            s_fast = build_text_fast(r, tok, 6144)
            sync()
            T["text_fast"] = time.perf_counter() - t0
            recs.append(
                {
                    "run": sp_pool,
                    "id": x["id"],
                    "k": k,
                    "k_batch": kb,
                    "k_fwd": kf,
                    "k_trace": tr[x["id"]]["k"],
                    "text_same": s_ref == s_fast,
                    "d_ref": d_ref,
                    "text": s_fast,
                    "T": T,
                }
            )
    from serve_opt import merge_decider

    merge_decider(dec)
    for r in recs:
        t0 = time.perf_counter()
        enc = tok([r.pop("text")], return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            r["d_merged"] = float(dec(enc.input_ids, enc.attention_mask)[0, 3])
        sync()
        r["T"]["dec_merged_fwd"] = time.perf_counter() - t0
    n = len(recs)
    A = lambda f: float(np.mean([f(r) for r in recs]))
    S = {
        "n": n,
        "k_batch_eq_seq": A(lambda r: r["k"] == r["k_batch"]),
        "k_fwd_eq_seq": A(lambda r: r["k"] == r["k_fwd"]),
        "k_seq_eq_trace": A(lambda r: r["k"] == r["k_trace"]),
        "k_batch_eq_trace": A(lambda r: r["k_batch"] == r["k_trace"]),
        "text_same": A(lambda r: r["text_same"]),
        "dEN_maxdiff": float(max(abs(r["d_ref"] - r["d_merged"]) for r in recs)),
        "decision_agree": A(lambda r: (r["d_ref"] > thr) == (r["d_merged"] > thr)),
        "ms": {k: 1e3 * A(lambda r: r["T"][k]) for k in recs[0]["T"]},
    }
    print(json.dumps(S, indent=1))
    json.dump({"summary": S, "rows": recs}, open(a.out, "w"))


if __name__ == "__main__":
    main()
