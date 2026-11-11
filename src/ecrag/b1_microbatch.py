"""B1: microbatch dynamic pruning, with the same decision rule as EC-RAG.

EC-RAG (slm_inference.judge_relevance + retriever loop):
- for i = min_k .. k-1: greedy generate(max_new_tokens=1) on the relevance prompt for chunk i
- decode; if == "No": stop, k = i; otherwise keep it and continue
Here:
- seq: the official judge_relevance, one chunk per call (reproduces stage 1)
- mb{b}: judge chunks i..i+b-1 in ONE left-padded forward (explicit position_ids). The decision is the same:
  full-vocab argmax of the last-position logits -> decode -> == "No". k = first No. Later chunks in the batch are wasted
  compute. If there is no No, move on to the next batch.
Per query and mode we record the judgments, k, prune latency (cuda-synchronized) and the number of chunks judged.
Agreement with the stage-1 trace (judgments and k) is reported, since padding and batching can change bf16/fp16 numerics.
Input: stage-1 traces (top-10 reranked chunk ids). Model: the official slm (Qwen3-4B, fp16).
"""

from paths import ECRAG_SRC, RUNS  # noqa: E402
import argparse, json, os, sys, time, statistics as st
import torch

SRC = ECRAG_SRC
sys.path.insert(0, SRC)
from slm_inference import slm, judge_relevance_prompt  # noqa: E402

TRACE = f"{RUNS}/pilot1/trace"


class N:
    def __init__(self, t):
        self.text = t
        self.metadata = {}


@torch.no_grad()
def judge_batch(texts, q):
    tok = slm.tokenizer
    ps = [judge_relevance_prompt(t, q, slm.model_type) for t in texts]
    tok.padding_side = "left"
    enc = tok(ps, return_tensors="pt", padding=True).to(slm.device)
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    logits = slm.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, position_ids=pos).logits[:, -1, :]
    ids = logits.argmax(-1).tolist()
    return [tok.decode([i], skip_special_tokens=True).strip() != "No" for i in ids]


def prune(mode, top, q, min_k, max_k):
    k = min(max_k, len(top))
    judg = []
    n_judged = 0
    if k <= min_k:
        return judg, k, n_judged
    i = min_k
    if mode == "seq":
        while i < k:
            rel = slm.judge_relevance(N(top[i]), q, False, None)
            judg.append(bool(rel))
            n_judged += 1
            if not rel:
                break
            i += 1
        return judg, i, n_judged
    b = int(mode[2:])
    while i < k:
        batch = top[i : min(i + b, k)]
        rels = judge_batch(batch, q)
        n_judged += len(batch)
        for r in rels:
            judg.append(bool(r))
            if not r:
                return judg, i, n_judged
            i += 1
    return judg, i, n_judged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["seq", "mb2", "mb4", "mb10"])
    ap.add_argument("--min_k", type=int, default=2)
    ap.add_argument("--max_k", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    slm.init("Qwen/Qwen3-4B")
    qs = []
    for p in a.pools:
        ch = {json.loads(l)["node_id"]: json.loads(l)["text"] for l in open(f"{TRACE}/{p}/chunks.jsonl")}
        for l in open(f"{TRACE}/{p}/trace.jsonl"):
            r = json.loads(l)
            qs.append((p, r, [ch[i] for i in r["topk_ids"]]))
    # warmup
    for m in a.modes:
        prune(m, qs[0][2], qs[0][1]["query"], a.min_k, a.max_k)
    res = {m: [] for m in a.modes}
    for p, r, top in qs:
        for m in a.modes:  # interleave modes per query so drift affects all modes equally
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            j, k, n = prune(m, top, r["query"], a.min_k, a.max_k)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            res[m].append(
                {
                    "pool": p,
                    "id": r["id"],
                    "k": k,
                    "judgments": j,
                    "n_judged": n,
                    "ms": dt * 1e3,
                    "k_trace": r["k"],
                    "judg_trace": r["judgments"],
                }
            )
    summ = {}
    for m, rows in res.items():
        ms = [x["ms"] for x in rows]
        summ[m] = {
            "n": len(rows),
            "ms_mean": st.mean(ms),
            "ms_median": st.median(ms),
            "ms_p90": sorted(ms)[int(0.9 * len(ms))],
            "chunks_judged_mean": st.mean(x["n_judged"] for x in rows),
            "k_agree_trace": st.mean(x["k"] == x["k_trace"] for x in rows),
            "k_agree_seq": st.mean(x["k"] == y["k"] for x, y in zip(rows, res["seq"])) if "seq" in res else None,
            "k_mean": st.mean(x["k"] for x in rows),
        }
        print(m, json.dumps(summ[m]), flush=True)
    json.dump({"summary": summ, "rows": res}, open(a.out, "w"))


if __name__ == "__main__":
    main()
