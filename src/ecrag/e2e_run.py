"""Integrated end-to-end run of the deployed HotpotQA policies. One process, one query at a
time (single-user latency), every stage executed live and timed (cuda-synchronized where on this GPU):
  retrieve (EC-RAG fusion retriever, pilot-1 index) -> rerank with early stopping (ONE compute_score per batch, as in the
  official RerankerWrapper; stage1_trace.py called it twice, which doubled the trace's rerank times but not its scores)
  -> dynamic pruning (official slm.judge_relevance, sequential) -> policy branch -> generation on the vLLM servers.
Policies (thresholds and router from --policy_json, written by analyze_cmp2.py; validation-only):
  rule            complexity score (slm.judge_complexity) and the EC-RAG lambda=0.8 rule
  router          F1 router (q+k) forward; cloud iff score > t_plain
  router+cmp/fix  router forward; date-comparison detection; if detected, edge extraction + guard v2b; a guarded answer is
                  final (no further generation); else cloud iff score > t_fix
Generation: edge = cfg3 'extract' prompt on :8195; cloud = cfg3 'orig_think' on :8193 (thinking on, max 6144). Truncation
policy B: if the cloud response ends with finish_reason 'length' (no final answer), the edge answer is generated and returned.
Saved per query: stage times, k, top-k ids (and agreement with the stage-1 trace), branch, guard verdict and extraction JSON,
final answer, cloud finish reason, token counts, F1 of the final answer.
"""

from paths import ECRAG_SRC, HF_HUB, RUNS  # noqa: E402
import argparse, json, os, sys, time, re

SRC = ECRAG_SRC
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402
from itertools import islice  # noqa: E402
import heapq  # noqa: E402
from openai import OpenAI  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from llama_index.core import Settings  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from slm_inference import slm  # noqa: E402
from reranker import local_reranker  # noqa: E402
from retriever import Retriever  # noqa: E402
from router import complexity_threshold  # noqa: E402
from train_router import Router, build_text  # noqa: E402
from regen_answers import QP, QS, body, parse, VARIANTS, CFG  # noqa: E402
from compare_path import detect, PROMPT, pdate  # noqa: E402
from cmp_guard2 import guard2b  # noqa: E402
from normalize_edge import normalize  # noqa: E402
from router_data import f1, DATA  # noqa: E402

EMB = f"{HF_HUB}/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
RERANK = f"{HF_HUB}/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
IDX = f"{RUNS}/pilot1/trace/_index"
TR = f"{RUNS}/pilot1/trace/hotpotqa"


def sync():
    torch.cuda.synchronize()


def rerank_es(query, nodes, top_k):
    heap, processed = [], 0
    it = iter(nodes)
    while True:
        batch = list(islice(it, top_k))
        if not batch:
            break
        scores = local_reranker.reranker.compute_score([(query, n.text) for n in batch])
        if not isinstance(scores, list):
            scores = [scores]
        if len(heap) == top_k and max(scores) <= heap[0][0]:
            break
        for s, n in zip(scores, batch):
            if len(heap) < top_k:
                heapq.heappush(heap, (s, -processed, n))
            elif s > heap[0][0]:
                heapq.heappushpop(heap, (s, -processed, n))
            processed += 1
    return [n for _, _, n in sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)[:top_k]]


def cmp_extract(edge, q, ctx, op):
    r = edge.completions.create(
        model="qwen3-4b", prompt=QP + PROMPT.format(ctx=ctx, q=q) + QS, temperature=0, max_tokens=128
    )
    raw = r.choices[0].text
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:
        j = {}
    x = {
        "A": j.get("A"),
        "B": j.get("B"),
        "date_A": j.get("date_A"),
        "date_B": j.get("date_B"),
        "raw": raw,
        "answer": None,
    }
    da, db = pdate(str(j.get("date_A"))), pdate(str(j.get("date_B")))
    if da and db and da != db and j.get("A") and j.get("B"):
        n = min(sum(1 for v in da if v), sum(1 for v in db if v))
        ka, kb = da[: max(1, n)], db[: max(1, n)]
        if ka != kb:
            first, last = (j["A"], j["B"]) if ka < kb else (j["B"], j["A"])
            x["answer"] = first if op == "early" else last
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_json", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="split/pool entries, e.g. test3/pool_00")
    ap.add_argument("--policies", nargs="+", default=["rule", "router", "router+cmp/fix"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N questions per pool (smoke test)")
    ap.add_argument(
        "--score_head", type=int, default=0, help="router output used as the score: 0 = pred_dq, 1 = pred_logit"
    )
    ap.add_argument("--normalize", action="store_true", help="apply normalize_edge.py to every edge answer (cfg3n)")
    a = ap.parse_args()
    P = json.load(open(a.policy_json))
    t_plain, t_fix = P["t_plain"], P["t_fix"]
    os.makedirs(a.out, exist_ok=True)
    json.dump(
        {
            "args": vars(a),
            "normalize": a.normalize,
            "score_head": a.score_head,
            "t_plain": t_plain,
            "t_fix": t_fix,
            "cand": P["cand"],
            "guard": "v2b",
            "truncation": "B",
        },
        open(f"{a.out}/config.json", "w"),
        indent=1,
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMB)
    Settings.llm = None
    slm.init("Qwen/Qwen3-4B")
    from FlagEmbedding import FlagReranker

    local_reranker.reranker = FlagReranker(RERANK, use_fp16=True, devices=["cuda:0"])
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    rt = Router("Qwen/Qwen3-4B")
    rt.load_state_dict(torch.load(f"{a.router}/best.pt"), strict=False)
    rt = rt.cuda().eval()
    edge = OpenAI(base_url="http://127.0.0.1:8195/v1", api_key="none", timeout=600)
    cloud = OpenAI(base_url="http://127.0.0.1:8193/v1", api_key="none", timeout=1200)
    ecfg, ccfg = CFG["extract"], CFG["orig_think"]

    def gen_edge(q, ch):
        t0 = time.perf_counter()
        r = edge.completions.create(
            model="qwen3-4b",
            prompt=QP + body(ch, q, VARIANTS["extract"]) + QS,
            temperature=0,
            max_tokens=ecfg["edge_max"],
        )
        ans = parse(r.choices[0].text, ecfg["parse"])
        if a.normalize:
            ans = normalize(q, ans)[0]
        return ans, time.perf_counter() - t0, r.usage.completion_tokens

    def gen_cloud(q, ch):
        t0 = time.perf_counter()
        r = cloud.chat.completions.create(
            model="q38-flash-next-fp8",
            temperature=0,
            max_tokens=ccfg["cloud_max"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": body(ch, q, VARIANTS["orig_think"]) + "\n"},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        c = r.choices[0]
        txt = c.message.content or ""
        return (
            parse(txt, None),
            time.perf_counter() - t0,
            r.usage.completion_tokens,
            r.usage.prompt_tokens,
            c.finish_reason,
            "</think>" in txt or c.finish_reason == "stop",
        )

    retr_cache = {}
    # warmup (not timed, not saved): every stage on the first 3 questions of the first pool, then 3 router forwards
    sp0, pool0 = a.runs[0].split("/")
    retr_cache[a.runs[0]] = Retriever(
        argparse.Namespace(
            docstore=f"{IDX}/hotpotqa_{sp0}_{pool0}",
            similarity_top_k=40,
            bm25_similarity_top_k=40,
            use_kvcache=False,
            preload_kvcache=False,
        )
    )
    for l in list(open(f"{DATA}/hotpotqa/{sp0}/{pool0}/questions/questions.jsonl"))[:3]:
        q = json.loads(l)["query"]
        top = rerank_es(q, retr_cache[a.runs[0]].fusion_retrieve(q), 10)
        slm.judge_relevance(top[2], q, False, None)
        slm.judge_complexity(q)
        enc = tok([build_text({"query": q, "k": 3}, tok, 6144, use_chunks=False)], return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rt(enc.input_ids, enc.attention_mask)
        gen_edge(q, [top[0].node.text])
    sync()
    for rep in range(a.reps):
        for pol in a.policies:
            for sp_pool in a.runs:
                sp, pool = sp_pool.split("/")
                fn = f"{a.out}/{sp}_{pool}_{pol.replace('/', '-').replace('+', 'p')}_r{rep}.jsonl"
                done = {json.loads(l)["id"] for l in open(fn)} if os.path.exists(fn) else set()
                if sp_pool not in retr_cache:
                    retr_cache[sp_pool] = Retriever(
                        argparse.Namespace(
                            docstore=f"{IDX}/hotpotqa_{sp}_{pool}",
                            similarity_top_k=40,
                            bm25_similarity_top_k=40,
                            use_kvcache=False,
                            preload_kvcache=False,
                        )
                    )
                retr = retr_cache[sp_pool]
                qs = [json.loads(l) for l in open(f"{DATA}/hotpotqa/{sp}/{pool}/questions/questions.jsonl")]
                qs = qs[: a.limit] if a.limit else qs
                refs = {
                    json.loads(l)["id"]: json.loads(l)["answers"]
                    for l in open(f"{DATA}/hotpotqa/{sp}/{pool}/answers/answers.jsonl")
                }
                trace = {json.loads(l)["id"]: json.loads(l) for l in open(f"{TR}/{sp}/{pool}/trace.jsonl")}
                with open(fn, "a") as f:
                    for qq in qs:
                        if qq["id"] in done:
                            continue
                        q = qq["query"]
                        T = {}
                        o = {"id": qq["id"], "policy": pol, "rep": rep}
                        sync()
                        t0 = time.perf_counter()
                        nodes = retr.fusion_retrieve(q)
                        T["retrieve"] = time.perf_counter() - t0
                        sync()
                        t0 = time.perf_counter()
                        top = rerank_es(q, nodes, 10)
                        sync()
                        T["rerank"] = time.perf_counter() - t0
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
                        T["prune"] = time.perf_counter() - t0
                        ch = [n.node.text for n in top[:k]]
                        ids = [n.node.node_id for n in top]
                        tr = trace.get(qq["id"], {})
                        o["k"] = k
                        o["k_trace"] = tr.get("k")
                        o["topk_same_as_trace"] = ids == tr.get("topk_ids")
                        branch, final = None, None
                        if pol == "rule":
                            t0 = time.perf_counter()
                            cs = slm.judge_complexity(q)
                            sync()
                            T["complexity"] = time.perf_counter() - t0
                            branch = "edge" if cs < complexity_threshold(k, 2, 10, lam=0.8) else "cloud"
                        else:
                            txt = build_text({"query": q, "k": k}, tok, 6144, use_chunks=False)
                            sync()
                            t0 = time.perf_counter()
                            enc = tok([txt], return_tensors="pt").to("cuda")
                            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                                s = float(rt(enc.input_ids, enc.attention_mask)[0, a.score_head])
                            sync()
                            T["router"] = time.perf_counter() - t0
                            o["router_score"] = s
                            thr = t_plain
                            if pol == "router+cmp/fix":
                                thr = t_fix
                                op = detect(q)
                                o["is_cmp"] = op is not None
                                if op:
                                    t0 = time.perf_counter()
                                    x = cmp_extract(edge, q, "\n\n".join(ch), op)
                                    T["cmp"] = time.perf_counter() - t0
                                    ok, why = guard2b(x, q, ch) if x["answer"] is not None else (False, "no-answer")
                                    o["cmp"] = {k_: x[k_] for k_ in ["A", "B", "date_A", "date_B", "answer"]}
                                    o["guard"] = why
                                    if ok:
                                        branch, final = "path", x["answer"]
                            if branch is None:
                                branch = "cloud" if s > thr else "edge"
                        if branch == "edge":
                            final, T["edge"], o["edge_out_tokens"] = gen_edge(q, ch)
                        elif branch == "cloud":
                            ans, T["cloud"], o["cloud_out_tokens"], o["cloud_in_tokens"], o["cloud_finish"], okc = (
                                gen_cloud(q, ch)
                            )
                            o["cloud_completed"] = bool(okc)
                            if okc:
                                final = ans
                            else:
                                final, T["edge_fallback"], o["edge_out_tokens"] = gen_edge(q, ch)
                        o.update(
                            {
                                "branch": branch,
                                "answer": final,
                                "f1": f1(final or "", refs[qq["id"]]),
                                "time": T,
                                "total": sum(T.values()),
                            }
                        )
                        f.write(json.dumps(o, ensure_ascii=False) + "\n")
                        f.flush()
                print(time.strftime("%FT%TZ", time.gmtime()), "done", fn, flush=True)


if __name__ == "__main__":
    main()
