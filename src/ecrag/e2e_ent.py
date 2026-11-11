"""Integrated end-to-end run of the frozen E/N/T and E/N policies. One process,
one query at a time (single user), every stage live and timed (cuda-synchronized):
  retrieve -> rerank (early stopping, one compute_score per batch) -> dynamic pruning (official slm.judge_relevance)
  -> policy branch -> generation on the vLLM servers.
Policies (thresholds from runs/pilot7/ent_policy.json, all chosen on validation; nothing chosen here):
  rule        EC-RAG complexity score + lambda=0.8 rule; cloud = T                      (baseline, = rule->T)
  ruleN       same rule; cloud = N                                                       (baseline, = rule->N)
  ens3_ENT    q+D+k decider, 3 seeds, scores averaged (3 forward passes); E/N/T, budget x1 (= the pre-registered test policy)
  s1234_EN    q+D+k decider, single seed 1234 (1 forward pass); E/N, budget x1          (deployment candidate)
Actions: E = edge 'extract' prompt on :8195 + normalize_edge.py; N = cloud 'orig', thinking off, max 256;
T = cloud 'orig_think', thinking on, max 6144, truncation B (no final answer -> E is generated and returned).
Saved per query: stage times, k, top-k agreement with the stage-1 trace, decider scores per seed (and the offline
prediction for the same query), branch, answer, F1, cloud tokens and cloud $ (0.27/M in, 1.1/M out; per 1k queries)."""

from paths import RUNS  # noqa: E402
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from e2e_run import (
    rerank_es,
    sync,
    Settings,
    HuggingFaceEmbedding,
    slm,
    local_reranker,
    Retriever,
    complexity_threshold,
    EMB,
    RERANK,
    IDX,
    TR,
)  # noqa: E402
import torch, torch.nn as nn  # noqa: E402
from openai import OpenAI  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from train_router import Router, build_text  # noqa: E402
from regen_answers import QP, QS, body, parse, VARIANTS, CFG  # noqa: E402
from normalize_edge import normalize  # noqa: E402
from router_data import f1, DATA  # noqa: E402

P7 = f"{RUNS}/pilot7"
POL = {"ens3_ENT": ("ens3_qd|E/N/T|x1.0", [1234, 2345, 3456]), "s1234_EN": ("s1234_qd|E/N|x1.0", [1234])}


def load_decider(seed):
    m = Router("Qwen/Qwen3-4B")
    h = m.head[0].in_features
    m.head = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Linear(512, 5)).float()
    m.load_state_dict(torch.load(f"{P7}/router/ent_qd_s{seed}/best.pt"), strict=False)
    return m.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="split/pool entries, e.g. test3/pool_00")
    ap.add_argument("--policies", nargs="+", default=["rule", "ruleN", "ens3_ENT", "s1234_EN"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    PJ = json.load(open(f"{P7}/ent_policy.json"))
    thr = {p: (PJ["policies"][k]["a_dEN"], PJ["policies"][k]["b_dNT"]) for p, (k, _) in POL.items()}
    json.dump(
        {
            "args": vars(a),
            "thresholds": thr,
            "policy_keys": {p: k for p, (k, _) in POL.items()},
            "deploy": PJ["deploy"],
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
    mem0 = torch.cuda.memory_allocated()
    D = {s: load_decider(s) for s in [1234, 2345, 3456]}
    mem_dec = (torch.cuda.memory_allocated() - mem0) / 3 / 2**30
    offline = {}
    for sp in {r.split("/")[0] for r in a.runs}:
        for s in D:
            for x in json.load(open(f"{P7}/router/ent_qd_s{s}/pred_{sp}.json")):
                offline[(sp, x["pool"], x["id"], s)] = (x["dEN"], x["dNT"])
    edge = OpenAI(base_url="http://127.0.0.1:8195/v1", api_key="none", timeout=600)
    cloud = OpenAI(base_url="http://127.0.0.1:8193/v1", api_key="none", timeout=1200)

    def gen_E(q, ch):
        t0 = time.perf_counter()
        r = edge.completions.create(
            model="qwen3-4b",
            prompt=QP + body(ch, q, VARIANTS["extract"]) + QS,
            temperature=0,
            max_tokens=CFG["extract"]["edge_max"],
        )
        return (
            normalize(q, parse(r.choices[0].text, CFG["extract"]["parse"]))[0],
            time.perf_counter() - t0,
            r.usage.completion_tokens,
        )

    def gen_C(q, ch, think):
        t0 = time.perf_counter()
        r = cloud.chat.completions.create(
            model="q38-flash-next-fp8",
            temperature=0,
            max_tokens=6144 if think else 256,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": body(ch, q, VARIANTS["orig_think" if think else "orig"]) + "\n"},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": think}},
        )
        c = r.choices[0]
        txt = c.message.content or ""
        ok = ("</think>" in txt or c.finish_reason == "stop") if think else True
        return (
            parse(txt, None),
            time.perf_counter() - t0,
            r.usage.completion_tokens,
            r.usage.prompt_tokens,
            c.finish_reason,
            ok,
        )

    def decide(seeds, q, k, ch):
        txt = build_text({"query": q, "k": k, "chunks": ch}, tok, 6144, use_chunks=True)
        S = {}
        sync()
        t0 = time.perf_counter()
        enc = tok([txt], return_tensors="pt").to("cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in seeds:
                o = D[s](enc.input_ids, enc.attention_mask)[0]
                S[s] = (float(o[3]), float(o[4]))
        sync()
        return S, time.perf_counter() - t0, enc.input_ids.shape[1], torch.cuda.max_memory_allocated() / 2**30

    retr_cache = {}

    def retriever(sp_pool):
        if sp_pool not in retr_cache:
            sp, pool = sp_pool.split("/")
            retr_cache[sp_pool] = Retriever(
                argparse.Namespace(
                    docstore=f"{IDX}/hotpotqa_{sp}_{pool}",
                    similarity_top_k=40,
                    bm25_similarity_top_k=40,
                    use_kvcache=False,
                    preload_kvcache=False,
                )
            )
        return retr_cache[sp_pool]

    sp0, pool0 = a.runs[0].split("/")  # warmup: every stage on 3 questions (not timed, not saved)
    for l in list(open(f"{DATA}/hotpotqa/{sp0}/{pool0}/questions/questions.jsonl"))[:3]:
        q = json.loads(l)["query"]
        top = rerank_es(q, retriever(a.runs[0]).fusion_retrieve(q), 10)
        slm.judge_relevance(top[2], q, False, None)
        slm.judge_complexity(q)
        ch = [n.node.text for n in top[:3]]
        decide([1234, 2345, 3456], q, 3, ch)
        gen_E(q, ch)
        gen_C(q, ch, False)
    sync()
    for rep in range(a.reps):
        for pol in a.policies:
            for sp_pool in a.runs:
                sp, pool = sp_pool.split("/")
                fn = f"{a.out}/{sp}_{pool}_{pol}_r{rep}.jsonl"
                done = {json.loads(l)["id"] for l in open(fn)} if os.path.exists(fn) else set()
                retr = retriever(sp_pool)
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
                        if pol in ("rule", "ruleN"):
                            t0 = time.perf_counter()
                            cs = slm.judge_complexity(q)
                            sync()
                            T["complexity"] = time.perf_counter() - t0
                            branch = (
                                "E" if cs < complexity_threshold(k, 2, 10, lam=0.8) else ("T" if pol == "rule" else "N")
                            )
                        else:
                            seeds = POL[pol][1]
                            S, T["decider"], o["decider_in_tokens"], o["decider_peak_gib"] = decide(seeds, q, k, ch)
                            dEN = sum(v[0] for v in S.values()) / len(S)
                            dNT = sum(v[1] for v in S.values()) / len(S)
                            o["scores"] = {str(s): v for s, v in S.items()}
                            o["dEN"], o["dNT"] = dEN, dNT
                            o["offline"] = {str(s): offline.get((sp, pool, qq["id"], s)) for s in seeds}
                            ta, tb = thr[pol]
                            branch = "T" if (dEN > ta and dNT > tb) else ("N" if dEN > ta else "E")
                        o["cost"] = 0.0
                        if branch == "E":
                            final, T["edge"], o["edge_out_tokens"] = gen_E(q, ch)
                        else:
                            ans, T["cloud"], o["cloud_out_tokens"], o["cloud_in_tokens"], o["cloud_finish"], ok = gen_C(
                                q, ch, branch == "T"
                            )
                            o["cost"] = (0.27 * o["cloud_in_tokens"] + 1.1 * o["cloud_out_tokens"]) / 1e3
                            o["cloud_completed"] = bool(ok)
                            final = ans
                            if not ok:
                                final, T["edge_fallback"], o["edge_out_tokens"] = gen_E(q, ch)
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
    json.dump({"decider_weights_gib_each": mem_dec}, open(f"{a.out}/memory.json", "w"))


if __name__ == "__main__":
    main()
