"""Integrated end-to-end run with wall-clock timing. Same pipeline as e2e_ent.py, plus:
  - 'wall': one wall-clock timer from request start (before retrieval) to the final answer, covering every Python step
    (text building, tokenization, policy logic). Per-stage times are still saved; the decider timer now includes build_text.
  - more policies (all thresholds fixed before the run, from validation; nothing chosen here):
      rule        EC-RAG rule (lambda 0.8), cloud = T                          rule->T
      ruleN       EC-RAG rule (lambda 0.8), cloud = N                          rule->N
      ruleN_tuned EC-RAG complexity rule with lam, c re-tuned on validation (rule_tuned.json, x1), cloud = N
      allN        every query to N
      tfidf_EN    TF-IDF + Ridge dEN (fit on train at start-up), E/N threshold of ent_policy.json tfidf|E/N|x1.0
      s1234_EN    q+D+k decider seed 1234, E/N threshold of ent_policy.json s1234_qd|E/N|x1.0  (primary)
      ens3_ENT    3-seed q+D+k ensemble, E/N/T thresholds ens3_qd|E/N/T|x1.0
      dec:<name>  extra decider given by --extra name=<ckpt dir>=<threshold a> (E/N), e.g. a retrained decider
  - --idx / --trace: index and stage-1 trace roots (pilot-1 for test2/test3, pilot-8 for test4).
Actions and truncation fallback as in e2e_ent.py: E = edge 'extract' + normalize_edge; N = cloud 'orig' no thinking (max 256);
T = cloud 'orig_think' thinking (max 6144), no final answer -> E."""

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
)  # noqa: E402
import numpy as np  # noqa: E402
import torch, torch.nn as nn  # noqa: E402
from openai import OpenAI  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from train_router import Router, build_text  # noqa: E402
from regen_answers import QP, QS, body, parse, VARIANTS, CFG  # noqa: E402
from normalize_edge import normalize  # noqa: E402
from router_data import f1, DATA  # noqa: E402

R = RUNS
P7 = f"{R}/pilot7"


def load_decider(ckpt, n_out=5):
    base = (
        json.load(open(f"{ckpt}/args.json")).get("base", "Qwen/Qwen3-4B")
        if os.path.exists(f"{ckpt}/args.json")
        else "Qwen/Qwen3-4B"
    )
    m = Router(base)
    h = m.head[0].in_features
    m.head = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Linear(512, n_out)).float()
    m.load_state_dict(torch.load(f"{ckpt}/best.pt"), strict=False)
    return m.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="split/pool entries, e.g. test4/pool_00")
    ap.add_argument("--policies", nargs="+", default=["rule", "ruleN", "ruleN_tuned", "allN", "tfidf_EN", "s1234_EN"])
    ap.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="name=ckpt_dir=a_dEN[=dEN_index] for extra E/N deciders (policy dec:name)",
    )
    ap.add_argument("--idx", default=f"{R}/pilot1/trace/_index")
    ap.add_argument("--trace", default=f"{R}/pilot1/trace/hotpotqa")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    PJ = json.load(open(f"{P7}/ent_policy.json"))
    RT = json.load(open(f"{P7}/rule_tuned.json"))
    thr = {
        "s1234_EN": (PJ["policies"]["s1234_qd|E/N|x1.0"]["a_dEN"], np.inf),
        "tfidf_EN": (PJ["policies"]["tfidf|E/N|x1.0"]["a_dEN"], np.inf),
        "ens3_ENT": (PJ["policies"]["ens3_qd|E/N/T|x1.0"]["a_dEN"], PJ["policies"]["ens3_qd|E/N/T|x1.0"]["b_dNT"]),
    }
    rtn = RT["policies"]["ruleN_tuned|x1.0"]
    extra = {}
    for e in a.extra:
        nm, ck, t, *ix = e.split("=")
        extra[f"dec:{nm}"] = (ck, float(t), int(ix[0]) if ix else 3)
    json.dump(
        {
            "args": vars(a),
            "thresholds": {k: list(v) for k, v in thr.items()},
            "ruleN_tuned": rtn,
            "extra": extra,
            "deploy": PJ["deploy"],
            "version": "v2",
        },
        open(f"{a.out}/config.json", "w"),
        indent=1,
        default=str,
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMB)
    Settings.llm = None
    slm.init("Qwen/Qwen3-4B")
    from FlagEmbedding import FlagReranker

    local_reranker.reranker = FlagReranker(RERANK, use_fp16=True, devices=["cuda:0"])
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    need = {
        s
        for p in a.policies
        if p in ("s1234_EN", "ens3_ENT")
        for s in ([1234] if p == "s1234_EN" else [1234, 2345, 3456])
    }
    D = {s: load_decider(f"{P7}/router/ent_qd_s{s}") for s in sorted(need)}
    for nm, (ck, _, _) in extra.items():
        n_out = json.load(open(f"{ck}/args.json")).get("n_out", 5)
        D[nm] = load_decider(ck, n_out)
    if "tfidf_EN" in a.policies:
        import ent_data
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge

        tr = ent_data.load("train", "A")
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in tr])
        X = lambda qs, ks: np.hstack([vec.transform(qs).toarray(), np.array([[k / 10] for k in ks])])
        ridge = Ridge(alpha=3.0).fit(X([r["query"] for r in tr], [r["k"] for r in tr]), [r["fN"] - r["fE"] for r in tr])
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

    def decide(keys, q, k, ch, idx=(3, 4)):
        sync()
        t0 = time.perf_counter()
        txt = build_text({"query": q, "k": k, "chunks": ch}, tok, 6144, use_chunks=True)
        enc = tok([txt], return_tensors="pt").to("cuda")
        S = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in keys:
                o = D[s](enc.input_ids, enc.attention_mask)[0].float().tolist()
                S[str(s)] = (o[idx[0]], o[idx[1]] if idx[1] is not None and idx[1] < len(o) else 0.0)
        sync()
        return S, time.perf_counter() - t0, enc.input_ids.shape[1]

    retr_cache = {}

    def retriever(sp_pool):
        if sp_pool not in retr_cache:
            sp, pool = sp_pool.split("/")
            retr_cache[sp_pool] = Retriever(
                argparse.Namespace(
                    docstore=f"{a.idx}/hotpotqa_{sp}_{pool}",
                    similarity_top_k=40,
                    bm25_similarity_top_k=40,
                    use_kvcache=False,
                    preload_kvcache=False,
                )
            )
        return retr_cache[sp_pool]

    sp0, pool0 = a.runs[0].split("/")  # warmup (not timed, not saved)
    for l in list(open(f"{DATA}/hotpotqa/{sp0}/{pool0}/questions/questions.jsonl"))[:3]:
        q = json.loads(l)["query"]
        top = rerank_es(q, retriever(a.runs[0]).fusion_retrieve(q), 10)
        slm.judge_relevance(top[2], q, False, None)
        slm.judge_complexity(q)
        ch = [n.node.text for n in top[:3]]
        if D:
            decide(list(D), q, 3, ch)
        gen_E(q, ch)
        gen_C(q, ch, False)
    sync()
    for rep in range(a.reps):
        for pol in a.policies:
            for sp_pool in a.runs:
                sp, pool = sp_pool.split("/")
                fn = f"{a.out}/{sp}_{pool}_{pol.replace(':', '-')}_r{rep}.jsonl"
                done = {json.loads(l)["id"] for l in open(fn)} if os.path.exists(fn) else set()
                retr = retriever(sp_pool)
                qs = [json.loads(l) for l in open(f"{DATA}/hotpotqa/{sp}/{pool}/questions/questions.jsonl")]
                qs = qs[: a.limit] if a.limit else qs
                refs = {
                    json.loads(l)["id"]: json.loads(l)["answers"]
                    for l in open(f"{DATA}/hotpotqa/{sp}/{pool}/answers/answers.jsonl")
                }
                tp = f"{a.trace}/{sp}/{pool}/trace.jsonl"
                trace = {json.loads(l)["id"]: json.loads(l) for l in open(tp)} if os.path.exists(tp) else {}
                with open(fn, "a") as f:
                    for qq in qs:
                        if qq["id"] in done:
                            continue
                        q = qq["query"]
                        T = {}
                        o = {"id": qq["id"], "policy": pol, "rep": rep}
                        sync()
                        w0 = time.perf_counter()
                        t0 = w0
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
                        if pol in ("rule", "ruleN", "ruleN_tuned"):
                            t0 = time.perf_counter()
                            cs = slm.judge_complexity(q)
                            sync()
                            T["complexity"] = time.perf_counter() - t0
                            o["complexity"] = cs
                            if pol == "ruleN_tuned":
                                lam = rtn["lam"]
                                branch = "N" if cs - (lam - (2 * lam - 1) * (k - 2) / 8) > rtn["c"] else "E"
                            else:
                                branch = (
                                    "E"
                                    if cs < complexity_threshold(k, 2, 10, lam=0.8)
                                    else ("T" if pol == "rule" else "N")
                                )
                        elif pol == "allN":
                            branch = "N"
                        elif pol == "tfidf_EN":
                            t0 = time.perf_counter()
                            s = float(ridge.predict(X([q], [k]))[0])
                            T["selector"] = time.perf_counter() - t0
                            o["dEN"] = s
                            branch = "N" if s > thr[pol][0] else "E"
                        else:
                            if pol in extra:
                                keys, ta, tb, ix = [pol], extra[pol][1], np.inf, (extra[pol][2], None)
                            else:
                                keys = [1234] if pol == "s1234_EN" else [1234, 2345, 3456]
                                (ta, tb), ix = thr[pol], (3, 4)
                            S, T["decider"], o["decider_in_tokens"] = decide(keys, q, k, ch, ix)
                            dEN = float(np.mean([v[0] for v in S.values()]))
                            dNT = float(np.mean([v[1] for v in S.values()]))
                            o["scores"] = S
                            o["dEN"], o["dNT"] = dEN, dNT
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
                        o["wall"] = time.perf_counter() - w0
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
