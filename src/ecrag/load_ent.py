"""Open-loop load test of the frozen E/N policies.
Model of the deployment: one edge device runs the local stages (retrieve -> rerank -> prune -> selector) for every request in
FIFO order on one worker (one GPU, as in the single-user runs); generation calls go to the edge vLLM (:8195) and the cloud vLLM
(:8193), which serve many requests concurrently (thread pool, 256 workers). Requests arrive as a Poisson process at rate r (fixed
seed per rate, the same arrival times and question order for every policy). Latency = answer time - arrival time (includes
queueing). Policies: rule (rule->T), ruleN, tfidf_EN, s1234_EN, allN (thresholds as in e2e_ent2.py; nothing chosen here).
Per request saved: arrival, local start/end, generation end, branch, F1, cloud tokens. Summary: analyze_load.py."""

from paths import RUNS  # noqa: E402
import argparse, json, os, queue, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

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
import torch  # noqa: E402
from openai import OpenAI  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from train_router import build_text  # noqa: E402
from regen_answers import QP, QS, body, parse, VARIANTS, CFG  # noqa: E402
from normalize_edge import normalize  # noqa: E402
from router_data import f1, DATA  # noqa: E402
from e2e_ent2 import load_decider  # noqa: E402

R = RUNS
P7 = f"{R}/pilot7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--rates", nargs="+", type=float, required=True)
    ap.add_argument("--policies", nargs="+", default=["s1234_EN", "rule", "ruleN", "tfidf_EN", "allN"])
    ap.add_argument("--idx", default=f"{R}/pilot1/trace/_index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    PJ = json.load(open(f"{P7}/ent_policy.json"))
    thr = {
        "s1234_EN": PJ["policies"]["s1234_qd|E/N|x1.0"]["a_dEN"],
        "tfidf_EN": PJ["policies"]["tfidf|E/N|x1.0"]["a_dEN"],
    }
    json.dump({"args": vars(a), "thr": thr}, open(f"{a.out}/config.json", "w"), indent=1)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMB)
    Settings.llm = None
    slm.init("Qwen/Qwen3-4B")
    from FlagEmbedding import FlagReranker

    local_reranker.reranker = FlagReranker(RERANK, use_fp16=True, devices=["cuda:0"])
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    dec = load_decider(f"{P7}/router/ent_qd_s1234")
    import ent_data
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    tr = ent_data.load("train", "A")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in tr])
    X = lambda qs, ks: np.hstack([vec.transform(qs).toarray(), np.array([[k / 10] for k in ks])])
    ridge = Ridge(alpha=3.0).fit(X([r["query"] for r in tr], [r["k"] for r in tr]), [r["fN"] - r["fE"] for r in tr])
    edge = OpenAI(base_url="http://127.0.0.1:8195/v1", api_key="none", timeout=1200, max_retries=0)
    cloud = OpenAI(base_url="http://127.0.0.1:8193/v1", api_key="none", timeout=1200, max_retries=0)
    Q = []
    retr = {}
    for sp_pool in a.runs:
        sp, pool = sp_pool.split("/")
        retr[sp_pool] = Retriever(
            argparse.Namespace(
                docstore=f"{a.idx}/hotpotqa_{sp}_{pool}",
                similarity_top_k=40,
                bm25_similarity_top_k=40,
                use_kvcache=False,
                preload_kvcache=False,
            )
        )
        refs = {
            json.loads(l)["id"]: json.loads(l)["answers"]
            for l in open(f"{DATA}/hotpotqa/{sp}/{pool}/answers/answers.jsonl")
        }
        qs = [json.loads(l) for l in open(f"{DATA}/hotpotqa/{sp}/{pool}/questions/questions.jsonl")]
        Q += [(sp_pool, x["id"], x["query"], refs[x["id"]]) for x in (qs[: a.limit] if a.limit else qs)]

    def gen(branch, q, ch):
        if branch == "E":
            r = edge.completions.create(
                model="qwen3-4b",
                prompt=QP + body(ch, q, VARIANTS["extract"]) + QS,
                temperature=0,
                max_tokens=CFG["extract"]["edge_max"],
            )
            return normalize(q, parse(r.choices[0].text, CFG["extract"]["parse"]))[0], 0, 0, True
        think = branch == "T"
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
        return parse(txt, None), r.usage.prompt_tokens, r.usage.completion_tokens, ok

    def local(pol, sp_pool, q):
        top = rerank_es(q, retr[sp_pool].fusion_retrieve(q), 10)
        k = min(10, len(top))
        if k > 2:
            i = 2
            while i < k:
                if not slm.judge_relevance(top[i], q, False, None):
                    break
                i += 1
            k = i
        ch = [n.node.text for n in top[:k]]
        if pol in ("rule", "ruleN"):
            cs = slm.judge_complexity(q)
            br = "E" if cs < complexity_threshold(k, 2, 10, lam=0.8) else ("T" if pol == "rule" else "N")
        elif pol == "allN":
            br = "N"
        elif pol == "tfidf_EN":
            br = "N" if float(ridge.predict(X([q], [k]))[0]) > thr[pol] else "E"
        else:
            enc = tok(
                [build_text({"query": q, "k": k, "chunks": ch}, tok, 6144, use_chunks=True)], return_tensors="pt"
            ).to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                s = float(dec(enc.input_ids, enc.attention_mask)[0, 3])
            br = "N" if s > thr[pol] else "E"
        sync()
        return br, ch

    for q in Q[:3]:  # warmup
        for pol in ["rule", "s1234_EN", "tfidf_EN"]:
            br, ch = local(pol, q[0], q[2])
            gen("E", q[2], ch)
            gen("N", q[2], ch)
    for rate in a.rates:
        arr = np.cumsum(np.random.default_rng(int(rate * 1000)).exponential(1 / rate, len(Q)))
        for pol in a.policies:
            fn = f"{a.out}/{pol}_r{rate:g}.jsonl"
            if os.path.exists(fn):
                continue
            res, lock = [], threading.Lock()
            inq = queue.Queue()
            pool = ThreadPoolExecutor(256)

            def finish(rec, branch, q, ch, refs):
                ans, ti, to, ok = gen(branch, q, ch)
                if not ok:
                    ans = gen("E", q, ch)[0]
                rec.update(
                    {
                        "t_done": time.perf_counter() - T0,
                        "answer": ans,
                        "f1": f1(ans or "", refs),
                        "cloud_in": ti,
                        "cloud_out": to,
                        "fallback": not ok,
                        "cost": (0.27 * ti + 1.1 * to) / 1e3,
                    }
                )
                with lock:
                    res.append(rec)

            def worker():
                while True:
                    it = inq.get()
                    if it is None:
                        return
                    i, t_arr = it
                    sp_pool, qid, q, refs = Q[i]
                    t_s = time.perf_counter() - T0
                    br, ch = local(pol, sp_pool, q)
                    t_e = time.perf_counter() - T0
                    rec = {
                        "i": i,
                        "run": sp_pool,
                        "id": qid,
                        "policy": pol,
                        "rate": rate,
                        "t_arr": t_arr,
                        "t_local_start": t_s,
                        "t_local_end": t_e,
                        "branch": br,
                    }
                    pool.submit(finish, rec, br, q, ch, refs)

            th = threading.Thread(target=worker)
            th.start()
            T0 = time.perf_counter()
            for i, t in enumerate(arr):
                d = T0 + t - time.perf_counter()
                if d > 0:
                    time.sleep(d)
                inq.put((i, float(t)))
            inq.put(None)
            th.join()
            pool.shutdown(wait=True)
            with open(fn, "w") as f:
                for rec in sorted(res, key=lambda x: x["i"]):
                    rec["latency"] = rec["t_done"] - rec["t_arr"]
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            L = np.array([r["latency"] for r in res])
            F = np.array([r["f1"] for r in res])
            print(
                time.strftime("%FT%TZ", time.gmtime()),
                f"rate {rate} {pol}: n {len(res)} F1 {F.mean():.4f} lat mean {L.mean():.2f} p50 {np.median(L):.2f} "
                f"p99 {np.percentile(L, 99):.2f} makespan {max(r['t_done'] for r in res):.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
