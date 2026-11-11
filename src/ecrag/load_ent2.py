"""Open-loop load test with configurable local serving. Same arrival model, questions, thresholds and generation
as load_ent.py; the local stages are served differently, identically for every policy:
- --workers W: W local worker processes on the one edge GPU, each with its own copy of the local models (retriever embedder,
  reranker, pruning SLM, decider), pulling from one shared FIFO request queue (load_ent.py = one worker, i.e. W=1 without --opt).
- --opt: decider text built with serve_opt.build_text_fast (identical string) and the decider LoRA merged into the base
  (serve_opt.merge_decider; E/N decision agreement 99-99.5% on test4, verify_serve_opt.py). Pruning is the reference sequential loop.
Policies: s1234_EN, tfidf_EN, ruleN_tuned (lam/c from pilot7/rule_tuned.json x1), rule (rule->T), ruleN, allN, and
adaptQ<q>: the s1234_EN decider when the shared local queue holds fewer than q waiting requests at the start of local work, else
the tfidf_EN selector (thresholds unchanged).
selU<u> (u in 25/50/75): TF-IDF first, the decider only when the TF-IDF score is within the val
u%-quantile of |score - threshold| (ambiguous); adaptS: decider when the queue is empty, else selU50. 'selector' records the one used. Output: --out/{pol}_r{rate}.jsonl (same records as load_ent.py plus worker, qlen, selector).
Summary: analyze_load.py --dir."""

from paths import RUNS  # noqa: E402
import argparse, json, os, sys, time, threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
R = RUNS
P7 = f"{R}/pilot7"


def worker_main(wid, a, inq, outq, ready):
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
    )
    import torch
    from transformers import AutoTokenizer
    from train_router import build_text
    from e2e_ent2 import load_decider
    from serve_opt import build_text_fast, merge_decider
    from router_data import DATA
    import ent_data
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    PJ = json.load(open(f"{P7}/ent_policy.json"))
    RT = json.load(open(f"{P7}/rule_tuned.json"))["policies"]["ruleN_tuned|x1.0"]
    thr = {
        "s1234_EN": PJ["policies"]["s1234_qd|E/N|x1.0"]["a_dEN"],
        "tfidf_EN": PJ["policies"]["tfidf|E/N|x1.0"]["a_dEN"],
    }
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMB)
    Settings.llm = None
    slm.init("Qwen/Qwen3-4B")
    from FlagEmbedding import FlagReranker

    local_reranker.reranker = FlagReranker(RERANK, use_fp16=True, devices=["cuda:0"])
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    dec = load_decider(f"{P7}/router/ent_qd_s1234")
    if a.opt:
        merge_decider(dec)
    tr = ent_data.load("train", "A")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in tr])
    X = lambda qs, ks: np.hstack([vec.transform(qs).toarray(), np.array([[k / 10] for k in ks])])
    ridge = Ridge(alpha=3.0).fit(X([r["query"] for r in tr], [r["k"] for r in tr]), [r["fN"] - r["fE"] for r in tr])
    vrows = ent_data.load("val", "A")
    vd = np.abs(ridge.predict(X([r["query"] for r in vrows], [r["k"] for r in vrows])) - thr["tfidf_EN"])
    DELTA = {
        u: float(np.quantile(vd, u / 100)) for u in (25, 50, 75)
    }  # selU<u>: decider share u% on val (selective_probe.py)
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

    def local(pol, sp_pool, q, qlen):
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
        sel = pol
        if pol.startswith("adaptQ"):
            sel = "s1234_EN" if qlen < int(pol[6:]) else "tfidf_EN"
        if pol == "adaptS":  # idle queue: decider; else selective (decider only on the ambiguous half)
            sel = "s1234_EN" if qlen < 1 else "selU50"
        if sel.startswith("selU"):
            st = float(ridge.predict(X([q], [k]))[0])
            sel = "s1234_EN" if abs(st - thr["tfidf_EN"]) <= DELTA[int(sel[4:])] else "tfidf_EN"
        if sel in ("rule", "ruleN"):
            cs = slm.judge_complexity(q)
            br = "E" if cs < complexity_threshold(k, 2, 10, lam=0.8) else ("T" if sel == "rule" else "N")
        elif sel == "ruleN_tuned":
            cs = slm.judge_complexity(q)
            lam = RT["lam"]
            br = "N" if cs - (lam - (2 * lam - 1) * (k - 2) / 8) > RT["c"] else "E"
        elif sel == "allN":
            br = "N"
        elif sel == "tfidf_EN":
            br = "N" if float(ridge.predict(X([q], [k]))[0]) > thr["tfidf_EN"] else "E"
        else:
            r = {"query": q, "k": k, "chunks": ch}
            txt = build_text_fast(r, tok, 6144) if a.opt else build_text(r, tok, 6144, use_chunks=True)
            enc = tok([txt], return_tensors="pt").to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                s = float(dec(enc.input_ids, enc.attention_mask)[0, 3])
            br = "N" if s > thr["s1234_EN"] else "E"
        sync()
        return br, ch, sel

    qs = {}
    for sp_pool in a.runs:
        sp, pool = sp_pool.split("/")
        qs[sp_pool] = [json.loads(l)["query"] for l in open(f"{DATA}/hotpotqa/{sp}/{pool}/questions/questions.jsonl")][
            :3
        ]
    for sp_pool, ql in qs.items():  # warmup
        for q in ql:
            for pol in ["rule", "s1234_EN", "tfidf_EN"]:
                local(pol, sp_pool, q, 0)
    ready.put(wid)
    while True:
        it = inq.get()
        if it is None:
            return
        i, pol, sp_pool, q = it
        qlen = inq.qsize()
        t_s = time.monotonic()
        br, ch, sel = local(pol, sp_pool, q, qlen)
        t_e = time.monotonic()
        outq.put((i, br, ch, sel, qlen, wid, t_s, t_e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--rates", nargs="+", type=float, required=True)
    ap.add_argument("--policies", nargs="+", default=["s1234_EN", "tfidf_EN", "ruleN_tuned", "rule", "allN"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--opt", action="store_true")
    ap.add_argument("--idx", default=f"{R}/pilot1/trace/_index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--arr_seed",
        type=int,
        default=0,
        help="arrival/order seed offset (0 = as before); >0 also shuffles the request order",
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump({"args": vars(a), "version": "load_ent2"}, open(f"{a.out}/config.json", "w"), indent=1)
    from openai import OpenAI
    from regen_answers import QP, QS, body, parse, VARIANTS, CFG
    from normalize_edge import normalize
    from router_data import f1, DATA

    edge = OpenAI(base_url="http://127.0.0.1:8195/v1", api_key="none", timeout=1200, max_retries=0)
    cloud = OpenAI(base_url="http://127.0.0.1:8193/v1", api_key="none", timeout=1200, max_retries=0)
    Q = []
    for sp_pool in a.runs:
        sp, pool = sp_pool.split("/")
        refs = {
            json.loads(l)["id"]: json.loads(l)["answers"]
            for l in open(f"{DATA}/hotpotqa/{sp}/{pool}/answers/answers.jsonl")
        }
        qs = [json.loads(l) for l in open(f"{DATA}/hotpotqa/{sp}/{pool}/questions/questions.jsonl")]
        Q += [(sp_pool, x["id"], x["query"], refs[x["id"]]) for x in (qs[: a.limit] if a.limit else qs)]
    if a.arr_seed:
        Q = [Q[j] for j in np.random.default_rng(a.arr_seed).permutation(len(Q))]
    ctx = mp.get_context("spawn")
    inq, outq, ready = ctx.Queue(), ctx.Queue(), ctx.Queue()
    W = [ctx.Process(target=worker_main, args=(w, a, inq, outq, ready), daemon=True) for w in range(a.workers)]
    for p in W:
        p.start()
    for _ in W:
        ready.get()
    print(time.strftime("%FT%TZ", time.gmtime()), f"{a.workers} workers ready", flush=True)

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

    for rate in a.rates:
        arr = np.cumsum(np.random.default_rng(int(rate * 1000) + a.arr_seed).exponential(1 / rate, len(Q)))
        for pol in a.policies:
            fn = f"{a.out}/{pol}_r{rate:g}.jsonl"
            if os.path.exists(fn):
                continue
            res, lock = [], threading.Lock()
            pool = ThreadPoolExecutor(256)
            T0 = time.monotonic()

            def finish(rec, branch, q, ch, refs):
                ans, ti, to, ok = gen(branch, q, ch)
                if not ok:
                    ans = gen("E", q, ch)[0]
                rec.update(
                    {
                        "t_done": time.monotonic() - T0,
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

            def collector():
                for _ in range(len(Q)):
                    i, br, ch, sel, qlen, wid, t_s, t_e = outq.get()
                    sp_pool, qid, q, refs = Q[i]
                    rec = {
                        "i": i,
                        "run": sp_pool,
                        "id": qid,
                        "policy": pol,
                        "rate": rate,
                        "t_arr": float(arr[i]),
                        "t_local_start": t_s - T0,
                        "t_local_end": t_e - T0,
                        "branch": br,
                        "selector": sel,
                        "qlen": qlen,
                        "worker": wid,
                    }
                    pool.submit(finish, rec, br, q, ch, refs)

            th = threading.Thread(target=collector)
            th.start()
            for i, t in enumerate(arr):
                d = T0 + t - time.monotonic()
                if d > 0:
                    time.sleep(d)
                sp_pool, qid, q, refs = Q[i]
                inq.put((i, pol, sp_pool, q))
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
            time.sleep(5)
    for _ in W:
        inq.put(None)
    for p in W:
        p.join(timeout=60)


if __name__ == "__main__":
    main()
