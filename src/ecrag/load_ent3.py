"""Open-loop load test: the deployed decider (vLLM pooling) vs general LLM selectors under
concurrent Poisson arrivals, any dataset, E/N/T. Selection and generation logic are those of e2e_ent5.py (same prompts,
thresholds, fallback/retry, cache_salt per request).
Serving (identical for every policy):
- --workers W local worker processes on one GPU (CUDA_VISIBLE_DEVICES), each with its own retriever embedder, reranker and pruning
  SLM, pulling from one shared FIFO queue: retrieve -> rerank -> dynamic pruning (+ the complexity call for rule policies).
- After the local stage the request is handed to a thread pool (non-blocking network clients): selector (vdec = HTTP /pooling to the
  vLLM decider + CPU MLP head; llm = Qwen3.8 cloud selector calls, which share the cloud server with answer generation; tfidf; rule;
  allN), then generation (edge E | cloud N | cloud T; truncated T -> retry per --fallback).
Policies (--pol name=spec, specs as e2e_ent5): vdec:<port>:<served>:<export dir>:a:b | llm:<zs|think|shared>:<probes>:a:b |
  tfidf:a:b | ruleN_tuned:lam:c | ruleT_tuned:lam:c | allN.
Arrivals: for each (rate, seed): question order = permutation(seed), exponential gaps (rng 1000*rate + seed); the SAME arrival
  sequence for every policy; the policy order is rotated per (rate, seed) cell.
--burst N: calibration mode (non-test data only): N requests at t=0 for each policy given, prints completion throughput.
Records per request: t_arr, t_local_start/end, t_sel_end, t_done, latency, branch, retry, answer, f1, cost (answer $), sel_cost.
Output: --out/<pol>_r<rate>_s<seed>.jsonl (+ config.json). Summary: analyze_load3.py."""

import argparse, json, os, sys, threading, time, uuid
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def worker_main(wid, a, inq, outq, ready):
    from e2e_run import rerank_es, sync, Settings, HuggingFaceEmbedding, slm, local_reranker, Retriever, EMB, RERANK
    from FlagEmbedding import FlagReranker
    from router_data import DATA

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMB)
    Settings.llm = None
    slm.init("Qwen/Qwen3-4B")
    local_reranker.reranker = FlagReranker(RERANK, use_fp16=True, devices=["cuda:0"])
    retr = {}
    for sp_pool in a.runs:
        sp, pool = sp_pool.split("/")
        retr[sp_pool] = Retriever(
            argparse.Namespace(
                docstore=f"{a.idx}/{a.ds}_{sp}_{pool}",
                similarity_top_k=40,
                bm25_similarity_top_k=40,
                use_kvcache=False,
                preload_kvcache=False,
            )
        )

    def local(sp_pool, q, need_cx):
        top = rerank_es(q, retr[sp_pool].fusion_retrieve(q), 10)
        k = min(10, len(top))
        if k > 2:
            i = 2
            while i < k:
                if not slm.judge_relevance(top[i], q, False, None):
                    break
                i += 1
            k = i
        cx = slm.judge_complexity(q) if need_cx else None
        sync()
        return k, [n.node.text for n in top[:k]], cx

    sp0, pool0 = a.runs[0].split("/")
    for l in list(open(f"{DATA}/{a.ds}/{sp0}/{pool0}/questions/questions.jsonl"))[:3]:
        local(a.runs[0], json.loads(l)["query"], True)
    ready.put(wid)
    while True:
        it = inq.get()
        if it is None:
            return
        i, sp_pool, q, need_cx = it
        qlen = inq.qsize()
        t_s = time.monotonic()
        k, ch, cx = local(sp_pool, q, need_cx)
        t_e = time.monotonic()
        outq.put((i, k, ch, cx, qlen, wid, t_s, t_e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pol", nargs="+", required=True)
    ap.add_argument("--fallback", default="E")
    ap.add_argument("--tfidf_train", default=None)
    ap.add_argument("--idx", required=True)
    ap.add_argument("--rates", nargs="+", type=float, default=[])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--burst", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cloud", default="http://127.0.0.1:8193/v1")
    ap.add_argument("--edge", default="http://127.0.0.1:8195/v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = {}
    for s in a.pol:
        nm, spec = s.split("=", 1)
        P[nm] = spec.split(":")
    json.dump(
        {"args": vars(a), "policies": P, "version": "load_ent3"},
        open(f"{a.out}/config{'_burst' if a.burst else ''}.json", "w"),
        indent=1,
    )
    import torch
    from openai import OpenAI
    from transformers import AutoTokenizer
    import requests
    from regen_answers import QP, QS, body, parse, VARIANTS, CFG
    from normalize_edge import normalize
    from router_data import f1, DATA
    from serve_opt import build_text_fast
    from llm_selector import prompt as sel_prompt, probs
    from llm_selector_shared import messages as shared_messages

    torch.set_num_threads(2)
    edge = OpenAI(base_url=a.edge, api_key="none", timeout=1800, max_retries=0)
    cloud = OpenAI(base_url=a.cloud, api_key="none", timeout=1800, max_retries=0)
    import queue

    TOKQ = queue.Queue()  # pre-loaded tokenizer + HTTP session slots (no per-thread loading inside timed requests)
    if any(sp[0] == "vdec" for sp in P.values()):
        for _ in range(48):
            TOKQ.put((AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"), requests.Session()))
    VH = {}
    for nm, sp in P.items():
        if sp[0] == "vdec":
            H = torch.load(f"{sp[3]}/head.pt")
            l1 = torch.nn.Linear(H["0.weight"].shape[1], H["0.weight"].shape[0])
            l2 = torch.nn.Linear(H["2.weight"].shape[1], H["2.weight"].shape[0])
            l1.load_state_dict({"weight": H["0.weight"], "bias": H["0.bias"]})
            l2.load_state_dict({"weight": H["2.weight"], "bias": H["2.bias"]})
            VH[nm] = torch.nn.Sequential(l1, torch.nn.GELU(), l2).eval()
    if any(sp[0] == "tfidf" for sp in P.values()):
        from analyze_en import load
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge

        trr = [r for s in a.tfidf_train.split(",") for r in load(s, "A")[0] if r["fT"] is not None]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([r["query"] for r in trr])
        X = lambda qs, ks: np.hstack([vec.transform(qs).toarray(), np.array([[k / 10] for k in ks])])
        RG = {
            h: Ridge(alpha=3.0).fit(X([r["query"] for r in trr], [r["k"] for r in trr]), [f(r) for r in trr])
            for h, f in [("dEN", lambda r: r["fN"] - r["fE"]), ("dNT", lambda r: r["fT"] - r["fN"])]
        }
        tf_lock = threading.Lock()

    def gen_E(q, ch, salt):
        r = edge.completions.create(
            model="qwen3-4b",
            prompt=QP + body(ch, q, VARIANTS["extract"]) + QS,
            temperature=0,
            max_tokens=CFG["extract"]["edge_max"],
            extra_body={"cache_salt": salt},
        )
        return normalize(q, parse(r.choices[0].text, CFG["extract"]["parse"]))[0]

    def gen_C(q, ch, think, salt):
        r = cloud.chat.completions.create(
            model="q38-flash-next-fp8",
            temperature=0,
            max_tokens=6144 if think else 256,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": body(ch, q, VARIANTS["orig_think" if think else "orig"]) + "\n"},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": think}, "cache_salt": salt},
        )
        c = r.choices[0]
        txt = c.message.content or ""
        ok = ("</think>" in txt or c.finish_reason == "stop") if think else True
        return parse(txt, None), (0.27 * r.usage.prompt_tokens + 1.1 * r.usage.completion_tokens) / 1e3, ok

    sel_pool = ThreadPoolExecutor(64)

    def llm_call(mode, kind, q, k, ch, salt):
        think = mode == "think"
        msgs = (
            shared_messages(kind, ch, q, k)
            if mode == "shared"
            else [{"role": "user", "content": sel_prompt(kind, {"chunks": ch, "query": q, "k": k})}]
        )
        r = cloud.chat.completions.create(
            model="q38-flash-next-fp8",
            temperature=0,
            max_tokens=3072 if think else 1,
            logprobs=True,
            top_logprobs=20,
            messages=msgs,
            extra_body={"chat_template_kwargs": {"enable_thinking": think}, "cache_salt": salt},
        )
        c = r.choices[0]
        lc = c.logprobs.content if c.logprobs and c.logprobs.content else []
        if not think:
            p, _ = probs(lc[0].top_logprobs if lc else [], ["Yes", "No"])
        else:
            toks = [x.token for x in lc]
            st = toks.index("</think>") + 1 if "</think>" in toks else len(toks)
            p = {"Yes": 0.5, "No": 0.5}
            for x in lc[st:]:
                if x.token.strip().strip(".:)*").lower() in ("yes", "no"):
                    p, _ = probs(x.top_logprobs, ["Yes", "No"])
                    break
        return p, (0.27 * r.usage.prompt_tokens + 1.1 * r.usage.completion_tokens) / 1e3

    def select(nm, q, k, ch, cx, salt):
        sp = P[nm]
        kind = sp[0]
        sc = 0.0
        if kind in ("ruleN_tuned", "ruleT_tuned"):
            lam, c = float(sp[1]), float(sp[2])
            return (kind[4] if cx - (lam - (2 * lam - 1) * (k - 2) / 8) > c else "E"), sc
        if kind == "allN":
            return "N", sc
        if kind == "tfidf":
            with tf_lock:
                x = X([q], [k])
                dEN, dNT = float(RG["dEN"].predict(x)[0]), float(RG["dNT"].predict(x)[0])
            ta, tb = float(sp[1]), float(sp[2])
        elif kind == "vdec":
            tk, ss = TOKQ.get()
            try:
                ids = tk(build_text_fast({"query": q, "k": k, "chunks": ch}, tk, 6144)).input_ids
                j = ss.post(
                    f"http://127.0.0.1:{sp[1]}/pooling",
                    json={"model": sp[2], "input": [ids], "cache_salt": salt},
                    timeout=600,
                ).json()
            finally:
                TOKQ.put((tk, ss))
            with torch.no_grad():
                o = VH[nm](torch.tensor(j["data"][0]["data"], dtype=torch.float32)[None])[0].tolist()
            dEN, dNT = o[3], o[4]
            ta, tb = float(sp[4]), float(sp[5])
        else:  # llm
            mode, kinds = sp[1], sp[2].split("+")
            R = list(sel_pool.map(lambda kd: llm_call(mode, kd, q, k, ch, salt), kinds))
            sc = sum(x[1] for x in R)
            dEN = R[0][0]["No"]
            dNT = R[1][0]["Yes"] if len(R) > 1 else 0.0
            ta, tb = float(sp[3]), float(sp[4])
        return ("T" if (dEN > ta and dNT > tb) else ("N" if dEN > ta else "E")), sc

    Q = []
    for sp_pool in a.runs:
        sp, pool = sp_pool.split("/")
        refs = {
            json.loads(l)["id"]: json.loads(l)["answers"]
            for l in open(f"{DATA}/{a.ds}/{sp}/{pool}/answers/answers.jsonl")
        }
        qs = [json.loads(l) for l in open(f"{DATA}/{a.ds}/{sp}/{pool}/questions/questions.jsonl")]
        Q += [(sp_pool, x["id"], x["query"], refs[x["id"]]) for x in (qs[: a.limit] if a.limit else qs)]
    ctx = mp.get_context("spawn")
    inq, outq, ready = ctx.Queue(), ctx.Queue(), ctx.Queue()
    W = [ctx.Process(target=worker_main, args=(w, a, inq, outq, ready), daemon=True) for w in range(a.workers)]
    for p in W:
        p.start()
    for _ in W:
        ready.get()
    for nm in P:  # warm the network paths of every policy (not timed)
        q = Q[0][2]
        select(nm, q, 3, ["warmup"] * 3, 0.5, uuid.uuid4().hex)
    print(time.strftime("%FT%TZ", time.gmtime()), f"{a.workers} workers ready", flush=True)
    pool_ex = ThreadPoolExecutor(1024)

    def one_run(nm, order, arr, fn):
        res, lock = [], threading.Lock()
        T0 = time.monotonic()
        need_cx = P[nm][0].startswith("rule")

        def finish(rec, q, k, ch, cx, refs):
            salt = uuid.uuid4().hex
            try:
                br, sc = select(nm, q, k, ch, cx, salt)
                rec["t_sel_end"] = time.monotonic() - T0
                cost = 0.0
                retry = None
                if br == "E":
                    ans = gen_E(q, ch, salt)
                else:
                    ans, cost, ok = gen_C(q, ch, br == "T", salt)
                    if not ok:
                        if a.fallback == "N":
                            ans, c2, _ = gen_C(q, ch, False, salt)
                            cost += c2
                            retry = "N"
                        else:
                            ans = gen_E(q, ch, salt)
                            retry = "E"
                rec.update(
                    {
                        "branch": br,
                        "retry": retry,
                        "answer": ans,
                        "f1": f1(ans or "", refs),
                        "cost": cost,
                        "sel_cost": sc,
                        "total_cost": cost + sc,
                    }
                )
            except Exception as e:  # noqa: BLE001
                rec.update(
                    {"error": repr(e)[:300], "f1": 0.0, "cost": 0.0, "sel_cost": 0.0, "total_cost": 0.0, "branch": None}
                )
            rec["t_done"] = time.monotonic() - T0
            rec["latency"] = rec["t_done"] - rec["t_arr"]
            with lock:
                res.append(rec)

        def collector():
            for _ in range(len(order)):
                j, k, ch, cx, qlen, wid, t_s, t_e = outq.get()
                sp_pool, qid, q, refs = Q[order[j]]
                rec = {
                    "i": int(order[j]),
                    "pos": j,
                    "run": sp_pool,
                    "id": qid,
                    "policy": nm,
                    "t_arr": float(arr[j]),
                    "t_local_start": t_s - T0,
                    "t_local_end": t_e - T0,
                    "k": k,
                    "qlen": qlen,
                    "worker": wid,
                }
                pool_ex.submit(finish, rec, q, k, ch, cx, refs)

        th = threading.Thread(target=collector)
        th.start()
        for j, t in enumerate(arr):
            d = T0 + t - time.monotonic()
            if d > 0:
                time.sleep(d)
            sp_pool, qid, q, refs = Q[order[j]]
            inq.put((j, sp_pool, q, need_cx))
        th.join()
        while True:
            with lock:
                if len(res) == len(order):
                    break
            time.sleep(0.2)
        with open(fn, "w") as f:
            for rec in sorted(res, key=lambda x: x["pos"]):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        L = np.array([r["latency"] for r in res])
        F = np.array([r["f1"] for r in res])
        ms = max(r["t_done"] for r in res)
        print(
            time.strftime("%FT%TZ", time.gmtime()),
            f"{os.path.basename(fn)}: n {len(res)} err {sum('error' in r for r in res)} F1 {F.mean():.4f} lat mean {L.mean():.2f} "
            f"p50 {np.median(L):.2f} p95 {np.percentile(L, 95):.2f} makespan {ms:.1f}s thr {len(res) / ms:.2f}/s",
            flush=True,
        )
        time.sleep(5)

    names = list(P)
    cell = 0
    if a.burst:
        order = np.random.default_rng(0).permutation(len(Q))[: a.burst]
        arr = np.zeros(len(order))
        for nm in names:
            fn = f"{a.out}/burst_{nm}_n{a.burst}.jsonl"
            if not os.path.exists(fn):
                one_run(nm, order, arr, fn)
    for rate in a.rates:
        for seed in a.seeds:
            order = np.random.default_rng(seed).permutation(len(Q))
            arr = np.cumsum(np.random.default_rng(int(rate * 1000) + seed).exponential(1 / rate, len(Q)))
            rot = names[cell % len(names) :] + names[: cell % len(names)]
            cell += 1
            for nm in rot:
                fn = f"{a.out}/{nm}_r{rate:g}_s{seed}.jsonl"
                if not os.path.exists(fn):
                    one_run(nm, order, arr, fn)
    for _ in W:
        inq.put(None)
    for p in W:
        p.join(timeout=60)


if __name__ == "__main__":
    main()
