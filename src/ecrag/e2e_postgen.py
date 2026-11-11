# Policy spec learnedpost:<E|N>:gain_threshold:port:served_name:export_dir (see the module docstring).
"""Integrated single-user run with replica-set ports and post-generation policies, on top of e2e_ent5.py.
--cloud_port/--edge_port select the replica set. Policy automix:<E|N>:<thr>[:<k samples, default 8>] = AutoMix-style
self-verification cascade (automix_verify.py: official 4-shot prompt, n=8, T=1.0, max 250, stop ---). E: edge answer -> edge
self-verification -> direct cloud call (N) iff score < thr. N: N answer -> cloud self-verification -> reasoning call (T) iff score < thr;
a truncated T returns the N answer already produced (no extra call). Every call is timed and every cloud call is charged (cost =
answer + verification + escalation); verification tokens/time recorded separately (ver_*). Thresholds come from validation.
Policy learnedpost:<E|N>:gain_threshold:port:served_name:export_dir = the same cascade with a learned post-generation gain
score (served by vLLM pooling + MLP head) instead of self-verification; escalate iff gain > gain_threshold.
Base (e2e_ent5): e2e_ent4.py + (1) policy interleaving: --block B questions per block, every policy runs the same block before
the next block starts, and the policy order is rotated by one each block, so slow drifts of the machine state hit all policies
alike; (2) decopt:<ckpt>:a:b = the same decider with serve_opt (build_text_fast + merged LoRA), i.e. e2e_ent4 --opt per
policy; (3) vdec:<port>:<served name>:<export dir>:a:b = the decider served by vLLM (pooling LAST, export_decider_vllm.py), client:
build_text_fast + tokenise + POST /pooling (with the request's cache_salt) + MLP head (head.pt) on CPU; thresholds are the HF decider's.
Decider sub-timings are recorded per query (decider_parts).
Base (e2e_ent4): integrated single-user run with live general-LLM selectors and prefix-cache accounting, on top of e2e_ent3.py.
New policy kind llm:<mode>:<probes>:a:b, mode zs (llm_selector.PROMPTS, thinking off, 1 token) | think
(same prompts, thinking on, label after </think>, max 3072 tokens) | shared (llm_selector_shared: answer-prompt prefix + selector
question); probes edge | edge+think (concurrent calls). Every request carries one fresh cache_salt on the edge and cloud servers (no
reuse across policies or questions; within a request the selector and answer calls share it, so only within-request reuse can hit).
Cloud prefix-cache hits per request = delta of vllm:prefix_cache_hits_total on :8193 (single user, sequential: exact attribution):
sel_cached (selector calls) and ans_cached (answer calls). Records sel_cost/sel_prompt_tokens/sel_completion_tokens and the
answer prompt tokens (ans_prompt_tokens) so costs with a cached-input price can be recomputed offline.
Base (e2e_ent3): e2e_ent2.py generalized to any dataset and E/N/T policies, including the T -> N retry after a truncated T.
Per query, all stages live and timed with a
wall clock: retrieve -> rerank -> dynamic pruning (reference sequential slm.judge_relevance) -> selector -> generation.
--ds hotpotqa|musique|2wiki (index <idx>/<ds>_<split>_<pool>, data data/ecrag/<ds>), --fallback E|N (truncated T: edge answer,
or a second cloud call with N; the cost and the time of both calls are counted). --opt: serve_opt (build_text_fast + merged LoRA).
Policies (--pol name=spec, thresholds copied from an analyze_ent2 output json, nothing is tuned here):
  rule | ruleN (EC-RAG lam 0.8, cloud = T | N); ruleT_tuned:lam:c | ruleN_tuned:lam:c; allN | allT;
  tfidf:a:b (Ridge on --tfidf_train question 1-2 grams + k, heads dEN and dNT, as analyze_ent2); dec:<ckpt dir>:a:b (q+D+k decider).
  E/N/T rule for scorers: cloud iff dEN > a; T iff cloud and dNT > b.
Output: --out/<split>_<pool>_<name>_r0.jsonl (wall, per-stage times, branch, retry, cost, F1)."""
import argparse, json, os, re, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor
import requests

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
from serve_opt import build_text_fast, merge_decider  # noqa: E402
from llm_selector import prompt as sel_prompt, probs  # noqa: E402
from llm_selector_shared import messages as shared_messages  # noqa: E402
from automix_verify import verify as am_verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pol", nargs="+", required=True)
    ap.add_argument("--fallback", default="E")
    ap.add_argument("--opt", action="store_true")
    ap.add_argument("--tfidf_train", default=None)
    ap.add_argument("--idx", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--block", type=int, default=25)
    ap.add_argument("--cloud_port", type=int, default=8193)
    ap.add_argument("--edge_port", type=int, default=8195)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = {}
    for s in a.pol:
        nm, spec = s.split("=", 1)
        P[nm] = spec.split(":")
    json.dump(
        {"args": vars(a), "policies": P, "version": "e2e_postgen_p0"}, open(f"{a.out}/config.json", "w"), indent=1
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMB)
    Settings.llm = None
    slm.init("Qwen/Qwen3-4B")
    from FlagEmbedding import FlagReranker

    local_reranker.reranker = FlagReranker(RERANK, use_fp16=True, devices=["cuda:0"])
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    D = {}
    VH = {}
    PH = {}
    for nm, sp in P.items():
        if sp[0] == "vdec":
            H = torch.load(f"{sp[3]}/head.pt")
            lin1 = torch.nn.Linear(H["0.weight"].shape[1], H["0.weight"].shape[0])
            lin2 = torch.nn.Linear(H["2.weight"].shape[1], H["2.weight"].shape[0])
            lin1.load_state_dict({"weight": H["0.weight"], "bias": H["0.bias"]})
            lin2.load_state_dict({"weight": H["2.weight"], "bias": H["2.bias"]})
            VH[nm] = torch.nn.Sequential(lin1, torch.nn.GELU(), lin2).eval()
        if sp[0] == "decopt":
            ck = sp[1]
            n_out = json.load(open(f"{ck}/args.json")).get("n_out", 5) if os.path.exists(f"{ck}/args.json") else 5
            D[nm] = merge_decider(load_decider(ck, n_out))
        if sp[0] == "dec":
            ck = sp[1]
            n_out = json.load(open(f"{ck}/args.json")).get("n_out", 5) if os.path.exists(f"{ck}/args.json") else 5
            D[nm] = load_decider(ck, n_out)
            if a.opt:
                merge_decider(D[nm])
    for nm, sp in P.items():
        if sp[0] == "learnedpost":
            H = torch.load(f"{sp[5]}/head.pt", map_location="cpu")
            PH[nm] = torch.nn.Sequential(
                torch.nn.Linear(H["0.weight"].shape[1], H["0.weight"].shape[0]),
                torch.nn.GELU(),
                torch.nn.Linear(H["2.weight"].shape[1], H["2.weight"].shape[0]),
            ).eval()
            PH[nm].load_state_dict(H)
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
    edge = OpenAI(base_url=f"http://127.0.0.1:{a.edge_port}/v1", api_key="none", timeout=600)
    cloud = OpenAI(base_url=f"http://127.0.0.1:{a.cloud_port}/v1", api_key="none", timeout=1200)

    HITS = re.compile(r"^vllm:prefix_cache_hits_total\{[^}]*\} ([0-9.e+]+)", re.M)
    ms = requests.Session()

    def hits():
        return sum(float(x) for x in HITS.findall(ms.get(f"http://127.0.0.1:{a.cloud_port}/metrics", timeout=10).text))

    SALT = {"v": None}

    def gen_E(q, ch):
        t0 = time.perf_counter()
        r = edge.completions.create(
            model="qwen3-4b",
            prompt=QP + body(ch, q, VARIANTS["extract"]) + QS,
            temperature=0,
            max_tokens=CFG["extract"]["edge_max"],
            extra_body={"cache_salt": SALT["v"]} if SALT["v"] else None,
        )
        return normalize(q, parse(r.choices[0].text, CFG["extract"]["parse"]))[0], time.perf_counter() - t0

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
            extra_body={
                "chat_template_kwargs": {"enable_thinking": think},
                **({"cache_salt": SALT["v"]} if SALT["v"] else {}),
            },
        )
        CU["ans_prompt_tokens"] = CU.get("ans_prompt_tokens", 0) + r.usage.prompt_tokens
        c = r.choices[0]
        txt = c.message.content or ""
        ok = ("</think>" in txt or c.finish_reason == "stop") if think else True
        return (
            parse(txt, None),
            time.perf_counter() - t0,
            (0.27 * r.usage.prompt_tokens + 1.1 * r.usage.completion_tokens) / 1e3,
            r.usage.completion_tokens,
            ok,
        )

    CU = {}
    DP = {}  # decider sub-timings of the last scorer call: text, tok(+H2D), model (forward | HTTP pooling), head
    vs = requests.Session()
    pool_ex = ThreadPoolExecutor(4)

    def llm_call(mode, kind, q, k, ch):
        think = mode == "think"
        t0 = time.perf_counter()
        msgs = (
            shared_messages(kind, ch, q, k)
            if mode == "shared"
            else [{"role": "user", "content": sel_prompt(kind, {"chunks": ch, "query": q, "k": k})}]
        )
        eb = {"chat_template_kwargs": {"enable_thinking": think}}
        if SALT["v"]:
            eb["cache_salt"] = SALT["v"]
        r = cloud.chat.completions.create(
            model="q38-flash-next-fp8",
            temperature=0,
            max_tokens=3072 if think else 1,
            logprobs=True,
            top_logprobs=20,
            messages=msgs,
            extra_body=eb,
        )
        c = r.choices[0]
        lc = c.logprobs.content if c.logprobs and c.logprobs.content else []
        found = True
        if not think:
            p, _ = probs(lc[0].top_logprobs if lc else [], ["Yes", "No"])
        else:
            toks = [x.token for x in lc]
            st = toks.index("</think>") + 1 if "</think>" in toks else len(toks)
            found = False
            p = {"Yes": 0.5, "No": 0.5}
            for x in lc[st:]:
                if x.token.strip().strip(".:)*").lower() in ("yes", "no"):
                    p, _ = probs(x.top_logprobs, ["Yes", "No"])
                    found = True
                    break
        return p, r.usage.prompt_tokens, r.usage.completion_tokens, found, time.perf_counter() - t0

    def llm_scorer(sp, q, k, ch, T, o):
        mode, kinds = sp[1], sp[2].split("+")
        t0 = time.perf_counter()
        R = list(pool_ex.map(lambda kd: llm_call(mode, kd, q, k, ch), kinds))
        T["selector"] = time.perf_counter() - t0
        pe = R[0][0]
        pt = R[1][0] if len(R) > 1 else {"Yes": 0.0}
        o["sel_prompt_tokens"] = [x[1] for x in R]
        o["sel_completion_tokens"] = [x[2] for x in R]
        o["sel_found"] = [x[3] for x in R]
        o["sel_call_sec"] = [x[4] for x in R]
        o["sel_cost"] = sum(0.27 * x[1] + 1.1 * x[2] for x in R) / 1e3
        return (pe["No"], pt["Yes"]), float(sp[3]), float(sp[4])

    def scorer(nm, q, k, ch, T):
        sp = P[nm]
        if sp[0] == "tfidf":
            t0 = time.perf_counter()
            x = X([q], [k])
            s = (float(RG["dEN"].predict(x)[0]), float(RG["dNT"].predict(x)[0]))
            T["selector"] = time.perf_counter() - t0
            return s, float(sp[1]), float(sp[2])
        if sp[0] == "vdec":
            t0 = time.perf_counter()
            txt = build_text_fast({"query": q, "k": k, "chunks": ch}, tok, 6144)
            t1 = time.perf_counter()
            ids = tok(txt).input_ids
            t2 = time.perf_counter()
            j = vs.post(
                f"http://127.0.0.1:{sp[1]}/pooling",
                json={"model": sp[2], "input": [ids], **({"cache_salt": SALT["v"]} if SALT["v"] else {})},
                timeout=60,
            ).json()
            t3 = time.perf_counter()
            with torch.no_grad():
                o = VH[nm](torch.tensor(j["data"][0]["data"], dtype=torch.float32)[None])[0].tolist()
            T["decider"] = time.perf_counter() - t0
            DP.update(text=t1 - t0, tok=t2 - t1, model=t3 - t2, head=T["decider"] - (t3 - t0), n_tok=len(ids))
            return (o[3], o[4]), float(sp[4]), float(sp[5])
        sync()
        t0 = time.perf_counter()
        r = {"query": q, "k": k, "chunks": ch}
        txt = (
            build_text_fast(r, tok, 6144) if (a.opt or sp[0] == "decopt") else build_text(r, tok, 6144, use_chunks=True)
        )
        t1 = time.perf_counter()
        enc = tok([txt], return_tensors="pt").to("cuda")
        sync()
        t2 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = D[nm](enc.input_ids, enc.attention_mask)[0].float().tolist()
        sync()
        T["decider"] = time.perf_counter() - t0
        DP.update(
            text=t1 - t0, tok=t2 - t1, model=T["decider"] - (t2 - t0), head=0.0, n_tok=int(enc.input_ids.shape[1])
        )
        return (o[3], o[4]), float(sp[2]), float(sp[3])

    def learned_post_score(nm, q, k, ch, ans, T):
        from postgen_input import build_postgen_text

        sp = P[nm]
        t0 = time.perf_counter()
        txt = build_postgen_text(
            {"query": q, "k": k, "chunks": ch, "candidate": ans, "candidate_plan": "E" if sp[1] == "E" else "C"},
            tok,
            lambda r, t, n, **kw: build_text_fast(r, t, n),
        )
        ids = tok(txt).input_ids
        t1 = time.perf_counter()
        response = vs.post(
            f"http://127.0.0.1:{sp[3]}/pooling",
            json={"model": sp[4], "input": [ids], **({"cache_salt": SALT["v"]} if SALT["v"] else {})},
            timeout=60,
        )
        response.raise_for_status()
        j = response.json()
        with torch.no_grad():
            value = PH[nm](torch.tensor(j["data"][0]["data"], dtype=torch.float32)[None])[0].tolist()
        T["verify"] = time.perf_counter() - t0
        return value[3 if sp[1] == "E" else 4], len(ids)

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
    sp0, pool0 = a.runs[0].split("/")
    for l in list(open(f"{DATA}/{a.ds}/{sp0}/{pool0}/questions/questions.jsonl"))[:3]:  # warmup
        q = json.loads(l)["query"]
        top = rerank_es(q, retr[a.runs[0]].fusion_retrieve(q), 10)
        slm.judge_relevance(top[2], q, False, None)
        slm.judge_complexity(q)
        ch = [n.node.text for n in top[:3]]
        for nm in list(D) + list(VH):
            scorer(nm, q, 3, ch, {})
        for sp in P.values():
            if sp[0] == "llm":
                llm_scorer(sp, q, 3, ch, {}, {})
        gen_E(q, ch)
        gen_C(q, ch, False)
        for sp in P.values():
            if sp[0] == "automix":
                am_verify(
                    edge if sp[1] == "E" else cloud,
                    "qwen3-4b" if sp[1] == "E" else "q38-flash-next-fp8",
                    ch,
                    q,
                    "unknown",
                    "warmup",
                )
    for nm in PH:
        learned_post_score(nm, q, 3, ch, "unknown", {})
    sync()
    names = list(P)
    blk = 0
    for sp_pool in a.runs:
        spl, pool = sp_pool.split("/")
        qs_all = [json.loads(l) for l in open(f"{DATA}/{a.ds}/{spl}/{pool}/questions/questions.jsonl")]
        qs_all = qs_all[: a.limit] if a.limit else qs_all
        refs = {
            json.loads(l)["id"]: json.loads(l)["answers"]
            for l in open(f"{DATA}/{a.ds}/{spl}/{pool}/answers/answers.jsonl")
        }
        tp = f"{a.trace}/{spl}/{pool}/trace.jsonl"
        trace = {json.loads(l)["id"]: json.loads(l) for l in open(tp)} if os.path.exists(tp) else {}
        for b0 in range(0, len(qs_all), a.block):
            order = names[blk % len(names) :] + names[: blk % len(names)]
            blk += 1
            for nm in order:
                sp = P[nm]
                fn = f"{a.out}/{spl}_{pool}_{nm}_r0.jsonl"
                done = {json.loads(l)["id"] for l in open(fn)} if os.path.exists(fn) else set()
                qs = qs_all[b0 : b0 + a.block]
                with open(fn, "a") as f:
                    for qq in qs:
                        if qq["id"] in done:
                            continue
                        o_block = blk - 1
                        q = qq["query"]
                        T = {}
                        o = {"id": qq["id"], "policy": nm, "sel_cost": 0.0}
                        SALT["v"] = uuid.uuid4().hex
                        CU.clear()
                        h0 = hits()
                        sync()
                        w0 = time.perf_counter()
                        nodes = retr[sp_pool].fusion_retrieve(q)
                        T["retrieve"] = time.perf_counter() - w0
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
                        tr = trace.get(qq["id"], {})
                        o.update(
                            {
                                "k": k,
                                "k_trace": tr.get("k"),
                                "topk_same_as_trace": [n.node.node_id for n in top] == tr.get("topk_ids"),
                            }
                        )
                        kind = sp[0]
                        if kind in ("rule", "ruleN", "ruleT_tuned", "ruleN_tuned"):
                            t0 = time.perf_counter()
                            cs = slm.judge_complexity(q)
                            sync()
                            T["complexity"] = time.perf_counter() - t0
                            o["complexity"] = cs
                            if kind.endswith("_tuned"):
                                lam, c = float(sp[1]), float(sp[2])
                                cl = cs - (lam - (2 * lam - 1) * (k - 2) / 8) > c
                                branch = kind[4] if cl else "E"
                            else:
                                branch = (
                                    "E"
                                    if cs < complexity_threshold(k, 2, 10, lam=0.8)
                                    else ("T" if kind == "rule" else "N")
                                )
                        elif kind in ("allN", "allT"):
                            branch = kind[3]
                        elif kind in ("automix", "learnedpost"):
                            branch = None
                        elif kind == "llm":
                            w1 = time.perf_counter()
                            (dEN, dNT), ta, tb = llm_scorer(sp, q, k, ch, T, o)
                            o["dEN"], o["dNT"] = dEN, dNT
                            hs = hits()
                            o["sel_cached"] = hs - h0
                            w0 += time.perf_counter() - w1 - T["selector"]
                            h0 = hs  # metrics read excluded from wall
                            branch = "T" if (dEN > ta and dNT > tb) else ("N" if dEN > ta else "E")
                        else:
                            DP.clear()
                            (dEN, dNT), ta, tb = scorer(nm, q, k, ch, T)
                            o["dEN"], o["dNT"] = dEN, dNT
                            if DP:
                                o["decider_parts"] = dict(DP)
                            branch = "T" if (dEN > ta and dNT > tb) else ("N" if dEN > ta else "E")
                        o["cost"] = 0.0
                        o["retry"] = None
                        if kind == "learnedpost":
                            cand, thr = sp[1], float(sp[2])
                            if cand == "E":
                                ans, T["edge"] = gen_E(q, ch)
                            else:
                                ans, T["cloud"], c1, o["cloud_out_tokens"], _ = gen_C(q, ch, False)
                                o["cost"] = c1
                            gain, ntok = learned_post_score(nm, q, k, ch, ans, T)
                            esc = gain > thr
                            o.update(
                                cand_answer=ans,
                                ver_score=gain,
                                ver_input_tokens=ntok,
                                ver_cost=0.0,
                                escalated=bool(esc),
                            )
                            if not esc:
                                final = ans
                                branch = cand
                            elif cand == "E":
                                final, T["cloud"], c2, o["cloud_out_tokens"], _ = gen_C(q, ch, False)
                                o["cost"] += c2
                                branch = "N"
                            else:
                                answer, T["cloud_T"], c2, o["T_out_tokens"], ok = gen_C(q, ch, True)
                                o["cost"] += c2
                                branch = "T"
                                o["cloud_completed"] = bool(ok)
                                final = answer if ok else ans
                                o["retry"] = None if ok else "reuseN"
                        elif kind == "automix":
                            cand, thr = sp[1], float(sp[2])
                            nk = int(sp[3]) if len(sp) > 3 else 8
                            o["ver_k"] = nk
                            if cand == "E":
                                ans, T["edge"] = gen_E(q, ch)
                                v = am_verify(edge, "qwen3-4b", ch, q, ans, qq["id"], SALT["v"], nk)
                                T["verify"] = v["ver_sec"]
                                vcost = 0.0
                            else:
                                ans, T["cloud"], c1, o["cloud_out_tokens"], _ = gen_C(q, ch, False)
                                o["cost"] = c1
                                v = am_verify(cloud, "q38-flash-next-fp8", ch, q, ans, qq["id"], SALT["v"], nk)
                                T["verify"] = v["ver_sec"]
                                vcost = v["ver_cost"]
                            o.update(
                                {
                                    "ver_score": v["score"],
                                    "ver_n_valid": v["n_valid"],
                                    "ver_in": v["ver_in"],
                                    "ver_out": v["ver_out"],
                                    "ver_cost": vcost,
                                    "cand_answer": ans,
                                    "ver_samples": [t[-160:] for t in v["samples"]],
                                }
                            )
                            o["cost"] += vcost
                            esc = v["score"] < thr
                            o["escalated"] = bool(esc)
                            if not esc:
                                final = ans
                                branch = cand
                            elif cand == "E":
                                final, T["cloud"], c2, o["cloud_out_tokens"], _ = gen_C(q, ch, False)
                                o["cost"] += c2
                                branch = "N"
                            else:
                                fT_, T["cloud_T"], c2, o["T_out_tokens"], ok = gen_C(q, ch, True)
                                o["cost"] += c2
                                branch = "T"
                                o["cloud_completed"] = bool(ok)
                                final = fT_ if ok else ans
                                o["retry"] = None if ok else "reuseN"
                        elif branch == "E":
                            final, T["edge"] = gen_E(q, ch)
                        else:
                            final, T["cloud"], o["cost"], o["cloud_out_tokens"], ok = gen_C(q, ch, branch == "T")
                            o["cloud_completed"] = bool(ok)
                            if not ok:
                                if a.fallback == "N":
                                    final, T["cloud_retry"], c2, _, _ = gen_C(q, ch, False)
                                    o["cost"] += c2
                                    o["retry"] = "N"
                                else:
                                    final, T["edge_fallback"] = gen_E(q, ch)
                                    o["retry"] = "E"
                        o["wall"] = time.perf_counter() - w0
                        o["ans_cached"] = hits() - h0
                        o["ans_prompt_tokens"] = CU.get("ans_prompt_tokens", 0)
                        o["total_cost"] = o["cost"] + o["sel_cost"]
                        o.update(
                            {
                                "branch": branch,
                                "answer": final,
                                "f1": f1(final or "", refs[qq["id"]]),
                                "time": T,
                                "total": sum(T.values()),
                                "block": o_block,
                                "block_pos": order.index(nm),
                            }
                        )
                        f.write(json.dumps(o, ensure_ascii=False) + "\n")
                        f.flush()
            print(time.strftime("%FT%TZ", time.gmtime()), "block", b0, sp_pool, "order", order, flush=True)


if __name__ == "__main__":
    main()
