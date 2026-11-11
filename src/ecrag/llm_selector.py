"""General-LLM E/N/T selectors. Same information as the decider: question + the
pruned passages D (k of them, stage-1 trace) + k. Zero-shot, temperature 0, thinking off; scores from first-token log-probs.
Three probes per question (see PROMPTS): edge (dEN = P(No)), think (dNT = P(Yes)), choice (A/B/C: dEN = 1 - P(A),
dNT = P(C) / (P(B) + P(C)); argmax kept as 'choice'). Output: --out/pred_<tag>.json in the decider pred format (keys dEN, dNT for
the two-probe scorer; c_dEN, c_dNT, choice for the choice probe), plus per-call wall seconds and prompt tokens. Resumable per spec.
--latency N: afterwards, N single sequential requests per probe (timing only) -> --out/latency.json.
Specs as train_router4.load (legacy:<split> | <pilotN>:<ds>:<split>)."""

import argparse, asyncio, json, math, os, sys, time
from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(__file__))
from train_router4 import load  # noqa: E402

CTX = "Passages retrieved for the question:\n\n{passages}\n\nQuestion: {q}\n\n"
PROMPTS = {
    "edge": CTX
    + (
        "A small 4B-parameter on-device language model will read these {k} passages and answer the question in a few words. "
        "Is the small model likely to answer correctly? Answer with exactly one word: Yes or No."
    ),
    "think": CTX
    + (
        "A large cloud language model will answer the question from these {k} passages. Would it need to reason step by step "
        "(chain of thought) to answer correctly, rather than answering directly? Answer with exactly one word: Yes or No."
    ),
    "choice": CTX
    + (
        "Choose how to answer this question from these {k} passages, balancing correctness and cost:\n"
        "A) a small 4B on-device model (free, fast)\n"
        "B) a large cloud model answering directly (costs money)\n"
        "C) a large cloud model reasoning step by step first (several times the cost of B, much slower)\n"
        "Pick the cheapest option that is likely to answer correctly. Answer with exactly one letter: A, B, or C."
    ),
}


def probs(lp, labels):
    """normalised probabilities of the label tokens among the first-token top log-probs."""
    m = {l: 0.0 for l in labels}
    for t in lp:
        s = t.token.strip().strip(".:)").lower()
        for l in labels:
            if s == l.lower():
                m[l] += math.exp(t.logprob)
    z = sum(m.values())
    return {l: (v / z if z > 0 else 1 / len(labels)) for l, v in m.items()}, z


async def call(cl, a, sem, text, labels):
    async with sem:
        for att in range(5):
            try:
                t0 = time.perf_counter()
                r = await cl.chat.completions.create(
                    model=a.model,
                    temperature=0,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=20,
                    messages=[{"role": "user", "content": text}],
                    extra_body=json.loads(a.extra_body),
                )
                dt = time.perf_counter() - t0
                c = r.choices[0]
                p, z = probs(c.logprobs.content[0].top_logprobs if c.logprobs and c.logprobs.content else [], labels)
                return p, z, dt, r.usage.prompt_tokens
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                await asyncio.sleep(2 * (att + 1))
        raise RuntimeError(err)


def prompt(kind, r):
    ps = "\n\n".join(f"Passage {i + 1}:\n{c}" for i, c in enumerate(r["chunks"]))
    return PROMPTS[kind].format(passages=ps, q=r["query"], k=r["k"])


async def run_spec(cl, a, sem, spec):
    tag = spec.replace(":", "_")
    fn = f"{a.out}/pred_{tag}.json"
    if os.path.exists(fn):
        print("exists, skip", fn, flush=True)
        return
    rows = load(spec)

    async def one(r):
        (pe, ze, te, ne), (pt, zt, tt, nt), (pc, zc, tc, nc) = await asyncio.gather(
            call(cl, a, sem, prompt("edge", r), ["Yes", "No"]),
            call(cl, a, sem, prompt("think", r), ["Yes", "No"]),
            call(cl, a, sem, prompt("choice", r), ["A", "B", "C"]),
        )
        return {
            "ds": r["ds"],
            "split": r["split"],
            "pool": r["pool"],
            "id": r["id"],
            "dEN": pe["No"],
            "dNT": pt["Yes"],
            "c_dEN": 1 - pc["A"],
            "c_dNT": pc["C"] / max(pc["B"] + pc["C"], 1e-9),
            "choice": max(pc, key=pc.get),
            "mass": [ze, zt, zc],
            "sec": [te, tt, tc],
            "prompt_tokens": [ne, nt, nc],
        }

    t0 = time.time()
    out = await asyncio.gather(*[one(r) for r in rows])
    json.dump(out, open(fn, "w"))
    print(time.strftime("%FT%TZ", time.gmtime()), spec, len(out), f"{time.time() - t0:.0f}s", flush=True)


async def latency(cl, a, spec, n):
    rows = load(spec)[:n]
    sem = asyncio.Semaphore(1)
    L = {k: [] for k in PROMPTS}
    T = {k: [] for k in PROMPTS}
    import uuid

    for r in rows:
        nonce = (
            f"[request {uuid.uuid4().hex}]\n" if not a.cold_each else None
        )  # unique prefix: the first probe of each question is a cold prefill (no prefix-cache hit
        for k in PROMPTS:  # from the main run); the later probes share the nonce + passages prefix, as in serving
            nz = (
                nonce if nonce is not None else f"[request {uuid.uuid4().hex}]\n"
            )  # --cold_each: every probe is a cold prefill
            _, _, dt, nt = await call(
                cl, a, sem, nz + prompt(k, r), ["Yes", "No"] if k != "choice" else ["A", "B", "C"]
            )
            L[k].append(dt)
            T[k].append(nt)
    S = {
        k: {
            "n": len(v),
            "mean_s": sum(v) / len(v),
            "p50_s": sorted(v)[len(v) // 2],
            "mean_prompt_tokens": sum(T[k]) / len(T[k]),
        }
        for k, v in L.items()
    }
    json.dump(
        {"spec": spec, "model": a.model, "cold_each": a.cold_each, "single_request": S},
        open(f"{a.out}/latency{'_cold_each' if a.cold_each else ''}.json", "w"),
        indent=1,
    )
    print(json.dumps(S), flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--extra_body", default='{"chat_template_kwargs": {"thinking": false, "enable_thinking": false}}')
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--latency", type=int, default=0)
    ap.add_argument("--latency_only", action="store_true")
    ap.add_argument("--cold_each", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.latency_only:
        json.dump({"args": vars(a), "prompts": PROMPTS}, open(f"{a.out}/config.json", "w"), indent=1)
    cl = AsyncOpenAI(base_url=a.base_url, api_key="none", timeout=600, max_retries=0)
    sem = asyncio.Semaphore(a.conc)
    for s in a.specs:
        if not a.latency_only:
            await run_spec(cl, a, sem, s)
    if a.latency:
        await latency(cl, a, a.specs[0], a.latency)


if __name__ == "__main__":
    asyncio.run(main())
