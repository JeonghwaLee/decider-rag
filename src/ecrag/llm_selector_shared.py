"""Shared-prefix Qwen3.8 selector (E2). The selector prompt starts with exactly the cloud ANSWER
prompt prefix (system 'You are a helpful assistant.' + passages + 'Given the above context, answer the question: q'), then asks the
selector question, so that in serving the answer call can reuse the selector's prefix-cache blocks. Zero-shot, thinking off,
first-token log-probs, probes edge (dEN = P(No)) and think (dNT = P(Yes)). Output pred_<tag>.json in llm_selector format
(dEN, dNT, c_dEN = dEN, c_dNT = dNT, prompt_tokens, completion_tokens), so llmsel_links.py / total_cost_cmp.py read it unchanged."""

import argparse, asyncio, json, os, sys, time
from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(__file__))
from train_router4 import load  # noqa: E402
from llm_selector import probs  # noqa: E402

SYS = "You are a helpful assistant."
SHARED = {
    "edge": (
        "Before answering: a small 4B-parameter on-device language model will read the {k} passages above and answer the question "
        "in a few words. Is the small model likely to answer correctly? Answer with exactly one word: Yes or No."
    ),
    "think": (
        "Before answering: would a large cloud language model need to reason step by step (chain of thought) to answer this "
        "question correctly from the {k} passages above, rather than answering directly? Answer with exactly one word: Yes or No."
    ),
}


def prefix(chunks, q):
    """identical to regen_answers.body(chunks, q, instr) up to the instruction"""
    return "\n\n".join(chunks) + f"\n\nGiven the above context, answer the question: {q}\n\n"


def messages(kind, chunks, q, k):
    return [
        {"role": "system", "content": SYS},
        {"role": "user", "content": prefix(chunks, q) + SHARED[kind].format(k=k) + "\n"},
    ]


async def call(cl, model, sem, msgs, labels, salt=None):
    async with sem:
        err = None
        for att in range(5):
            try:
                t0 = time.perf_counter()
                eb = {"chat_template_kwargs": {"enable_thinking": False}}
                if salt:
                    eb["cache_salt"] = salt
                r = await cl.chat.completions.create(
                    model=model,
                    temperature=0,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=20,
                    messages=msgs,
                    extra_body=eb,
                )
                c = r.choices[0]
                p, z = probs(c.logprobs.content[0].top_logprobs if c.logprobs and c.logprobs.content else [], labels)
                return p, z, time.perf_counter() - t0, r.usage.prompt_tokens, r.usage.completion_tokens
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                await asyncio.sleep(2 * (att + 1))
        raise RuntimeError(err)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://127.0.0.1:8193/v1")
    ap.add_argument("--model", default="q38-flash-next-fp8")
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump({"args": vars(a), "system": SYS, "prompts": SHARED}, open(f"{a.out}/config.json", "w"), indent=1)
    cl = AsyncOpenAI(base_url=a.base_url, api_key="none", timeout=600, max_retries=0)
    sem = asyncio.Semaphore(a.conc)
    for spec in a.specs:
        fn = f"{a.out}/pred_{spec.replace(':', '_')}.json"
        if os.path.exists(fn):
            print("exists, skip", fn, flush=True)
            continue
        rows = load(spec)

        async def one(r):
            (pe, ze, te, ne, oe), (pt, zt, tt, nt, ot) = await asyncio.gather(
                *[
                    call(cl, a.model, sem, messages(kd, r["chunks"], r["query"], r["k"]), ["Yes", "No"])
                    for kd in ["edge", "think"]
                ]
            )
            return {
                "ds": r["ds"],
                "split": r["split"],
                "pool": r["pool"],
                "id": r["id"],
                "dEN": pe["No"],
                "dNT": pt["Yes"],
                "c_dEN": pe["No"],
                "c_dNT": pt["Yes"],
                "mass": [ze, zt, 0.0],
                "sec": [te, tt, 0.0],
                "prompt_tokens": [ne, nt, 0],
                "completion_tokens": [oe, ot, 0],
            }

        t0 = time.time()
        out = await asyncio.gather(*[one(r) for r in rows])
        json.dump(out, open(fn, "w"))
        print(
            time.strftime("%FT%TZ", time.gmtime()),
            spec,
            len(out),
            f"{time.time() - t0:.0f}s low-mass {sum(x['mass'][0] < .5 for x in out)}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
