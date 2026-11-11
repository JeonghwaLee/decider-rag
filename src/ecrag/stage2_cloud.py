"""Stage 2 of the EC-RAG router pilot: cloud answers on the SAME pruned chunks D recorded in stage 1.

Cloud = local Qwen3.5-122B-A10B-FP8 (vLLM, OpenAI API). It replaces the paper's DeepSeek-V3 API, so this is a new model pair,
not a reproduction.
- Prompt: identical to EC-RAG llm_inference.query_prompt; system "You are a helpful assistant."
- Decoding: temperature 0, thinking off, max_tokens 256.
- Recorded per query: answer, prompt/completion tokens, latency. The cost model (hypothetical, DeepSeek-V3 list prices used
  in EC-RAG utils.calc_cost: $0.27/M input, $1.1/M output) is applied later in analysis, on these token counts.
Output: <trace pool dir>/cloud.jsonl (append, resumable)
"""

from paths import RUNS  # noqa: E402
import argparse, asyncio, json, os, time
from openai import AsyncOpenAI


def query_prompt(chunk_list, query):  # copy of EC-RAG llm_inference.query_prompt
    chunks = "\n\n".join(chunk_list)
    return f"""\
{chunks}

Given the above context, answer the question: {query}

Only give me the answer and do not output any other words.
"""


async def one(client, sem, a, rec, chunks, f):
    prompt = query_prompt([chunks[i] for i in rec["topk_ids"][: rec["k"]]], rec["query"])
    async with sem:
        for attempt in range(5):
            try:
                t0 = time.perf_counter()
                r = await client.chat.completions.create(
                    model=a.model,
                    temperature=0,
                    max_tokens=256,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                dt = time.perf_counter() - t0
                out = {
                    "id": rec["id"],
                    "cloud_answer": r.choices[0].message.content or "",
                    "finish": r.choices[0].finish_reason,
                    "cloud_in_tokens": r.usage.prompt_tokens,
                    "cloud_out_tokens": r.usage.completion_tokens,
                    "latency": dt,
                    "attempts": attempt + 1,
                }
                break
            except Exception as e:  # record the failure, never a fabricated answer
                out = {"id": rec["id"], "error": repr(e)[:300], "attempts": attempt + 1}
                await asyncio.sleep(5)
    f.write(json.dumps(out, ensure_ascii=False) + "\n")
    f.flush()


async def run_pool(d, a, client):
    chunks = {}
    for l in open(f"{d}/chunks.jsonl"):
        c = json.loads(l)
        chunks[c["node_id"]] = c["text"]
    recs = [json.loads(l) for l in open(f"{d}/trace.jsonl")]
    cp = f"{d}/cloud{a.tag}.jsonl"
    done = set()
    if os.path.exists(cp):
        done = {json.loads(l)["id"] for l in open(cp) if "error" not in json.loads(l)}
    todo = [r for r in recs if r["id"] not in done]
    sem = asyncio.Semaphore(a.concurrency)
    with open(cp, "a") as f:
        await asyncio.gather(*(one(client, sem, a, r, chunks, f) for r in todo))
    return len(todo)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=f"{RUNS}/pilot1/trace")
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--base_url", default="http://127.0.0.1:8191/v1")
    ap.add_argument("--model", default="qwen35-122b")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--tag", default="", help="output suffix, e.g. _q38flash -> cloud_q38flash.jsonl")
    a = ap.parse_args()
    client = AsyncOpenAI(base_url=a.base_url, api_key="none", timeout=600)
    for p in a.pools:
        d = f"{a.trace}/{p}"
        if not os.path.exists(f"{d}/DONE"):
            print("skip (stage 1 not done)", p, flush=True)
            continue
        n = await run_pool(d, a, client)
        print(time.strftime("%FT%TZ", time.gmtime()), "cloud done", p, n, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
