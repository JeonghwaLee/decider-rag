"""Pilot 2: answer-format / answerability prompt variants on the SAME pruned chunks D (stage-1 traces).

Both generators get the same instruction for a variant:
- edge: Qwen3-4B on a local vLLM server (raw completion, the exact EC-RAG Qwen3 chat prefix/suffix with thinking off),
  greedy, max_tokens 50 (EC-RAG setting)
- cloud: Qwen3.8-Flash-Next FP8 (chat, system "You are a helpful assistant."), temperature 0, thinking off,
  max_tokens 256 (stage2 setting)
The 'orig' variant regenerates the EC-RAG prompt on the same engine as a control.
Then, optionally, every answer is graded by an LLM judge with the blind rubric from judge_rubric.py, and F1 is computed
with EC-RAG cal_f1.
Output: runs/pilot2/answers/<ds>/<split>/<pool>/<variant>.jsonl (resumable)
"""

from paths import RUNS  # noqa: E402
import argparse, asyncio, json, os, re, sys
from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(__file__))
from router_data import DATA, TRACE, f1  # noqa: E402
from judge_rubric import prompt as judge_prompt  # noqa: E402

OUT = os.environ.get("ECRAG_ANS_OUT", f"{RUNS}/pilot2/answers")
ORIG = "Only give me the answer and do not output any other words."
MIN = (
    "Give the minimal complete answer to the question. Do not repeat the question or add an introduction, "
    "but keep any conditions, negations, or explanations the answer requires."
)
ABST = " If the context does not contain the information needed to answer the question, reply exactly: Unanswerable"
TYPED = (
    "Answer with exactly the information the question asks for, as specifically as possible: if it asks for a name, give "
    "the specific name (not a description or a count); if it compares options (e.g. who is older, which came first), give "
    "the chosen option; if it asks what something is, state its distinguishing function; if it asks for several items or "
    "methods, list them by name. Do not repeat the question and do not explain unless the question asks for an explanation."
)
EXTRACT = (
    "First, after 'Facts:', list the facts from the context that are needed to answer, one per line, quoting the key "
    "words (for comparisons, give the value for each option). Then, on the last line, write 'Answer:' followed only by "
    "the answer, as specific as the question requires."
)
VARIANTS = {
    "orig": ORIG,
    "orig_abst": ORIG + ABST,
    "min": MIN,
    "min_abst": MIN + ABST,
    "typed": TYPED,
    "extract": EXTRACT,
    "orig_think": ORIG,
    "typed_think": TYPED,
}
# per-variant generation settings: edge/cloud max tokens, cloud thinking, answer parsing
CFG = {
    "extract": {"edge_max": 256, "cloud_max": 512, "parse": "answer_line"},
    "orig_think": {"cloud_think": True, "cloud_max": 6144},
    "typed_think": {"cloud_think": True, "cloud_max": 6144},
}


def parse(text, how):
    t = text
    if "</think>" in t:
        t = t.split("</think>")[-1]
    t = t.strip()
    if how == "answer_line":
        m = re.findall(r"Answer:\s*(.*)", t)
        t = m[-1].strip() if m else t.splitlines()[-1].strip() if t else ""
    return t


QP = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
QS = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n\n"  # = EC-RAG _build_suffix('qwen3')
S = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


def body(chunks, q, instr):
    return "\n\n".join(chunks) + f"\n\nGiven the above context, answer the question: {q}\n\n{instr}"


async def retry(fn):
    err = None
    for _ in range(5):
        try:
            return await fn()
        except Exception as e:
            err = e
            await asyncio.sleep(5)
    raise err


async def gen_one(a, cl, sem, rec, chunks, refs, instr, cfg):
    ch = [chunks[i] for i in rec["topk_ids"][: rec["k"]]]
    async with sem:
        e = await retry(
            lambda: cl["edge"].completions.create(
                model="qwen3-4b",
                prompt=QP + body(ch, rec["query"], instr) + QS,
                temperature=0,
                max_tokens=cfg.get("edge_max", 50),
            )
        )
        c = await retry(
            lambda: cl["cloud"].chat.completions.create(
                model="q38-flash-next-fp8",
                temperature=0,
                max_tokens=cfg.get("cloud_max", 256),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": body(ch, rec["query"], instr) + "\n"},
                ],
                extra_body={"chat_template_kwargs": {"enable_thinking": bool(cfg.get("cloud_think", False))}},
            )
        )
        e_raw, c_raw = e.choices[0].text, (c.choices[0].message.content or "")
        ea, ca = parse(e_raw, cfg.get("parse")), parse(c_raw, cfg.get("parse"))
        out = {
            "id": rec["id"],
            "edge_answer": ea,
            "cloud_answer": ca,
            "edge_raw": e_raw,
            "cloud_raw": c_raw[-3000:],
            "cloud_finish": c.choices[0].finish_reason,
            "edge_out_tokens": e.usage.completion_tokens,
            "cloud_in_tokens": c.usage.prompt_tokens,
            "cloud_out_tokens": c.usage.completion_tokens,
            "f1_edge": f1(ea, refs),
            "f1_cloud": f1(ca, refs),
        }
        if a.judge:
            for who, ans in [("edge", ea), ("cloud", ca)]:
                lab = None
                for _ in range(3):
                    r = await retry(
                        lambda: cl["judge"].chat.completions.create(
                            model=a.judge_model,
                            temperature=0,
                            max_tokens=a.judge_max_tokens,
                            messages=[{"role": "user", "content": judge_prompt(rec["query"], refs, ans)}],
                            extra_body=json.loads(a.judge_extra),
                        )
                    )
                    raw = (r.choices[0].message.content or "").upper()
                    if raw.strip():
                        lab = "INCORRECT" if "INCORRECT" in raw else (re.search(r"CORRECT|PARTIAL", raw) or [None])[0]
                        break
                out[f"j_{who}"] = lab
        return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--limit", type=int, default=0, help="first N questions per pool (0 = all)")
    ap.add_argument("--edge_url", default="http://127.0.0.1:8195/v1")
    ap.add_argument("--cloud_url", default="http://127.0.0.1:8193/v1")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge_url", default="http://127.0.0.1:8194/v1")
    ap.add_argument("--judge_model", default="glm53-flash")
    ap.add_argument("--judge_max_tokens", type=int, default=2048)
    ap.add_argument("--judge_extra", default='{"chat_template_kwargs": {"reasoning_effort": "low"}}')
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    cl = {
        "edge": AsyncOpenAI(base_url=a.edge_url, api_key="none", timeout=600),
        "cloud": AsyncOpenAI(base_url=a.cloud_url, api_key="none", timeout=600),
        "judge": AsyncOpenAI(base_url=a.judge_url, api_key="none", timeout=600),
    }
    sem = asyncio.Semaphore(a.concurrency)
    for p in a.pools:
        ds, split, pool = p.split("/")
        chunks = {json.loads(l)["node_id"]: json.loads(l)["text"] for l in open(f"{TRACE}/{p}/chunks.jsonl")}
        recs = [json.loads(l) for l in open(f"{TRACE}/{p}/trace.jsonl")]
        recs = sorted(recs, key=lambda r: r["id"])[: a.limit or None]
        refs = {json.loads(l)["id"]: json.loads(l)["answers"] for l in open(f"{DATA}/{p}/answers/answers.jsonl")}
        os.makedirs(f"{OUT}/{p}", exist_ok=True)
        for v in a.variants:
            fp = f"{OUT}/{p}/{v}.jsonl"
            done = {json.loads(l)["id"] for l in open(fp)} if os.path.exists(fp) else set()
            todo = [r for r in recs if r["id"] not in done]

            async def one(r, f):
                try:
                    o = await gen_one(a, cl, sem, r, chunks, refs[r["id"]], VARIANTS[v], CFG.get(v, {}))
                except Exception as e:
                    print("ERROR", p, v, r["id"], repr(e)[:200], flush=True)
                    return
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
                f.flush()

            with open(fp, "a") as f:
                await asyncio.gather(*(one(r, f) for r in todo))
            print(p, v, "done", len(todo), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
