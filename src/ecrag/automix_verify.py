"""AutoMix-style self-verification on stored candidate answers (post-generation verification baseline).
Verifier = the model that generated the candidate (HotpotQA: edge Qwen3-4B verifies the E answer; MuSiQue: cloud Qwen3.8 verifies the
N answer). Prompt = the official AutoMix 4-shot verification prompt (github.com/automix-llm/automix, colabs/Step2_SelfVerify.ipynb,
Apache-2.0) as a raw completion for the edge Qwen3-4B (official notebook style) and as the user turn of the cloud chat call (thinking off, as
in our cloud answer calls) for Qwen3.8, whose raw completions ended most samples before a verdict (format compliance, not accuracy, decided this), temperature 1.0, n=8 samples, max 250 tokens, stop '---'. Score = fraction of samples whose verdict is
'Correct' among samples with a parsable verdict (official compute_fraction_correct; 0 when none parses). Inputs: question, the pruned
evidence passages (the same k passages the answer was generated from), the candidate answer; never references, F1 or judge labels.
Usage: automix_verify.py --spec pilot9:musique:val --cand N --port 8393 --model q38-flash-next-fp8 --out <jsonl>"""

import argparse, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

SHOTS = [
    (
        "The manuscript, discovered in 1980 in a dusty attic, turned out to be a lost work of Shakespeare.",
        "Whose lost work was discovered in a dusty attic in 1980?",
        "Shakespeare",
        "The context specifically mentions that a lost work of Shakespeare was discovered in 1980 in a dusty attic.",
        "Correct",
    ),
    (
        "The celestial event, known as the Pink Moon, is unique to the month of April and has cultural significance in many indigenous tribes.",
        "In which month does the celestial event, the Pink Moon, occur?",
        "July",
        "The context clearly states that the Pink Moon is unique to the month of April.",
        "Incorrect",
    ),
    (
        "The Mona Lisa, housed in the Louvre Museum, is believed to be a portrait of Lisa Gherardini, painted by Leonardo da Vinci in the early 16th century.",
        "Who is believed to have painted the Mona Lisa in the early 16th century?",
        "Vincent van Gogh",
        "The context specifies that the Mona Lisa was painted by Leonardo da Vinci in the early 16th century.",
        "Incorrect",
    ),
    (
        "The planet Kepler-442b, located 1,100 light-years away, is one of the most Earth-like planets ever discovered, having a similar size and orbiting within its star's habitable zone.",
        "How far away is the planet Kepler-442b?",
        "1,100 light-years",
        "The context states that Kepler-442b is located 1,100 light-years away.",
        "Correct",
    ),
]
INSTR = (
    "Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. "
    "Provide the judgement and reasoning for each case. Choose between Correct or Incorrect."
)


def block(c, q, a):
    return f"Context: {c}\n\nQuestion: {q}\n\nAI Generated Answer: {a}\n\n{INSTR}\n\n"


PREFIX = "".join(
    block(c, q, a) + f"Evaluation: {e}\n\nVerification Decision: The AI generated answer is {d}.\n\n---\n\n"
    for c, q, a, e, d in SHOTS
)
NS, TEMP, MAXT, STOP = 8, 1.0, 250, "---"
QP = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
QS = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n\n"  # = regen_answers.QP/QS


def prompt(chunks, q, ans):
    return PREFIX + block("\n\n".join(chunks), q, ans if ans else "") + "Evaluation:"


def frac_correct(texts):
    v = [t for t in texts if "the ai generated answer is" in t.lower()]
    return 0.0 if not v else sum("the ai generated answer is correct" in t.lower() for t in v) / len(v), len(v)


def seed_of(qid):
    return int(hashlib.md5(str(qid).encode()).hexdigest()[:8], 16) % (2**31)


def verify(client, model, chunks, q, ans, qid, salt=None, n=NS):
    t0 = time.perf_counter()
    eb = {"seed": seed_of(qid)}
    if salt:
        eb["cache_salt"] = salt
    if (
        model == "qwen3-4b"
    ):  # raw completion as in the official notebook (Qwen3-4B follows the verdict format: 7.6/8 parsable on val)
        r = client.completions.create(
            model=model,
            prompt=prompt(chunks, q, ans),
            temperature=TEMP,
            max_tokens=MAXT,
            n=n,
            stop=[STOP],
            extra_body=eb,
        )
        txt = [c.text for c in r.choices]
    else:  # same chat call as the cloud answer (system prompt, thinking off)
        eb["chat_template_kwargs"] = {"enable_thinking": False}
        r = client.chat.completions.create(
            model=model,
            temperature=TEMP,
            max_tokens=MAXT,
            n=n,
            stop=[STOP],
            extra_body=eb,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt(chunks, q, ans)},
            ],
        )
        txt = [c.message.content or "" for c in r.choices]
    s, nv = frac_correct(txt)
    return {
        "score": s,
        "n_valid": nv,
        "ver_in": r.usage.prompt_tokens,
        "ver_out": r.usage.completion_tokens,
        "ver_cost": (0.27 * r.usage.prompt_tokens + 1.1 * r.usage.completion_tokens) / 1e3,
        "ver_sec": time.perf_counter() - t0,
        "samples": txt,
    }


def candidates(spec, cand):
    """(pool, id, query, chunks, candidate answer) from the stored traces/answers the offline tables use."""
    from router_data import DATA  # noqa: F401
    from normalize_edge import normalize

    p = spec.split(":")
    out = []
    if p[0] == "legacy":
        import ent_data

        R = ent_data.R
        for r in ent_data.load(p[1], "B", with_text=True):
            t = TR[(p[1], r["pool"])][r["id"]]
            out.append((r["pool"], r["id"], r["query"], r["chunks"], t["edge_answer"] if cand == "E" else None))
        return out
    import ent_data2, glob

    root = f"{ent_data2.R}/{p[0]}"
    for r in ent_data2.load(root, p[1], p[2], "B", with_text=True):
        A = f"{root}/answers/{p[1]}/{p[2]}/{r['pool']}"
        if (A, "E") not in CACHE:
            CACHE[(A, "E")] = ent_data2.rd(f"{A}/extract.jsonl")
            CACHE[(A, "N")] = ent_data2.rd(f"{A}/orig.jsonl")
        if cand == "E":
            a = normalize(r["query"], CACHE[(A, "E")][r["id"]]["edge_answer"])[0]
        else:
            a = CACHE[(A, "N")][r["id"]]["cloud_answer"]
        out.append((r["pool"], r["id"], r["query"], r["chunks"], a))
    return out


CACHE = {}


class _TR(dict):
    def __missing__(self, key):
        import glob, ent_data

        split, pool = key
        d = f"{ent_data.R}/pilot6/trace_cfg3nB/hotpotqa/{split}/{pool}"
        self[key] = {json.loads(l)["id"]: json.loads(l) for l in open(f"{d}/trace.jsonl")}
        return self[key]


TR = _TR()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--cand", choices=["E", "N"], required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from openai import OpenAI

    cl = OpenAI(base_url=f"http://127.0.0.1:{a.port}/v1", api_key="none", timeout=600)
    C = candidates(a.spec, a.cand)
    C = C[: a.limit] if a.limit else C
    done = {(json.loads(l)["pool"], json.loads(l)["id"]) for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [c for c in C if (c[0], c[1]) not in done]
    print(f"{a.spec} cand {a.cand}: {len(C)} rows, {len(todo)} to verify", flush=True)

    def one(c):
        pool, qid, q, ch, ans = c
        return {"pool": pool, "id": qid, "cand": a.cand, "cand_answer": ans, **verify(cl, a.model, ch, q, ans, qid)}

    with open(a.out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, o in enumerate(ex.map(one, todo)):
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                f.flush()
                print(time.strftime("%FT%TZ", time.gmtime()), i, flush=True)


if __name__ == "__main__":
    main()
