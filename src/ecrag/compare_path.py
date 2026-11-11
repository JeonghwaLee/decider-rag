"""Comparison-correction path: for date-comparison questions, extract each option's date with an LLM,
compare in code, output the chosen option. Falls back (no answer) when it is not a date comparison or a date is missing.

Detection is from the question text only (deployable): two options joined by ' or ' plus a temporal comparison word.
- Earlier date wins: older, oldest, born first, born earlier, first, earlier, came first, founded first, ...
- Later date wins: younger, youngest, more recent(ly), most recent(ly), later, newer, latest
Extraction prompt (same chunks D as the normal answer) returns JSON {"A": name, "date_A": "...", "B": ..., "date_B": ...}.
Dates are parsed to (year, month, day); ties or unparsable -> fallback.
Model: --model edge (Qwen3-4B via the vLLM completion server) or cloud (Qwen3.8 chat). Output per pool:
runs/pilot2/answers/<pool>/cmp_<model>.jsonl with {id, is_cmp, op, A, B, date_A, date_B, answer (or null), raw}.
"""

import argparse, asyncio, json, os, re, sys
from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(__file__))
from router_data import DATA, TRACE, f1  # noqa: E402
from regen_answers import QP, QS, OUT, retry  # noqa: E402

EARLY = r"older|oldest|born first|born earlier|first|earlier|came first|founded first|established first|formed first|released first|longer ago"
LATE = r"younger|youngest|more recent(?:ly)?|most recent(?:ly)?|later|newer|latest|born later|born last"
MONTHS = {
    m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
}
PROMPT = (
    "{ctx}\n\nQuestion: {q}\n\nThe question compares two options by a date (for people: the birth date). Using only the "
    "context above, give each option and its date. Reply with JSON only, in this form:\n"
    '{{"A": "<first option as written in the question>", "date_A": "<date, e.g. 1956-04-06 or 1956, or unknown>", '
    '"B": "<second option>", "date_B": "<date or unknown>"}}'
)


def detect(q):
    ql = q.lower()
    if " or " not in ql:
        return None
    late, early = re.search(rf"\b({LATE})\b", ql), re.search(rf"\b({EARLY})\b", ql)
    if late and (not early or late.start() <= early.start()):
        return "late"
    return "early" if early else None


def pdate(s):
    s = (s or "").lower()
    y = re.search(r"\b(\d{3,4})\b", s)
    if not y or "unknown" in s:
        return None
    m = re.search(r"\d{3,4}-(\d{1,2})(?:-(\d{1,2}))?", s)
    if m:
        return (int(y.group(1)), int(m.group(1)), int(m.group(2) or 0))
    mon = next((v for k, v in MONTHS.items() if re.search(rf"\b{k}", s)), 0)
    d = re.search(r"\b(\d{1,2})\b(?!\d)", s.replace(y.group(1), ""))
    return (int(y.group(1)), mon, int(d.group(1)) if d and mon else 0)


async def one(a, cl, sem, rec, chunks, refs):
    op = detect(rec["query"])
    out = {"id": rec["id"], "is_cmp": op is not None, "op": op, "answer": None}
    if op is None:
        return out
    ctx = "\n\n".join(chunks[i] for i in rec["topk_ids"][: rec["k"]])
    p = PROMPT.format(ctx=ctx, q=rec["query"])
    async with sem:
        if a.model == "edge":
            r = await retry(
                lambda: cl.completions.create(model="qwen3-4b", prompt=QP + p + QS, temperature=0, max_tokens=128)
            )
            raw = r.choices[0].text
        else:
            r = await retry(
                lambda: cl.chat.completions.create(
                    model="q38-flash-next-fp8",
                    temperature=0,
                    max_tokens=256,
                    messages=[{"role": "user", "content": p}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            )
            raw = r.choices[0].message.content or ""
    out["raw"] = raw
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:
        j = {}
    da, db = pdate(str(j.get("date_A"))), pdate(str(j.get("date_B")))
    out.update({"A": j.get("A"), "B": j.get("B"), "date_A": j.get("date_A"), "date_B": j.get("date_B")})
    if da and db and da != db and j.get("A") and j.get("B"):
        # compare at the precision both dates share
        n = min(sum(1 for x in da if x), sum(1 for x in db if x))
        ka, kb = da[: max(1, n)], db[: max(1, n)]
        if ka != kb:
            first = j["A"] if ka < kb else j["B"]
            last = j["B"] if ka < kb else j["A"]
            out["answer"] = first if op == "early" else last
    if out["answer"] is not None:
        out["f1"] = f1(out["answer"], refs)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--model", choices=["edge", "cloud"], default="edge")
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    cl = AsyncOpenAI(
        base_url="http://127.0.0.1:8195/v1" if a.model == "edge" else "http://127.0.0.1:8193/v1",
        api_key="none",
        timeout=600,
    )
    sem = asyncio.Semaphore(a.concurrency)
    for p in a.pools:
        chunks = {json.loads(l)["node_id"]: json.loads(l)["text"] for l in open(f"{TRACE}/{p}/chunks.jsonl")}
        recs = [json.loads(l) for l in open(f"{TRACE}/{p}/trace.jsonl")]
        refs = {json.loads(l)["id"]: json.loads(l)["answers"] for l in open(f"{DATA}/{p}/answers/answers.jsonl")}
        res = await asyncio.gather(*(one(a, cl, sem, r, chunks, refs[r["id"]]) for r in recs))
        os.makedirs(f"{OUT}/{p}", exist_ok=True)
        with open(f"{OUT}/{p}/cmp_{a.model}.jsonl", "w") as f:
            for o in res:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(
            p, "cmp", sum(o["is_cmp"] for o in res), "answered", sum(o["answer"] is not None for o in res), flush=True
        )


if __name__ == "__main__":
    asyncio.run(main())
