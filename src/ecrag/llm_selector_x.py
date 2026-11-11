"""Stronger general-LLM selectors.
Same information, probes and output format as llm_selector.py (pred_<tag>.json: dEN, dNT, c_dEN, c_dNT, choice, sec, ...).
--mode fewshot: 6 solved TRAIN examples of the same dataset as a fixed prefix (seed 7; 2 per cheapest-correct option A/B/C,
  among rows with <= median passage characters), thinking off, first-token scores.
--mode think: zero-shot prompts with thinking ON (temperature 0, --max_tokens); scores from the top-20 log-probs of the first
  label token after </think>; none found -> uniform (counted in 'nolabel').
--ctk: chat_template_kwargs override, e.g. GLM-5.3 '{"reasoning_effort": "low"}' (its template cannot turn thinking off;
  'low' answers right after an empty </think>; scored with the think-mode parser).
--latency N --latency_only: N single sequential requests per probe on the first spec, each with a unique nonce AFTER the
  few-shot prefix (the prefix stays cached, as in serving) -> --out/latency.json."""

import argparse, asyncio, json, math, os, random, sys, time, uuid
from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(__file__))
from train_router4 import load  # noqa: E402
from llm_selector import PROMPTS, prompt, probs  # noqa: E402

LABELS = {"edge": ["Yes", "No"], "think": ["Yes", "No"], "choice": ["A", "B", "C"]}
TRAIN = {"hotpotqa": "legacy:train", "musique": "pilot9:musique:train", "2wiki": "pilot10:2wiki:train"}


def option(r):
    if r["fE"] >= 0.5:
        return "A"
    if r["fN"] >= 0.5:
        return "B"
    if r["fT"] is not None and r["fT"] >= 0.5:
        return "C"
    return None


def shots(ds):
    rows = [r for r in load(TRAIN[ds]) if r["fT"] is not None]
    L = sorted(sum(len(c) for c in r["chunks"]) for r in rows)
    med = L[len(L) // 2]
    rows = [r for r in rows if sum(len(c) for c in r["chunks"]) <= med and option(r)]
    rng = random.Random(7)
    rng.shuffle(rows)
    out = []
    for o in "ABC":
        out += [r for r in rows if option(r) == o][:2]
    rng.shuffle(out)
    return out


def shot_label(kind, r):
    o = option(r)
    return {"edge": "Yes" if o == "A" else "No", "think": "Yes" if o == "C" else "No", "choice": o}[kind]


def fewshot_prefix(kind, S):
    return (
        "Here are solved examples of the task.\n\n"
        + "".join(f"### Example {i + 1}\n{prompt(kind, r)}\nAnswer: {shot_label(kind, r)}\n\n" for i, r in enumerate(S))
        + "### Now the actual task\n"
    )


async def call(cl, a, sem, text, labels):
    async with sem:
        err = None
        for att in range(5):
            try:
                t0 = time.perf_counter()
                think = a.mode == "think"
                r = await cl.chat.completions.create(
                    model=a.model,
                    temperature=0,
                    max_tokens=a.max_tokens if think else 1,
                    logprobs=True,
                    top_logprobs=20,
                    messages=[{"role": "user", "content": text}],
                    extra_body={
                        "chat_template_kwargs": (
                            json.loads(a.ctk) if a.ctk else {"thinking": think, "enable_thinking": think}
                        )
                    },
                )
                dt = time.perf_counter() - t0
                c = r.choices[0]
                lc = c.logprobs.content if c.logprobs and c.logprobs.content else []
                if not think:
                    p, z = probs(lc[0].top_logprobs if lc else [], labels)
                    found = True
                else:
                    toks = [x.token for x in lc]
                    st = toks.index("</think>") + 1 if "</think>" in toks else len(toks)
                    found = False
                    p, z = {l: 1 / len(labels) for l in labels}, 0.0
                    for x in lc[st:]:
                        if x.token.strip().strip(".:)*").lower() in [l.lower() for l in labels]:
                            p, z = probs(x.top_logprobs, labels)
                            found = True
                            break
                return p, z, dt, r.usage.prompt_tokens, r.usage.completion_tokens, found
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                await asyncio.sleep(2 * (att + 1))
        raise RuntimeError(err)


def text(a, kind, r, PRE, nonce=""):
    return (PRE[r["ds"]][kind] if a.mode == "fewshot" else "") + nonce + prompt(kind, r)


async def run_spec(cl, a, sem, spec, PRE):
    fn = f"{a.out}/pred_{spec.replace(':', '_')}.json"
    if os.path.exists(fn):
        print("exists, skip", fn, flush=True)
        return
    rows = load(spec)

    async def one(r):
        (pe, ze, te, ne, oe, fe), (pt, zt, tt, nt, ot, ft), (pc, zc, tc, nc, oc, fc) = await asyncio.gather(
            *[call(cl, a, sem, text(a, k, r, PRE), LABELS[k]) for k in ["edge", "think", "choice"]]
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
            "completion_tokens": [oe, ot, oc],
            "found": [fe, ft, fc],
        }

    t0 = time.time()
    out = await asyncio.gather(*[one(r) for r in rows])
    json.dump(out, open(fn, "w"))
    nol = sum(not f for x in out for f in x["found"])
    print(time.strftime("%FT%TZ", time.gmtime()), spec, len(out), f"{time.time() - t0:.0f}s nolabel {nol}", flush=True)


async def latency(cl, a, spec, n, PRE):
    rows = load(spec)[:n]
    sem = asyncio.Semaphore(1)
    L = {k: [] for k in PROMPTS}
    T = {k: [] for k in PROMPTS}
    O = {k: [] for k in PROMPTS}
    for k in PROMPTS:  # warm the few-shot prefix once per probe (it is shared by every request in serving)
        await call(cl, a, sem, text(a, k, rows[-1], PRE, f"[request {uuid.uuid4().hex}]\n"), LABELS[k])
    for r in rows:
        for k in PROMPTS:
            _, _, dt, nt, no, _ = await call(
                cl, a, sem, text(a, k, r, PRE, f"[request {uuid.uuid4().hex}]\n"), LABELS[k]
            )
            L[k].append(dt)
            T[k].append(nt)
            O[k].append(no)
    S = {
        k: {
            "n": len(v),
            "mean_s": sum(v) / len(v),
            "p50_s": sorted(v)[len(v) // 2],
            "mean_prompt_tokens": sum(T[k]) / len(v),
            "mean_completion_tokens": sum(O[k]) / len(v),
        }
        for k, v in L.items()
    }
    json.dump(
        {"spec": spec, "model": a.model, "mode": a.mode, "single_request": S},
        open(f"{a.out}/latency_{spec.replace(':', '_')}.json", "w"),
        indent=1,
    )
    print(json.dumps(S), flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["fewshot", "think"], required=True)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--ctk", default=None, help="chat_template_kwargs JSON override (GLM: reasoning_effort low)")
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--latency", type=int, default=0)
    ap.add_argument("--latency_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    PRE = {}
    if a.mode == "fewshot":
        for ds in TRAIN:
            S = shots(ds)
            PRE[ds] = {k: fewshot_prefix(k, S) for k in PROMPTS}
        json.dump(
            {ds: {"ids": [[r["pool"], r["id"]] for r in shots(ds)]} for ds in TRAIN},
            open(f"{a.out}/shots.json", "w"),
            indent=1,
        )
    if not a.latency_only:
        json.dump({"args": vars(a)}, open(f"{a.out}/config.json", "w"), indent=1)
    cl = AsyncOpenAI(base_url=a.base_url, api_key="none", timeout=1200, max_retries=0)
    sem = asyncio.Semaphore(a.conc)
    for s in a.specs:
        if not a.latency_only:
            await run_spec(cl, a, sem, s, PRE)
    if a.latency:
        for s in a.specs:
            await latency(cl, a, s, a.latency, PRE)


if __name__ == "__main__":
    asyncio.run(main())
