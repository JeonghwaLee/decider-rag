"""Per-action generation latency for the E/N/T decider. One query at a time, same chunks D:
  E = edge 'extract' prompt (vLLM :8195), N = cloud 'orig' thinking off (:8193), T = cloud 'orig_think' thinking on (max 6144).
Records wall time, input/output tokens, finish reason. Order of the three calls is rotated per query to avoid drift bias.
Output: runs/pilot7/latency_actions_<split>_<pool>.jsonl"""

from paths import RUNS  # noqa: E402
import argparse, json, os, sys, time
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from regen_answers import QP, QS, body, VARIANTS, CFG  # noqa: E402

TR = f"{RUNS}/pilot1/trace/hotpotqa"
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--pool", default="pool_00")
a = ap.parse_args()
edge = OpenAI(base_url="http://127.0.0.1:8195/v1", api_key="none", timeout=600)
cloud = OpenAI(base_url="http://127.0.0.1:8193/v1", api_key="none", timeout=1200)
d = f"{TR}/{a.split}/{a.pool}"
ch = {json.loads(l)["node_id"]: json.loads(l)["text"] for l in open(f"{d}/chunks.jsonl")}
out = f"{RUNS}/pilot7/latency_actions_{a.split}_{a.pool}.jsonl"
done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()


def E(q, c):
    t0 = time.perf_counter()
    r = edge.completions.create(
        model="qwen3-4b",
        prompt=QP + body(c, q, VARIANTS["extract"]) + QS,
        temperature=0,
        max_tokens=CFG["extract"]["edge_max"],
    )
    return {
        "t": time.perf_counter() - t0,
        "in": r.usage.prompt_tokens,
        "out": r.usage.completion_tokens,
        "finish": r.choices[0].finish_reason,
    }


def C(q, c, think):
    t0 = time.perf_counter()
    r = cloud.chat.completions.create(
        model="q38-flash-next-fp8",
        temperature=0,
        max_tokens=6144 if think else 256,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": body(c, q, VARIANTS["orig"]) + "\n"},
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": think}},
    )
    return {
        "t": time.perf_counter() - t0,
        "in": r.usage.prompt_tokens,
        "out": r.usage.completion_tokens,
        "finish": r.choices[0].finish_reason,
    }


for _ in range(3):  # warmup
    E("warmup?", ["warmup"])
    C("warmup?", ["warmup"], False)
with open(out, "a") as f:
    for n, l in enumerate(open(f"{d}/trace.jsonl")):
        t = json.loads(l)
        if t["id"] in done:
            continue
        c = [ch[i] for i in t["topk_ids"][: t["k"]]]
        acts = [
            ("E", lambda: E(t["query"], c)),
            ("N", lambda: C(t["query"], c, False)),
            ("T", lambda: C(t["query"], c, True)),
        ]
        acts = acts[n % 3 :] + acts[: n % 3]
        o = {"id": t["id"], "k": t["k"]}
        for name, fn in acts:
            o[name] = fn()
        f.write(json.dumps(o) + "\n")
        f.flush()
