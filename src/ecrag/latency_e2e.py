"""End-to-end latency components for the fixed HotpotQA policy (cfg3), measured one query at a time (single-user latency).
Per query (HotpotQA, given split/pools), sequentially:
  router   : F1 q+k router (Qwen3-4B + LoRA + head, bf16, bs=1) forward, cuda-synchronized; the score is checked against
             the saved prediction
  cmp      : date-comparison extraction on the edge server, only when the question is detected (regex, timed as well)
  edge     : edge generation with the cfg3 edge prompt ('extract') on the vLLM edge server
  cloud    : cloud generation with the cfg3 cloud prompt ('orig_think', thinking on) on the Qwen3.8 server
Upstream stages (retrieve, rerank, prune, complexity) are taken from the stage-1 trace timings of the same queries.
Policies are composed offline (analyze_latency.py) from these per-query times. Output: one jsonl row per query.
"""

from paths import RUNS  # noqa: E402
import argparse, json, os, sys, time, re
import torch
from openai import OpenAI
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from train_router import Router, build_text  # noqa: E402
from regen_answers import QP, QS, body, parse, VARIANTS, CFG  # noqa: E402
from compare_path import detect, PROMPT  # noqa: E402

TR = f"{RUNS}/pilot4/trace_cfg3/hotpotqa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test3")
    ap.add_argument("--pools", nargs="+", default=["pool_00", "pool_01"])
    ap.add_argument("--router", default="runs/pilot4/router/f1_qk_s1234")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    tok.padding_side = "right"
    model = Router("Qwen/Qwen3-4B")
    model.load_state_dict(torch.load(f"{a.router}/best.pt"), strict=False)
    model = model.cuda().eval()
    saved = {
        (x["pool"], x["id"]): x["pred_dq"]
        for x in json.load(open(f"{a.router}/pred_{a.split}.json"))
        if x["ds"] == "hotpotqa"
    }
    edge = OpenAI(base_url="http://127.0.0.1:8195/v1", api_key="none", timeout=600)
    cloud = OpenAI(base_url="http://127.0.0.1:8193/v1", api_key="none", timeout=1200)
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["pool"], json.loads(l)["id"]) for l in open(a.out)}
    # warmup
    for _ in range(3):
        ids = tok("warmup question?", return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(ids.input_ids, ids.attention_mask)
    f = open(a.out, "a")
    for pool in a.pools:
        d = f"{TR}/{a.split}/{pool}"
        chunks = {json.loads(l)["node_id"]: json.loads(l)["text"] for l in open(f"{d}/chunks.jsonl")}
        for l in open(f"{d}/trace.jsonl"):
            t = json.loads(l)
            if (pool, t["id"]) in done:
                continue
            ch = [chunks[i] for i in t["topk_ids"][: t["k"]]]
            o = {"pool": pool, "id": t["id"], "k": t["k"], "upstream": t["time"]}
            # router
            txt = build_text({"query": t["query"], "k": t["k"]}, tok, 6144, use_chunks=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            enc = tok([txt], return_tensors="pt").to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(enc.input_ids, enc.attention_mask)
            torch.cuda.synchronize()
            o["router"] = time.perf_counter() - t0
            o["router_dq"] = float(y[0, 0])
            o["router_dq_saved"] = saved.get((pool, t["id"]))
            # cmp extraction
            op = detect(t["query"])
            o["is_cmp"] = op is not None
            if op:
                p = PROMPT.format(ctx="\n\n".join(ch), q=t["query"])
                t0 = time.perf_counter()
                r = edge.completions.create(model="qwen3-4b", prompt=QP + p + QS, temperature=0, max_tokens=128)
                o["cmp"] = time.perf_counter() - t0
                o["cmp_out_tokens"] = r.usage.completion_tokens
            # edge (cfg3 edge prompt)
            cfg = CFG.get("extract", {})
            t0 = time.perf_counter()
            r = edge.completions.create(
                model="qwen3-4b",
                prompt=QP + body(ch, t["query"], VARIANTS["extract"]) + QS,
                temperature=0,
                max_tokens=cfg.get("edge_max", 50),
            )
            o["edge"] = time.perf_counter() - t0
            o["edge_out_tokens"] = r.usage.completion_tokens
            o["edge_answer"] = parse(r.choices[0].text, cfg.get("parse"))
            # cloud (cfg3 cloud prompt, thinking on); time to the full answer
            cc = CFG["orig_think"]
            t0 = time.perf_counter()
            r = cloud.chat.completions.create(
                model="q38-flash-next-fp8",
                temperature=0,
                max_tokens=cc["cloud_max"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": body(ch, t["query"], VARIANTS["orig_think"]) + "\n"},
                ],
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            o["cloud"] = time.perf_counter() - t0
            o["cloud_out_tokens"] = r.usage.completion_tokens
            o["cloud_in_tokens"] = r.usage.prompt_tokens
            o["cloud_answer"] = parse(r.choices[0].message.content or "", None)
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
