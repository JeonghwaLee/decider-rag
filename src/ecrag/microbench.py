"""System microbenchmark for the EC-RAG router pilot (one GPU, 16 GB memory budget).

A. Router (Qwen3-4B + LoRA + head, our train_router.Router): single-query latency and peak memory vs input length.
B. B1 pointwise relevance:
   - EC-RAG style: sequential generate(max_new_tokens=1), one call per chunk, for 1..10 chunks.
   - Batched: the same 10 prompts in ONE batched forward, reading the next-token logits (Yes vs No).
   Latency only, no accuracy.
C. Shared-base 16 GB budget:
   - One Qwen3-4B causal LM in bf16 is used for edge generation with the adapter disabled, and the same weights plus a LoRA
     adapter and head serve the router.
   - Cap: torch.cuda.set_per_process_memory_fraction(16 GiB / device memory).
   - Run router forward at 6k tokens, then generate 50 tokens on a 6k prompt. Record peak memory and OOM.
Prompts are synthetic text from real chunks, so token counts are exact. Timings: median of N runs after warmup.
"""

from paths import ECRAG_SRC, RUNS  # noqa: E402
import json, os, sys, time, statistics as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(__file__))
from train_router import Router  # noqa: E402

sys.path.insert(0, ECRAG_SRC)

M = "Qwen/Qwen3-4B"
CH = f"{RUNS}/pilot1/trace/hotpotqa/test/pool_00/chunks.jsonl"
OUT = f"{RUNS}/pilot1/microbench.json"


def timeit(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return st.median(ts)


def text_of(tok, chunks, ntok):
    ids = tok("\n".join(chunks), add_special_tokens=False).input_ids
    while len(ids) < ntok:
        ids = ids + ids
    return tok.decode(ids[:ntok])


@torch.no_grad()
def main():
    res = {"gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16"}
    tok = AutoTokenizer.from_pretrained(M)
    chunks = [json.loads(l)["text"] for l in open(CH)][:200]
    # A. router latency vs length
    torch.cuda.reset_peak_memory_stats()
    r = Router(M).cuda().eval()
    res["router_weights_gb"] = torch.cuda.memory_allocated() / 2**30
    res["A_router"] = {}
    for L in [512, 1024, 2048, 4096, 6144]:
        ids = tok(text_of(tok, chunks, L), return_tensors="pt", add_special_tokens=False).input_ids[:, :L].cuda()
        m = torch.ones_like(ids)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = timeit(lambda: r(ids, m))
        res["A_router"][L] = {"latency_ms": t * 1e3, "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
        print("A", L, res["A_router"][L], flush=True)
    del r
    torch.cuda.empty_cache()
    # B. relevance: sequential generate vs one batched forward
    lm = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
    from slm_inference import judge_relevance_prompt

    q = "Which case was brought to court first Miller v. California or Gates v. Collier ?"
    prompts = [judge_relevance_prompt(c, q, "qwen3") for c in chunks[:10]]
    enc1 = [tok(p, return_tensors="pt").to("cuda") for p in prompts]
    res["B_relevance"] = {"prompt_tokens_mean": st.mean(e.input_ids.shape[1] for e in enc1)}
    for k in [1, 4, 8, 10]:
        t = timeit(lambda: [lm.generate(**enc1[i], max_new_tokens=1, do_sample=False) for i in range(k)], n=5, warm=2)
        res["B_relevance"][f"sequential_generate_{k}_ms"] = t * 1e3
    tok.padding_side = "left"
    encb = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    t = timeit(lambda: lm(**encb).logits[:, -1, :], n=5, warm=2)
    res["B_relevance"]["batched_forward_10_ms"] = t * 1e3
    tok.padding_side = "right"
    print("B", res["B_relevance"], flush=True)
    del lm
    torch.cuda.empty_cache()
    # C. shared base under a 16 GB cap
    cap = 16 * 2**30 / torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(cap)
    torch.cuda.reset_peak_memory_stats()
    c = {"cap_gb": 16}
    try:
        lm = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
        pm = get_peft_model(
            lm, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        ).eval()
        head = torch.nn.Sequential(
            torch.nn.Linear(lm.config.hidden_size, 512), torch.nn.GELU(), torch.nn.Linear(512, 2)
        ).cuda()
        c["weights_gb"] = torch.cuda.memory_allocated() / 2**30
        ids = tok(text_of(tok, chunks, 6144), return_tensors="pt", add_special_tokens=False).input_ids[:, :6144].cuda()

        def route():
            h = pm.base_model.model.model(input_ids=ids).last_hidden_state[:, -1].float()
            return head(h)

        c["router_6k_ms"] = timeit(route, n=5, warm=2) * 1e3
        c["peak_after_router_gb"] = torch.cuda.max_memory_allocated() / 2**30

        def gen():
            with pm.disable_adapter():
                return pm.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=50,
                    min_new_tokens=50,
                    do_sample=False,
                )

        c["edge_gen_6k_50tok_ms"] = timeit(gen, n=3, warm=1) * 1e3
        c["peak_total_gb"] = torch.cuda.max_memory_allocated() / 2**30
        c["oom"] = False
    except torch.OutOfMemoryError as e:
        c["oom"] = True
        c["error"] = str(e)[:300]
        c["peak_total_gb"] = torch.cuda.max_memory_allocated() / 2**30
    res["C_shared_base_16gb"] = c
    print("C", c, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
