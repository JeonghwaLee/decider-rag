"""Decider forward latency by backbone: merged LoRA, B=1, same test4 texts (build_text_fast), ms/request incl. tokenization.
Usage: bench_backbone.py <ckpt dir> [...]"""

from paths import RUNS  # noqa: E402
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer  # noqa: E402
from e2e_ent2 import load_decider  # noqa: E402
from serve_opt import build_text_fast, merge_decider  # noqa: E402
import ent_data2  # noqa: E402

R = RUNS
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
rows = ent_data2.load(f"{R}/pilot8", "hotpotqa", "test4", "B", with_text=True)[:120]
texts = [build_text_fast(r, tok, 6144) for r in rows]
for ck in sys.argv[1:]:
    dec = merge_decider(load_decider(ck, json.load(open(f"{ck}/args.json")).get("n_out", 5)))
    ts = []
    for rep in range(2):
        for t in texts:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            enc = tok([t], return_tensors="pt").to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                dec(enc.input_ids, enc.attention_mask)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0) if rep else None
    print(ck.split("/")[-1], f"{1e3 * np.mean(ts):.1f} ms mean, p90 {1e3 * np.percentile(ts, 90):.1f}", flush=True)
    del dec
    torch.cuda.empty_cache()
