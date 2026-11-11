"""Decider with CUDA graphs: merged LoRA, input right-padded to a multiple of 512 tokens (causal attention: the last real
token's hidden state does not see the padding), torch.compile(mode="reduce-overhead") per bucket. Reports ms/request and score agreement
vs the eager merged decider on the same texts. Usage: bench_graph.py <ckpt dir>"""

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
rows = ent_data2.load(f"{R}/pilot8", "hotpotqa", "test4", "B", with_text=True)[:200]
texts = [build_text_fast(r, tok, 6144) for r in rows]
ck = sys.argv[1]
dec = merge_decider(load_decider(ck, json.load(open(f"{ck}/args.json")).get("n_out", 5)))
enc_m = dec.enc


def run_eager(t):
    e = tok([t], return_tensors="pt").to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(dec(e.input_ids, e.attention_mask)[0, 3])


def core(ids, mask, last):
    out = enc_m(input_ids=ids, attention_mask=mask).last_hidden_state
    return dec.head(out[0, last].float())


cc = torch.compile(core, mode="reduce-overhead", dynamic=False)


def run_graph(t):
    ids = tok([t], return_tensors="pt").input_ids[0]
    n = len(ids)
    L = ((n + 511) // 512) * 512
    x = torch.full((1, L), tok.pad_token_id or 0, dtype=torch.long)
    x[0, :n] = ids
    m = torch.zeros((1, L), dtype=torch.long)
    m[0, :n] = 1
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(cc(x.cuda(), m.cuda(), torch.tensor(n - 1, device="cuda"))[3])


ref = [run_eager(t) for t in texts]
t0 = time.perf_counter()
[run_graph(t) for t in texts]
print(f"compile/warm pass {time.perf_counter() - t0:.0f} s", flush=True)
for nm, fn in [("eager", run_eager), ("graph", run_graph)]:
    ts, s = [], []
    for t in texts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s.append(fn(t))
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    s = np.array(s)
    print(
        nm,
        f"{1e3 * np.mean(ts):.1f} ms mean p90 {1e3 * np.percentile(ts, 90):.1f} | max|d - eager| {np.abs(s - np.array(ref)).max():.4f}",
        flush=True,
    )
