"""Decider decision time on the same questions as llm_selector.py --latency: single request, idle GPU.
Per question: build_text_fast (q + D + k, the same pruned passages the LLM selectors read) + tokenize + forward of the merged-LoRA
decider (serve_opt), synchronised. Reports mean / p50 seconds of (a) the whole decision and (b) the forward only, and the token
count. Usage: decider_latency.py --ckpt <dir> --spec <train_router4 spec> --n 100 --out file.json"""

import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer  # noqa: E402
from train_router4 import load  # noqa: E402
from e2e_ent2 import load_decider  # noqa: E402
from serve_opt import build_text_fast, merge_decider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--warm",
        type=int,
        default=0,
        help="warm-up on this many OTHER questions of the spec before timing (as a running service)",
    )
    a = ap.parse_args()
    n_out = json.load(open(f"{a.ckpt}/args.json")).get("n_out", 5) if os.path.exists(f"{a.ckpt}/args.json") else 5
    dec = load_decider(a.ckpt, n_out)
    merge_decider(dec)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    rows = load(a.spec)[: a.n + a.warm]
    W, Fw, N = [], [], []
    WU = (
        rows[a.n : a.n + a.warm] if a.warm else rows[:5]
    )  # warm-up on OTHER questions (shapes not reused), default 5 of the same
    T_ROWS = rows[: a.n]
    for it, r in enumerate(WU + T_ROWS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        txt = build_text_fast({"query": r["query"], "k": r["k"], "chunks": r["chunks"]}, tok, 6144)
        enc = tok([txt], return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            dec(enc.input_ids, enc.attention_mask)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if it >= len(WU):
            W.append(t2 - t0)
            Fw.append(t2 - t1)
            N.append(enc.input_ids.shape[1])
    med = lambda v: sorted(v)[len(v) // 2]
    S = {
        "spec": a.spec,
        "ckpt": a.ckpt,
        "n": len(W),
        "decision_mean_s": sum(W) / len(W),
        "decision_p50_s": med(W),
        "forward_mean_s": sum(Fw) / len(Fw),
        "forward_p50_s": med(Fw),
        "mean_tokens": sum(N) / len(N),
        "gpu": torch.cuda.get_device_name(0),
    }
    json.dump(S, open(a.out, "w"), indent=1)
    print(json.dumps(S), flush=True)


if __name__ == "__main__":
    main()
