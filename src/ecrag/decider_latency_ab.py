"""Decider-only latency A/B: the E2 HotpotQA run showed decider 85 ms (big2e4) vs 52 ms (s1234)
for the same 4B architecture; big2e4 was run while CPU-heavy E3 jobs ran. Here both checkpoints score the SAME rows (pruned chunks of
--spec), alternating the order per row (ABBA), single request, synchronised, in three serving modes:
  hf      = e2e_ent4 default (train_router.build_text + unmerged LoRA + bf16 autocast)   [what E2 HotpotQA used]
  opt     = serve_opt (build_text_fast + merged LoRA + bf16 autocast)                     [e2e_ent4 --opt]
Per mode and checkpoint: mean/p50/p95 of text build, tokenisation+H2D, forward; input tokens; and the max |score diff| between modes.
Usage: decider_latency_ab.py --spec pilot8:hotpotqa:test6 --n 200 --ckpt name=dir ... --out file.json"""

import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer  # noqa: E402
from train_router import build_text  # noqa: E402
from train_router4 import load  # noqa: E402
from e2e_ent2 import load_decider  # noqa: E402
from serve_opt import build_text_fast, merge_decider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    rows = load(a.spec)[: a.warm + a.n]
    C = dict(s.split("=", 1) for s in a.ckpt)
    D = {}
    for nm, ck in C.items():
        n_out = json.load(open(f"{ck}/args.json")).get("n_out", 5)
        D[nm] = load_decider(ck, n_out)
    sync = torch.cuda.synchronize
    names = list(C)
    res = {}
    scores = {}

    def one(nm, r, fast):
        sync()
        t0 = time.perf_counter()
        txt = build_text_fast(r, tok, 6144) if fast else build_text(r, tok, 6144, use_chunks=True)
        t1 = time.perf_counter()
        enc = tok([txt], return_tensors="pt").to("cuda")
        sync()
        t2 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = D[nm](enc.input_ids, enc.attention_mask)[0].float().tolist()
        sync()
        t3 = time.perf_counter()
        return (t1 - t0, t2 - t1, t3 - t2, t3 - t0), enc.input_ids.shape[1], (o[3], o[4])

    for mode in ["hf", "opt"]:
        if mode == "opt":
            for nm in names:
                merge_decider(D[nm])
        T = {nm: [] for nm in names}
        L = {nm: [] for nm in names}
        S = {nm: [] for nm in names}
        for i, r in enumerate(rows):
            order = names if i % 2 == 0 else names[::-1]
            for nm in order:
                t, n, s = one(nm, r, mode == "opt")
                if i >= a.warm:
                    T[nm].append(t)
                    L[nm].append(n)
                    S[nm].append(s)
        for nm in names:
            t = np.array(T[nm]) * 1e3
            scores[(mode, nm)] = np.array(S[nm])
            res[f"{mode}|{nm}"] = {
                "tokens_mean": float(np.mean(L[nm])),
                **{f"{k}_mean_ms": float(t[:, j].mean()) for j, k in enumerate(["text", "tok", "fwd", "total"])},
                "total_p50_ms": float(np.median(t[:, 3])),
                "total_p95_ms": float(np.percentile(t[:, 3], 95)),
            }
            print(mode, nm, json.dumps(res[f"{mode}|{nm}"]), flush=True)
    for nm in names:
        d = np.abs(scores[("hf", nm)] - scores[("opt", nm)])
        res[f"hf_vs_opt_maxdiff|{nm}"] = {"dEN": float(d[:, 0].max()), "dNT": float(d[:, 1].max())}
    res["args"] = vars(a)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if "maxdiff" in k}))


if __name__ == "__main__":
    main()
