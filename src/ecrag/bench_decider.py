"""Decider serving micro-benchmark: same checkpoint (pilot7 ent_qd_s1234), same inputs
(build_text, max 6144), on test4 rows. Variants: base (LoRA unmerged, as in e2e_ent2/load_ent), merged (peft merge_and_unload),
and micro-batches B in {1,2,4,8} (right padding; last-token index from the mask, as in training). Reports ms/request and the
score agreement (max |d dEN|, decision agreement at the frozen threshold) against base B=1. Output: stdout + --out json."""

from paths import RUNS  # noqa: E402
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer  # noqa: E402
from train_router import build_text  # noqa: E402
from e2e_ent2 import load_decider  # noqa: E402
import ent_data2  # noqa: E402

R = RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    rows = ent_data2.load(f"{R}/pilot8", "hotpotqa", "test4", "B", with_text=True)[: a.n]
    texts = [build_text(r, tok, 6144, use_chunks=True) for r in rows]
    thr = json.load(open(f"{R}/pilot7/ent_policy.json"))["policies"]["s1234_qd|E/N|x1.0"]["a_dEN"]
    dec = load_decider(f"{R}/pilot7/router/ent_qd_s1234")

    def run(B):
        out, ts = [], []
        for i in range(0, len(texts), B):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            enc = tok(texts[i : i + B], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                s = dec(enc.input_ids, enc.attention_mask)[:, 3].float().cpu().numpy()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
            out += list(s)
        return np.array(out), 1e3 * np.sum(ts) / len(texts)

    tok.padding_side = "right"
    run(1)  # warmup
    res = {}
    ref, ms = run(1)
    res["base_B1"] = {"ms": ms}
    print(f"base B1: {ms:.1f} ms/req | mean tokens {np.mean([len(tok(t).input_ids) for t in texts]):.0f}", flush=True)
    for B in [2, 4, 8]:
        s, ms = run(B)
        res[f"base_B{B}"] = {
            "ms": ms,
            "maxdiff": float(np.abs(s - ref).max()),
            "agree": float(((s > thr) == (ref > thr)).mean()),
        }
        print(f"base B{B}: {res[f'base_B{B}']}", flush=True)
    dec.enc = dec.enc.merge_and_unload()
    run(1)
    for B in [1, 2, 4, 8]:
        s, ms = run(B)
        res[f"merged_B{B}"] = {
            "ms": ms,
            "maxdiff": float(np.abs(s - ref).max()),
            "agree": float(((s > thr) == (ref > thr)).mean()),
        }
        print(f"merged B{B}: {res[f'merged_B{B}']}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
