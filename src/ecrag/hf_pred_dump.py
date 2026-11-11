"""HF predictions of an existing decider checkpoint (E3 reference): e2e_ent2.load_decider + train_router.build_text
(use_chunks, 6144) + bf16 autocast, batch 1, for each --spec; writes --out/pred_<spec with _>.json (decider format) plus split-name
copies for HotpotQA (val/test4/test5/test6). Same scoring path as the train_router5 E3 ablation predictions."""

import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer  # noqa: E402
from train_router import build_text  # noqa: E402
from train_router4 import load  # noqa: E402
from e2e_ent2 import load_decider  # noqa: E402

H = {
    "legacy_val": "val",
    "pilot8_hotpotqa_test4": "test4",
    "pilot8_hotpotqa_test5": "test5",
    "pilot8_hotpotqa_test6": "test6",
}
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--specs", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
n_out = json.load(open(f"{a.ckpt}/args.json")).get("n_out", 5)
m = load_decider(a.ckpt, n_out)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
os.makedirs(a.out, exist_ok=True)
for s in a.specs:
    rows = load(s)
    out = []
    for r in rows:
        enc = tok([build_text(r, tok, 6144, use_chunks=True)], return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(enc.input_ids, enc.attention_mask)[0].float().tolist()
        out.append(
            {
                "ds": r["ds"],
                "split": r["split"],
                "pool": r["pool"],
                "id": r["id"],
                **{k: o[i] for i, k in enumerate(["fE", "fN", "fT", "dEN", "dNT"])},
            }
        )
    tag = s.replace(":", "_")
    json.dump(out, open(f"{a.out}/pred_{tag}.json", "w"))
    if tag in H:
        json.dump(out, open(f"{a.out}/pred_{H[tag]}.json", "w"))
    print(s, len(out), flush=True)
