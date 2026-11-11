"""Predict with a trained router (best.pt from train_router.py) on another split, e.g. test2. No retraining.
It uses the run's own args (max_len, no_chunks). The cloud tag comes from ECRAG_CLOUD_TAG, the same as at training time.
Output: <run>/pred_<split>.json
"""

import argparse, json, os, sys
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from train_router import Router, build_text, predict  # noqa: E402
from router_data import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="test2")
    a = ap.parse_args()
    for run in a.runs:
        ra = json.load(open(f"{run}/args.json"))
        out = f"{run}/pred_{a.split}.json"
        if os.path.exists(out):
            print("exists", out)
            continue
        tok = AutoTokenizer.from_pretrained(ra["model"])
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        rows = [r for ds in ra["datasets"] for r in load(ds, a.split)]
        texts = [build_text(r, tok, ra["max_len"], not ra["no_chunks"]) for r in rows]
        model = Router(ra["model"]).to("cuda")
        sd = torch.load(f"{run}/best.pt")
        res = model.load_state_dict(sd, strict=False)
        assert not res.unexpected_keys and not any("lora" in k or k.startswith("head") for k in res.missing_keys), res
        p = predict(model, tok, rows, texts, ra["bs"] * 2, "cuda")
        json.dump(
            [
                {"ds": r["ds"], "pool": r["pool"], "id": r["id"], "pred_dq": x[0], "pred_logit": x[1]}
                for r, x in zip(rows, p)
            ],
            open(out, "w"),
        )
        print("wrote", out, len(rows), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
