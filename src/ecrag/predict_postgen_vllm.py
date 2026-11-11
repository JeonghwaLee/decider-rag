"""Dump decider scores from the vLLM-served model (LAST-token pooling + MLP head), with or without
post-generation candidates (--post); the same pooling/head is used for both arms."""

import argparse, json, sys, uuid, time
from pathlib import Path
import requests, torch
from transformers import AutoTokenizer
from train_router4 import load
from train_router import build_text
from postgen_input import attach_candidates, build_postgen_text
from concurrent.futures import ThreadPoolExecutor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--specs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--post", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(1)
    h = torch.load(Path(a.export) / "head.pt", map_location="cpu")
    head = torch.nn.Sequential(
        torch.nn.Linear(h["0.weight"].shape[1], h["0.weight"].shape[0]),
        torch.nn.GELU(),
        torch.nn.Linear(h["2.weight"].shape[1], h["2.weight"].shape[0]),
    ).eval()
    head.load_state_dict(h)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    out = Path(a.out)
    out.mkdir(exist_ok=True, parents=True)
    for spec in a.specs:
        fn = out / ("pred_" + spec.replace(":", "_") + ".json")
        if fn.exists():
            raise FileExistsError(fn)
        rows = load(spec)
        if a.post:
            rows = attach_candidates(rows, spec)

        def predict(r):
            text = build_postgen_text(r, tok, build_text) if a.post else build_text(r, tok, 6144)
            ids = tok(text).input_ids
            response = requests.post(
                f"http://127.0.0.1:{a.port}/pooling",
                json={"model": a.model, "input": [ids], "cache_salt": uuid.uuid4().hex},
                timeout=120,
            )
            response.raise_for_status()
            with torch.no_grad():
                v = head(torch.tensor(response.json()["data"][0]["data"], dtype=torch.float32)[None])[0].tolist()
            return {
                "split": r["split"],
                "pool": r["pool"],
                "id": r["id"],
                "n_input_tokens": len(ids),
                **dict(zip(["fE", "fN", "fT", "dEN", "dNT"], v)),
            }

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(predict, rows))
        fn.write_text(json.dumps(res) + "\n")
        print("saved", fn, len(res), flush=True)


if __name__ == "__main__":
    main()
