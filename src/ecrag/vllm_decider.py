"""Decider served by vLLM (pooling, LAST token, no normalisation; export_decider_vllm.py) + the MLP head on the client.
1. Agreement: on --spec rows, scores vs the HF decider predictions (--ref pred json: dEN/dNT max |diff|, decision agreement at the
   frozen threshold --thr).
2. Latency, single request, cold (each question is a new prompt; vLLM prefix caching is disabled per request by a unique cache
   salt): build_text_fast + HTTP + head, as one decision. --n questions after --warm others.
Output json --out.
3. --dump <dir> --dump_specs S...: pred_<tag>.json for every row of each spec (decider pred format: split, pool, id, fE, fN, fT, dEN,
   dNT), --conc concurrent requests, so the rank comparison can be redone with the served model's own scores."""

import argparse, json, os, sys, time, uuid
import numpy as np
import requests
import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer  # noqa: E402
from train_router4 import load  # noqa: E402
from serve_opt import build_text_fast  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8196")
    ap.add_argument("--model", default="dec-s1234")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--dump_specs", nargs="*", default=[])
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--no_latency", action="store_true")
    a = ap.parse_args()
    H = torch.load(f"{a.dir}/head.pt")
    W1, b1, W2, b2 = (H[k].numpy() for k in ["0.weight", "0.bias", "2.weight", "2.bias"])
    gelu = lambda x: 0.5 * x * (1 + np.vectorize(__import__("math").erf)(x / np.sqrt(2)))
    head = lambda h: (gelu(h @ W1.T + b1) @ W2.T + b2)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    rows = load(a.spec)
    s = requests.Session()

    def decide(r):
        t0 = time.perf_counter()
        txt = build_text_fast({"query": r["query"], "k": r["k"], "chunks": r["chunks"]}, tok, 6144)
        ids = tok(txt).input_ids  # same tokenisation as training (no special tokens added by the Qwen tokenizer)
        j = s.post(f"{a.url}/pooling", json={"model": a.model, "input": [ids], "cache_salt": uuid.uuid4().hex}).json()
        h = np.array(j["data"][0]["data"], dtype=np.float32)
        o = head(h)
        return o, time.perf_counter() - t0, len(ids)

    res = {}
    if a.dump:
        from concurrent.futures import ThreadPoolExecutor

        os.makedirs(a.dump, exist_ok=True)
        for sp in a.dump_specs:
            fn = f"{a.dump}/pred_{sp.replace(':', '_')}.json"
            if os.path.exists(fn):
                print("exists, skip", fn, flush=True)
                continue
            R2 = load(sp)
            with ThreadPoolExecutor(a.conc) as ex:
                O = list(ex.map(lambda r: decide(r)[0], R2))
            json.dump(
                [
                    {
                        "split": r["split"],
                        "pool": r["pool"],
                        "id": r["id"],
                        **{k: float(o[i]) for i, k in enumerate(["fE", "fN", "fT", "dEN", "dNT"])},
                    }
                    for r, o in zip(R2, O)
                ],
                open(fn, "w"),
            )
            print("dumped", fn, len(O), flush=True)
    # agreement on all rows of the spec
    if a.ref:
        R = {(x["pool"], x["id"]): x for x in json.load(open(a.ref))}
        D = []
        for r in rows:
            o, _, _ = decide(r)
            x = R[(r["pool"], r["id"])]
            D.append((o[3], x["dEN"], o[4], x["dNT"]))
        D = np.array(D)
        res["agree"] = {
            "n": len(D),
            "dEN_maxdiff": float(np.abs(D[:, 0] - D[:, 1]).max()),
            "dNT_maxdiff": float(np.abs(D[:, 2] - D[:, 3]).max()),
            "decision_agree": float(((D[:, 0] > a.thr) == (D[:, 1] > a.thr)).mean()) if a.thr is not None else None,
        }
        print(json.dumps(res["agree"]), flush=True)
    if a.no_latency:
        json.dump(res, open(a.out, "w"), indent=1)
        return
    for r in rows[a.n : a.n + a.warm]:
        decide(r)
    T, N = [], []
    for r in rows[: a.n]:
        _, dt, n = decide(r)
        T.append(dt)
        N.append(n)
    res["latency"] = {
        "n": len(T),
        "mean_s": float(np.mean(T)),
        "p50_s": float(np.median(T)),
        "mean_tokens": float(np.mean(N)),
    }
    print(json.dumps(res["latency"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
