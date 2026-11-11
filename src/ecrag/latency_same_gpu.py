"""Same-GPU decider latency A/B: merged-LoRA HF decider (serve_opt) in this process and a vLLM pooling
decider served on the SAME GPU (started separately on that GPU), scored on the same stored inputs (--spec rows, pruned chunks),
alternating the order per row (ABBA), single request, synchronised. Parts per call: text (build_text_fast), tok (+H2D for HF), model
(HF forward | HTTP POST /pooling incl. server time), head (client MLP for vLLM). Also the HTTP round trip of an 8-token input (server
overhead floor). Records nvidia-smi compute processes on the GPU at start and end (other jobs present or not).
Usage: latency_same_gpu.py --spec S --ckpt <decider dir> --vllm_url http://127.0.0.1:P --vllm_model name --head <export dir> --n 200 --out f.json"""

import argparse, json, os, subprocess, sys, time, uuid
import numpy as np
import requests
import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer  # noqa: E402
from train_router4 import load  # noqa: E402
from e2e_ent2 import load_decider  # noqa: E402
from serve_opt import build_text_fast, merge_decider  # noqa: E402


def gpu_procs():
    try:
        return (
            subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except Exception as e:  # noqa: BLE001
        return [repr(e)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vllm_url", required=True)
    ap.add_argument("--vllm_model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    rows = load(a.spec)[: a.warm + a.n]
    s = requests.Session()
    sync = torch.cuda.synchronize
    dec = merge_decider(load_decider(a.ckpt, json.load(open(f"{a.ckpt}/args.json")).get("n_out", 5)))
    H = torch.load(f"{a.head}/head.pt")
    l1 = torch.nn.Linear(H["0.weight"].shape[1], H["0.weight"].shape[0])
    l2 = torch.nn.Linear(H["2.weight"].shape[1], H["2.weight"].shape[0])
    l1.load_state_dict({"weight": H["0.weight"], "bias": H["0.bias"]})
    l2.load_state_dict({"weight": H["2.weight"], "bias": H["2.bias"]})
    head = torch.nn.Sequential(l1, torch.nn.GELU(), l2).eval()
    res = {
        "gpu_procs_start": gpu_procs(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "args": vars(a),
    }

    def hf(r):
        sync()
        t0 = time.perf_counter()
        txt = build_text_fast(r, tok, 6144)
        t1 = time.perf_counter()
        enc = tok([txt], return_tensors="pt").to("cuda")
        sync()
        t2 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = dec(enc.input_ids, enc.attention_mask)[0].float().tolist()
        sync()
        t3 = time.perf_counter()
        return [t1 - t0, t2 - t1, t3 - t2, 0.0, t3 - t0], o[3]

    def vl(r):
        t0 = time.perf_counter()
        txt = build_text_fast(r, tok, 6144)
        t1 = time.perf_counter()
        ids = tok(txt).input_ids
        t2 = time.perf_counter()
        j = s.post(
            f"{a.vllm_url}/pooling",
            json={"model": a.vllm_model, "input": [ids], "cache_salt": uuid.uuid4().hex},
            timeout=60,
        ).json()
        t3 = time.perf_counter()
        with torch.no_grad():
            o = head(torch.tensor(j["data"][0]["data"], dtype=torch.float32)[None])[0].tolist()
        t4 = time.perf_counter()
        return [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0], o[3]

    T = {"hf_merged": [], "vllm": []}
    S = {"hf_merged": [], "vllm": []}
    F = {"hf_merged": hf, "vllm": vl}
    for i, r in enumerate(rows):
        for nm in ["hf_merged", "vllm"] if i % 2 == 0 else ["vllm", "hf_merged"]:
            t, sc = F[nm](r)
            if i >= a.warm:
                T[nm].append(t)
                S[nm].append(sc)
    tiny = []
    for _ in range(50):
        t0 = time.perf_counter()
        s.post(
            f"{a.vllm_url}/pooling",
            json={"model": a.vllm_model, "input": [[1, 2, 3, 4, 5, 6, 7, 8]], "cache_salt": uuid.uuid4().hex},
            timeout=60,
        ).json()
        tiny.append(time.perf_counter() - t0)
    for nm in T:
        t = np.array(T[nm]) * 1e3
        res[nm] = {
            **{f"{k}_mean_ms": float(t[:, i].mean()) for i, k in enumerate(["text", "tok", "model", "head", "total"])},
            "total_p50_ms": float(np.median(t[:, 4])),
            "total_p95_ms": float(np.percentile(t[:, 4], 95)),
            "model_p10_p50_p90_ms": [float(x) for x in np.percentile(t[:, 2], [10, 50, 90])],
        }
        print(nm, json.dumps(res[nm]), flush=True)
    res["http_8tok_roundtrip_ms"] = {"mean": float(np.mean(tiny) * 1e3), "p50": float(np.median(tiny) * 1e3)}
    d = np.abs(np.array(S["hf_merged"]) - np.array(S["vllm"]))
    res["dEN_absdiff"] = {"mean": float(d.mean()), "max": float(d.max())}
    res["gpu_procs_end"] = gpu_procs()
    print(json.dumps({k: res[k] for k in ["http_8tok_roundtrip_ms", "dEN_absdiff"]}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
