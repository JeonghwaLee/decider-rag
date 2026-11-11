"""E/N/T decider: Qwen3-4B + LoRA + head on the last token, q+k input (same text as the q+k F1 router).
Five regression outputs: F1 of E, N, T and the gains N-E, T-N (MSE, equal weights). Labels from ent_data.load(var='A')
(truncated T answers score 0). Train: HotpotQA train; epoch selection: validation loss. Predictions for val/test2/test3."""

import argparse, json, os, random, sys, time
import torch, torch.nn as nn
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from train_router import Router, build_text, batches, encode  # noqa: E402
import ent_data  # noqa: E402

KEYS = ["fE", "fN", "fT", "dEN", "dNT"]


def targets(r):
    return [r["fE"], r["fN"], r["fT"], r["fN"] - r["fE"], r["fT"] - r["fN"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--use_chunks", action="store_true", help="q+D+k input (pruned chunks, max 6144 tokens)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    tok.padding_side = "right"
    data = {s: ent_data.load(s, "A", with_text=a.use_chunks) for s in ["train", "val", "test2", "test3"]}
    print({s: len(v) for s, v in data.items()}, flush=True)
    text = lambda r: build_text(
        {"query": r["query"], "k": r["k"], "chunks": r["chunks"]}, tok, 6144, use_chunks=a.use_chunks
    )
    model = Router("Qwen/Qwen3-4B")
    h = model.head[0].in_features
    model.head = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Linear(512, len(KEYS))).float()
    model = model.cuda()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    @torch.no_grad()
    def predict(rows):
        model.eval()
        out = []
        for b in batches(rows, 4 if a.use_chunks else 16, False, None):
            ids, m = encode(tok, [text(r) for r in b], "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out += model(ids, m).float().tolist()
        model.train()
        return out

    def vloss(rows):
        P = torch.tensor(predict(rows))
        Y = torch.tensor([targets(r) for r in rows])
        return float(((P - Y) ** 2).mean())

    best, log = 1e9, []
    for ep in range(a.epochs):
        t0 = time.time()
        tl = []
        for b in batches(data["train"], a.bs, True, rng):
            ids, m = encode(tok, [text(r) for r in b], "cuda")
            y = torch.tensor([targets(r) for r in b], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(ids, m).float()
            loss = ((p - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(float(loss))
        vl = vloss(data["val"])
        log.append({"epoch": ep + 1, "train_loss": sum(tl) / len(tl), "val_mse": vl, "sec": time.time() - t0})
        print(log[-1], flush=True)
        if vl < best:
            best = vl
            torch.save(
                {k: v for k, v in model.state_dict().items() if "lora" in k or k.startswith("head")}, f"{a.out}/best.pt"
            )
    json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    model.load_state_dict(torch.load(f"{a.out}/best.pt"), strict=False)
    for s in ["val", "test2", "test3"]:
        P = predict(data[s])
        json.dump(
            [{"split": r["split"], "pool": r["pool"], "id": r["id"], **dict(zip(KEYS, p))} for r, p in zip(data[s], P)],
            open(f"{a.out}/pred_{s}.json", "w"),
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
