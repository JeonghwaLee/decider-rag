"""q+D decider router for the EC-RAG pilot: Qwen3-4B + LoRA (attention projections, r=16, alpha=32, like our C decider)
+ a score head on the last token.

Two heads, trained jointly:
- reg: predicts dq = F1_cloud - F1_edge (MSE).
- cls: predicts P(cloud better by more than t) (BCE). Hybrid-LLM-style relaxed label y = 1[dq > t], with t fixed beforehand
  (--t, F1 units).
Routing score for the budget curves = predicted dq, or the cls logit (both saved; choose on validation).
Input text = question + pruned chunks (EC-RAG's D, in rerank order) + k, truncated to --max_len tokens. The chunks are cut
from the end first, so the question is never truncated.
Training uses the train split of both datasets together. Validation is used for epoch selection only. Test predictions are
written once at the end.
"""

import argparse, json, math, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(__file__))
from router_data import load  # noqa: E402


def build_text(r, tok, max_len, use_chunks=True):
    head = f"Question: {r['query']}\n\nRetrieved context ({r['k']} chunks):\n"
    tail = f"\n\nNumber of retained chunks: {r['k']}\nWill a much larger cloud model answer this question better than the small edge model?"
    if not use_chunks:
        return head.split("\n\nRetrieved")[0] + tail
    budget = max_len - len(tok(head + tail, add_special_tokens=False).input_ids) - 8
    body, used = [], 0
    for i, c in enumerate(r["chunks"]):
        ids = tok(f"[{i + 1}] {c}\n", add_special_tokens=False).input_ids
        if used + len(ids) > budget:
            ids = ids[: max(0, budget - used)]
            if ids:
                body.append(tok.decode(ids))
            break
        body.append(f"[{i + 1}] {c}\n")
        used += len(ids)
    return head + "".join(body) + tail


class Router(nn.Module):
    def __init__(self, path, r=16, alpha=32):
        super().__init__()
        base = AutoModel.from_pretrained(path, dtype=torch.bfloat16)
        base.config.use_cache = False
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        self.enc = get_peft_model(
            base,
            LoraConfig(
                r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
        )
        h = base.config.hidden_size
        self.head = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Linear(512, 2)).float()

    def forward(self, ids, mask):
        out = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        last = mask.sum(1) - 1
        x = out[torch.arange(out.size(0), device=out.device), last].float()
        return self.head(x)  # [:,0]=dq regression, [:,1]=cls logit


def batches(items, bs, shuffle, rng):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i : i + bs]]


def encode(tok, texts, device):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=8192)
    return enc.input_ids.to(device), enc.attention_mask.to(device)


@torch.no_grad()
def predict(model, tok, rows, texts, bs, device):
    model.eval()
    out = []
    order = sorted(range(len(rows)), key=lambda i: len(texts[i]))
    preds = [None] * len(rows)
    for i in range(0, len(order), bs):
        b = order[i : i + bs]
        ids, m = encode(tok, [texts[j] for j in b], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(ids, m)
        for j, v in zip(b, y.tolist()):
            preds[j] = v
    model.train()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=6144)
    ap.add_argument("--t", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no_chunks", action="store_true", help="q-only control")
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "qasper"])
    ap.add_argument(
        "--pred_splits",
        nargs="+",
        default=["test"],
        help="splits predicted once at the end (missing splits are skipped)",
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    from router_data import TRACE

    have = lambda s: all(os.path.isdir(f"{TRACE}/{ds}/{s}") for ds in a.datasets)
    data = {
        s: [r for ds in a.datasets for r in load(ds, s)]
        for s in ["train", "val"] + [p for p in a.pred_splits if have(p)]
    }
    texts = {s: [build_text(r, tok, a.max_len, not a.no_chunks) for r in data[s]] for s in data}
    print({s: len(v) for s, v in data.items()}, flush=True)
    model = Router(a.model).to(device)
    model.train()  # gradient checkpointing is only active in training mode
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * math.ceil(len(data["train"]) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    items = list(zip(data["train"], texts["train"]))
    best, log = None, []
    for ep in range(a.epochs):
        t0 = time.time()
        tot = 0.0
        n = 0
        for b in batches(items, a.bs, True, rng):
            ids, m = encode(tok, [x[1] for x in b], device)
            dq = torch.tensor([x[0]["dq"] for x in b], device=device)
            y = (dq > a.t).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(ids, m)
            loss = F.mse_loss(o[:, 0], dq) + F.binary_cross_entropy_with_logits(o[:, 1], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            tot += loss.item() * len(b)
            n += len(b)
        pv = predict(model, tok, data["val"], texts["val"], a.bs * 2, device)
        vmse = sum((p[0] - r["dq"]) ** 2 for p, r in zip(pv, data["val"])) / len(pv)
        vbce = sum(
            F.binary_cross_entropy_with_logits(torch.tensor(p[1]), torch.tensor(float(r["dq"] > a.t))).item()
            for p, r in zip(pv, data["val"])
        ) / len(pv)
        rec = {
            "epoch": ep + 1,
            "train_loss": tot / n,
            "val_mse": vmse,
            "val_bce": vbce,
            "val_loss": vmse + vbce,
            "sec": time.time() - t0,
        }
        log.append(rec)
        print(rec, flush=True)
        if best is None or rec["val_loss"] < best:
            best = rec["val_loss"]
            torch.save(
                {k: v for k, v in model.state_dict().items() if "lora" in k or k.startswith("head")}, f"{a.out}/best.pt"
            )
            json.dump(
                [
                    {"ds": r["ds"], "pool": r["pool"], "id": r["id"], "pred_dq": p[0], "pred_logit": p[1]}
                    for r, p in zip(data["val"], pv)
                ],
                open(f"{a.out}/pred_val.json", "w"),
            )
    json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    sd = torch.load(f"{a.out}/best.pt")
    model.load_state_dict(sd, strict=False)
    for s in [p for p in a.pred_splits if p in data]:
        pt = predict(model, tok, data[s], texts[s], a.bs * 2, device)
        json.dump(
            [
                {"ds": r["ds"], "pool": r["pool"], "id": r["id"], "pred_dq": p[0], "pred_logit": p[1]}
                for r, p in zip(data[s], pt)
            ],
            open(f"{a.out}/pred_{s}.json", "w"),
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
