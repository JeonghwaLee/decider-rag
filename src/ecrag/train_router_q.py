"""Quality-target decider for the EC-RAG pilot. Same backbone and input as train_router.py
(Qwen3-4B + LoRA r16 on attention projections, last-token head; q+D or q+k input).

Labels: GLM-5.3-Flash judge scores of the fixed pilot-1 answers (CORRECT 1 / PARTIAL 0.5 / INCORRECT 0), per answer.
Outputs (one head, 4 numbers):
  0 qe: logit of edge quality      (BCE with soft target j_edge)
  1 qc: logit of cloud quality     (BCE with soft target j_cloud)
  2 g : judge gain j_cloud - j_edge (MSE)
  3 f : F1 dq, auxiliary           (MSE, weight --w_f1)
Routing scores saved for validation-side selection: sigmoid(qc) - sigmoid(qe), g, f.
- Training sample: the train split, with decisive pairs (exactly one side CORRECT) repeated --oversample times. Thresholds
  are always set later on the original validation distribution.
- Epoch selection: validation loss of the judge terms only (qe + qc + g).
- Validation per-group errors are logged every epoch: cloud-win, edge-win, both-high, both-low.
- Predictions on every split in --pred_splits that has stage-1 traces + cloud answers + (for metrics only) judge labels.
"""

import argparse, json, math, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from router_data import load, TRACE  # noqa: E402
from train_router import Router, build_text, batches, encode  # noqa: E402

S = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
TAG = os.environ.get("ECRAG_CLOUD_TAG", "")


def attach_judge(rows, name):
    cache = {}
    for r in rows:
        key = (r["ds"], r["split"], r["pool"])
        if key not in cache:
            p = f"{TRACE}/{key[0]}/{key[1]}/{key[2]}/judge_{name}{TAG}.jsonl"
            lab = {}
            if os.path.exists(p):
                for l in open(p):
                    x = json.loads(l)
                    if x["label"]:
                        lab[(x["id"], x["who"])] = S[x["label"]]
            cache[key] = lab
        lab = cache[key]
        r["je"], r["jc"] = lab.get((r["id"], "edge")), lab.get((r["id"], "cloud"))
    return rows


class Router4(Router):
    def __init__(self, path):
        super().__init__(path)
        h = self.head[0].in_features
        self.head = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Linear(512, 4)).float()


@torch.no_grad()
def predict(model, tok, texts, bs, device):
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    preds = [None] * len(texts)
    for i in range(0, len(order), bs):
        b = order[i : i + bs]
        ids, m = encode(tok, [texts[j] for j in b], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(ids, m)
        for j, v in zip(b, y.tolist()):
            preds[j] = v
    model.train()
    return preds


def group(r):
    if r["jc"] == 1 and r["je"] < 1:
        return "cloud_win"
    if r["je"] == 1 and r["jc"] < 1:
        return "edge_win"
    if r["je"] >= 0.5 and r["jc"] >= 0.5:
        return "both_high"
    return "both_low"


def judge_loss(o, rows, device):
    je = torch.tensor([r["je"] for r in rows], device=device)
    jc = torch.tensor([r["jc"] for r in rows], device=device)
    return (
        F.binary_cross_entropy_with_logits(o[:, 0], je)
        + F.binary_cross_entropy_with_logits(o[:, 1], jc)
        + F.mse_loss(o[:, 2], jc - je)
    )


def dump(rows, preds, path):
    json.dump(
        [
            {
                "ds": r["ds"],
                "split": r["split"],
                "pool": r["pool"],
                "id": r["id"],
                "qe": p[0],
                "qc": p[1],
                "g": p[2],
                "f": p[3],
                "score_q": 1 / (1 + math.exp(-p[1])) - 1 / (1 + math.exp(-p[0])),
            }
            for r, p in zip(rows, preds)
        ],
        open(path, "w"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="glm53")
    ap.add_argument("--max_len", type=int, default=6144)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--w_f1", type=float, default=0.25)
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--no_chunks", action="store_true", help="q+k input")
    ap.add_argument("--pred_splits", nargs="+", default=["test", "test2", "test3"])
    ap.add_argument("--predict_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda"
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = Router4(a.model).to(device)
    model.train()
    T = lambda rows: [build_text(r, tok, a.max_len, not a.no_chunks) for r in rows]
    if not a.predict_only:
        json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
        tr = attach_judge([r for ds in ["hotpotqa", "qasper"] for r in load(ds, "train")], a.judge)
        va = attach_judge([r for ds in ["hotpotqa", "qasper"] for r in load(ds, "val")], a.judge)
        miss = sum(r["je"] is None or r["jc"] is None for r in tr + va)
        assert miss == 0, f"{miss} rows without judge labels"
        items = [(r, t) for r, t in zip(tr, T(tr))]
        items += [x for x in items if (x[0]["je"] == 1) != (x[0]["jc"] == 1)] * (a.oversample - 1)
        tv = T(va)
        print({"train": len(tr), "train_items": len(items), "val": len(va)}, flush=True)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        steps = a.epochs * math.ceil(len(items) / a.bs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
        best, log = None, []
        for ep in range(a.epochs):
            t0 = time.time()
            tot = n = 0
            for b in batches(items, a.bs, True, rng):
                rows = [x[0] for x in b]
                ids, m = encode(tok, [x[1] for x in b], device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(ids, m)
                f1dq = torch.tensor([r["dq"] for r in rows], device=device)
                loss = judge_loss(o, rows, device) + a.w_f1 * F.mse_loss(o[:, 3], f1dq)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                tot += loss.item() * len(b)
                n += len(b)
            pv = predict(model, tok, tv, a.bs * 2, device)
            vl = judge_loss(torch.tensor(pv), va, "cpu").item()
            grp = {}
            for r, p in zip(va, pv):
                g = group(r)
                e = grp.setdefault(g, [0, 0.0, 0.0])
                e[0] += 1
                e[1] += abs(torch.sigmoid(torch.tensor(p[0])).item() - r["je"])
                e[2] += abs(torch.sigmoid(torch.tensor(p[1])).item() - r["jc"])
            rec = {
                "epoch": ep + 1,
                "train_loss": tot / n,
                "val_judge_loss": vl,
                "sec": time.time() - t0,
                "val_groups_mae_edge_cloud": {g: [e[0], e[1] / e[0], e[2] / e[0]] for g, e in grp.items()},
            }
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if best is None or vl < best:
                best = vl
                torch.save(
                    {k: v for k, v in model.state_dict().items() if "lora" in k or k.startswith("head")},
                    f"{a.out}/best.pt",
                )
                dump(va, pv, f"{a.out}/pred_val.json")
        json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    sd = torch.load(f"{a.out}/best.pt")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all("lora" not in k and not k.startswith("head") for k in missing)
    for s in a.pred_splits:
        try:
            rows = [r for ds in ["hotpotqa", "qasper"] for r in load(ds, s)]
        except Exception as e:
            print("skip split", s, repr(e)[:120], flush=True)
            continue
        if not rows:
            continue
        dump(rows, predict(model, tok, T(rows), a.bs * 2, device), f"{a.out}/pred_{s}.json")
        print("predicted", s, len(rows), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
