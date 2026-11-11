"""Generic E/N(/T) decider training, extending train_router3.py to several data sources and an E/N-only head set.
Same model as train_router3 (Qwen3-4B + LoRA r16 + MLP head on the last token, q+D+k input via build_text, max 6144 tokens),
MSE on the targets, epoch chosen by validation MSE, labels with truncation A.
Data specs: 'legacy:<split>' = ent_data.load (HotpotQA pilot-6/7 roots), '<pilotN>:<ds>:<split>' = ent_data2.load(runs/<pilotN>).
--heads EN: [fE, fN, dEN] (dEN index 2; no T labels needed). ENT: [fE, fN, fT, dEN, dNT] (as train_router3; dEN index 3).
--tvar N: a truncated T is labelled with the N answer's F1 (the deployed T -> N retry); default A = 0.
--soft/--alpha/--unl: distillation from teacher predictions (labelled rows: alpha*label + (1-alpha)*teacher; unlabelled
  trace:<pilotN>:<ds>:<split> rows: teacher only); validation always on true labels.
--ema d: exponential moving average of the trainable weights (validation, best.pt); off by default.
--predict_only: load <out>/best.pt (or --ckpt) and write predictions for --pred specs only.
Outputs in --out: args.json (with n_out, keys), log.json, best.pt, pred_<tag>.json for every --pred spec."""

from paths import RUNS  # noqa: E402
import argparse, json, os, random, sys, time
import torch, torch.nn as nn
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from train_router import Router, build_text, batches, encode  # noqa: E402
import ent_data, ent_data2  # noqa: E402

R = RUNS
HEADS = {"EN": ["fE", "fN", "dEN"], "ENT": ["fE", "fN", "fT", "dEN", "dNT"]}


TVAR = {
    "v": "A"
}  # --tvar: label of a truncated T; A = 0 (default, as before), N = the N answer's F1 (deployed T -> N retry)


def load_trace(root, ds, split):
    """unlabeled rows (decider inputs only) from stage-1 traces: 'trace:<pilotN>:<ds>:<split>'."""
    import glob

    rows = []
    for d in sorted(glob.glob(f"{R}/{root}/trace/{ds}/{split}/pool_*")):
        if not os.path.exists(f"{d}/DONE"):
            continue
        pool = d.split("/")[-1]
        ch = {json.loads(l)["node_id"]: json.loads(l)["text"] for l in open(f"{d}/chunks.jsonl")}
        for l in open(f"{d}/trace.jsonl"):
            t = json.loads(l)
            rows.append(
                {
                    "ds": ds,
                    "split": split,
                    "pool": pool,
                    "id": t["id"],
                    "query": t["query"],
                    "k": t["k"],
                    "chunks": [ch[c] for c in t["topk_ids"][: t["k"]]],
                    "fE": None,
                    "fN": None,
                    "fT": None,
                    "unl": True,
                }
            )
    return rows


def load(spec):
    p = spec.split(":")
    if p[0] == "trace":
        return load_trace(p[1], p[2], p[3])
    if p[0] == "legacy":
        rows = ent_data.load(p[1], "A", with_text=True)
        for r in rows:
            r["ds"] = "hotpotqa"
    else:
        rows = ent_data2.load(f"{R}/{p[0]}", p[1], p[2], "A", with_text=True)
    if TVAR["v"] == "N":
        for r in rows:
            if r["fT"] is not None and r.get("T_trunc"):
                r["fT"] = r["fN"]
    return rows


def targets(r, keys):
    if r["fE"] is None:  # unlabeled row
        v = {}
    else:
        v = {
            "fE": r["fE"],
            "fN": r["fN"],
            "fT": r["fT"],
            "dEN": r["fN"] - r["fE"],
            "dNT": (r["fT"] - r["fN"]) if r["fT"] is not None else None,
        }
    if "soft" in r:  # distillation: alpha * label + (1 - alpha) * teacher mean; teacher only where the label is missing
        al = SOFT["alpha"]
        return [r["soft"][k] if v.get(k) is None else al * v[k] + (1 - al) * r["soft"][k] for k in keys]
    return [v.get(k) for k in keys]


SOFT = {"alpha": 1.0}


def attach_soft(rows, spec, pats):
    """teacher mean of the five heads for each row of a train spec, from pred files '<pattern with {tag}>' (tag = spec with _)."""
    tag = spec.replace(":", "_")
    T = [{(x["pool"], x["id"]): x for x in json.load(open(p.format(tag=tag)))} for p in pats]
    for r in rows:
        r["soft"] = {k: sum(t[(r["pool"], r["id"])][k] for t in T) / len(T) for k in HEADS["ENT"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--pred", nargs="*", default=[])
    ap.add_argument("--heads", default="EN")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--predict_only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument(
        "--mask_missing",
        action="store_true",
        help="ENT heads: rows without T labels (e.g. train2) keep fE/fN/dEN, T terms masked",
    )
    ap.add_argument("--base", default="Qwen/Qwen3-4B", help="decider backbone (same tokenizer family)")
    ap.add_argument(
        "--n_out",
        type=int,
        default=None,
        help="head size of --ckpt when it differs from --heads (5 = train_router3 deciders)",
    )
    ap.add_argument(
        "--soft",
        default=None,
        help="distillation: comma-separated teacher pred patterns with {tag} (e.g. <dir>/pred_{tag}.json)",
    )
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the true label in the distillation target")
    ap.add_argument(
        "--unl", nargs="*", default=[], help="unlabeled train specs (trace:<pilotN>:<ds>:<split>), teacher targets only"
    )
    ap.add_argument(
        "--tvar", default="A", choices=["A", "N"], help="truncated-T label: A = 0 (default), N = F1 of the N retry"
    )
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N train and val rows only")
    ap.add_argument(
        "--ema",
        type=float,
        default=0.0,
        help="EMA decay of the trainable weights (0 = off); validation and best.pt use the EMA weights",
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    keys = HEADS[a.heads]
    n_out = a.n_out or len(keys)
    TVAR["v"] = a.tvar
    if not a.predict_only:
        json.dump({**vars(a), "n_out": n_out, "keys": keys}, open(f"{a.out}/args.json", "w"), indent=1)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    tok.padding_side = "right"
    text = lambda r: build_text({"query": r["query"], "k": r["k"], "chunks": r["chunks"]}, tok, 6144, use_chunks=True)
    model = Router(a.base)
    h = model.head[0].in_features
    model.head = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Linear(512, n_out)).float()
    model = model.cuda()

    @torch.no_grad()
    def predict(rows):
        model.eval()
        out = []
        for b in batches(rows, 4, False, None):
            ids, m = encode(tok, [text(r) for r in b], "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out += model(ids, m).float().tolist()
        model.train()
        return out

    if not a.predict_only:
        tr = []
        for s in a.train + a.unl:
            rs = load(s)
            if a.soft:
                attach_soft(rs, s, a.soft.split(","))
            tr += rs
        va = [r for s in a.val for r in load(s)]
        SOFT["alpha"] = a.alpha if a.soft else 1.0
        if "dNT" in keys and not a.mask_missing:
            tr = [r for r in tr if r["fT"] is not None or "soft" in r]
        if a.limit:
            tr, va = tr[: a.limit], va[: a.limit]
        print({"train": len(tr), "val": len(va)}, flush=True)
        tp = {n: p for n, p in model.named_parameters() if p.requires_grad}
        opt = torch.optim.AdamW(list(tp.values()), lr=a.lr)
        best, log = 1e9, []
        ema = {n: p.detach().float().clone() for n, p in tp.items()} if a.ema > 0 else None

        def swap():  # exchange the live and EMA weights (call twice to restore)
            with torch.no_grad():
                for n, p in tp.items():
                    cur = p.detach().float().clone()
                    p.copy_(ema[n].to(p.dtype))
                    ema[n] = cur

        for ep in range(a.epochs):
            t0 = time.time()
            tl = []
            for b in batches(tr, a.bs, True, rng):
                ids, m = encode(tok, [text(r) for r in b], "cuda")
                yv = [targets(r, keys) for r in b]
                mk = torch.tensor([[0.0 if v is None else 1.0 for v in t] for t in yv], device="cuda")
                y = torch.tensor([[0.0 if v is None else v for v in t] for t in yv], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model(ids, m).float()
                loss = (((p - y) ** 2) * mk).sum() / mk.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl.append(float(loss))
                if ema is not None:
                    with torch.no_grad():
                        for n, q in tp.items():
                            ema[n].mul_(a.ema).add_(q.detach().float(), alpha=1 - a.ema)
            if ema is not None:
                swap()
            sv = SOFT["alpha"]
            SOFT["alpha"] = 1.0  # validation always on the true labels
            P = torch.tensor(predict(va))
            Y = torch.tensor([targets(r, keys) for r in va])
            SOFT["alpha"] = sv
            vl = float(((P - Y) ** 2).mean())
            log.append({"epoch": ep + 1, "train_loss": sum(tl) / len(tl), "val_mse": vl, "sec": time.time() - t0})
            print(log[-1], flush=True)
            if vl < best:
                best = vl
                torch.save(
                    {k: v for k, v in model.state_dict().items() if "lora" in k or k.startswith("head")},
                    f"{a.out}/best.pt",
                )
            if ema is not None:
                swap()
        json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    model.load_state_dict(torch.load(f"{a.ckpt or a.out}/best.pt"), strict=False)
    ok = HEADS["ENT"] if n_out == 5 else keys
    for s in a.pred:
        rows = load(s)
        tag = s.replace(":", "_")
        fn = f"{a.out}/pred_{tag}.json"
        if os.path.exists(fn):
            print("exists, skip", fn)
            continue
        P = predict(rows)
        json.dump(
            [
                {"ds": r["ds"], "split": r["split"], "pool": r["pool"], "id": r["id"], **dict(zip(ok, p))}
                for r, p in zip(rows, P)
            ],
            open(fn, "w"),
        )
        print("pred", s, len(rows), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
