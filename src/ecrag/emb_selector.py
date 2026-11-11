"""Cheap learned selector on passage embeddings. Features: bge-small-en-v1.5 (the
retriever's embedder, normalised) question embedding, mean of the k pruned passage embeddings, max/mean/min question-passage cosine,
k/10. Models: Ridge (alpha 1/10/100 picked on val MSE) and MLP (256 hidden, early stopping), heads dEN = fN - fE and dNT = fT - fN
(rows without T labels: dNT not trained). Output: --out/<model>/pred_<spec with _>.json in decider format (fE/fN/fT = 0),
plus split-name copies for HotpotQA (pred_val/test4/test5/test6.json).
Usage: emb_selector.py --train S... --val S --pred S... --out dir"""

from paths import HF_HUB  # noqa: E402
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_router4 import load  # noqa: E402

EMB = f"{HF_HUB}/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # = e2e_run.EMB (retriever embedder)

H = {
    "legacy_val": "val",
    "pilot8_hotpotqa_test4": "test4",
    "pilot8_hotpotqa_test5": "test5",
    "pilot8_hotpotqa_test6": "test6",
}


class Enc:
    """bge-small-en-v1.5: CLS pooling + L2 normalisation (as sentence-transformers / llama-index HuggingFaceEmbedding)"""

    def __init__(self, path):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModel.from_pretrained(path).cuda().eval().half()

    @torch.no_grad()
    def encode(self, texts, normalize_embeddings=True, batch_size=256):
        out = []
        for i in range(0, len(texts), batch_size):
            b = self.tok(
                texts[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            e = self.m(**b).last_hidden_state[:, 0].float()
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out)


def featurize(rows, model):
    qs = model.encode([r["query"] for r in rows], normalize_embeddings=True, batch_size=256)
    flat = [c for r in rows for c in r["chunks"]]
    pe = model.encode(flat, normalize_embeddings=True, batch_size=256)
    X = []
    i = 0
    for r, q in zip(rows, qs):
        p = pe[i : i + len(r["chunks"])]
        i += len(r["chunks"])
        p = p if len(p) else np.zeros((1, len(q)))
        s = p @ q
        X.append(np.concatenate([q, p.mean(0), [s.max(), s.mean(), s.min(), r["k"] / 10]]))
    return np.array(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor

    m = Enc(EMB)
    tr = [r for s in a.train for r in load(s) if r["fE"] is not None]
    va = load(a.val)
    Xt, Xv = featurize(tr, m), featurize(va, m)
    Y = {
        "dEN": (
            np.array([r["fN"] - r["fE"] for r in tr]),
            np.array([r["fN"] - r["fE"] for r in va]),
            np.ones(len(tr), bool),
        ),
        "dNT": (
            np.array([(r["fT"] - r["fN"]) if r["fT"] is not None else 0 for r in tr]),
            np.array([(r["fT"] - r["fN"]) if r["fT"] is not None else 0 for r in va]),
            np.array([r["fT"] is not None for r in tr]),
        ),
    }
    M = {"ridge": {}, "mlp": {}}
    log = {}
    for h, (yt, yv, mk) in Y.items():
        best = None
        for al in [1.0, 10.0, 100.0]:
            r = Ridge(alpha=al).fit(Xt[mk], yt[mk])
            e = float(((r.predict(Xv) - yv) ** 2).mean())
            if best is None or e < best[0]:
                best = (e, al, r)
        M["ridge"][h] = best[2]
        log[f"ridge_{h}"] = {"val_mse": best[0], "alpha": best[1]}
        mlp = MLPRegressor(hidden_layer_sizes=(256,), early_stopping=True, random_state=1234, max_iter=300).fit(
            Xt[mk], yt[mk]
        )
        M["mlp"][h] = mlp
        log[f"mlp_{h}"] = {"val_mse": float(((mlp.predict(Xv) - yv) ** 2).mean())}
    os.makedirs(a.out, exist_ok=True)
    json.dump({"args": vars(a), "n_train": len(tr), "log": log}, open(f"{a.out}/log.json", "w"), indent=1)
    print(log, flush=True)
    for s in a.pred:
        rows = load(s)
        X = featurize(rows, m)
        tag = s.replace(":", "_")
        for nm, hh in M.items():
            os.makedirs(f"{a.out}/{nm}", exist_ok=True)
            pe, pt = hh["dEN"].predict(X), hh["dNT"].predict(X)
            Yp = [
                {
                    "ds": r["ds"],
                    "split": r["split"],
                    "pool": r["pool"],
                    "id": r["id"],
                    "fE": 0.0,
                    "fN": 0.0,
                    "fT": 0.0,
                    "dEN": float(x),
                    "dNT": float(y),
                }
                for r, x, y in zip(rows, pe, pt)
            ]
            json.dump(Yp, open(f"{a.out}/{nm}/pred_{tag}.json", "w"))
            if tag in H:
                json.dump(Yp, open(f"{a.out}/{nm}/pred_{H[tag]}.json", "w"))
        print("pred", s, len(rows), flush=True)


if __name__ == "__main__":
    main()
