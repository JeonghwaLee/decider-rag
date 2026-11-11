"""Verification-sample-count sensitivity of the AutoMix-style cascade from the stored 8 samples.
For k in {1,2,4} (k=8 = the stored run), a subset of k of the 8 stored samples per question gives the score (official fraction
'Correct' among parsable ones in the subset; 0 if none parses). Subsets: --subset first (samples 0..k-1, the main setting) or
rand<seed> (k random samples per question, seeded). Verification cloud cost of a k-sample call = prompt tokens once + the output
tokens of the k used samples (counted with the verifier's tokenizer; the stored total output count of 8 rescales the token sum so
the k=8 cost equals the stored one). Latency is NOT derived from these files (k-sample calls are measured live).
Writes <out_dir>/<spec>.jsonl with the fields automix_cmp.ver() reads (+ parse flags).
Usage: automix_k.py --in_dir ver --out_dir ver_k2_first --k 2 --subset first --specs ..."""

from paths import CLOUD_MODEL  # noqa: E402
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(__file__))
from automix_verify import frac_correct  # noqa: E402

TOK = {"E": "Qwen/Qwen3-4B", "N": CLOUD_MODEL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--subset", default="first")
    ap.add_argument("--specs", nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    from transformers import AutoTokenizer

    toks = {}
    for spec in a.specs:
        fn = spec.replace(":", "_") + ".jsonl"
        out = f"{a.out_dir}/{fn}"
        if os.path.exists(out):
            print("exists, skip", out)
            continue
        X = [json.loads(l) for l in open(f"{a.in_dir}/{fn}")]
        cand = X[0]["cand"]
        tk = toks.setdefault(cand, AutoTokenizer.from_pretrained(TOK[cand]))
        rng = random.Random(int(a.subset[4:]) if a.subset.startswith("rand") else 0)
        with open(out, "w") as f:
            for x in X:
                S = x["samples"]
                idx = list(range(a.k)) if a.subset == "first" else sorted(rng.sample(range(len(S)), a.k))
                sub = [S[i] for i in idx]
                s, nv = frac_correct(sub)
                nt = [len(tk(t).input_ids) for t in S]
                scale = x["ver_out"] / max(1, sum(nt))
                out_k = scale * sum(nt[i] for i in idx)
                o = {
                    "pool": x["pool"],
                    "id": x["id"],
                    "cand": cand,
                    "k": a.k,
                    "subset": a.subset,
                    "idx": idx,
                    "score": s,
                    "n_valid": nv,
                    "all_unparsable": nv == 0,
                    "ver_in": x["ver_in"],
                    "ver_out": out_k,
                    "ver_cost": (0.27 * x["ver_in"] + 1.1 * out_k) / 1e3,
                    "ver_sec": x["ver_sec"],
                }
                f.write(json.dumps(o) + "\n")
        print("wrote", out, len(X))


if __name__ == "__main__":
    main()
