"""Extra held-out test pools ("test2") to enlarge the test set (n=200 per dataset is too small for CIs).

Sources:
- hotpotqa: HotpotQA *validation* (distractor). LongBench questions are excluded. Validation is disjoint from our train/val
  pools, which were drawn from the HotpotQA train split.
- qasper: Qasper *test* split. LongBench papers and questions are excluded as in build_data.py. It is disjoint from
  Qasper train/validation.
Construction is identical to build_data.py: 200 questions per pool, LongBench-style documents, Qasper pools padded to
200 documents. The original LongBench test (split "test") stays the paper-comparable set; test2 is reported separately
and pooled.
"""

from paths import DATA  # noqa: E402
import argparse, json, os, random
import pandas as pd
import build_data as B


def hotpot_dev(out, n_pools, rng, lb_q):
    df = pd.read_parquet(f"{B.HOTPOT}/validation-00000-of-00001.parquet")
    df = df[~df.question.map(B.norm).isin(lb_q)].reset_index(drop=True)
    lb_len = [json.loads(l)["length"] for l in open(f"{B.LB}/hotpotqa.jsonl")]
    idx = rng.sample(range(len(df)), n_pools * B.POOL)
    paras = [(t, " ".join(s)) for c in df.context for t, s in zip(c["title"], c["sentences"])]
    for p in range(n_pools):
        docs, qs = [], []
        for j, r in enumerate(idx[p * B.POOL : (p + 1) * B.POOL]):
            row = df.iloc[r]
            own = [(t, " ".join(s)) for t, s in zip(row.context["title"], row.context["sentences"])]
            target = rng.choice(lb_len)
            ps = list(own)
            while sum(len(x[1].split()) for x in ps) < target:
                ps.append(rng.choice(paras))
            rng.shuffle(ps)
            docs.append("\n".join(f"Passage {i + 1}:\n{t}\n{x}" for i, (t, x) in enumerate(ps)))
            qs.append(
                {
                    "query": row.question,
                    "answers": [row.answer],
                    "doc": j + 1,
                    "meta": {
                        "src_id": row.id,
                        "type": row.type,
                        "level": row.level,
                        "support_titles": sorted(set(row.supporting_facts["title"].tolist())),
                    },
                }
            )
        B.write_pool(f"{out}/hotpotqa/test2/pool_{p:02d}", docs, qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DATA)
    ap.add_argument("--pools", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4321)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lb = {n: [json.loads(l) for l in open(f"{B.LB}/{n}.jsonl")] for n in ["hotpotqa", "qasper"]}
    lb_q = {B.norm(r["input"]) for n in lb for r in lb[n]}
    hotpot_dev(a.out, a.pools, rng, lb_q)
    n = B.qasper(a.out, "test", "test2", a.pools, rng, lb_q, set())
    # audit: overlap of test2 questions with LongBench and with our train/val pools
    seen = {}
    for ds in ["hotpotqa", "qasper"]:
        for split in ["train", "val"]:
            base = f"{a.out}/{ds}/{split}"
            for p in os.listdir(base):
                for l in open(f"{base}/{p}/questions/questions.jsonl"):
                    seen.setdefault(ds, set()).add(B.norm(json.loads(l)["query"]))
    audit = {"qasper_pools": n}
    for ds in ["hotpotqa", "qasper"]:
        base = f"{a.out}/{ds}/test2"
        q = []
        for p in sorted(os.listdir(base)):
            q += [B.norm(json.loads(l)["query"]) for l in open(f"{base}/{p}/questions/questions.jsonl")]
        audit[ds] = {
            "n": len(q),
            "in_longbench": sum(x in lb_q for x in q),
            "in_trainval": sum(x in seen[ds] for x in q),
        }
    json.dump(audit, open(f"{a.out}/leakage_audit_test2.json", "w"), indent=1)
    print(audit)


if __name__ == "__main__":
    main()
