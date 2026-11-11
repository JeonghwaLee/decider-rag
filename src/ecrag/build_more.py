"""More HotpotQA pools, construction identical to build_data.hotpot / build_test3._hotpot (POOL 200, LongBench-length
padding with random distractor paragraphs), new seeds:
- train2: from the HotpotQA *train* split, excluding every question already in hotpotqa/train or /val (src_id and text) and
  LongBench questions. For training a larger E/N decider (selection on the existing val only).
- test4: fresh confirmation pools from the HotpotQA *validation* split, excluding test2/test3 questions (src_id and text) and
  LongBench questions. Opened once, after the policies were frozen.
Distractor paragraphs are drawn from the same source split as in the earlier builders. Writes data/ecrag/leakage_audit_more.json."""

from paths import DATA  # noqa: E402
import argparse, json, os, random
import pandas as pd
import build_data as B

D = DATA


def used(split):
    ids, q = set(), set()
    for p in os.listdir(f"{D}/hotpotqa/{split}"):
        if os.path.exists(f"{D}/hotpotqa/{split}/{p}/meta.jsonl"):  # hotpotqa/test (LongBench copy) has no meta
            for l in open(f"{D}/hotpotqa/{split}/{p}/meta.jsonl"):
                ids.add(json.loads(l).get("src_id"))
        for l in open(f"{D}/hotpotqa/{split}/{p}/questions/questions.jsonl"):
            q.add(B.norm(json.loads(l)["query"]))
    return ids, q


def build(df, split, n_pools, rng, lb_len):
    idx = rng.sample(range(len(df)), n_pools * B.POOL)
    paras = [(t, " ".join(s)) for c in df.context for t, s in zip(c["title"], c["sentences"])]
    for p in range(n_pools):
        d = f"{D}/hotpotqa/{split}/pool_{p:02d}"
        assert not os.path.exists(d), d
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
        B.write_pool(d, docs, qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pools", type=int, default=20)
    ap.add_argument("--test_pools", type=int, default=4)
    ap.add_argument("--audit_only", action="store_true")
    a = ap.parse_args()
    lb_q = {B.norm(json.loads(l)["input"]) for n in ["hotpotqa", "qasper"] for l in open(f"{B.LB}/{n}.jsonl")}
    lb_len = [json.loads(l)["length"] for l in open(f"{B.LB}/hotpotqa.jsonl")]
    ex_ids, ex_q = set(), set()
    for s in [] if a.audit_only else ["train", "val"]:
        i, q = used(s)
        ex_ids |= i
        ex_q |= q
    if not a.audit_only:
        build_all(a, lb_q, lb_len)
    audit_(lb_q)


def build_all(a, lb_q, lb_len):
    ex_ids, ex_q = set(), set()
    for s in ["train", "val"]:
        i, q = used(s)
        ex_ids |= i
        ex_q |= q
    tr = pd.concat([pd.read_parquet(f"{B.HOTPOT}/train-0000{i}-of-00002.parquet") for i in range(2)], ignore_index=True)
    tr = tr[~tr.question.map(B.norm).isin(lb_q | ex_q) & ~tr.id.isin(ex_ids)].reset_index(drop=True)
    build(tr, "train2", a.train_pools, random.Random(9101), lb_len)
    ex_ids, ex_q = set(), set()
    for s in ["test2", "test3"]:
        i, q = used(s)
        ex_ids |= i
        ex_q |= q
    dv = pd.read_parquet(f"{B.HOTPOT}/validation-00000-of-00001.parquet")
    dv = dv[~dv.question.map(B.norm).isin(lb_q | ex_q) & ~dv.id.isin(ex_ids)].reset_index(drop=True)
    build(dv, "test4", a.test_pools, random.Random(9202), lb_len)


def audit_(lb_q):
    audit = {}
    allq = {}
    for s in os.listdir(f"{D}/hotpotqa"):
        allq[s] = used(s)
    for s in ["train2", "test4"]:
        ids, q = allq[s]
        audit[s] = {
            "n_text": len(q),
            "in_longbench": len(q & lb_q),
            "text_in_other_splits": {o: len(q & allq[o][1]) for o in allq if o != s},
            "src_id_in_other_splits": {o: len(ids & allq[o][0]) for o in allq if o != s},
        }
    json.dump(audit, open(f"{D}/leakage_audit_more.json", "w"), indent=1)
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
