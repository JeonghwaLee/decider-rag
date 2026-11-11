"""Fresh confirmation pools ("test3"). test2 has been looked at; every new policy decided after that must be reconfirmed here.

- hotpotqa: HotpotQA validation, excluding LongBench questions and every test2 question (by source id and by text).
- qasper: Qasper test split, excluding LongBench papers/questions and every paper used in test2 (by paper id), so no
  test3 question shares a paper with test2. As many pools as the remaining papers allow (up to --pools).
Construction is otherwise identical to build_test2.py / build_data.py (seed differs). Writes data/ecrag/leakage_audit_test3.json.
"""

from paths import DATA  # noqa: E402
import argparse, json, os, random
import pandas as pd
import build_data as B  # _hotpot below is build_test2.hotpot_dev with test2 exclusions and split test3

D = DATA


def used(ds, split, key):
    out = set()
    for p in os.listdir(f"{D}/{ds}/{split}"):
        f = f"{D}/{ds}/{split}/{p}/meta.jsonl" if key else f"{D}/{ds}/{split}/{p}/questions/questions.jsonl"
        for l in open(f):
            x = json.loads(l)
            out.add(x[key] if key else B.norm(x["query"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", type=int, default=4)
    ap.add_argument("--seed", type=int, default=5678)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lb = {n: [json.loads(l) for l in open(f"{B.LB}/{n}.jsonl")] for n in ["hotpotqa", "qasper"]}
    lb_q = {B.norm(r["input"]) for n in lb for r in lb[n]}
    # hotpotqa: drop test2 questions, then reuse the test2 constructor on the filtered frame
    t2_ids, t2_q = used("hotpotqa", "test2", "src_id"), used("hotpotqa", "test2", None)
    orig = pd.read_parquet

    def filtered(path, *x, **k):
        df = orig(path, *x, **k)
        if "validation-00000" in path:
            df = df[~df.id.isin(t2_ids) & ~df.question.map(B.norm).isin(t2_q)].reset_index(drop=True)
        return df

    _hotpot(a.pools, rng, lb_q, filtered)
    # qasper: exclude test2 papers
    t2_papers = used("qasper", "test2", "paper_id")
    lb_titles = set()
    qdf = orig(f"{B.QASPER}/test/0000.parquet")
    keep_titles = set(qdf[qdf.id.isin(t2_papers)].title.map(B.norm))
    n = B.qasper(D, "test", "test3", a.pools, rng, lb_q, lb_titles | keep_titles)
    # audit
    audit = {"qasper_pools": n}
    seen = {ds: set() for ds in ["hotpotqa", "qasper"]}
    for ds in seen:
        for split in ["train", "val", "test", "test2"]:
            for p in os.listdir(f"{D}/{ds}/{split}"):
                f = f"{D}/{ds}/{split}/{p}/questions/questions.jsonl"
                if os.path.exists(f):
                    seen[ds] |= {B.norm(json.loads(l)["query"]) for l in open(f)}
    for ds in seen:
        q, papers = [], set()
        for p in sorted(os.listdir(f"{D}/{ds}/test3")):
            for l in open(f"{D}/{ds}/test3/{p}/questions/questions.jsonl"):
                q.append(B.norm(json.loads(l)["query"]))
            for l in open(f"{D}/{ds}/test3/{p}/meta.jsonl"):
                papers.add(json.loads(l).get("paper_id"))
        audit[ds] = {
            "n": len(q),
            "q_in_longbench": sum(x in lb_q for x in q),
            "q_in_other_splits": sum(x in seen[ds] for x in q),
        }
        if ds == "qasper":
            audit[ds]["papers_shared_with_test2"] = len(papers & t2_papers)
    json.dump(audit, open(f"{D}/leakage_audit_test3.json", "w"), indent=1)
    print(audit)


def _hotpot(n_pools, rng, lb_q, reader):
    df = reader(f"{B.HOTPOT}/validation-00000-of-00001.parquet")
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
        B.write_pool(f"{D}/hotpotqa/test3/pool_{p:02d}", docs, qs)


if __name__ == "__main__":
    main()
