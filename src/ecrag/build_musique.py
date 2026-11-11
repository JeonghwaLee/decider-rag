"""MuSiQue-Ans (bdsaglam/musique, v1.0) in the EC-RAG pool format, built like the HotpotQA pools
(build_data.hotpot): each question contributes one document = its own 20 paragraphs (supporting + MuSiQue's distractors), padded
with random paragraphs of the same source split up to a length drawn from LongBench musique, shuffled; POOL = 200 questions
(= 200 documents) per pool. Answers = [answer] + answer_aliases.
  train  10 pools, val 2 pools  from MuSiQue train (disjoint questions)
  test   4 pools                from MuSiQue dev, excluding LongBench musique questions (LongBench is drawn from dev)
Pools: data/ecrag/musique/<split>/pool_XX. Audit: data/ecrag/leakage_audit_musique.json."""

from paths import DATA, HF_HUB  # noqa: E402
import json, os, random, zipfile
import build_data as B

D = DATA
SRC = f"{HF_HUB}/datasets--bdsaglam--musique/snapshots/22873a405dd809893b22ada0b499299fb612d2df"
LBZ = f"{HF_HUB}/datasets--THUDM--LongBench/snapshots/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip"


def build(rows, split, n_pools, rng, lb_len):
    paras = [(p["title"], p["paragraph_text"]) for r in rows for p in r["paragraphs"]]
    for p in range(n_pools):
        d = f"{D}/musique/{split}/pool_{p:02d}"
        assert not os.path.exists(d), d
        docs, qs = [], []
        for j, r in enumerate(rows[p * B.POOL : (p + 1) * B.POOL]):
            own = [(x["title"], x["paragraph_text"]) for x in r["paragraphs"]]
            ps = list(own)
            target = rng.choice(lb_len)
            while sum(len(x[1].split()) for x in ps) < target:
                ps.append(rng.choice(paras))
            rng.shuffle(ps)
            docs.append("\n".join(f"Passage {i + 1}:\n{t}\n{x}" for i, (t, x) in enumerate(ps)))
            qs.append(
                {
                    "query": r["question"],
                    "answers": [r["answer"]] + list(r.get("answer_aliases") or []),
                    "doc": j + 1,
                    "meta": {
                        "src_id": r["id"],
                        "hops": r["id"].split("__")[0],
                        "support_titles": sorted({x["title"] for x in r["paragraphs"] if x["is_supporting"]}),
                    },
                }
            )
        B.write_pool(d, docs, qs)


def main():
    z = zipfile.ZipFile(LBZ)
    lb = [json.loads(l) for l in z.open("data/musique.jsonl")]
    os.makedirs(f"{D}/raw/longbench/data", exist_ok=True)
    if not os.path.exists(f"{D}/raw/longbench/data/musique.jsonl"):
        with open(f"{D}/raw/longbench/data/musique.jsonl", "w") as f:
            for r in lb:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lb_q = {B.norm(r["input"]) for r in lb}
    lb_len = [int(r["length"]) for r in lb]
    tr = [json.loads(l) for l in open(f"{SRC}/musique_ans_v1.0_train.jsonl")]
    dv = [json.loads(l) for l in open(f"{SRC}/musique_ans_v1.0_dev.jsonl")]
    tr = [r for r in tr if B.norm(r["question"]) not in lb_q]
    dvk = [r for r in dv if B.norm(r["question"]) not in lb_q]
    rng = random.Random(7301)
    rng.shuffle(tr)
    rng.shuffle(dvk)
    build(tr[:2000], "train", 10, rng, lb_len)
    build(tr[2000:2400], "val", 2, rng, lb_len)
    build(dvk[:800], "test", 4, rng, lb_len)
    seen = {}
    for s in ["train", "val", "test"]:
        seen[s] = set()
        for p in os.listdir(f"{D}/musique/{s}"):
            seen[s] |= {B.norm(json.loads(l)["query"]) for l in open(f"{D}/musique/{s}/{p}/questions/questions.jsonl")}
    audit = {
        "dev_total": len(dv),
        "dev_in_longbench_removed": len(dv) - len(dvk),
        **{
            s: {"n": len(q), "in_longbench": len(q & lb_q), **{f"in_{o}": len(q & seen[o]) for o in seen if o != s}}
            for s, q in seen.items()
        },
    }
    json.dump(audit, open(f"{D}/leakage_audit_musique.json", "w"), indent=1)
    print(audit)


if __name__ == "__main__":
    main()
