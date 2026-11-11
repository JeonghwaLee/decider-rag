"""Build EC-RAG-format train/val pools for the edge-cloud router pilot.

Layout (same as EC-RAG data/<ds>/): <out>/<ds>/<split>/pool_XX/{documents/doc_i.txt, questions/questions.jsonl, answers/answers.jsonl, meta.jsonl}
- Each pool has 200 questions and is indexed separately, as in EC-RAG, where the corpus is the documents of 200 questions.
- test = the LongBench 200 shipped in the EC-RAG repo, copied unchanged.
- hotpotqa train/val come from the HotpotQA *train* split (distractor), because LongBench hotpotqa is drawn from dev.
  Each question's document is LongBench-style "Passage i:\nTitle\ntext".
  It holds the 10 own paragraphs plus random paragraphs from other questions in the same split, padded to about the LongBench
  length (sampled from the LongBench word-length distribution).
- qasper train comes from Qasper train and val from Qasper validation. LongBench qasper is not built from these splits; this is checked
  below by exact question and title overlap. Each document = title + abstract + sections.
  Answers follow the LongBench convention: free-form, else extractive spans joined with ', ', else yes/no, else 'Unanswerable'.
Evidence (HotpotQA supporting titles, Qasper evidence paragraphs) is stored in meta.jsonl for later chunk-relevance work.
"""

from paths import DATA, ECRAG_DATA, HF_HUB  # noqa: E402
import argparse, json, os, random, re, shutil
import pandas as pd

HF = HF_HUB
HOTPOT = f"{HF}/datasets--hotpotqa--hotpot_qa/snapshots/1908d6afbbead072334abe2965f91bd2709910ab/distractor"
QASPER = f"{HF}/datasets--allenai--qasper/snapshots/06806e4608976fc2fac0a090ac425d5b2b29caf4/qasper"
ECRAG_DATA = ECRAG_DATA
LB = f"{DATA}/raw/longbench/data"
POOL = 200


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def write_pool(d, docs, qs):
    """docs: list of str; qs: list of dict(query, answers, doc, meta)."""
    os.makedirs(f"{d}/documents", exist_ok=True)
    os.makedirs(f"{d}/questions", exist_ok=True)
    os.makedirs(f"{d}/answers", exist_ok=True)
    for i, t in enumerate(docs, 1):
        open(f"{d}/documents/doc_{i}.txt", "w", encoding="utf-8").write(t)
    with (
        open(f"{d}/questions/questions.jsonl", "w") as fq,
        open(f"{d}/answers/answers.jsonl", "w") as fa,
        open(f"{d}/meta.jsonl", "w") as fm,
    ):
        for i, q in enumerate(qs):
            fq.write(json.dumps({"id": i, "query": q["query"]}, ensure_ascii=False) + "\n")
            fa.write(json.dumps({"id": i, "answers": q["answers"]}, ensure_ascii=False) + "\n")
            fm.write(json.dumps({"id": i, "doc": q["doc"], **q["meta"]}, ensure_ascii=False) + "\n")


def hotpot(out, n_train_pools, n_val_pools, rng, lb_q):
    df = pd.concat([pd.read_parquet(f"{HOTPOT}/train-0000{i}-of-00002.parquet") for i in range(2)], ignore_index=True)
    df = df[~df.question.map(norm).isin(lb_q)].reset_index(drop=True)
    lb_len = [json.loads(l)["length"] for l in open(f"{LB}/hotpotqa.jsonl")]
    idx = rng.sample(range(len(df)), (n_train_pools + n_val_pools) * POOL)
    paras = [(t, " ".join(s)) for c in df.context for t, s in zip(c["title"], c["sentences"])]
    for p in range(n_train_pools + n_val_pools):
        split = "train" if p < n_train_pools else "val"
        k = p if p < n_train_pools else p - n_train_pools
        docs, qs = [], []
        for j, r in enumerate(idx[p * POOL : (p + 1) * POOL]):
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
        write_pool(f"{out}/hotpotqa/{split}/pool_{k:02d}", docs, qs)


def qasper_answer(a):
    if a["unanswerable"]:
        return "Unanswerable"
    if a["free_form_answer"]:
        return a["free_form_answer"]
    if len(a["extractive_spans"]):
        return ", ".join(s.strip() for s in a["extractive_spans"])
    if a["yes_no"] is not None:
        return "Yes" if a["yes_no"] else "No"
    return None


def qasper(out, split_src, split, n_pools, rng, lb_q, lb_titles):
    df = pd.read_parquet(f"{QASPER}/{split_src}/0000.parquet")
    df = df[~df.title.map(norm).isin(lb_titles)].reset_index(drop=True)
    lb_ctx = " ".join(norm(json.loads(l)["context"]) for l in open(f"{LB}/qasper.jsonl"))
    df = df[~df.abstract.map(lambda s: norm(s)[:200] in lb_ctx)].reset_index(
        drop=True
    )  # drop papers whose abstract appears in LongBench
    order = list(range(len(df)))
    rng.shuffle(order)
    pools, cur_docs, cur_qs = [], [], []
    for r in order:
        row = df.iloc[r]
        ft = row.full_text
        text = (
            row.title
            + "\n\nAbstract\n"
            + row.abstract
            + "\n\n"
            + "\n\n".join(
                (n or "") + "\n" + "\n".join(p or "" for p in ps) for n, ps in zip(ft["section_name"], ft["paragraphs"])
            )
        )
        qs = []
        for q, ans in zip(row.qas["question"], row.qas["answers"]):
            if norm(q) in lb_q:
                continue
            refs = [x for x in (qasper_answer(a) for a in ans["answer"]) if x]
            if not refs:
                continue
            ev = sorted(set(e for a in ans["answer"] for e in a["evidence"]))
            qs.append({"query": q, "answers": refs, "meta": {"paper_id": row.id, "evidence": ev}})
        if not qs:
            continue
        cur_docs.append(text)
        for q in qs:
            q["doc"] = len(cur_docs)
        cur_qs += qs
        if len(cur_qs) >= POOL:
            pools.append((cur_docs, cur_qs[:POOL]))
            cur_docs, cur_qs = [], []
            if len(pools) == n_pools:
                break
    # pad each pool to 200 documents with other papers of the same split (distractors), matching the LongBench test pool corpus size
    all_text = [
        row.title
        + "\n\nAbstract\n"
        + row.abstract
        + "\n\n"
        + "\n\n".join(
            (n or "") + "\n" + "\n".join(p or "" for p in ps)
            for n, ps in zip(row.full_text["section_name"], row.full_text["paragraphs"])
        )
        for row in df.itertuples()
    ]
    for k, (docs, qs) in enumerate(pools):
        have = set(docs)
        extra = [x for x in rng.sample(all_text, len(all_text)) if x not in have][: max(0, POOL - len(docs))]
        write_pool(f"{out}/qasper/{split}/pool_{k:02d}", docs + extra, qs)
    return len(pools)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DATA)
    ap.add_argument("--hotpot_train_pools", type=int, default=10)
    ap.add_argument("--val_pools", type=int, default=2)
    ap.add_argument("--qasper_train_pools", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lb = {n: [json.loads(l) for l in open(f"{LB}/{n}.jsonl")] for n in ["hotpotqa", "qasper"]}
    lb_q = {norm(r["input"]) for n in lb for r in lb[n]}
    lb_titles = set()
    for r in lb["qasper"]:
        m = re.search(r"\[\s*(.+?)(?:[A-Z][a-z]+ [A-Z]|\(|$)", r["context"][:400])
        if m:
            lb_titles.add(norm(m.group(1)))
    for ds in ["hotpotqa", "qasper"]:
        d = f"{a.out}/{ds}/test/pool_00"
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(f"{ECRAG_DATA}/{ds}", d)
    hotpot(a.out, a.hotpot_train_pools, a.val_pools, rng, lb_q)
    nt = qasper(a.out, "train", "train", a.qasper_train_pools, rng, lb_q, lb_titles)
    nv = qasper(a.out, "validation", "val", a.val_pools, rng, lb_q, lb_titles)
    # leakage audit: exact question overlap and Qasper paper text overlap (first 300 chars of abstract) against LongBench contexts
    lb_ctx = " ".join(norm(r["context"]) for r in lb["qasper"])
    leak = {"qasper_pools_train": nt, "qasper_pools_val": nv, "question_overlap": 0, "qasper_abstract_in_longbench": 0}
    for ds in ["hotpotqa", "qasper"]:
        for split in ["train", "val"]:
            base = f"{a.out}/{ds}/{split}"
            for p in sorted(os.listdir(base)):
                for l in open(f"{base}/{p}/questions/questions.jsonl"):
                    leak["question_overlap"] += norm(json.loads(l)["query"]) in lb_q
                if ds == "qasper":
                    for f in os.listdir(f"{base}/{p}/documents"):
                        t = open(f"{base}/{p}/documents/{f}").read()
                        abs_ = norm(t.split("Abstract\n", 1)[1][:300]) if "Abstract\n" in t else ""
                        leak["qasper_abstract_in_longbench"] += bool(abs_) and abs_[:200] in lb_ctx
    json.dump(leak, open(f"{a.out}/leakage_audit.json", "w"), indent=1)
    print(leak)


if __name__ == "__main__":
    main()
