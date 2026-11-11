"""Third dataset: 2WikiMultihopQA (framolfese/2WikiMultihopQA parquet; same context format as HotpotQA) in the EC-RAG
pool format, built exactly like the HotpotQA pools (build_more.build: own paragraphs + random paragraphs of the same source split up
to a LongBench 2wikimqa length). train 10 pools + val 2 pools from 2Wiki train; test 4 pools from 2Wiki validation minus LongBench
2wikimqa questions. Pools: data/ecrag/2wiki/<split>/pool_XX. Audit: data/ecrag/leakage_audit_2wiki.json."""

from paths import DATA, HF_HUB  # noqa: E402
import glob, json, os, random, zipfile
import pandas as pd
import build_data as B
import build_more as M

D = DATA
SRC = glob.glob(f"{HF_HUB}/datasets--framolfese--2WikiMultihopQA/snapshots/*/data")[0]
LBZ = f"{HF_HUB}/datasets--THUDM--LongBench/snapshots/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip"


def build(df, split, n, rng, lb_len):  # build_more.build writes under hotpotqa/; redirect the dataset name
    old = M.D
    try:
        os.makedirs(f"{D}/2wiki", exist_ok=True)
        M.D = D.replace("/ecrag", "/ecrag/_2wiki_tmp")
        os.makedirs(f"{M.D}/hotpotqa", exist_ok=True)
        if not os.path.exists(f"{M.D}/hotpotqa/{split}") and os.path.exists(f"{D}/2wiki/{split}"):
            raise SystemExit("exists")
        M.build(df, split, n, rng, lb_len)
        os.rename(f"{M.D}/hotpotqa/{split}", f"{D}/2wiki/{split}")
    finally:
        M.D = old


def main():
    z = zipfile.ZipFile(LBZ)
    lb = [json.loads(l) for l in z.open("data/2wikimqa.jsonl")]
    with open(f"{D}/raw/longbench/data/2wikimqa.jsonl", "w") as f:
        for r in lb:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lb_q = {B.norm(r["input"]) for r in lb}
    lb_len = [int(r["length"]) for r in lb]
    conv = lambda d: d.assign(
        context=d.context.map(lambda c: {"title": list(c["title"]), "sentences": [list(s) for s in c["sentences"]]}),
        supporting_facts=d.supporting_facts.map(lambda s: {"title": pd.Series(list(s["title"]))}),
        level="-",
    )
    tr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SRC}/train-*.parquet"))], ignore_index=True)
    tr = conv(tr[~tr.question.map(B.norm).isin(lb_q)].drop_duplicates("question").reset_index(drop=True))
    dv = pd.read_parquet(glob.glob(f"{SRC}/validation-*.parquet")[0])
    dv = conv(dv[~dv.question.map(B.norm).isin(lb_q)].drop_duplicates("question").reset_index(drop=True))
    rng = random.Random(7401)
    idx = rng.sample(range(len(tr)), 2400)
    t_tr = tr.iloc[idx[:2000]].reset_index(drop=True)
    t_va = tr.iloc[idx[2000:]].reset_index(drop=True)
    build(t_tr, "train", 10, random.Random(7402), lb_len)
    build(t_va, "val", 2, random.Random(7403), lb_len)
    build(dv, "test", 4, random.Random(7404), lb_len)
    seen = {}
    for s in ["train", "val", "test"]:
        seen[s] = set()
        for p in os.listdir(f"{D}/2wiki/{s}"):
            seen[s] |= {B.norm(json.loads(l)["query"]) for l in open(f"{D}/2wiki/{s}/{p}/questions/questions.jsonl")}
    audit = {
        s: {"n": len(q), "in_longbench": len(q & lb_q), **{f"in_{o}": len(q & seen[o]) for o in seen if o != s}}
        for s, q in seen.items()
    }
    json.dump(audit, open(f"{D}/leakage_audit_2wiki.json", "w"), indent=1)
    print(audit)


if __name__ == "__main__":
    main()
