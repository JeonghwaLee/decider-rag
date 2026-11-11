"""2Wiki test2: 8 fresh pools (1,600 q) from 2Wiki validation minus
LongBench 2wikimqa minus every question / src id already in data/ecrag/2wiki/{train,val,test}. Same builder as build_2wiki.py, seed 7405.
Audit -> data/ecrag/leakage_audit_2wiki_test2.json."""

import glob, json, os, random
import pandas as pd
import build_data as B
import build_2wiki as W


def main():
    lb = [json.loads(l) for l in open(f"{W.D}/raw/longbench/data/2wikimqa.jsonl")]
    lb_q = {B.norm(r["input"]) for r in lb}
    lb_len = [int(r["length"]) for r in lb]
    ex_q, ex_id = set(), set()
    for s in ["train", "val", "test"]:
        for p in glob.glob(f"{W.D}/2wiki/{s}/pool_*"):
            ex_q |= {B.norm(json.loads(l)["query"]) for l in open(f"{p}/questions/questions.jsonl")}
            ex_id |= {json.loads(l)["src_id"] for l in open(f"{p}/meta.jsonl")}
    conv = lambda d: d.assign(
        context=d.context.map(lambda c: {"title": list(c["title"]), "sentences": [list(s) for s in c["sentences"]]}),
        supporting_facts=d.supporting_facts.map(lambda s: {"title": pd.Series(list(s["title"]))}),
        level="-",
    )
    dv = pd.read_parquet(glob.glob(f"{W.SRC}/validation-*.parquet")[0])
    dv = (
        dv[~dv.question.map(B.norm).isin(lb_q | ex_q) & ~dv.id.isin(ex_id)]
        .drop_duplicates("question")
        .reset_index(drop=True)
    )
    print("candidates", len(dv), flush=True)
    W.build(conv(dv), "test2", 8, random.Random(7405), lb_len)
    q2, id2 = set(), set()
    for p in glob.glob(f"{W.D}/2wiki/test2/pool_*"):
        q2 |= {B.norm(json.loads(l)["query"]) for l in open(f"{p}/questions/questions.jsonl")}
        id2 |= {json.loads(l)["src_id"] for l in open(f"{p}/meta.jsonl")}
    audit = {
        "n": len(q2),
        "in_longbench": len(q2 & lb_q),
        "q_in_train_val_test": len(q2 & ex_q),
        "id_in_train_val_test": len(id2 & ex_id),
    }
    json.dump(audit, open(f"{W.D}/leakage_audit_2wiki_test2.json", "w"), indent=1)
    print(audit)


if __name__ == "__main__":
    main()
