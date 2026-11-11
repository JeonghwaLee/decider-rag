"""unl1 (unlabeled pool for distilling the seed ensemble into one decider): 4,000 extra HotpotQA *train*-split
questions, construction identical to build_more.py (POOL 200, LongBench-length padding), seed 9404, excluding every question in
hotpotqa/train, /val, /train2 (src_id and text) and LongBench. Only the decider input (stage-1 trace: k and chunks) is needed; no
edge/cloud answers are generated and the answers are not used for training (teacher scores replace labels)."""

import json, random
import pandas as pd
import build_data as B
from build_more import used, build


def main():
    lb_q = {B.norm(json.loads(l)["input"]) for n in ["hotpotqa", "qasper"] for l in open(f"{B.LB}/{n}.jsonl")}
    lb_len = [json.loads(l)["length"] for l in open(f"{B.LB}/hotpotqa.jsonl")]
    ex_ids, ex_q = set(), set()
    for s in ["train", "val", "train2"]:
        i, q = used(s)
        ex_ids |= i
        ex_q |= q
    tr = pd.concat([pd.read_parquet(f"{B.HOTPOT}/train-0000{i}-of-00002.parquet") for i in range(2)], ignore_index=True)
    tr = tr[~tr.question.map(B.norm).isin(lb_q | ex_q) & ~tr.id.isin(ex_ids)].reset_index(drop=True)
    build(tr, "unl1", 20, random.Random(9404), lb_len)
    iu, qu = used("unl1")
    print(
        {
            s: (len(qu & used(s)[1]), len(iu & used(s)[0]))
            for s in ["train", "val", "train2", "test2", "test3", "test4", "test5"]
        },
        len(qu & lb_q),
    )


if __name__ == "__main__":
    main()
