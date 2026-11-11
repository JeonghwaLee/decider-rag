"""test5: a further fresh HotpotQA confirmation sample for policies chosen after test4 was opened (e.g. retrained or
ensembled deciders). Same construction as build_more.py; HotpotQA validation minus test2/test3/test4 (src_id and text) and LongBench."""

import json, random
import pandas as pd
import build_data as B
from build_more import used, build, audit_


def main():
    lb_q = {B.norm(json.loads(l)["input"]) for n in ["hotpotqa", "qasper"] for l in open(f"{B.LB}/{n}.jsonl")}
    lb_len = [json.loads(l)["length"] for l in open(f"{B.LB}/hotpotqa.jsonl")]
    ex_ids, ex_q = set(), set()
    for s in ["test2", "test3", "test4"]:
        i, q = used(s)
        ex_ids |= i
        ex_q |= q
    dv = pd.read_parquet(f"{B.HOTPOT}/validation-00000-of-00001.parquet")
    dv = dv[~dv.question.map(B.norm).isin(lb_q | ex_q) & ~dv.id.isin(ex_ids)].reset_index(drop=True)
    build(dv, "test5", 4, random.Random(9303), lb_len)
    i5, q5 = used("test5")
    print(
        {
            s: (len(q5 & used(s)[1]), len(i5 & used(s)[0]))
            for s in ["train", "val", "train2", "test2", "test3", "test4"]
        },
        len(q5 & lb_q),
    )


if __name__ == "__main__":
    main()
