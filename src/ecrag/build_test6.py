"""test6: 8 fresh HotpotQA pools (1,600 q) for selector-comparison power.
Same construction as build_test5.py; HotpotQA validation minus test2..test5 (src_id and text) and LongBench; seed 9404."""

import json, random
import pandas as pd
import build_data as B
from build_more import used, build, D


def main():
    lb_q = {B.norm(json.loads(l)["input"]) for n in ["hotpotqa", "qasper"] for l in open(f"{B.LB}/{n}.jsonl")}
    lb_len = [json.loads(l)["length"] for l in open(f"{B.LB}/hotpotqa.jsonl")]
    ex_ids, ex_q = set(), set()
    for s in ["test2", "test3", "test4", "test5"]:
        i, q = used(s)
        ex_ids |= i
        ex_q |= q
    dv = pd.read_parquet(f"{B.HOTPOT}/validation-00000-of-00001.parquet")
    dv = dv[~dv.question.map(B.norm).isin(lb_q | ex_q) & ~dv.id.isin(ex_ids)].reset_index(drop=True)
    print("candidates", len(dv), flush=True)
    build(dv, "test6", 8, random.Random(9404), lb_len)
    i6, q6 = used("test6")
    audit = {
        s: [len(q6 & used(s)[1]), len(i6 & used(s)[0])]
        for s in ["train", "val", "train2", "test", "test2", "test3", "test4", "test5", "unl1"]
    }
    audit["longbench"] = len(q6 & lb_q)
    audit["n"] = len(q6)
    print(audit)
    json.dump(audit, open(f"{D}/leakage_audit_test6.json", "w"), indent=1)


if __name__ == "__main__":
    main()
