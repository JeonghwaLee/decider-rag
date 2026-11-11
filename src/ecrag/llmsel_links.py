"""Turn llm_selector outputs into decider-format pred files per probe family: <dir>/two/pred_<tag>.json (dEN, dNT from
the edge/think probes) and <dir>/choice/pred_<tag>.json (dEN, dNT from the choice probe), plus HotpotQA split-name copies
(pred_val/test4/test5.json) for analyze_en."""

import json, os, sys

d = sys.argv[1]
H = {
    "legacy_val": "val",
    "pilot8_hotpotqa_test4": "test4",
    "pilot8_hotpotqa_test5": "test5",
    "pilot8_hotpotqa_test6": "test6",
}
for fam in ["two", "choice"]:
    os.makedirs(f"{d}/{fam}", exist_ok=True)
for f in os.listdir(d):
    if not (f.startswith("pred_") and f.endswith(".json")):
        continue
    X = json.load(open(f"{d}/{f}"))
    tag = f[5:-5]
    for fam in ["two", "choice"]:
        Y = [
            {
                "ds": x["ds"],
                "split": x["split"],
                "pool": x["pool"],
                "id": x["id"],
                "fE": 0.0,
                "fN": 0.0,
                "fT": 0.0,
                "dEN": x["dEN"] if fam == "two" else x["c_dEN"],
                "dNT": x["dNT"] if fam == "two" else x["c_dNT"],
            }
            for x in X
        ]
        json.dump(Y, open(f"{d}/{fam}/pred_{tag}.json", "w"))
        if tag in H:
            json.dump(Y, open(f"{d}/{fam}/pred_{H[tag]}.json", "w"))
print("ok", d)
