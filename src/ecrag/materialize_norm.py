"""cfg3n: cfg3 + HotpotQA edge answer normalized with normalize_edge.py (Qasper unchanged), from the truncation-policy trace
roots runs/pilot6/trace_cfg3t{A,B} into trace_cfg3n{A,B} (tag _cfg3n). Per HotpotQA trace row: edge_answer
normalized, edge_answer_raw and edge_norm (rule name or null) kept. Policy B's fallback cloud answer (truncated rows) becomes
the normalized edge answer. Judge files are NOT written here: judge_norm.py writes them after re-grading changed answers."""

from paths import RUNS  # noqa: E402
import glob, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(__file__))
from normalize_edge import normalize  # noqa: E402

R = f"{RUNS}/pilot6"
for v in ["A", "B"]:
    n = 0
    for d in sorted(glob.glob(f"{R}/trace_cfg3t{v}/*/*/pool_*")):
        ds, sp, pool = d.split("/")[-3:]
        od = f"{R}/trace_cfg3n{v}/{ds}/{sp}/{pool}"
        if os.path.exists(f"{od}/DONE"):
            continue
        os.makedirs(od, exist_ok=True)
        shutil.copy(f"{d}/chunks.jsonl", f"{od}/chunks.jsonl")
        tr = [json.loads(l) for l in open(f"{d}/trace.jsonl")]
        E = {}
        with open(f"{od}/trace.jsonl", "w") as f:
            for t in tr:
                if ds == "hotpotqa":
                    na, why = normalize(t["query"], t["edge_answer"])
                    t = dict(t, edge_answer_raw=t["edge_answer"], edge_answer=na, edge_norm=why)
                    n += why is not None
                E[t["id"]] = t["edge_answer"]
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        with open(f"{od}/cloud_cfg3n.jsonl", "w") as f:
            for l in open(f"{d}/cloud_cfg3t.jsonl"):
                c = json.loads(l)
                if v == "B" and c.get("truncated"):
                    c["cloud_answer"] = E[c["id"]]
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        open(f"{od}/DONE", "w").write("cfg3n\n")
    print(v, "normalized edge answers:", n)
