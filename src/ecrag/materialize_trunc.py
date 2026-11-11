"""Truncation policy for cloud thinking, fixed before re-scoring.
A cloud response with finish_reason == 'length' (equivalently: no '</think>' in the output) has NO final answer.
Two trace roots are built from runs/pilot4/trace_cfg3 (tag _cfg3) into runs/pilot6/ (tag _cfg3t):
  trace_cfg3tA : truncated cloud answer -> "" (F1 0); judge labels for that cloud answer -> INCORRECT. Token cost kept.
                 Used for router training labels (the router should learn to avoid these queries).
  trace_cfg3tB : deployable fallback: truncated cloud answer -> the edge answer of the same query; cloud judge label -> the
                 edge label. Cloud token cost kept (the cloud call was made); the extra edge latency is counted in the e2e run.
Every cloud row gets 'cloud_finish'. Only HotpotQA ('orig_think') has truncations; Qasper rows are copied unchanged.
"""

from paths import RUNS  # noqa: E402
import glob, json, os, shutil

R = RUNS
SRC, ANS = f"{R}/pilot4/trace_cfg3", f"{R}/pilot2/answers"
for var in ["A", "B"]:
    out = f"{R}/pilot6/trace_cfg3t{var}"
    tot = 0
    for d in sorted(glob.glob(f"{SRC}/*/*/pool_*")):
        ds, sp, pool = d.split("/")[-3:]
        od = f"{out}/{ds}/{sp}/{pool}"
        if os.path.exists(f"{od}/DONE"):
            continue
        os.makedirs(od, exist_ok=True)
        for f in ["chunks.jsonl", "trace.jsonl"]:
            shutil.copy(f"{d}/{f}", f"{od}/{f}")
        fin = {}
        if ds == "hotpotqa":
            for l in open(f"{ANS}/hotpotqa/{sp}/{pool}/orig_think.jsonl"):
                x = json.loads(l)
                fin[x["id"]] = x.get("cloud_finish")
        tr = {json.loads(l)["id"]: json.loads(l) for l in open(f"{d}/trace.jsonl")}
        cut = {i for i, f in fin.items() if f == "length"}
        tot += len(cut)
        with open(f"{od}/cloud_cfg3t.jsonl", "w") as fc:
            for l in open(f"{d}/cloud_cfg3.jsonl"):
                c = json.loads(l)
                c["cloud_finish"] = fin.get(c["id"], "stop")
                c["truncated"] = c["id"] in cut
                if c["truncated"]:
                    c["cloud_answer"] = "" if var == "A" else tr[c["id"]]["edge_answer"]
                fc.write(json.dumps(c, ensure_ascii=False) + "\n")
        for jf in glob.glob(f"{d}/judge_*_cfg3.jsonl"):
            name = os.path.basename(jf).replace("_cfg3.jsonl", "_cfg3t.jsonl")
            rows = [json.loads(l) for l in open(jf)]
            elab = {r["id"]: r["label"] for r in rows if r["who"] == "edge"}
            with open(f"{od}/{name}", "w") as fj:
                for r in rows:
                    if r["who"] == "cloud" and r["id"] in cut:
                        r = dict(
                            r, label="INCORRECT" if var == "A" else elab.get(r["id"]), raw=f"truncation policy {var}"
                        )
                    fj.write(json.dumps(r, ensure_ascii=False) + "\n")
        open(f"{od}/DONE", "w").write(f"truncation policy {var}\n")
    print(var, "truncated rows changed:", tot)
