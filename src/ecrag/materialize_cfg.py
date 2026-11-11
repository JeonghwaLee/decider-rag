"""Build a trace root for a chosen generation configuration so the existing tools (router_data.load, train_router_q,
analyze2, analyze_final) run unchanged with ECRAG_TRACE=<out> ECRAG_CLOUD_TAG=<tag>.

Configurations: cfg3 is primary (F1-selected); cfg2 is the judge-selected variant kept for comparison.
Configuration cfg2 (pilot 3, selected per dataset on train pools 00-01 by mean GLM quality of edge and cloud):
- hotpotqa: retrieval = pilot-1 traces (EC-RAG pooled corpus); edge = 'extract' prompt; cloud = 'orig_think' (thinking on)
- qasper:   retrieval = pilot-3 paper-scoped traces;          edge = 'orig' prompt (vLLM);   cloud = 'extract'
Per pool, written only when both answer files cover every trace question:
  chunks.jsonl (copied), trace.jsonl (edge_answer / edge_out_tokens replaced), DONE,
  cloud<tag>.jsonl (cloud_answer + token counts), judge_glm53<tag>.jsonl (labels from the answer files)
"""

from paths import RUNS  # noqa: E402
import argparse, glob, json, os, shutil

ROOT = RUNS
CFG = {
    "cfg2": {
        "hotpotqa": {
            "trace": f"{ROOT}/pilot1/trace",
            "ans": f"{ROOT}/pilot2/answers",
            "edge": "extract",
            "cloud": "orig_think",
        },
        "qasper": {
            "trace": f"{ROOT}/pilot3/trace",
            "ans": f"{ROOT}/pilot3/answers",
            "edge": "orig",
            "cloud": "extract",
        },
    },
    # cfg3 (F1 is the primary metric, judges auxiliary): same as cfg2 but the per-dataset prompt pair is the
    # F1-best one on the pilot pools. hotpotqa is unchanged (extract / orig_think is also F1-best); qasper cloud = 'orig'.
    "cfg3": {
        "hotpotqa": {
            "trace": f"{ROOT}/pilot1/trace",
            "ans": f"{ROOT}/pilot2/answers",
            "edge": "extract",
            "cloud": "orig_think",
        },
        "qasper": {"trace": f"{ROOT}/pilot3/trace", "ans": f"{ROOT}/pilot3/answers", "edge": "orig", "cloud": "orig"},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfg2")
    ap.add_argument("--out", default=f"{ROOT}/pilot4/trace")
    a = ap.parse_args()
    tag = f"_{a.cfg}"
    for ds, c in CFG[a.cfg].items():
        for td in sorted(glob.glob(f"{c['trace']}/{ds}/*/pool_*")):
            split, pool = td.split("/")[-2:]
            ef, cf = (
                f"{c['ans']}/{ds}/{split}/{pool}/{c['edge']}.jsonl",
                f"{c['ans']}/{ds}/{split}/{pool}/{c['cloud']}.jsonl",
            )
            if not (os.path.exists(f"{td}/DONE") and os.path.exists(ef) and os.path.exists(cf)):
                continue
            E = {json.loads(l)["id"]: json.loads(l) for l in open(ef)}
            C = {json.loads(l)["id"]: json.loads(l) for l in open(cf)}
            tr = [json.loads(l) for l in open(f"{td}/trace.jsonl")]
            ok = [
                t
                for t in tr
                if t["id"] in E and t["id"] in C and E[t["id"]].get("j_edge") and C[t["id"]].get("j_cloud")
            ]
            if len(ok) < len(tr):
                print("incomplete, skip", ds, split, pool, len(ok), "/", len(tr))
                continue
            od = f"{a.out}/{ds}/{split}/{pool}"
            if os.path.exists(f"{od}/DONE"):
                continue
            os.makedirs(od, exist_ok=True)
            shutil.copy(f"{td}/chunks.jsonl", f"{od}/chunks.jsonl")
            with (
                open(f"{od}/trace.jsonl", "w") as ft,
                open(f"{od}/cloud{tag}.jsonl", "w") as fc,
                open(f"{od}/judge_glm53{tag}.jsonl", "w") as fj,
            ):
                for t in tr:
                    e, cl = E[t["id"]], C[t["id"]]
                    t = dict(
                        t, edge_answer=e["edge_answer"], edge_out_tokens=e["edge_out_tokens"], edge_variant=c["edge"]
                    )
                    ft.write(json.dumps(t, ensure_ascii=False) + "\n")
                    fc.write(
                        json.dumps(
                            {
                                "id": t["id"],
                                "cloud_answer": cl["cloud_answer"],
                                "cloud_in_tokens": cl["cloud_in_tokens"],
                                "cloud_out_tokens": cl["cloud_out_tokens"],
                                "cloud_variant": c["cloud"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    fj.write(
                        json.dumps({"id": t["id"], "who": "edge", "label": e["j_edge"], "raw": e["j_edge"]}) + "\n"
                    )
                    fj.write(
                        json.dumps({"id": t["id"], "who": "cloud", "label": cl["j_cloud"], "raw": cl["j_cloud"]}) + "\n"
                    )
            open(f"{od}/DONE", "w").write("materialized\n")
            print("ok", ds, split, pool, len(tr), flush=True)


if __name__ == "__main__":
    main()
