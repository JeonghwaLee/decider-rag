"""Final public-judge pass.
Auxiliary only: policies, thresholds and test items are frozen; the labels are never used to choose anything.
Judges, blind rubric (judge_rubric.grade):
  glm53   GLM-5.3-Flash-FP8 :8194, reasoning_effort low, max 2048 (same settings as the existing glm53 labels -> reused)
  dsv4    DeepSeek-V4-Flash-0731 :8196, chat mode (thinking false), max 8
  llama33 Llama-3.3-70B-Instruct :8199, max 8
Items:
  offline  stored per-action answers of the roots (as judge_root.py): E = normalized edge 'extract', N = cloud 'orig', T = cloud
           'orig_think' (truncated T not graded; analysis gives it the fallback label)
  e2       final answers actually returned in the E2 v5 live runs (field 'answer', after any retry)
Dedup: one grade per (ds, split, pool, id, answer string) per judge, cached in judge_final/cache_<jn>.jsonl (resumable); the
glm53 cache is seeded with the existing glm53 labels of the offline roots (same rubric, settings and answer).
Outputs: offline labels in the judge_root layout <root>/judge/<ds>/<split>/<pool>/judge_<jn>.jsonl (written only if absent) and
judge_final/e2_<name>_<jn>.jsonl rows {file, id, policy, label}.
--precheck N: grade N random items of the NON-test splits per judge only and print the parse rate + label mix (no outputs).
Usage: judge_final.py --judges glm53 dsv4 [--precheck 40]"""

from paths import RUNS  # noqa: E402
import argparse, asyncio, glob, json, os, random, sys
from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(__file__))
from judge_rubric import grade  # noqa: E402
from router_data import DATA  # noqa: E402
from normalize_edge import normalize  # noqa: E402

R = RUNS
OUT = f"{R}/judge_final"
JUDGES = {
    "glm53": ("http://127.0.0.1:8194/v1", "glm53-flash", 2048, {"chat_template_kwargs": {"reasoning_effort": "low"}}),
    "dsv4": ("http://127.0.0.1:8196/v1", "dsv4-flash", 8, {"chat_template_kwargs": {"thinking": False}}),
    "llama33": ("http://127.0.0.1:8199/v1", "llama33-70b", 8, {}),
}
OFFLINE_TEST = [("pilot8", "hotpotqa", s) for s in ("test4", "test5", "test6")] + [
    ("pilot9", "musique", "test"),
    ("pilot10", "2wiki", "test"),
    ("pilot10", "2wiki", "test2"),
]
OFFLINE_NONTEST = [("pilot8", "hotpotqa", "train2"), ("pilot9", "musique", "val"), ("pilot10", "2wiki", "val")]
E2 = {"hotpot_test6": ("hotpotqa", f"{R}/e2e5/hotpot_test6"), "musique_test": ("musique", f"{R}/e2e5/musique_test")}


def qa(ds, sp, pool, C={}):
    k = (ds, sp, pool)
    if k not in C:
        b = f"{DATA}/{ds}/{sp}/{pool}"
        C[k] = (
            {json.loads(l)["id"]: json.loads(l)["query"] for l in open(f"{b}/questions/questions.jsonl")},
            {json.loads(l)["id"]: json.loads(l)["answers"] for l in open(f"{b}/answers/answers.jsonl")},
        )
    return C[k]


def offline_items(root, ds, sp):
    out = []
    for d in sorted(glob.glob(f"{R}/{root}/answers/{ds}/{sp}/pool_*")):
        pool = d.split("/")[-1]
        qs, _ = qa(ds, sp, pool)
        for v, who in [("extract", "E"), ("orig", "N"), ("orig_think", "T")]:
            if not os.path.exists(f"{d}/{v}.jsonl"):
                continue
            for l in open(f"{d}/{v}.jsonl"):
                x = json.loads(l)
                if who == "E":
                    out.append((root, ds, sp, pool, x["id"], who, normalize(qs[x["id"]], x["edge_answer"])[0]))
                elif who == "N" or not (x["cloud_finish"] == "length" or "</think>" not in (x.get("cloud_raw") or "")):
                    out.append((root, ds, sp, pool, x["id"], who, x["cloud_answer"]))
    return out


def e2_items(name):
    ds, d = E2[name]
    out = []
    for fn in sorted(glob.glob(f"{d}/*_r0.jsonl")):
        b = os.path.basename(fn)
        sp, _, pn = b.split("_", 2)
        pool = "pool_" + pn.split("_")[0]
        for l in open(fn):
            x = json.loads(l)
            out.append((fn, ds, sp, pool, x["id"], x["policy"], x["answer"] or ""))
    return out


def key(ds, sp, pool, i, ans):
    return json.dumps([ds, sp, pool, i, ans], ensure_ascii=False)


def load_cache(jn):
    C = {}
    fn = f"{OUT}/cache_{jn}.jsonl"
    if os.path.exists(fn):
        for l in open(fn):
            x = json.loads(l)
            C[x["key"]] = x["label"]
    return C


async def run(jn, todo, C):
    url, model, mt, eb = JUDGES[jn]
    cl = AsyncOpenAI(base_url=url, api_key="none", timeout=900)
    sem = asyncio.Semaphore(48)
    cfg = type("a", (), {"model": model, "max_tokens": mt, "extra_body": json.dumps(eb)})
    fn = f"{OUT}/cache_{jn}.jsonl"
    n = 0
    with open(fn, "a") as f:

        async def one(k):
            ds, sp, pool, i, ans = json.loads(k)
            qs, refs = qa(ds, sp, pool)
            lab, raw = await grade(cl, sem, cfg, qs[i], refs[i], ans)
            return k, lab, raw

        for coro in asyncio.as_completed([one(k) for k in todo]):
            k, lab, raw = await coro
            C[k] = lab
            n += 1
            f.write(json.dumps({"key": k, "label": lab, "raw": raw[-200:]}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 2000 == 0:
                print(jn, n, "/", len(todo), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--precheck", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.precheck:
        items = [x for t in OFFLINE_NONTEST for x in offline_items(*t)]
        random.Random(0).shuffle(items)
        for jn in a.judges:
            C = {}
            sm = items[: a.precheck]
            ks = [key(*x[1:5], x[6]) for x in sm]
            url, model, mt, eb = JUDGES[jn]
            cl = AsyncOpenAI(base_url=url, api_key="none", timeout=900)
            cfg = type("a", (), {"model": model, "max_tokens": mt, "extra_body": json.dumps(eb)})
            sem = asyncio.Semaphore(32)

            async def go():
                return await asyncio.gather(
                    *(
                        grade(
                            cl,
                            sem,
                            cfg,
                            qa(*json.loads(k)[:3])[0][json.loads(k)[3]],
                            qa(*json.loads(k)[:3])[1][json.loads(k)[3]],
                            json.loads(k)[4],
                        )
                        for k in ks
                    )
                )

            R_ = asyncio.run(go())
            labs = [r[0] for r in R_]
            print(
                jn,
                "precheck n",
                len(labs),
                "unparsed",
                sum(x is None for x in labs),
                {l: labs.count(l) for l in set(labs)},
                "| raw examples:",
                [r[1][-40:] for r in R_[:3]],
                flush=True,
            )
        return
    off = [x for t in OFFLINE_TEST for x in offline_items(*t)]
    e2 = [x for nm in E2 for x in e2_items(nm)]
    keys = sorted({key(*x[1:5], x[6]) for x in off} | {key(*x[1:5], x[6]) for x in e2})
    print(f"items: offline {len(off)} e2 {len(e2)} -> unique (q, answer) {len(keys)}", flush=True)
    for jn in a.judges:
        C = load_cache(jn)
        if jn == "glm53" and not C:  # seed with the existing glm53 labels (same rubric/settings/answer)
            seeded = 0
            with open(f"{OUT}/cache_glm53.jsonl", "a") as f:
                for root, ds, sp in OFFLINE_TEST:
                    L = {}
                    for fn in glob.glob(f"{R}/{root}/judge/{ds}/{sp}/pool_*/judge_glm53.jsonl"):
                        pool = fn.split("/")[-2]
                        for l in open(fn):
                            x = json.loads(l)
                            L[(pool, x["id"], x["who"])] = x["label"]
                    for x in offline_items(root, ds, sp):
                        lb = L.get((x[3], x[4], x[5]))
                        k = key(*x[1:5], x[6])
                        if lb is not None and k not in C:
                            C[k] = lb
                            seeded += 1
                            f.write(
                                json.dumps(
                                    {"key": k, "label": lb, "raw": "seed:existing glm53 label"}, ensure_ascii=False
                                )
                                + "\n"
                            )
            print("glm53 seeded from existing labels:", seeded, flush=True)
        todo = [k for k in keys if C.get(k) is None]
        print(jn, "to grade", len(todo), "of", len(keys), flush=True)
        asyncio.run(run(jn, todo, C))
        for root, ds, sp in OFFLINE_TEST:  # judge_root layout
            G = {}
            for x in off:
                if x[:3] == (root, ds, sp):
                    G.setdefault(x[3], []).append({"id": x[4], "who": x[5], "label": C.get(key(*x[1:5], x[6]))})
            for pool, rows in G.items():
                fn = f"{R}/{root}/judge/{ds}/{sp}/{pool}/judge_{jn}.jsonl"
                if not os.path.exists(fn):
                    os.makedirs(os.path.dirname(fn), exist_ok=True)
                    with open(fn, "w") as f:
                        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        for nm in E2:
            with open(f"{OUT}/e2_{nm}_{jn}.jsonl", "w") as f:
                for x in e2_items(nm):
                    f.write(
                        json.dumps(
                            {
                                "file": os.path.basename(x[0]),
                                "id": x[4],
                                "policy": x[5],
                                "label": C.get(key(*x[1:5], x[6])),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        print(jn, "done; unlabeled", sum(C.get(k) is None for k in keys), flush=True)


if __name__ == "__main__":
    main()
