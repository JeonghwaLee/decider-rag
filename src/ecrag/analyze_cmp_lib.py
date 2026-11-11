"""(helpers copied from analyze_cmp.py) cfg3 HotpotQA + date-comparison path. Policies (thresholds from validation only):
  router         : PRIMARY score, validation threshold at the rule's validation rate (as in analyze_cfg3 F1@rule)
  router+cmp     : same routing, edge answer replaced by the comparison-path answer when it answered
  router+cmp/fix : comparison-answered queries pinned to edge; threshold re-fit on validation (others) so the validation
                   cloud rate equals the rule's validation rate (the freed budget goes to the next-highest scores)
  rule+cmp       : EC-RAG rule with the edge answer replaced (shows how much comes from the path alone)
All compared with the plain EC-RAG rule (paired bootstrap 10k)."""

from paths import RUNS  # noqa: E402
import sys, json, math, argparse, re
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from router_data import load, f1, DATA, TRACE  # noqa: E402

P = RUNS
A = f"{P}/pilot2/answers/hotpotqa"


def boot(d, B=10000):
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def thr_rate(s, r):
    q = np.sort(s)[::-1]
    m = int(math.ceil(r * len(q) - 1e-9))
    return float((q[m - 1] + q[m]) / 2)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()


def guard_ok(x, q, ctx):
    """Deployment guard: both extracted options appear in the question (>= half of each option's tokens),
    and each date's year appears in the retrieved chunks D."""
    qt = set(norm(q))
    for o in [x.get("A"), x.get("B")]:
        t = norm(o)
        if not t or sum(w in qt for w in t) < max(1, len(t) / 2):
            return False
    for d in [x.get("date_A"), x.get("date_B")]:
        y = re.search(r"\b(\d{3,4})\b", str(d))
        if not y or y.group(1) not in ctx:
            return False
    return True


def cmp_f1(sp, keys, model, guard=False):
    out = {}
    for p in sorted({k[1] for k in keys}):
        refs = {
            json.loads(l)["id"]: json.loads(l)["answers"]
            for l in open(f"{DATA}/hotpotqa/{sp}/{p}/answers/answers.jsonl")
        }
        if guard:
            d = f"{TRACE}/hotpotqa/{sp}/{p}"
            ch = {json.loads(l)["node_id"]: json.loads(l)["text"] for l in open(f"{d}/chunks.jsonl")}
            tr = {json.loads(l)["id"]: json.loads(l) for l in open(f"{d}/trace.jsonl")}
        for l in open(f"{A}/{sp}/{p}/cmp_{model}.jsonl"):
            x = json.loads(l)
            if x["answer"] is not None:
                if guard:
                    t = tr[x["id"]]
                    if not guard_ok(x, t["query"], "\n".join(ch[i] for i in t["topk_ids"][: t["k"]])):
                        continue
                out[(sp, p, x["id"])] = f1(x["answer"], refs[x["id"]])
    return out
