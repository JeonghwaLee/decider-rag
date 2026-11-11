"""Evidence perturbation diagnostic. Does the decider's score move with the REAL relative gain of
the actions when the evidence changes (not: does it send more to the cloud)?
Items: the first --n questions of --spec (stored rows: query, k, pruned chunks). Two inputs per question, same k and chunk order:
  orig  the stored pruned chunks
  swap  every chunk containing a gold answer string (normalized substring) is replaced by a chunk of an UNRELATED question of the
        same split (question offset +97, the candidate chunk with the closest character length that contains none of this
        question's gold answers); if no chunk contains an answer, the top-1 chunk is replaced. A leading "Passage i:" label keeps
        the original position's label.
For each input: E (edge extract, normalized), N (cloud orig), T (cloud orig_think; truncated -> retry N, as fallback N) answers,
generated now on the perturbation replicas (--cloud/--edge), F1 vs gold; decider scores (HF, merged LoRA, build_text_fast) dEN, dNT.
Output: --out/<name>.jsonl rows {id, cond, n_swapped, has_ans_chunk, fE, fN, fT, T_trunc, dEN, dNT}. Analysis: --analyze.
Usage: perturb_evidence.py --spec pilot9:musique:test --ckpt <decider> --n 150 --actions E N T --name musique --out dir"""

import argparse, json, os, re, string, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

PFX = re.compile(r"^(Passage \d+:\s*)")


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def analyze(out, name):
    from bootlib import boot

    R = [json.loads(l) for l in open(f"{out}/{name}.jsonl")]
    O = {r["id"]: r for r in R if r["cond"] == "orig"}
    S = {r["id"]: r for r in R if r["cond"] == "swap"}
    ids = [i for i in O if i in S]
    acts = [a for a in ("E", "N", "T") if O[ids[0]].get("f" + a) is not None]
    print(
        f"{name}: n {len(ids)} | answer-bearing chunk present {np.mean([S[i]['has_ans_chunk'] for i in ids]):.2f} | mean swapped {np.mean([S[i]['n_swapped'] for i in ids]):.2f}"
    )
    for a in acts:
        o = np.array([O[i]["f" + a] for i in ids])
        s = np.array([S[i]["f" + a] for i in ids])
        w = boot(s - o)
        print(f"  F1 {a}: orig {o.mean():.4f} swap {s.mean():.4f} diff {w[0]:+.4f} [{w[1]:+.4f},{w[2]:+.4f}]")
    pairs = [("dEN", "E", "N")] + ([("dNT", "N", "T")] if "T" in acts else [])
    for h, lo, hi in pairs:
        tg = lambda X, i: X[i]["f" + hi] - X[i]["f" + lo]
        to, ts = np.array([tg(O, i) for i in ids]), np.array([tg(S, i) for i in ids])
        po, ps = np.array([O[i][h] for i in ids]), np.array([S[i][h] for i in ids])
        dt, dp = ts - to, ps - po
        wt, wp = boot(dt), boot(dp)
        print(
            f"  {h}: true gain orig {to.mean():+.4f} swap {ts.mean():+.4f} change {wt[0]:+.4f} [{wt[1]:+.4f},{wt[2]:+.4f}] | decider orig {po.mean():+.4f} swap {ps.mean():+.4f} change {wp[0]:+.4f} [{wp[1]:+.4f},{wp[2]:+.4f}]"
        )
        rs = lambda x, y: float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
        bs = []
        rng = np.random.default_rng(0)
        for _ in range(2000):
            b = rng.integers(0, len(ids), len(ids))
            if np.std(dt[b]) > 0 and np.std(dp[b]) > 0:
                bs.append(np.corrcoef(dt[b], dp[b])[0, 1])
        print(
            f"    per-question change: corr(true change, decider change) pearson {np.corrcoef(dt, dp)[0, 1]:+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}] spearman {rs(dt, dp):+.3f}"
            f" | level corr(decider, true gain): orig {np.corrcoef(po, to)[0, 1]:+.3f} swap {np.corrcoef(ps, ts)[0, 1]:+.3f}"
        )
        for lab, m in [("true gain fell (<0)", dt < 0), ("unchanged (=0)", dt == 0), ("true gain rose (>0)", dt > 0)]:
            if m.sum():
                print(f"      {lab:22s} n {int(m.sum()):3d} decider change mean {dp[m].mean():+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--actions", nargs="+", default=["E", "N", "T"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cloud", default="http://127.0.0.1:8293/v1")
    ap.add_argument("--edge", default="http://127.0.0.1:8295/v1")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.analyze:
        return analyze(a.out, a.name)
    import torch
    from openai import OpenAI
    from transformers import AutoTokenizer
    from train_router4 import load
    from e2e_ent2 import load_decider
    from serve_opt import build_text_fast, merge_decider
    from regen_answers import QP, QS, body, parse, VARIANTS, CFG
    from normalize_edge import normalize
    from router_data import f1, DATA

    rows_all = load(a.spec)
    rows = rows_all[: a.n]
    ds, sp = rows[0]["ds"], rows[0]["split"]
    G = {}
    for pool in {r["pool"] for r in rows_all}:
        for l in open(f"{DATA}/{ds}/{sp}/{pool}/answers/answers.jsonl"):
            x = json.loads(l)
            G[(pool, str(x["id"]))] = x["answers"]
    gold = lambda r: [g for g in G[(r["pool"], str(r["id"]))] if len(norm(g)) >= 2]

    def swap(idx):
        r = rows_all[idx]
        gs = [norm(g) for g in gold(r)]
        ch = list(r["chunks"])
        hit = [j for j, c in enumerate(ch) if any(g in norm(c) for g in gs)]
        tgt = hit if hit else [0]
        donor = rows_all[(idx + 97) % len(rows_all)]
        pool_c = [c for c in donor["chunks"] if not any(g in norm(c) for g in gs)]
        used = set()
        for j in tgt:
            cand = [c for c in pool_c if c not in used] or pool_c
            if not cand:
                continue
            body_len = len(PFX.sub("", ch[j]))
            c = min(cand, key=lambda c: abs(len(PFX.sub("", c)) - body_len))
            used.add(c)
            m = PFX.match(ch[j])
            ch[j] = (m.group(1) if m else "") + PFX.sub("", c)
        return ch, len(tgt), bool(hit)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    dec = None
    if a.ckpt:
        n_out = json.load(open(f"{a.ckpt}/args.json")).get("n_out", 5) if os.path.exists(f"{a.ckpt}/args.json") else 5
        dec = merge_decider(load_decider(a.ckpt, n_out))
    edge = OpenAI(base_url=a.edge, api_key="none", timeout=1800)
    cloud = OpenAI(base_url=a.cloud, api_key="none", timeout=1800)

    def gen(act, q, ch):
        if act == "E":
            r = edge.completions.create(
                model="qwen3-4b",
                prompt=QP + body(ch, q, VARIANTS["extract"]) + QS,
                temperature=0,
                max_tokens=CFG["extract"]["edge_max"],
            )
            return normalize(q, parse(r.choices[0].text, CFG["extract"]["parse"]))[0], False
        think = act == "T"
        r = cloud.chat.completions.create(
            model="q38-flash-next-fp8",
            temperature=0,
            max_tokens=6144 if think else 256,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": body(ch, q, VARIANTS["orig_think" if think else "orig"]) + "\n"},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": think}},
        )
        c = r.choices[0]
        txt = c.message.content or ""
        ok = ("</think>" in txt or c.finish_reason == "stop") if think else True
        return parse(txt, None), not ok

    fn = f"{a.out}/{a.name}.jsonl"
    done = {(json.loads(l)["id"], json.loads(l)["cond"]) for l in open(fn)} if os.path.exists(fn) else set()
    json.dump(vars(a), open(f"{a.out}/{a.name}_args.json", "w"), indent=1)
    items = []
    for idx, r in enumerate(rows):
        sw, nsw, has = swap(idx)
        items += [(r, "orig", list(r["chunks"]), 0, None), (r, "swap", sw, nsw, has)]
    items = [it for it in items if (str(it[0]["id"]) + "|" + it[0]["pool"], it[1]) not in done]

    def run(it):
        r, cond, ch, nsw, has = it
        q = r["query"]
        refs = G[(r["pool"], str(r["id"]))]
        o = {"id": str(r["id"]) + "|" + r["pool"], "cond": cond, "n_swapped": nsw, "has_ans_chunk": has, "k": r["k"]}
        A = {}
        for act in a.actions:
            A[act] = gen(act, q, ch)
        for act in a.actions:
            ans, tr = A[act]
            if act == "T" and tr:
                ans = A["N"][0] if "N" in A else gen("N", q, ch)[0]
            o["f" + act] = f1(ans or "", refs)
            o["ans_" + act] = ans
        o["T_trunc"] = bool(A["T"][1]) if "T" in A else None
        o["chunks"] = ch
        return o

    with ThreadPoolExecutor(32) as ex, open(fn, "a") as f:
        for o in ex.map(run, items):
            if dec is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    enc = tok(
                        [
                            build_text_fast(
                                {"query": "", "k": o["k"], "chunks": o["chunks"]}
                                | {
                                    "query": next(r["query"] for r in rows if str(r["id"]) + "|" + r["pool"] == o["id"])
                                },
                                tok,
                                6144,
                            )
                        ],
                        return_tensors="pt",
                    ).to("cuda")
                    s = dec(enc.input_ids, enc.attention_mask)[0].float().tolist()
                o["dEN"], o["dNT"] = s[3], s[4]
            o.pop("chunks")
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            f.flush()
    analyze(a.out, a.name)


if __name__ == "__main__":
    main()
