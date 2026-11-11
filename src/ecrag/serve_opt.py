"""Serving optimizations for the local stages. Same policy, same inputs; only the
implementation changes. Each is checked for agreement with the reference implementation in verify_serve_opt.py.
- build_text_fast: the same string as train_router.build_text, but the chunks are tokenized in one batched call (the per-chunk
  ids are identical; the fast tokenizer is deterministic per string).
- prune_batched: EC-RAG dynamic pruning (keep chunks 0,1; then keep chunk i while slm.judge_relevance says not "No"; stop at
  the first "No") computed with one left-padded forward over all candidate chunks 2..k-1 and greedy next token, instead of one
  generate(max_new_tokens=1) per chunk. The result is the first-"No" prefix, as in the sequential loop.
- merged decider: peft merge_and_unload on the LoRA encoder (fewer matmuls)."""

import torch
from slm_inference import judge_relevance_prompt


def build_text_fast(r, tok, max_len):
    head = f"Question: {r['query']}\n\nRetrieved context ({r['k']} chunks):\n"
    tail = f"\n\nNumber of retained chunks: {r['k']}\nWill a much larger cloud model answer this question better than the small edge model?"
    budget = max_len - len(tok(head + tail, add_special_tokens=False).input_ids) - 8
    pieces = [f"[{i + 1}] {c}\n" for i, c in enumerate(r["chunks"])]
    if not pieces:
        return head + tail
    ids_all = tok(pieces, add_special_tokens=False).input_ids
    body, used = [], 0
    for p, ids in zip(pieces, ids_all):
        if used + len(ids) > budget:
            ids = ids[: max(0, budget - used)]
            if ids:
                body.append(tok.decode(ids))
            break
        body.append(p)
        used += len(ids)
    return head + "".join(body) + tail


_NO = {}


def prune_batched(slm, top, q, k_max):
    """top: reranked nodes (already cut to <= 10). Returns k (same rule as the sequential loop in e2e_run / load_ent)."""
    k = min(10, len(top), k_max)
    if k <= 2:
        return k
    tk = slm.tokenizer
    prompts = [judge_relevance_prompt(top[i].text, q, slm.model_type) for i in range(2, k)]
    old = tk.padding_side
    tk.padding_side = "left"
    enc = tk(prompts, return_tensors="pt", padding=True).to(slm.device)
    tk.padding_side = old
    with torch.no_grad():
        pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
        logits = slm.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, position_ids=pos).logits[
            :, -1, :
        ]
    nxt = logits.argmax(-1).tolist()
    for j, t in enumerate(nxt):
        if t not in _NO:
            _NO[t] = tk.decode([t], skip_special_tokens=True).strip() == "No"
        if _NO[t]:
            return 2 + j
    return k


def merge_decider(dec):
    dec.enc = dec.enc.merge_and_unload()
    return dec


def prune_fwd(slm, top, q, k_max):
    """Sequential, early-stopping like the reference, but one plain forward (greedy next token) per chunk instead of generate()."""
    k = min(10, len(top), k_max)
    if k <= 2:
        return k
    tk = slm.tokenizer
    for i in range(2, k):
        ids = tk(judge_relevance_prompt(top[i].text, q, slm.model_type), return_tensors="pt").input_ids.to(slm.device)
        with torch.no_grad():
            t = int(slm.model(input_ids=ids, use_cache=False).logits[0, -1].argmax())
        if t not in _NO:
            _NO[t] = tk.decode([t], skip_special_tokens=True).strip() == "No"
        if _NO[t]:
            return i
    return k
