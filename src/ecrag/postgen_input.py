"""Candidate-only extension of the existing Decider input (no reference/quality features)."""

import json
from pathlib import Path
from paths import RUNS

ROOT = Path(RUNS)


def attach_candidates(rows, spec):
    from normalize_edge import normalize

    parts = spec.split(":")
    cached = {}
    for r in rows:
        if parts[0] == "legacy":
            path = ROOT / "pilot6" / "trace_cfg3nA" / "hotpotqa" / parts[1] / r["pool"] / "trace.jsonl"
            field = "edge_answer"
            kind = "E"
        else:
            kind = "E" if r["ds"] == "hotpotqa" else "C"
            name = "extract.jsonl" if kind == "E" else "orig.jsonl"
            path = ROOT / parts[0] / "answers" / parts[1] / parts[2] / r["pool"] / name
            field = "edge_answer" if kind == "E" else "cloud_answer"
        if path not in cached:
            items = [json.loads(l) for l in path.read_text().splitlines()]
            cached[path] = {x["id"]: x for x in items}
            if len(items) != len(cached[path]):
                raise ValueError(f"duplicate candidate IDs: {path}")
        ans = cached[path][r["id"]][field]
        if not isinstance(ans, str):
            raise TypeError(f'non-string candidate: {spec}/{r["pool"]}/{r["id"]}')
        if kind == "E":
            ans = normalize(r["query"], ans)[0]
        r["candidate"] = ans
        r["candidate_plan"] = kind
    return rows


def build_postgen_text(row, tok, base_builder, candidate_max_tokens=512):
    # Preserve the exact pre-generation question/evidence prefix. Extra candidate tokens
    # enlarge input rather than remove evidence. The existing encoder's 8192 limit applies.
    base = base_builder({"query": row["query"], "k": row["k"], "chunks": row["chunks"]}, tok, 6144, use_chunks=True)
    candids = tok(row["candidate"], add_special_tokens=False).input_ids
    candidate = row["candidate"] if len(candids) <= candidate_max_tokens else tok.decode(candids[:candidate_max_tokens])
    text = (
        base
        + f"\n\nCandidate answer from plan {row['candidate_plan']}:\n{candidate}\n\nEstimate answer quality and gains from escalation."
    )
    if len(tok(text, add_special_tokens=True).input_ids) > 8192:
        raise ValueError("postgen input would be silently truncated")
    return text
