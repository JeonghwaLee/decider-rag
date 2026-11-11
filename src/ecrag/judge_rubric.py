"""Blind grading rubric shared by the LLM-judge passes.

Each answer is graded on its own against the reference answers; the prompt never reveals which
model or plan produced it. Labels: CORRECT=1, PARTIAL=0.5, INCORRECT=0.
"""

import asyncio
import json
import re

RUBRIC = """You are grading an answer to a question about a document.

Question: {q}

Reference answer(s) from human annotators (any one of them counts as correct):
{refs}

Candidate answer:
{a}

Grading rules:
- CORRECT: the candidate conveys the same essential information as at least one reference (wording may differ; extra correct detail is fine).
- PARTIAL: the candidate contains part of the essential information, or is mostly right with a minor error or omission.
- INCORRECT: the candidate is wrong, unrelated, or misses the essential information.
{unans_rule}
Reply with exactly one word: CORRECT, PARTIAL, or INCORRECT."""
UN_ALL = "- All references say the question cannot be answered from the document: the candidate is CORRECT only if it says the information is not available; any concrete answer is INCORRECT.\n"
UN_SOME = "- Some annotators judged the question unanswerable: grade against the answerable references; a candidate saying the information is not available is PARTIAL.\n"
SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


def prompt(q, refs, a):
    un = [r for r in refs if r.strip().lower() == "unanswerable"]
    ans = [r for r in refs if r.strip().lower() != "unanswerable"]
    rule = UN_ALL if un and not ans else (UN_SOME if un else "")
    shown = ans if ans else ["(unanswerable)"]
    return RUBRIC.format(q=q, refs="\n".join(f"- {r}" for r in shown), a=a.strip() or "(empty)", unans_rule=rule)


async def grade(client, sem, a, q, refs, ans):
    async with sem:
        for att in range(5):
            try:
                r = await client.chat.completions.create(
                    model=a.model,
                    temperature=0,
                    max_tokens=a.max_tokens,
                    messages=[{"role": "user", "content": prompt(q, refs, ans)}],
                    extra_body=json.loads(a.extra_body),
                )
                raw = (r.choices[0].message.content or "").strip()
                if not raw:  # e.g. reasoning ran out of tokens: retry, then record as unlabeled
                    err = f"empty content finish={r.choices[0].finish_reason}"
                    continue
                m = re.search(r"CORRECT|PARTIAL|INCORRECT", raw.upper())
                # INCORRECT contains CORRECT: check the longer label first
                lab = "INCORRECT" if "INCORRECT" in raw.upper() else (m.group(0) if m else None)
                return lab, raw
            except Exception as e:
                err = repr(e)[:200]
                await asyncio.sleep(5)
        return None, "ERROR " + err
