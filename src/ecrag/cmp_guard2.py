"""Date-path guard v2. A comparison answer is used only when
  1. the two extracted options are distinct after normalization and every token of each option occurs in the question;
  2. for each option there is a sentence of the retrieved chunks D that contains the option (all its tokens, or for options
     of >= 2 tokens at least its last token) AND the year of that option's extracted date.
Otherwise the query falls back to the router. guard v1 (analyze_cmp_lib.guard_ok) checked only half-token presence in the
question and year presence anywhere in D."""

import re

STOP = {"the", "a", "an", "of", "and"}


def toks(s):
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split() if w not in STOP]


def sentences(ctx):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", ctx) if s.strip()]


def guard2(x, q, ctx):
    a, b = toks(x.get("A")), toks(x.get("B"))
    qt = set(toks(q))
    if not a or not b or a == b or not set(a) <= qt or not set(b) <= qt:
        return False, "options"
    S = [(set(toks(s)), s) for s in sentences(ctx)]
    for opt, d in [(a, x.get("date_A")), (b, x.get("date_B"))]:
        y = re.search(r"\b(\d{3,4})\b", str(d))
        if not y:
            return False, "no-year"
        ok = False
        for st, raw in S:
            name_ok = set(opt) <= st or (len(opt) >= 2 and opt[-1] in st)
            if name_ok and re.search(rf"\b{y.group(1)}\b", raw):
                ok = True
                break
        if not ok:
            return False, "date-link"
    return True, "ok"


def guard2b(x, q, chunks):
    """v2b (post-hoc: designed after reading test3 v2 rejections, where pronoun sentences such as 'He was born in 1950' failed):
    same option check as v2; the date link is checked per retrieved chunk (one Wikipedia passage) instead of per sentence:
    some chunk must contain the option (all tokens, or its last token for >= 2-token options) and the option's year."""
    a, b = toks(x.get("A")), toks(x.get("B"))
    qt = set(toks(q))
    if not a or not b or a == b or not set(a) <= qt or not set(b) <= qt:
        return False, "options"
    C = [(set(toks(c)), c) for c in chunks]
    for opt, d in [(a, x.get("date_A")), (b, x.get("date_B"))]:
        y = re.search(r"\b(\d{3,4})\b", str(d))
        if not y:
            return False, "no-year"
        if not any(
            (set(opt) <= st or (len(opt) >= 2 and opt[-1] in st)) and re.search(rf"\b{y.group(1)}\b", raw)
            for st, raw in C
        ):
            return False, "date-link"
    return True, "ok"
