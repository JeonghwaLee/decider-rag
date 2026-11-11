"""Deterministic answer-format normalization for the HotpotQA edge answer (deployable: uses the question and the answer only).
Motivation (pilot6 diagnosis, f1_vs_judge_diag.txt): the F1 router spends cloud calls on queries where the edge answer is a
correct sentence ('Yes, both X and Y are ...', 'Baz Luhrmann was born first.') that token F1 penalizes. Rules:
  yesno : the question has no ' or ', starts with an auxiliary verb (is/are/was/were/do/does/did/has/have/had/can/could/will/would/should)
          and the answer starts with 'yes' or 'no'  ->  'yes' / 'no'
  choice: the question contains ' or ' and the answer has >= 4 words and matches '^(X) (is|was|are|were|has|had|did|does|came|
          aired|...) ...' where every token of X occurs in the question  ->  X
Otherwise the answer is unchanged. Rules were written from the pilot6 diagnosis examples (test3) and are evaluated on train
pools first (selection), then val; test numbers are reported once."""

import re

AUX = r"^(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b"
VERB = r"(is|was|are|were|has|had|did|does|do|came|aired|premiered|released|founded|formed|established|started|opened|published|born|died|won|played|served|lived|located|older|younger|larger|bigger|longer|taller)"


def toks(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def normalize(q, a):
    ql, al = q.strip().lower(), a.strip()
    if re.match(AUX, ql) and " or " not in ql:  # train-pool fix: "Did X or Y ...?" is a choice question
        m = re.match(r"^(yes|no)\b", al.lower())
        if m:
            return m.group(1), "yesno"
    if " or " in ql and len(al.split()) >= 4:
        m = re.match(rf"^(.+?)\s+{VERB}\b", al, re.I)
        if m:
            x = m.group(1).strip().strip(",.;:")
            xt = toks(x)
            qt = set(toks(q))
            if xt and len(xt) <= 8 and set(xt) <= qt:
                return x, "choice"
    return a, None
