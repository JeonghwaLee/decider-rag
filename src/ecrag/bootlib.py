"""Shared paired bootstrap. Fixed seed, B resamples, 95% percentile CI.
cluster=None resamples questions; cluster=<array of group ids> resamples whole groups (e.g. Qasper paper_id), keeping every
question of a drawn group (paired cluster bootstrap). Returns (mean, lo, hi)."""

from paths import DATA  # noqa: E402
import json, glob
import numpy as np

DATA = DATA


def boot(d, cluster=None, B=10000, seed=0, stat=np.mean):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    if cluster is None:
        idx = rng.integers(0, len(d), (B, len(d)))
        m = np.array([stat(d[i]) for i in idx]) if stat is not np.mean else d[idx].mean(1)
    else:
        cluster = np.asarray(cluster)
        g = {c: np.where(cluster == c)[0] for c in np.unique(cluster)}
        G = list(g)
        m = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, len(G), len(G))
            ii = np.concatenate([g[G[j]] for j in pick])
            m[b] = stat(d[ii])
    return float(stat(d)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paper_ids(split, keys):
    """keys: (split, pool, id) tuples for Qasper -> paper_id per key (from data/ecrag/qasper/<split>/<pool>/meta.jsonl)."""
    m = {}
    for p in glob.glob(f"{DATA}/qasper/{split}/pool_*/meta.jsonl"):
        pool = p.split("/")[-2]
        for l in open(p):
            x = json.loads(l)
            m[(pool, x["id"])] = x["paper_id"]
    return np.array([m[(k[1], k[2])] for k in keys])
