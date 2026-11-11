"""CPU-only checks of candidate isolation, target parity, and recovery charging."""

import unittest
from types import SimpleNamespace
import numpy as np
from postgen_input import build_postgen_text
from evaluate_learned_postgen import pair_arrays, choose, execute, online


class Tokenizer:
    def __call__(self, text, **kw):
        return SimpleNamespace(input_ids=list(map(ord, text)))

    def decode(self, ids):
        return "".join(map(chr, ids))


class Contract(unittest.TestCase):
    def test_input_does_not_read_labels_or_reference(self):
        row = {
            "query": "Q",
            "chunks": ["D"],
            "k": 1,
            "candidate": "candidate",
            "candidate_plan": "E",
            "answers": ["SECRET_REFERENCE"],
            "fE": 999,
        }
        base = lambda r, t, n, **kw: "Question: " + r["query"] + "\nEvidence: " + r["chunks"][0]
        out = build_postgen_text(row, Tokenizer(), base)
        self.assertTrue(out.startswith(base(row, None, None)))
        self.assertIn("candidate", out)
        self.assertNotIn("SECRET", out)
        self.assertNotIn("999", out)

    def test_candidate_cap_does_not_remove_evidence(self):
        row = {"query": "Q", "chunks": ["D"], "k": 1, "candidate": "a" * 700, "candidate_plan": "C"}
        out = build_postgen_text(row, Tokenizer(), lambda *args, **kwargs: "fixed evidence")
        self.assertTrue(out.startswith("fixed evidence"))
        self.assertIn("a" * 512, out)
        self.assertNotIn("a" * 513, out)

    def test_reasoning_reuses_paid_candidate(self):
        rows = [
            {
                "fE": 0.1,
                "fN": 0.7,
                "fT": 0.9,
                "N_in": 100,
                "N_out": 10,
                "T_in": 100,
                "T_out": 100,
                "T_trunc": tr,
                "ecrag_edge": False,
                "complexity": 0.5,
                "k": 3,
            }
            for tr in [False, True]
        ]
        pre = pair_arrays(rows, "musique", False)
        post = pair_arrays(rows, "musique", True)
        c = (0.27 * 100 + 1.1 * 10) / 1e3
        r = (0.27 * 100 + 1.1 * 100) / 1e3
        np.testing.assert_allclose(post[3], [c + r, c + r])
        np.testing.assert_allclose(pre[3], [r, c + r])
        np.testing.assert_allclose(post[1], [0.9, 0.7])

    def test_no_budget_infeasibility_fallback_to_edge(self):
        s = np.array([0.0, 1.0])
        g = (np.zeros(2), np.ones(2), np.ones(2), np.full(2, 2.0))
        self.assertIsNone(choose(s, g, 0.5))

    def test_controller_never_uses_quality_labels_for_decisions(self):
        n = 150
        s = np.linspace(-1, 1, n)
        g = (np.zeros(n), np.ones(n), np.zeros(n), np.ones(n))
        policy = {
            "primary": {"threshold": 0},
            "frontier": [
                {"multiple": 0.5, "fraction": 0.25},
                {"multiple": 1, "fraction": 0.5},
                {"multiple": 2, "fraction": 1},
            ],
        }
        _, a = online(s, g, policy, 0.5, 1234)
        _, b = online(s, (np.ones(n) * 99, np.ones(n) * -99, *g[2:]), policy, 0.5, 1234)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
