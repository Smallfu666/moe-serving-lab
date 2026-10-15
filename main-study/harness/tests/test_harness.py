import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from replay_bounds import build_prompts, common_prefix_len, truncate_session, turn_bounds  # noqa: E402
from sample_sessions import sample_ids  # noqa: E402


class ReplayBounds(unittest.TestCase):
    turns = [([1, 2, 3], [4, 5]), ([6], [7, 8, 9]), ([10, 11], [12])]

    def test_prompts_carry_recorded_answers(self):
        self.assertEqual(build_prompts(self.turns), [
            [1, 2, 3],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ])

    def test_bounds(self):
        rows = turn_bounds(self.turns)
        self.assertEqual([r["turn"] for r in rows], [1, 2])
        self.assertAlmostEqual(rows[0]["replay_bound"], 3 / 6)
        self.assertAlmostEqual(rows[0]["serving_bound"], 5 / 6)
        self.assertAlmostEqual(rows[1]["replay_bound"], 6 / 11)
        self.assertAlmostEqual(rows[1]["serving_bound"], 9 / 11)

    def test_replay_never_exceeds_serving(self):
        for r in turn_bounds(self.turns):
            self.assertLessEqual(r["replay_bound"], r["serving_bound"])

    def test_single_turn_has_no_bound(self):
        self.assertEqual(turn_bounds([([1], [2])]), [])

    def test_common_prefix(self):
        self.assertEqual(common_prefix_len([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(common_prefix_len([], [1]), 0)

    def test_truncate_drops_tail(self):
        self.assertEqual(truncate_session(self.turns, 9), self.turns[:2])
        self.assertEqual(truncate_session(self.turns, 4), [])


class Sampling(unittest.TestCase):
    def test_deterministic_and_order_free(self):
        ids = [f"s{i}" for i in range(100)]
        a = sample_ids(ids, 10, seed=7)
        b = sample_ids(list(reversed(ids)), 10, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 10)

    def test_too_many(self):
        with self.assertRaises(ValueError):
            sample_ids(["a", "b"], 3, seed=0)


if __name__ == "__main__":
    unittest.main()
