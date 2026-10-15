import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2_score import (aggregate, bracket, config_view, frontier, interpolate, matched_ratio, mean_in_flight,  # noqa: E402
                      med, point_metrics, rel_spread, request_rate, separation, union_view)


def rec(t_send, ttft, e2e, n, cached=0, prompt=10, error=None):
    return {"t_send": t_send, "ttft_s": ttft, "e2e_s": e2e, "completion_tokens": n, "cached_tokens": cached,
            "prompt_tokens": prompt, "error": error}


class Rates(unittest.TestCase):
    def test_request_rate(self):
        self.assertAlmostEqual(request_rate(rec(0, 0.1, 1.1, 101)), 100.0)
        self.assertIsNone(request_rate(rec(0, 0.1, 1.1, 1)))
        self.assertIsNone(request_rate(rec(0, 0.5, 0.5, 10)))
        self.assertIsNone(request_rate(rec(0, 0.1, 1.1, 101, error="x")))

    def test_in_flight(self):
        self.assertAlmostEqual(mean_in_flight([rec(0, 0, 2, 2), rec(1, 0, 2, 2)]), 4 / 3)

    def test_point_metrics(self):
        m = point_metrics([rec(0, 0.1, 1.1, 101, cached=5), rec(0, 0.1, 2.1, 101, cached=5)], wall_s=2.0, n_gpus=2)
        self.assertAlmostEqual(m["tput_per_gpu"], 202 / 2 / 2)
        self.assertAlmostEqual(m["interactivity"], 100.0)
        self.assertAlmostEqual(m["prefix_hit"], 0.5)


class Frontier(unittest.TestCase):
    def test_dominated_removed(self):
        self.assertEqual(frontier([(10, 100), (20, 80), (15, 70), (30, 40)]), [(10, 100), (20, 80), (30, 40)])

    def test_duplicates_and_none(self):
        self.assertEqual(frontier([(10, 100), (10, 100), (None, 5)]), [(10, 100)])

    def test_interpolate(self):
        f = [(10, 100), (30, 40)]
        self.assertAlmostEqual(interpolate(f, 20), 70)
        self.assertEqual(interpolate(f, 10), 100)
        self.assertIsNone(interpolate(f, 5))
        self.assertIsNone(interpolate(f, 31))
        self.assertEqual(interpolate([(25, 7)], 25), 7)
        self.assertIsNone(interpolate([(25, 7)], 26))
        self.assertIsNone(interpolate([], 25))


class Separation(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(separation(1.20, [1.1, 1.2, 1.3], 0.04), "SEPARATED")
        self.assertEqual(separation(1.05, [1.0, 1.05, 1.1], 0.04), "NOT_SEPARATED")
        self.assertEqual(separation(1.20, [0.95, 1.2, 1.3], 0.04), "NOT_SEPARATED")
        self.assertEqual(separation(1.20, [1.1, 1.2], 0.2), "NOT_EVALUABLE")
        self.assertEqual(separation(None, [], 0.01), "NOT_EVALUABLE")

    def test_rel_spread(self):
        self.assertAlmostEqual(rel_spread([9, 10, 11]), 0.2)
        self.assertAlmostEqual(rel_spread([100, 120]), 20 / 110)
        self.assertIsNone(rel_spread([10]))

    def test_med(self):
        self.assertEqual(med([100, 120]), 110)
        self.assertEqual(med([3, 1, 2]), 2)
        self.assertIsNone(med([5]))


def P(i, t, complete=True):
    return {"interactivity": i, "tput_per_gpu": t, "complete": complete}


class Aggregation(unittest.TestCase):
    def setUp(self):
        self.a = [{1: P(100, 10), 8: P(50, 40), 64: P(20, 80)},
                  {1: P(102, 11), 8: P(52, 41), 64: P(21, 82)},
                  {1: P(98, 9), 8: P(48, 39.5), 64: P(19, 78, complete=False)}]
        self.b = [{1: P(110, 8), 8: P(60, 30), 64: P(22, 50)} for _ in range(3)]

    def test_aggregate_skips_incomplete(self):
        agg = aggregate(self.a)
        self.assertEqual(agg[64]["n"], 2)
        self.assertEqual(agg[64]["point"], (20.5, 81))
        self.assertEqual(agg[1]["point"], (100, 10))

    def test_bracket(self):
        f = [(10, 1), (20, 2), (30, 3)]
        self.assertEqual(bracket(f, 20), [(20, 2)])
        self.assertEqual(bracket(f, 25), [(20, 2), (30, 3)])
        self.assertEqual(bracket(f, 35), [])

    def test_matched_ratio_separated(self):
        r = matched_ratio(config_view(self.a), config_view(self.b), 50)
        self.assertAlmostEqual(r["ratio"], 40 / (50 - 20 * 28 / 38))
        self.assertEqual(len(r["pairs"]), 9)
        self.assertEqual(r["label"], "SEPARATED")

    def test_cut_launch_dropped_from_pairs(self):
        r = matched_ratio(config_view(self.a), config_view(self.b), 30)
        self.assertEqual(len(r["pairs"]), 6)
        self.assertIsNotNone(r["ratio"])

    def test_too_few_pairs(self):
        r = matched_ratio(config_view(self.a[:1] * 1), config_view(self.b), 50)
        self.assertEqual(r["label"], "NOT_EVALUABLE")

    def test_union_view(self):
        u = union_view(config_view(self.a), config_view(self.b))
        self.assertIn((110, 8), u["front"])
        self.assertEqual(len(u["launch_fronts"]), 3)


if __name__ == "__main__":
    unittest.main()
