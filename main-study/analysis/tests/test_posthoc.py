import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e4_per_gpu_step import per_gpu_step  # noqa: E402
from prefill_decode_coupling import classify, overlap_fraction, summarize_engine  # noqa: E402
from serverlog import DECODE, PREFILL, config_of, launch_label, stamp  # noqa: E402

T0 = datetime(2026, 10, 16, 4, 0, 0)


def at(s):
    return T0 + timedelta(seconds=s)


class Classify(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(classify([], 0, at(0), at(2)), "none")
        self.assertEqual(classify([{"t": at(1), "dp": 3}], 0, at(0), at(2)), "other")
        self.assertEqual(classify([{"t": at(2), "dp": 0}, {"t": at(1), "dp": 3}], 0, at(0), at(2)), "own")
        self.assertEqual(classify([{"t": at(0), "dp": 3}], 0, at(0), at(2)), "ambiguous")
        self.assertEqual(classify([{"t": at(0), "dp": 0}, {"t": at(1), "dp": 3}], 0, at(0), at(2)), "ambiguous")


class EqualSpan(unittest.TestCase):
    def test_stratum(self):
        lines = [{"cls": "other", "graph": True, "ms": ms, "span_s": sp} for ms, sp in ((20.0, 1.0), (24.0, 1.0),
                                                                                       (90.0, 6.0))]
        r = summarize_engine(lines)["other"]
        self.assertEqual((r["lines_span_1s"], r["median_ms_span_1s"], r["max_span_s"]), (2, 22.0, 6.0))
        self.assertIsNone(summarize_engine(lines)["none"]["median_ms_span_1s"])


class Overlap(unittest.TestCase):
    def rec(self, t_send, ttft, e2e):
        return {"t_send": t_send, "ttft_s": ttft, "e2e_s": e2e}

    def test_union_of_waits(self):
        r = self.rec(0.0, 1.0, 5.0)
        others = [self.rec(1.5, 1.0, 3.0), self.rec(2.0, 1.0, 3.0), self.rec(10.0, 1.0, 2.0)]
        self.assertAlmostEqual(overlap_fraction(r, others), 1.5 / 4.0)

    def test_no_decode_interval(self):
        self.assertIsNone(overlap_fraction(self.rec(0.0, 1.0, 1.0), []))


class Parsing(unittest.TestCase):
    def test_lines(self):
        line = ("[2026-10-16 04:04:30 DP2 TP2 EP2] Decode batch, #running-req: 8, #token: 1792, token usage: 0.00, "
                "cuda graph: True, gen throughput (token/s): 36.29, #queue-req: 0")
        self.assertEqual(stamp(line), (datetime(2026, 10, 16, 4, 4, 30), 2))
        m = DECODE.search(line)
        self.assertEqual((m.group(1), m.group(3), m.group(4)), ("8", "True", "36.29"))
        p = "[2026-10-16 04:04:24 TP0] Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, token usage: 0.00"
        self.assertEqual(stamp(p)[1], 0)
        self.assertEqual(PREFILL.search(p).group(2), "6")

    def test_tags(self):
        self.assertEqual(config_of("e1-hfp8tp4-l1"), "E1 H-FP8-TP4")
        self.assertEqual(config_of("e1-hfp8-l2"), "E1 H-FP8")
        self.assertIsNone(config_of("e2-h8fp8-l1"))
        self.assertEqual(config_of("e2-h8fp8ep-l1"), "H8-FP8")
        self.assertEqual(launch_label("e3-r16fp8-l2-r0"), "l2-r0")


class PerGpuStep(unittest.TestCase):
    def test_scale(self):
        row = {"config": "H8-FP8", "B": 4, **{k: 8.0 for k in ("comm", "moe_gemm", "moe_other", "other_known",
                                                               "attention", "dense_gemm", "other", "idle", "overlap")}}
        self.assertAlmostEqual(per_gpu_step(row)["attention"], 4.0)


if __name__ == "__main__":
    unittest.main()
