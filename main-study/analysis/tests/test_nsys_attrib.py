import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsys_attrib import analyse, classify  # noqa: E402

NAMES = {1: "fused_moe_kernel", 2: "ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 3: "flash_fwd_splitkv_kernel",
         4: "nvjet_tst_64x8_64x16_4x1_v_bz_TNT", 5: "rmsnorm_kernel", 6: "top_k_top_p_sampling"}


def make_db(path, per_dev):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", NAMES.items())
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INT, start INT, end INT, correlationId INT, "
               "demangledName INT, graphNodeId INT)")
    for dev, rows in per_dev.items():
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
                       [(dev, *r) for r in rows])
    db.commit()


def step(t0, corr, comm_ns):
    # one decode graph execution: moe 3000, comm comm_ns, attention 2000, gemm 1000 ns, then a 500 ns gap,
    # then a non-graph sampling kernel of 500 ns; the next step starts at t0 + 10000
    rows, t = [], t0
    for nid, d in ((1, 3000), (2, comm_ns), (3, 2000), (4, 1000)):
        rows.append((t, t + d, corr, nid, 7))
        t += d
    rows.append((t + 500, t + 1000, 10_000 + corr, 6, None))
    return rows


class Attrib(unittest.TestCase):
    def test_classify(self):
        self.assertEqual([classify(NAMES[i]) for i in range(1, 7)],
                         ["moe_gemm", "comm", "attention", "dense_gemm", "other", "other"])
        self.assertEqual(classify("void moe::dev::routing::routingIndicesClusterKernel<float>"), "moe_other")
        self.assertEqual(classify("bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s6_et64x8_m64x8x32"), "moe_gemm")
        self.assertEqual(classify("allreduce_fusion_kernel_oneshot_lamport<bf16>"), "comm")
        self.assertEqual(classify("fmhaSm100fKernel_QkvE4m3OBfloat16"), "attention")

    def test_decode_window(self):
        per_dev = {0: [], 1: []}
        for k in range(4):
            per_dev[0] += step(k * 10_000, k, 1000)
            per_dev[1] += step(k * 10_000 + 100, 100 + k, 1500)
        with tempfile.TemporaryDirectory() as d:
            make_db(d + "/r.sqlite", per_dev)
            res = analyse(d + "/r.sqlite", "decode", 2, expect_steps=4)
            self.assertFalse(analyse(d + "/r.sqlite", "decode", 2, expect_steps=64)["gate_steps"])
        self.assertTrue(res["gate_devices"])
        self.assertTrue(res["gate_steps"])
        d0 = res["per_device"]["0"]
        self.assertEqual(d0["steps"], 3)
        self.assertAlmostEqual(d0["step_us_mean"], 10.0)
        self.assertAlmostEqual(d0["class_us_per_step"]["moe_gemm"], 3.0)
        self.assertAlmostEqual(d0["class_us_per_step"]["comm"], 1.0)
        self.assertAlmostEqual(d0["class_us_per_step"]["other"], 0.5)
        self.assertAlmostEqual(d0["idle_us_per_step"], 10.0 - 7.5)
        self.assertAlmostEqual(d0["idle_between_executions_us_per_step"], 3.0 - 0.5)
        self.assertAlmostEqual(d0["idle_inside_executions_us_per_step"], 0.0)
        self.assertAlmostEqual(d0["closure_us"], 0.0)
        skew = res["comm_skew"]
        self.assertAlmostEqual(skew["transfer_estimate_us_per_step"], 1.0)
        self.assertAlmostEqual(skew["wait_us_per_step"]["1"], 0.5)
        self.assertAlmostEqual(skew["wait_us_per_step"]["0"], 0.0)

    def test_device_gate_fails(self):
        per_dev = {0: step(0, 1, 1000) + step(10_000, 2, 1000)}
        with tempfile.TemporaryDirectory() as d:
            make_db(d + "/r.sqlite", per_dev)
            self.assertFalse(analyse(d + "/r.sqlite", "decode", 2)["gate_devices"])

    def test_mixed_window(self):
        rows = step(0, 1, 1000) + [(20_000, 60_000, 50, 4, None)] + step(70_000, 2, 1000)
        with tempfile.TemporaryDirectory() as d:
            make_db(d + "/r.sqlite", {0: rows})
            res = analyse(d + "/r.sqlite", "mixed", 1)["per_device"]["0"]
        self.assertEqual(res["decode_executions"], 2)
        self.assertAlmostEqual(res["class_us_outside_decode"]["dense_gemm"], 40.0)
        self.assertAlmostEqual(res["class_us_inside_decode"]["moe_gemm"], 6.0)


if __name__ == "__main__":
    unittest.main()
