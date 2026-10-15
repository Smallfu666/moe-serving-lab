# E2 results — throughput per GPU vs interactivity, each platform at its own configuration

All numbers are quoted from an independent scoring of the raw per-request records under the frozen preregistration. Rules:
[`../prereg/e2-prereg.md`](../prereg/e2-prereg.md), including erratum E-1.

**Scope:** one SGLang nightly (`a6a8c560`); Qwen3-235B-A22B-Instruct-2507, public FP8 and NVFP4 checkpoints; bf16 KV;
closed-loop replay of SWE agent sessions (concurrency decays as sessions finish); three launches per configuration;
shared nodes. Throughput per GPU counts output tokens only and leaves out host CPUs, networking and power. Nothing
here is a cost statement or a ranking of GPUs, and no difference is attributed to a single cause.

## Configurations (each point and ratio names its parallelism)

| Config | GPUs | Parallelism | KV pool (tokens) |
|---|---|---|---|
| G2-FP8 | 2 × GB200 | TP2 (the largest single-node GB200 job this cluster accepts) | 514,048 |
| G2-NV4 | 2 × GB200 | TP2, NVFP4 | 1,019,072 |
| H4-FP8 | 4 × H200 | TP4 | 1,522,852 |
| H8-FP8 | 8 × H200 | TP8 + EP8 (TP8 alone cannot start with this FP8 checkpoint on the pinned software; erratum E-1) | 2,127,459 |
| H100-8-FP8 | 8 × H100 | TP8 + EP8; **different cluster, driver (550 + CUDA-13 compatibility libraries) and container runtime — not a controlled comparison** | 778,277 |

## Matched-interactivity ratios (throughput per GPU, numerator / denominator)

Every ratio with a value is `SEPARATED` under the preregistered noise rule (three launches per side).

| Ratio | 50 tok/s/user | 75 | 100 |
|---|---|---|---|
| G2-FP8 (TP2) / H4-FP8 (TP4) | 2.0080 | 2.1141 | 2.1756 |
| G2-FP8 (TP2) / H8-FP8 (TP8+EP8) | 3.2609 | 3.9540 | 7.6952 |
| G2-NV4 (TP2) / H4-FP8 (TP4) | 3.3202 | 3.9475 | 4.3252 |
| G2-NV4 (TP2) / H8-FP8 (TP8+EP8) | 5.3919 | 7.3830 | 15.2985 |
| G2-NV4 (TP2) / G2-FP8 (TP2) | 1.6535 | 1.8672 | 1.9880 |
| H8-FP8 (TP8+EP8) / H4-FP8 (TP4) | 0.6158 | 0.5347 | 0.2827 |
| H100-8-FP8 / H8-FP8 (both TP8+EP8; different cluster, driver and runtime — not controlled) | 0.2599 | 0.3005 | not evaluable |
| G2-FP8 (TP2) / H200 platform (union of H4 TP4 and H8 TP8+EP8) | 2.0080 | 2.1141 | 3.4240 |
| G2-NV4 (TP2) / H200 platform | 3.3202 | 3.9475 | 6.8071 |

At 25 tok/s/user every ratio involving GB200 is not evaluable: the GB200 frontiers start at 42.33 (FP8) and 35.89
(NVFP4) tok/s/user, and the preregistration forbids extrapolation. H8/H4 at 25 is 0.6318 (it uses H4's c = 64 point, whose KV pool is below the working set
(0.76), and has exactly the minimum 4 launch pairs); H100-8/H8 (different cluster, driver and runtime) at 25 is 0.2189.

Two readings to keep in mind:

- At 100 tok/s/user the H200-platform value (30.06 tok/s/GPU) is below H4 alone (47.31): H8's c = 4 point has 0.13
  tok/s/user more interactivity than H4's and so enters the union frontier. That is the registered rule; taking the
  better of H4 and H8 instead would give 2.1756 and 4.3252, which is not the registered rule.
- H8 (8 GPUs, TP8 + EP8) gives less throughput per GPU than H4 (4 GPUs, TP4) at every target. This is not attributed
  to GPU count or to expert parallelism alone.

## Per-configuration frontiers (interactivity tok/s/user, throughput tok/s/GPU; launch medians)

- G2-FP8: (42.33, 272.39), (61.87, 186.04), (83.18, 129.34), (126.71, 60.99)
- G2-NV4: (35.89, 446.98), (56.85, 368.81), (82.58, 245.97), (113.57, 172.43), (152.61, 73.63)
- H4-FP8: (17.47, 219.38), (26.82, 166.54), (44.06, 138.25), (59.01, 89.24), (82.15, 63.53), (120.41, 28.77)
- H8-FP8: (20.06, 125.80), (30.89, 94.94), (48.66, 75.35), (65.69, 47.20), (82.29, 31.18), (101.11, 12.26)
- H100-8-FP8: (9.50, 31.46), (16.44, 28.37), (28.22, 22.99), (49.71, 19.09), (80.81, 9.74)

## Completeness and failures

- Two launches each lost one request to a client-side connection drop (the server stayed up), so H4's c = 64 and
  H8's c = 32 and c = 64 are medians of two launches.
- H100-8's c = 64 hit the 1-hour harness end time in all three launches ("did not complete within the 1 h launch").
- Five earlier H8 jobs are VOID and not data: two node faults and three with the original TP8-only setting.

## Correctness (descriptive)

No configuration is flagged. Mean GSM8K accuracy: G2-FP8 95.426 %, G2-NV4 95.451 %, H4 95.350 %, H8 95.375 %, H100-8
95.325 %. Top-1 agreement with E1's BF16 reference: 0.969666, 0.945923, 0.972839, 0.975159 and 0.975159. The flag is only
a 3-point accuracy test; it does not show the configurations give the same answers (NVFP4 is not shown to be
accuracy-neutral).

## Capacity

KV pool below the working set: G2-FP8 from c = 32, G2-NV4 at c = 64, H4 at c = 64, H100-8 from c = 32. At those points
throughput reflects KV capacity at that GPU count (for GB200, the cluster's 2-GPU limit), not kernel or platform
speed. G2-FP8's c = 32 and c = 64 points and G2-NV4's c = 64 point are dominated and not on their frontiers.

## What I learned

_To be written by the author._

---

Provenance: private research notes, main study E2, 2026-10-16; measurements re-quoted verbatim from the independent score of those runs.
