# E3 results — GB200 at node scale (4 GPUs) and one server across four nodes (16 GPUs)

All numbers are quoted from an independent scoring of the raw per-request records under the frozen preregistration. Rules:
[`../prereg/e3-prereg.md`](../prereg/e3-prereg.md); definitions as in E2.

**Scope:** as E2 (one SGLang nightly, two public checkpoints, bf16 KV, closed-loop SWE session replay), plus: N4 is one
GB200 node running the `--tp 4 --ep 4` setting SGLang documents for a larger model of the same family (Qwen3-Coder
480B); its four replicas ran at the same time on four nodes, so their spread is **within-job, across concurrent nodes**,
with no job-to-job variation. R16 is one untuned multi-node configuration with round-robin request dispatch. No
cause is attributed to any difference; no cost or ranking claim.

| Config | GPUs | Parallelism | KV pool (tokens) |
|---|---|---|---|
| N4-FP8 | 4 × GB200 (one node) | TP4 + EP4 | 2,235,840 |
| N4-NV4 | 4 × GB200 (one node) | TP4 + EP4, NVFP4 | 2,750,976 |
| R16-FP8 | 16 × GB200 (four nodes, one server) | TP16 + attention-DP16 + EP16, FlashInfer all-to-all, round-robin dispatch | 539,136 per attention rank |

## Registered ratios (throughput per GPU, numerator / denominator)

Every ratio with a value is `SEPARATED` under the preregistered noise rule (N4 spreads are within-job).

| Ratio | 50 tok/s/user | 75 | 100 |
|---|---|---|---|
| N4-FP8 (TP4+EP4) / G2-FP8 (TP2) | 0.8503 | 0.7974 | 0.7078 |
| N4-NV4 (TP4+EP4) / G2-NV4 (TP2) | 0.7953 | 0.8681 | 0.7976 |
| N4-FP8 (TP4+EP4) / H8-FP8 (TP8+EP8) | 2.7728 | 3.1529 | 5.4466 |
| N4-NV4 (TP4+EP4) / H8-FP8 (TP8+EP8) | 4.2881 | 6.4094 | 12.2019 |
| R16-FP8 (TP16+attn-DP16+EP16; round-robin) / N4-FP8 (TP4+EP4) | 0.02028 | 0.02781 | not evaluable |

At 25 tok/s/user every ratio is not evaluable (the N4 frontiers start at 26.11 and 33.03 tok/s/user, the GB200 2-GPU
side was already not evaluable there in E2, and extrapolation is not allowed). R16/N4 at 100 is not evaluable because
R16's frontier ends at 94.94 tok/s/user.

## Frontiers (interactivity tok/s/user, throughput tok/s/GPU)

- N4-FP8: (26.11, 327.53), (39.30, 233.82), (57.66, 180.58), (77.14, 113.07), (97.72, 75.20), (138.74, 32.87)
- N4-NV4: (33.03, 422.92), (50.77, 308.68), (76.44, 241.17), (103.99, 150.00), (133.06, 99.59), (169.39, 39.97)
- R16-FP8: (8.10, 7.16), (12.86, 5.32), (28.73, 4.76), (94.94, 2.74)

## Completeness

- N4: every point complete on all four replicas of both jobs.
- R16: GSM8K, logprob and c = 1, 4, 8, 16 complete in all three launches; c = 32 was cut by the 1-hour harness end
  time in each, and c = 64 and c = 128 never started (the preregistration expected c = 128 not to complete and allowed c = 64 to be cut).

## Correctness (descriptive)

No configuration is flagged. Mean GSM8K accuracy: N4-FP8 95.489 %, N4-NV4 94.977 %, R16-FP8 95.123 %. Top-1 agreement
with E1's BF16 reference: 0.970581, 0.944580, 0.970093. The flag is only a 3-point accuracy test: NVFP4 at TP4+EP4
is not shown to be accuracy-neutral, and R16 is not shown to give the same answers as N4.

## Read these numbers with their limits

- N4's pool is above the working set at every point (N4-FP8: 1.11× at c = 64), so unlike the 2-GPU GB200 setup it is
  not capacity-limited in this sweep.
- R16's very low throughput per GPU, its prefix hit rate of 49–60 % and its TTFT p50 of about 1.7–1.9 s come with
  round-robin dispatch across 16 attention ranks, which the preregistration expected to hurt prefix reuse. That is a
  caveat, not an explanation: E3 does not isolate the cause.
- N4's replicas ran concurrently in one job; their small spreads say nothing about variation between jobs.
- N4 and G2 differ in GPU count, parallelism and node pool (different node groups of the same cluster), so N4/G2 is
  not a GPU-count scaling result.
- No capacity label applies to R16 in either direction (16 per-rank pools under round-robin). No rack-scale or
  interconnect claim.

## What I learned

_To be written by the author._

---

Provenance: private research notes, main study E3, 2026-10-16; measurements re-quoted verbatim from the independent score of those runs.
