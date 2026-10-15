# E1 results — 2 × GB200 FP8, 2 × GB200 NVFP4, 2 × H200 FP8 (same commit, same replay)

All numbers are quoted from the independent score of E1, which an independent reviewer recomputed from the raw records
(verdict: confirmed, wording corrections only). Rules: [`../prereg/e1-prereg.md`](../prereg/e1-prereg.md).

**Scope for everything below:** one SGLang nightly (`a6a8c560`); Qwen3-235B-A22B-Instruct-2507, public FP8 and NVFP4
checkpoints; bf16 KV cache; TP2 on 2 GPUs per arm on shared nodes; one replay sample of SWE agent trajectories with
recorded assistant content. GB200 here is not B200. The arms differ in more than one factor (kernel path, HBM
bandwidth, KV capacity, host CPU), so no difference is attributed to a single cause.

## Gates

All gates passed on all nine jobs (eight arm launches and the BF16 reference); no VOIDs, no re-runs, registered order
kept. On H200 no MoE backend-selection line was logged; the score reads the backend gate as applying to GB200 only, as
the preregistration's gate row states.

## Predictions

| ID | Verdict | Numbers |
|---|---|---|
| P1 | HIT | c = 1 TPOT p50: GB200 FP8 7.943 ms, H200 FP8 9.932 ms, ratio 0.800 (band 0.50–0.85) |
| P2 | HIT | c = 1 TPOT p50: GB200 NVFP4 6.625 ms, GB200 FP8 7.943 ms, ratio 0.834 (band 0.55–0.95) |
| P3 | HIGHER | c = 32 prefix-cache hit rate: GB200 FP8 92.64 %, H200 FP8 31.07 %, +61.58 points |
| P4a | MOVED | GSM8K, GB200 NVFP4 vs FP8: 30 questions differ per launch pair (limit 20); accuracies 95.45 % vs 95.53 % |
| P4b | NO_CHANGE_BEYOND_AA | GSM8K, H200 FP8 vs GB200 FP8: 20 questions differ, exactly at the limit of 20 |
| P5 | HIT | top-1 agreement with a BF16 reference: GB200 FP8 0.96967, GB200 NVFP4 0.94592 |

P1–P4 confirm directions already seen in the feasibility runs before the preregistration was frozen; only P5 was
blind.

## Per-arm values (mean of two launches)

| Arm | c | TTFT p50 ms | TPOT p50 ms | output tok/s per GPU | prefix hit % |
|---|---|---|---|---|---|
| GB200 FP8 | 1 | 63.5 | 7.943 | 59.6 | 95.79 |
| GB200 NVFP4 | 1 | 53.4 | 6.625 | 71.7 | 95.79 |
| H200 FP8 | 1 | 81.9 | 9.932 | 46.7 | 95.94 |
| GB200 FP8 | 8 | 81.8 | 16.335 | 195.4 | 95.66 |
| GB200 NVFP4 | 8 | 64.6 | 12.299 | 255.9 | 95.66 |
| H200 FP8 | 8 | 120.4 | 24.459 | 117.4 | 91.41 |
| GB200 FP8 | 32 | 118.2 | 41.998 | 286.4 | 92.64 |
| GB200 NVFP4 | 32 | 84.9 | 29.439 | 422.7 | 95.90 |
| H200 FP8 | 32 | 27,730.2 | 95.177 | 47.3 | 31.07 |

A descriptive 4-GPU H200 arm (TP4) is in the score; it decides nothing and is not compared with the 2-GPU arms here.

## Read these numbers with their limits

- At c = 32 every arm's KV pool is below the working set (pool ÷ working set: H200 0.16, GB200 FP8 0.50, GB200 NVFP4
  0.99), and at c = 8 the H200 arm is too. Those points are capacity-limited serving on 2 GPUs, including eviction and
  queueing; they are not kernel or architecture speed. The H200 c = 32 row in particular is a cache too small for the
  load, not a slow GPU.
- "NVFP4 is lossless" is not supported: P4a moved on per-question flips and P5 shows lower agreement with BF16, even
  though GSM8K accuracy is equal.
- Which kernels executed was not traced. No cost, ranking or B200 statement.

## What I learned

_To be written by the author._

---

Provenance: private research notes, main study E1, 2026-10-16; measurements re-quoted verbatim from the independent score of those runs.
