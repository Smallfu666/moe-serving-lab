# E1 results — 2 × GB200 FP8, 2 × GB200 NVFP4, 2 × H200 FP8 (same commit, same replay)

All numbers are quoted from an independent scoring of the raw per-request records under the frozen preregistration. Rules: [`../prereg/e1-prereg.md`](../prereg/e1-prereg.md).

**Scope for everything below:** one SGLang nightly (`a6a8c560`); Qwen3-235B-A22B-Instruct-2507, public FP8 and NVFP4
checkpoints; bf16 KV cache; TP2 on 2 GPUs per arm on shared nodes; one replay sample of SWE agent trajectories with
recorded assistant content. GB200 here is not B200. The arms differ in more than one factor (kernel path, HBM
bandwidth, KV capacity, host CPU), so no difference is attributed to a single cause.

## Gates

All gates passed on all nine jobs (eight arm launches and the BF16 reference; the backend gate applies to the four
GB200 launches only, see below); no VOIDs, no re-runs, registered order kept.

**MoE backend on H200.** No MoE backend-selection line was logged on the H200 launches. The empty backend record is
partly our own harness's fault: its search pattern looked for `moe_runner_backend=`, but the server prints its
settings as a dictionary, so the pattern never matched. Under the preregistration's literal reading the
backend gate applies to the GB200 launches only (they logged `Use flashinfer_trtllm as MoE runner backend`), so the
empty H200 record is not a gate failure. Under a stricter reading (an empty record counts as a failure), P1, P3, P4b
and P5 would be NOT_EVALUABLE and only P2 and P4a would stand. The literal reading was applied. Descriptive evidence from
the H200 server logs: the settings show `'moe_runner_backend': 'auto'`, and at startup the server looked up kernel-configuration
files for the Triton fused-MoE runner on H200 (none for the current Triton version; it fell back to an older one). That is consistent with the Triton fused-MoE path, but it is
not a logged selection and does not show which kernels executed: **the backend on H200 is not established.** The pattern is fixed in the harness for future experiments; E2 and E3 kept their
frozen harness, and their backend records were read from the server logs.

## Predictions

| ID | Verdict | Numbers |
|---|---|---|
| P1 | HIT | c = 1 TPOT p50: GB200 FP8 7.943 ms, H200 FP8 9.932 ms, ratio 0.800 (band 0.50–0.85) |
| P2 | HIT | c = 1 TPOT p50: GB200 NVFP4 6.625 ms, GB200 FP8 7.943 ms, ratio 0.834 (band 0.55–0.95) |
| P3 | HIGHER | c = 32 prefix-cache hit rate: GB200 FP8 92.64 %, H200 FP8 31.07 %, +61.58 points |
| P4a | MOVED | GSM8K, GB200 NVFP4 vs FP8: 30 questions differ per launch pair (limit 20); accuracies 95.45 % vs 95.53 % |
| P4b | NO_CHANGE_BEYOND_AA | GSM8K, H200 FP8 vs GB200 FP8: 20 questions differ, exactly at the limit of 20 (boundary: one more flip would give MOVED) |
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

A descriptive 4-GPU H200 arm (TP4) is in the full score.

## Read these numbers with their limits

- At c = 32 every arm's KV pool is below the working set (pool ÷ working set: H200 0.16, GB200 FP8 0.50, GB200 NVFP4
  0.99), and at c = 8 the H200 arm is too. Those points are capacity-limited serving on 2 GPUs, including eviction and
  queueing; they are not kernel or architecture speed. The H200 c = 32 row is capacity-limited serving on a 160,140-token KV pool; it does not
  measure GPU speed.
- "NVFP4 is lossless" is not supported: P4a moved on per-question flips and P5 shows lower agreement with BF16, even
  though GSM8K accuracy is equal.
- Prefix hit rates are not estimates for real serving: the replay uses recorded assistant content.
- Which kernels executed was not traced. No cost, ranking or B200 statement.

## What I learned

_To be written by the author._

---

Provenance: private research notes, main study E1, 2026-10-16; measurements re-quoted verbatim from the independent score of those runs.
