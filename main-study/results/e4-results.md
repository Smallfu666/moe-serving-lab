# E4 results — where decode-step GPU time goes (nsys attribution of the E2/E3 configurations)

All numbers are quoted from an independent scoring of the harvested nsys attribution summaries and server logs under the
frozen preregistration, after an adversarial review of that score (GO-with-fixes, no blocking finding; the fixes changed
no verdict). Rules: [`../prereg/e4-prereg.md`](../prereg/e4-prereg.md), including erratum E4-E1 (capture mechanics
only); the attribution code is [`../analysis/nsys_attrib.py`](../analysis/nsys_attrib.py).

**Scope:** one SGLang nightly (`a6a8c56015b8`), the E2 configurations with their exact server flags, short
profiler-attached windows (nsys, CUDA graphs traced per node). Decode windows are 64-step captures (63–64 full executions per device) at batch B ∈ {1, 4, 8, 16}
with 16,384-token prompts; one mixed window per configuration comes from the replay at c = 8. E4 is descriptive: it
splits GPU time into kernel classes and idle time and runs no ablation, so **no class is called the cause of any
ratio**. Each configuration ran in one job, so nothing here measures variation between nodes or jobs. No cost, energy
or ranking claim.

| Config | GPUs | Parallelism | Checkpoint |
|---|---|---|---|
| G2-FP8 | 2 × GB200 | TP2 | Qwen3-235B-A22B-Instruct-2507-FP8 |
| G2-NV4 | 2 × GB200 | TP2 | Qwen3-235B-A22B-Instruct-2507-NVFP4 |
| H4-FP8 | 4 × H200 (GPUs 0–3 of an 8-GPU allocation; the other 4 idle, recorded not corrected) | TP4 | FP8 |
| H8-FP8 | 8 × H200 | TP8 + EP8 | FP8 |
| 30B reference | 1 × H200 (TP1), 2 × H200 (TP2), 2 × GB200 (TP2) | — | Qwen3-30B-A3B-FP8, with a profiler-free twin arm |

Hardware facts (H200 sm_90 x86 nodes, GB200 sm_100 aarch64 nodes, one Slurm cluster): [`../../docs/hardware.md`](../../docs/hardware.md),
from the author's hardware inventory (H200 and GB200 rows of that cluster).

## Instrument outcome

- **Every report passed G-CAP** (27 reports: 22 decode windows + 5 mixed windows), **and every decode window passed
  G-BATCH and G-BACKEND**. Every
  configuration has 4 of 4 valid decode windows, so nothing is `NOT_EVALUABLE` under F-EMPTY. No window had
  `cuda graph: False`.
- **G-DISTORT** (profiled median step ÷ the same server's log step outside the window): (a) 0.9863–1.0235 over the 22
  decode windows. (b), against the profiler-free 30B twin, measured on 30B only: J1 TP1 H200 1.0120–1.0244, TP2 H200
  0.9835, TP2 GB200 1.0172. Transferred to 235B **as an assumption**: H200 1.0244, GB200 1.0172. All are ≤ 1.15 and
  ≤ 1.05, so no window is `profiler-distorted`. (a) and (b) share their numerator, so they are not independent
  evidence. Both TP2 profiler-free references ran while earlier arms' exports were still running on the same node,
  which could only understate (b). GB200's transferred 1.0172 rests on this single pair; the log-only attached ÷ free
  ratios (1.0045 GB200, 0.9972 H200 TP2) show no visible effect.
- **G-CLASS: `CLASSIFIER-GAP` flag for G2-FP8** (all 5 windows) and the GB200 30B gate arm. The GB200 MoE activation
  kernel `moe::dev::activation::activationDeepSeekKernel` (4.13–5.29 % of the decode kernel sum, 4.24 % mixed; 6.97 % in
  the 30B arm) lands in `moe_other` only through the generic `moe` token, while the prereg puts activation in
  `other_known`, where the H200 kernel `sglang::act_and_mul_kernel` sits (that it is the MoE activation, and so the
  counterpart, is an inference from the model configuration, not a trace fact). It is shown on its own row, not
  reassigned, and excludes no window. Moving it would not change the F-DOM verdict.
- Two GB200 kernels are shown on their own rows with identity unresolved (no shapes or NVTX in the summaries), not
  flagged: an `nvjet_sm100 … splitK` GEMM in `dense_gemm` (382–407 µs/step G2-FP8, 356–396 µs/step G2-NV4; H200 has a
  similar `nvjet_sm90 … splitK` row in `dense_gemm`), and a bf16 `TgvGemmCuteExtKernel` in G2-NV4 (10.70 % at B = 1).
- Seven H200 reports (H8 B = 4, 8, 16 and mixed; H4 B = 8, 16 and mixed) were attributed on CPU after the job, as the
  prereg provides for exports unfinished at job end, with the same commands and the job's own window-to-report mapping. Three in-job reports (H8 B = 1, H4 B = 1 and 4)
  have no recorded post rc; their validity rests on an empty attribution log plus the gate flags in their summaries.

## Per-token GPU time (sum basis, µs per output token, device means)

G = step time × GPUs ÷ B. Classes plus idle minus overlap add up to G. Profiled windows only.

| Config | B | step µs | thr/GPU (tok/s) | G | comm | moe_gemm | moe_other | other_known | attention | dense_gemm | other | idle | −overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G2-FP8 | 4 | 10529.2 | 189.95 | 5264.6 | 493.0 | 2463.1 | 838.9 | 421.6 | 763.9 | 1055.6 | 25.0 | 69.7 | −866.2 |
| G2-FP8 | 8 | 12870.7 | 310.78 | 3217.7 | 262.6 | 1669.2 | 457.4 | 218.9 | 519.0 | 526.3 | 12.8 | 30.9 | −479.5 |
| G2-FP8 | 16 | 16284.1 | 491.28 | 2035.5 | 135.7 | 1104.2 | 254.0 | 114.3 | 397.6 | 267.7 | 6.9 | 14.1 | −259.1 |
| G2-NV4 | 4 | 7980.5 | 250.61 | 3990.2 | 498.8 | 1359.1 | 376.5 | 386.7 | 777.7 | 1002.1 | 25.4 | 53.9 | −490.0 |
| G2-NV4 | 8 | 9552.1 | 418.76 | 2388.0 | 260.4 | 938.0 | 202.8 | 195.1 | 519.4 | 508.4 | 13.6 | 22.6 | −272.3 |
| G2-NV4 | 16 | 11687.3 | 684.50 | 1460.9 | 132.1 | 584.9 | 111.4 | 97.9 | 397.8 | 259.8 | 7.4 | 11.1 | −141.4 |
| H4-FP8 | 4 | 11247.1 | 88.91 | 11247.1 | 1313.8 | 4138.4 | 1109.7 | 3914.6 | 1716.0 | 2108.7 | 249.0 | 454.9 | −3758.1 |
| H4-FP8 | 8 | 14172.3 | 141.12 | 7086.1 | 731.2 | 3194.3 | 481.6 | 1909.1 | 1150.2 | 970.9 | 128.6 | 206.8 | −1686.7 |
| H4-FP8 | 16 | 16573.2 | 241.35 | 4143.3 | 381.9 | 1851.7 | 234.0 | 1455.5 | 833.5 | 489.2 | 68.0 | 135.4 | −1305.9 |
| H8-FP8 | 4 | 11462.2 | 43.62 | 22924.5 | 4205.4 | 6643.0 | 3311.2 | 8158.5 | 3386.4 | 3511.5 | 992.8 | 1199.0 | −8483.4 |
| H8-FP8 | 8 | 12927.6 | 77.35 | 12927.6 | 2478.2 | 3805.9 | 1805.9 | 4680.4 | 2296.5 | 1694.2 | 532.8 | 592.8 | −4959.1 |
| H8-FP8 | 16 | 15567.4 | 128.47 | 7783.7 | 1616.7 | 2275.8 | 958.0 | 2777.2 | 1672.0 | 846.6 | 281.8 | 291.8 | −2936.2 |

B = 1 rows are in the score; B = 1 is not an F-DOM or F-COMM batch.

Own-row kernels (not reassigned; still counted in the class shown), µs per token [share of kernel sum]:

| Config | Kernel (class) | Status | B=4 | B=8 | B=16 |
|---|---|---|---|---|---|
| G2-FP8 | `moe::dev::activation::activationDeepSeekKernel` (moe_other) | **CLASSIFIER-GAP** | 320.9 [5.29 %] | 164.1 [4.48 %] | 94.3 [4.13 %] |
| G2-FP8 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` (dense_gemm) | identity unresolved | 192.9 [3.18 %] | 97.8 [2.67 %] | 50.9 [2.23 %] |
| G2-NV4 | same nvjet sm100 splitK (dense_gemm) | identity unresolved | 195.0 [4.41 %] | 96.2 [3.65 %] | 49.5 [3.11 %] |
| G2-NV4 | `kernel_cutlass_kernel_TgvGemmCuteExtKernel…outbfloat16…` (dense_gemm) | identity unresolved | 401.3 [9.07 %] | 198.0 [7.51 %] | 99.8 [6.27 %] |

## Q1 — GB200 TP2 vs H200 TP4 (registered readings)

Share of the per-token difference ΔG = G(H4) − G(G2) carried by each class (sum basis; the shares add to more than 1
before the overlap term is subtracted, because H4's kernels overlap more than G2's):

| Pair | B | G_H4 / G_G2 | comm | moe_gemm | moe_other | other_known | attention | dense_gemm | other | idle | −overlap | own row: activation / nvjet splitK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G2-FP8/H4 | 4 | 2.1364 | 0.1372 | 0.2800 | 0.0453 | **0.5839** | 0.1592 | 0.1760 | 0.0374 | 0.0644 | −0.4834 | −0.0536 / −0.0322 |
| G2-FP8/H4 | 8 | 2.2022 | 0.1211 | 0.3942 | 0.0063 | **0.4369** | 0.1632 | 0.1149 | 0.0299 | 0.0455 | −0.3121 | −0.0424 / — |
| G2-FP8/H4 | 16 | 2.0355 | 0.1168 | 0.3546 | −0.0095 | **0.6363** | 0.2068 | 0.1051 | 0.0290 | 0.0575 | −0.4966 | −0.0447 / — |
| G2-NV4/H4 | 4 | 2.8187 | 0.1123 | 0.3830 | 0.1010 | **0.4861** | 0.1293 | 0.1525 | 0.0308 | 0.0553 | −0.4503 | — / −0.0269 |
| G2-NV4/H4 | 8 | 2.9674 | 0.1002 | **0.4803** | 0.0594 | 0.3648 | 0.1343 | 0.0984 | 0.0245 | 0.0392 | −0.3011 | — / −0.0205 |
| G2-NV4/H4 | 16 | 2.8361 | 0.0931 | 0.4723 | 0.0457 | **0.5061** | 0.1625 | 0.0855 | 0.0226 | 0.0463 | −0.4341 | — / −0.0185 |

- **F-DOM: "no single dominant class" for both pairs.** The rule needs one class ≥ 0.40 with the next ≤ half of it at
  every B ∈ {4, 8, 16}. G2-FP8/H4 meets it only at B = 4 (0.5839 vs 0.2800); at B = 8 (0.4369 vs 0.3942) and B = 16
  (0.6363 vs 0.3546) it fails. G2-NV4/H4 fails at every B (0.4861 vs 0.3830; 0.4803 vs 0.3648; 0.5061 vs 0.4723).
- **What `other_known` holds here (description only):** on the H200 side it is almost entirely two kernels, the activation kernel
  `sglang::act_and_mul_kernel` (act-and-mul) and the FP8 activation quantization
  `sglang::per_token_group_quant_flat_kernel` — not norm, RoPE or sampling. For G2-FP8/H4 at B = 4/8/16, other_known = act_and_mul + quant + rest =
  0.4343 + 0.1245 + 0.0251 / 0.3135 + 0.1035 + 0.0200 / 0.5197 + 0.0981 + 0.0185. Kernel values come from the per-device
  top-15 lists, so "rest" also absorbs any variant outside the top 15. Read these shares together with the overlap term,
  which is about as large.
- **F-BRIDGE** (decode-only throughput-per-GPU ratio, interpolated in r = 10⁶ ÷ step µs, against E2's ratio at the same
  tok/s/user; holds when the quotient is in [0.77, 1.30]):

| Pair | tok/s/user | Decode-only | E2 | Quotient | Reading |
|---|---|---|---|---|---|
| G2-FP8/H4 | 50 | — | 2.0080 | — | `NOT_EVALUABLE` (outside both r ranges) |
| G2-FP8/H4 | 75 | 2.6512 | 2.1141 | 1.2541 | holds |
| G2-FP8/H4 | 100 | 2.4949 | 2.1756 | 1.1468 | holds |
| G2-NV4/H4 | 50 | — | 3.3202 | — | `NOT_EVALUABLE` |
| G2-NV4/H4 | 75 | — | 3.9475 | — | `NOT_EVALUABLE` (below G2-NV4's r range) |
| G2-NV4/H4 | 100 | 7.0909 | 4.3252 | 1.6394 | **fires** |

  At G2-NV4/H4 100 tok/s/user the decode-only ratio is above E2's (quotient 1.6394; the rule is two-sided): decode
  steps do not account for the E2 ratio; the rest lies outside decode steps
  (prefill, interleaving, KV capacity), and nothing further is attributed. In the mixed window, decode executions cover
  93.89 % of G2-NV4's window and 86.17 % of H4's (description).

## Q2 — H200 TP8 + EP8 vs TP4

| B | G_H8 | G_H4 | ΔG | comm H8 / H4 | (Δcomm) ÷ ΔG | Largest other terms of ΔG |
|---|---|---|---|---|---|---|
| 4 | 22924.5 | 11247.1 | 11677.3 | 4205.4 / 1313.8 | **0.2476** | other_known 0.3634, moe_gemm 0.2145, moe_other 0.1885; idle 0.0637; −overlap −0.4047 |
| 8 | 12927.6 | 7086.1 | 5841.4 | 2478.2 / 731.2 | **0.2991** | other_known 0.4744, moe_other 0.2267, attention 0.1962; idle 0.0661; −overlap −0.5602 |
| 16 | 7783.7 | 4143.3 | 3640.4 | 1616.7 / 381.9 | **0.3392** | other_known 0.3631, attention 0.2303, moe_other 0.1989; idle 0.0429; −overlap −0.4478 |

Kernel-busy (union) time per token, H8 / H4: 21725.5 / 10792.2 (B=4), 12334.8 / 6879.4 (B=8), 7491.9 / 4007.9 (B=16).

- **F-COMM (decode steps): "partial".** The ratio is neither ≥ 0.50 at every B nor < 0.25 at every B. The largest single term of the
  rest is other_known at every B (0.36–0.47); idle time carries 0.04–0.07.
- **Transfer vs wait** (per token): most of the H8 − H4 communication difference is in the transfer estimate
  (+2675.2 / +1685.2 / +1157.5 µs at B = 4/8/16), not in waiting for other ranks (+216.9 / +59.9 / +76.8). The transfer
  estimate is the fastest rank's communication time, so it is an upper bound on pure transfer. Transfer and wait are two
  readouts of the same communication kernels, not independent evidence.
- On H200 the communication class is the fused all-reduce + RMSNorm kernel
  (`flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_ones…`), which the prereg counts as communication.
- F-BRIDGE for H8/H4: holds at 75 tok/s/user (0.6734 vs E2 0.5347, quotient 1.2594); `NOT_EVALUABLE` at 50 and at 100
  (H8's highest r is 99.69).

## Mixed window (replay c = 8; description only)

| Config | Window ms | Decode-execution share | Rest share | Largest outside-decode classes (ms) |
|---|---|---|---|---|
| G2-FP8 | 2519.6 | 0.7859 | 0.2141 | attention 249.1, moe_gemm 165.7, comm 43.7, moe_other 42.6 |
| G2-NV4 | 1652.2 | 0.9389 | 0.0611 | attention 64.6, moe_gemm 13.6 |
| H4-FP8 | 2441.3 | 0.8617 | 0.1383 | moe_gemm 130.3, attention 92.2, comm 48.6, other_known 47.3 |
| H8-FP8 | 2326.5 | 0.8224 | 0.1776 | comm 117.4, attention 103.9, moe_gemm 95.8 |

The H8 replay was cut by its 150 s deadline (267/273 turns, 0 bad requests) about 100 s after the profiler window
closed, so the cut does not truncate the recorded window.

## Q3 — R16 vs N4 (CPU only, from the E3 logs; `source+runtime, observational`)

Not preregistered (the lead had looked at these logs first); recomputed by the independent scorer with the registered
script. There is no counterfactual run. Worker logs are mapped by 1 s timestamps, and the per-line decode rate
(`gen throughput` ÷ `#running-req`) is not comparable to E2 interactivity. At the complete points c = 1–16:

- **Prefill:** R16 computes about 9.4–13× more new prefill tokens than N4 for the same turns; its cached fraction is
  0.48–0.61 against 0.95–0.96.
- **Decode lines without a CUDA graph:** R16's fraction rises with c (0 → 0.125 → 0.24 → 0.38); N4 has none.
- **Running requests:** R16's median per-rank line holds 1 running request, against 3–14 on N4 at c = 4–16.
- **Per-line decode rate (graph lines) at c = 1:** R16 93.6–93.8 against N4 132.0–133.0.

These co-occur with R16's 0.02028 / 0.02781 ratios to N4. E4 does not establish which of them, if any, produces those
ratios. The scorer did not re-check the source anchors.

## Read these numbers with their limits

- The strongest allowed sentence is: in profiler-attached decode windows at B ∈ {1, 4, 8, 16} and 16,384-token
  prompts, on SGLang `a6a8c56015b8` with the E2 configurations, class C accounted for x % of the per-token GPU-time
  difference between X and Y (sum basis); idle time accounted for y %.
- No causes. No other batch sizes, context lengths, workloads or SGLang versions. Prefill appears only as the mixed-window
  description; Q3's prefill token counts are observational E3 log counts, not E4 GPU readings.
- Absolute µs are quoted inside profiled windows only. For 235B they rely on the 30B G-DISTORT (b) value transferred as
  an assumption.
- One job per configuration: no node-to-node or job-to-job variation. H4 ran beside four idle GPUs of its node.
- E2 bridging holds only for G2-FP8/H4 at 75 and 100 tok/s/user and H8/H4 at 75.
- G2-FP8 tables carry the `CLASSIFIER-GAP` activation row.

## Post-hoc, not scored: the same times per GPU per decode step

Added 2026-10-21 on CPU, after the score; it changes no verdict. [`../analysis/e4_per_gpu_step.py`](../analysis/e4_per_gpu_step.py)
multiplies the per-token table above by B ÷ GPUs, giving µs that one device spends in each class during one decode
step (device means, sum basis; same data, different denominator; output in
[`../analysis/derived/e4_per_gpu_step.md`](../analysis/derived/e4_per_gpu_step.md)).

| Config | B | step µs | comm | moe_gemm | moe_other | other_known | attention | dense_gemm | other | idle | overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G2-FP8 | 4 | 10529.2 | 986.0 | 4926.2 | 1677.8 | 843.2 | 1527.8 | 2111.2 | 50.0 | 139.4 | -1732.4 |
| G2-FP8 | 8 | 12870.7 | 1050.4 | 6676.8 | 1829.6 | 875.6 | 2076.0 | 2105.2 | 51.2 | 123.6 | -1918.0 |
| G2-FP8 | 16 | 16284.1 | 1085.6 | 8833.6 | 2032.0 | 914.4 | 3180.8 | 2141.6 | 55.2 | 112.8 | -2072.8 |
| G2-NV4 | 4 | 7980.5 | 997.6 | 2718.2 | 753.0 | 773.4 | 1555.4 | 2004.2 | 50.8 | 107.8 | -980.0 |
| G2-NV4 | 8 | 9552.1 | 1041.6 | 3752.0 | 811.2 | 780.4 | 2077.6 | 2033.6 | 54.4 | 90.4 | -1089.2 |
| G2-NV4 | 16 | 11687.3 | 1056.8 | 4679.2 | 891.2 | 783.2 | 3182.4 | 2078.4 | 59.2 | 88.8 | -1131.2 |
| H4-FP8 | 4 | 11247.1 | 1313.8 | 4138.4 | 1109.7 | 3914.6 | 1716.0 | 2108.7 | 249.0 | 454.9 | -3758.1 |
| H4-FP8 | 8 | 14172.3 | 1462.4 | 6388.6 | 963.2 | 3818.2 | 2300.4 | 1941.8 | 257.2 | 413.6 | -3373.4 |
| H4-FP8 | 16 | 16573.2 | 1527.6 | 7406.8 | 936.0 | 5822.0 | 3334.0 | 1956.8 | 272.0 | 541.6 | -5223.6 |
| H8-FP8 | 4 | 11462.2 | 2102.7 | 3321.5 | 1655.6 | 4079.2 | 1693.2 | 1755.8 | 496.4 | 599.5 | -4241.7 |
| H8-FP8 | 8 | 12927.6 | 2478.2 | 3805.9 | 1805.9 | 4680.4 | 2296.5 | 1694.2 | 532.8 | 592.8 | -4959.1 |
| H8-FP8 | 16 | 15567.4 | 3233.4 | 4551.6 | 1916.0 | 5554.4 | 3344.0 | 1693.2 | 563.6 | 583.6 | -5872.4 |

H8-FP8 (TP8 + EP8) ÷ H4-FP8 (TP4), per GPU per step; 1.00 = unchanged, 0.50 = halved by twice the GPUs:

| B | step | comm | moe_gemm | moe_other | other_known | attention | dense_gemm | other | idle | overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1.02 | 1.60 | 0.80 | 1.49 | 1.04 | 0.99 | 0.83 | 1.99 | 1.32 | 1.13 |
| 8 | 0.91 | 1.69 | 0.60 | 1.87 | 1.23 | 1.00 | 0.87 | 2.07 | 1.43 | 1.47 |
| 16 | 0.94 | 2.12 | 0.61 | 2.05 | 0.95 | 1.00 | 0.87 | 2.07 | 1.08 | 1.12 |

Description: at the same B, an H8 decode step takes 0.91–1.02× as long as an H4 step on twice the GPUs, so each H8 GPU
produces about half as many tokens per second. Per GPU, attention (0.99–1.00) and other_known (0.95–1.23) do not
shrink, moe_gemm shrinks to 0.60–0.80 and dense_gemm to 0.83–0.87, and comm (1.60–2.12) and moe_other (1.49–2.05) grow.

**One reading for attention, labelled as inference.**

- *Fact, model config.* Qwen3-235B-A22B has 64 query heads and 4 KV heads (`num_attention_heads: 64`,
  `num_key_value_heads: 4`, `head_dim: 128`, 94 layers; HF `config.json` of the FP8 checkpoint at `e156cb4e`).
- *Fact, server logs (raw, not in this repository).* The `KV Cache is allocated` line of the E4 launches (job 557426) gives 24,064 bytes of K per
  token per GPU on both H4 (34.12 GB for 1,522,305 tokens) and H8 (47.67 GB for 2,126,914 tokens; SGLang prints GiB
  as "GB"). That is 94 × 128 × 2 bytes,
  i.e. one KV head per GPU in both: with TP8 each of the 4 KV heads is held by two GPUs (per-configuration values in
  [`../README.md`](../README.md#configurations-and-server-flags)).
- *Inference, not measured.* A decode step's attention on one GPU reads that GPU's KV head for every cached token of
  every request in the batch. With one KV head per GPU in both settings, TP8 reads the same KV bytes per GPU per step
  as TP4 at the same B and context, and only the query heads per GPU halve (16 → 8). If decode attention here is bound
  by those KV reads, its per-GPU time stays flat, as the table shows. No KV-read byte count (DRAM counters) was
  collected, so this is consistent with the data, not shown by it.
- *What it does not cover.* Attention is 15–21 % of an H8 step (1693 of 11462 µs at B = 4, 3344 of 15567 µs at
  B = 16). The flat other_known, the larger comm and moe_other, and the larger −overlap term are not explained by KV
  heads.

**Alternative or additional explanations, none tested:**

- *Sum basis counts waiting as work.* The −overlap term per GPU per step (−4242 to −5872 µs on H8) is as large as the
  largest class. A short kernel launched early (for example with programmatic dependent launch) can be "running" while
  it waits for its predecessor, so flat class times may be partly waiting. Exclusive-time attribution would separate
  this; it needs the `.nsys-rep`/sqlite exports, which are not in this repository.
- *Latency-bound kernels* (our addition, not from the review). At 4–16 decode tokens per step, attention and the
  small elementwise kernels may be bound by launch and memory latency rather than bytes, which would also keep their
  time flat when work per GPU halves, with or without KV replication.
- *other_known under EP8.* With EP8 each GPU applies the activation and FP8 activation quantization only to the tokens
  routed to its 16 experts, which halves the work per GPU against TP4 in an ideal split; the time does not halve. The
  summaries carry no kernel shapes, so whether these kernels run on padded buffers is not known.
- *A test that separates these* (GPU, not run): H8 with attention-DP8 + EP8 instead of TP8 + EP8. Each rank would hold
  all 4 KV heads for only its own requests, which changes KV bytes per GPU without changing the MoE layout.

## What I learned

_To be written by the author._

---

Provenance: private research notes, main study E4, 2026-10-16; measurements re-quoted verbatim from the independent score
of those runs.
