# Main study summary

**Question.** One MoE model, one agentic multi-turn workload, one SGLang commit: how do GB200 FP8, GB200 NVFP4 and
H200 FP8 differ in latency, throughput per GPU and prefix-cache reuse, and where does decode-step GPU time go?

**Setup.**
- Model: Qwen3-235B-A22B-Instruct-2507, public FP8 and NVFP4 checkpoints (quantized by different parties), bf16 KV cache.
- Workload: closed-loop replay of SWE agent trajectories (`nebius/SWE-rebench-openhands-trajectories`) with recorded assistant content.
- Platforms: 2 × GB200 TP2; 4 × GB200 TP4 + EP4; 16 × GB200 over four nodes; 2 × H200 TP2, 4 × H200 TP4, 8 × H200 TP8 + EP8; 8 × H100 TP8 + EP8 on another cluster.
- Software: one SGLang nightly, `a6a8c56015b8`; MoE and attention backends on `auto`.
- Evidence: frozen preregistrations, independent scoring, nsys attribution in E4. Results: [E1](results/e1-results.md), [E2](results/e2-results.md), [E3](results/e3-results.md), [E4](results/e4-results.md).

## Headline findings

1. **Same 2 GPUs, low load (E1, c = 1).** TPOT p50 is 7.943 ms on GB200 FP8, 9.932 ms on H200 FP8 (ratio 0.800) and
   6.625 ms on GB200 NVFP4 (0.834 of FP8). *Caveat:* the arms differ in kernel path, HBM bandwidth, KV capacity and
   host CPU, so no single cause; the H200 MoE backend is not established. NVFP4 is not shown lossless: 30 GSM8K
   questions differ against FP8 (limit 20), and top-1 agreement with BF16 is 0.94592 against 0.96967.
2. **At c = 32 on 2 GPUs the KV cache decides the result (E1).** H200 FP8 TTFT p50 is 27,730.2 ms and its prefix hit
   rate 31.07 %, against 92.64 % on GB200 FP8. *Caveat:* the KV pool is 0.16 of the working set on H200 and 0.50 on
   GB200 FP8. This is capacity-limited serving, not GPU speed.
3. **Each platform at its own configuration (E2, throughput per GPU at 50 / 75 / 100 tok/s/user).** GB200 FP8 TP2 ÷
   H200 TP4 = 2.0080 / 2.1141 / 2.1756; GB200 NVFP4 TP2 ÷ H200 TP4 = 3.3202 / 3.9475 / 4.3252. H200 TP8 + EP8 ÷ TP4 =
   0.6158 / 0.5347 / 0.2827; this drop is not attributed to GPU count or expert parallelism alone. *Caveat:* output
   tokens only; not a cost ratio or a GPU ranking. H100 ÷ H200 (0.2599 / 0.3005) ran on a different cluster, driver
   and container runtime, so **it is not controlled**.
4. **No single kernel class explains GB200 TP2 vs H200 TP4 (E4).** The F-DOM verdict is "no single dominant class" for
   both pairs. For G2-FP8 vs H4 the largest share of the per-token GPU-time difference is `other_known` (on H200 almost
   all activation and FP8 activation-quantization kernels): 0.5839 / 0.4369 / 0.6363 at B = 4 / 8 / 16. But
   `moe_gemm` comes close (0.2800 / 0.3942 / 0.3546), and the overlap term is about as large. For H8 vs H4,
   communication carries 0.2476 / 0.2991 / 0.3392 of the difference ("partial"). *Caveat:* descriptive, in
   profiler-attached decode windows only, one job per configuration, no ablation. For GB200 NVFP4 vs H4 at 100
   tok/s/user, decode steps do not account for E2's ratio (quotient 1.6394).
5. **The 16-GPU result is a configuration effect (E3).** One untuned server across four nodes (TP16 + attention-DP16 +
   EP16, round-robin dispatch) gives 0.02028 / 0.02781 of the 4-GPU node's throughput per GPU at 50 / 75 tok/s/user.
   Its logs show a prefill cached fraction of 0.48–0.61 against 0.95–0.96, and 9.4–13× more new prefill tokens.
   *Caveat:* these signals co-occur with the low ratio, but the cause is not isolated, and this is not a 16-GPU,
   rack-scale or interconnect claim. Separately, 4 GPUs ÷ 2 GPUs on GB200 FP8 is 0.8503 / 0.7974 / 0.7078. This is
   not GPU-count scaling, because parallelism and node pool also differ.

Figures (built by [`figures/make_figures.py`](figures/make_figures.py) from the results files only):
[E2 ratios](figures/fig1_e2_per_gpu_ratios.png) · [E3 scaling](figures/fig2_e3_scaling.png) ·
[E4 attribution and Q3](figures/fig3_e4_attribution_q3.png).

## What this study does not show

- No cost, price-per-token, energy or hardware ranking. GB200 here is not B200.
- No single-cause attribution of any ratio. E4 splits time into classes; it does not test causes.
- Prefix hit rates are not estimates for real agentic serving, because the replay reuses recorded answers rather than server-generated KV.
- No other model, context length, workload or SGLang version.

## What I would do next

Re-run the four-node, 16-GPU configuration with cache-aware (prefix-affinity) request dispatch in place of
round-robin, keeping every other flag fixed, under a new preregistration. This tests whether lost prefix reuse
accounts for a large part of the 0.02028 / 0.02781 ratios. The result could go either way.
