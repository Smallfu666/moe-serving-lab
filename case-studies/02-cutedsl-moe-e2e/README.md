# Case 02: a Hopper MoE kernel wired end to end into a serving engine

> **Private draft — owner decision pending.** Whether this case study may ever be published is the author's decision
> and has not been made. It describes measurements of a private, unpublished integration; the integration code itself
> is not in this repository. Numbers are quoted from the independent scores of the runs.

## Question

FlashInfer ships several new SM90 (Hopper) MoE backends. SGLang's automatic backend choice does not reach the
CuTe-DSL bf16 one (`cute_dsl_fused_moe_bf16`).

1. **Op level.** On H200, at real MoE shapes, is that kernel faster or slower than the Triton MoE path SGLang uses?
2. **End to end.** Wired into an SGLang server through a private opt-in runner, what happens to TTFT, TPOT and
   accuracy? Is any gain the kernel's, or only a sign that SGLang's shipped Triton config is stale?
3. **On the pinned version.** Does the result survive on the FlashInfer version SGLang actually pinned at the time?

## Setup

| | |
|---|---|
| GPU | H200 (SM90): 1 GPU for op level and 30B; 4 GPUs for 235B |
| Models | Qwen3-30B-A3B bf16 at TP1; Qwen3-235B-A22B bf16 at TP4 (the usual launch recipe for this model on H200 is TP8) |
| SGLang | upstream `0b635266d4` plus a private local opt-in runner (never pushed) |
| FlashInfer | 0.7.1rc4 for the op-level and the main end-to-end runs. **This was not SGLang's pin at the time (0.7.0.post1).** |
| Non-default switches | `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=1`, `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1` |
| Baseline | SGLang's Triton MoE path with its shipped H200 config (a fallback config file that SGLang itself logs as "might be sub-optimal") |
| End-to-end protocol | CUDA graphs on; radix cache flushed before each run; fixed-length token-id prompts, prefill-heavy 4096 → 32 and decode-heavy 256 → 1024; two fresh launches per arm in `B C C B` order; GSM8K 1,314 questions at temperature 0 |
| Labels | Per cell `FASTER` / `WITHIN` / `SLOWER` against `T = max(0.03, 3 × A/A)`; a cell whose A/A exceeds 0.05 is `NOT_EVALUABLE` |

## Result

**1. Op level** (RUNTIME, median of 60 interleaved CUDA-graph replays, random routing, cold L2, synthetic weights).
At both shapes, the CuTe-DSL bf16 kernel took ≤ 0.90 × the whole-MoE-op time of SGLang's Triton path at every
M ∈ {1024, 4096, 16384}:

| Shape | r at M = 1024 / 4096 / 16384 |
|---|---|
| Qwen3-30B-A3B TP1 (H 2048, I 768, E 128, top-8) | 0.878 / 0.768 / 0.743 |
| Qwen3-235B-A22B TP4-local slice (H 4096, I 384, E 128, top-8) | 0.858 / 0.781 / 0.755 |

At small batches there was no band win (30B: within ±10 % at M = 1–256; 235B: faster at M = 1 only, within ±10 % at
M = 4–256). Two other new backends took ≥ 1.10 × the time of their Triton counterparts at every M: cuTile bf16 against
Triton bf16, and CUTLASS FP8 block against SGLang's Triton block-FP8 path.

**2. End to end, 30B** (RUNTIME, run J5).

| Cell | c = 1 | c = 8 | c = 32 | c = 128 |
|---|---|---|---|---|
| Prefill-heavy TTFT, r | 0.873 FASTER | 0.883 FASTER | 0.888 FASTER | 0.884 FASTER |
| Decode-heavy TPOT, r | 1.031 **SLOWER** | 0.975 WITHIN | 0.971 WITHIN | 0.957 FASTER |

- An earlier run with the same protocol (J2) gave TTFT r = 0.870 / 0.874 / 0.882 / 0.884. The runs are scored
  separately and never pooled.
- **Attribution.** J5 also ran a third arm: Triton with a config re-tuned by SGLang's own tuner. The CuTe-DSL TTFT
  ratio vs the re-tuned ratio was 0.873 vs 1.002 (c = 1), 0.883 vs 0.999 (c = 8), 0.888 vs 1.025 (c = 32) and 0.884
  vs 1.000 (c = 128). So in this run the prefill gain was not explained away by re-tuning Triton. But the re-tuned
  config was not shown faster than the shipped one in this run, so this does not mean Triton cannot be made faster.
- **Accuracy.** GSM8K moved by −0.0011 (0.9053 vs 0.9064), within the A/A level (`NO_CHANGE_BEYOND_AA`).
- **Not lossless.** Two launches of the CuTe-DSL arm were not identical to each other (mean |Δlogp| 0.0277 on 64
  sequences), while the two baseline launches were token-identical.
- **Without CUDA graphs** (run J3, its own table, never mixed with the graph table): decode TPOT was SLOWER at every
  concurrency (r = 1.148 / 1.147 / 1.145 / 1.168), prefill TTFT at c = 1 was SLOWER (r = 1.155), and the prefill band
  was `NOT_EVALUABLE`. No mechanism was measured for this.
- **Dispatch** (TRACE, profiled steps at 2048/64 tokens, c = 8): every one of the 48 MoE layers ran the CuTe-DSL GEMM1
  and GEMM2 kernels once per step, and no Triton MoE kernel ran.

**3. End to end, 235B on 4 × H200** (RUNTIME, run J8). Prefill TTFT r = 0.876 / 0.886 / 0.924 at c = 1 / 8 / 16 (all
FASTER); decode TPOT r = 0.892 (FASTER) / 0.957 / 0.967 (WITHIN). The pre-registered pass gate passed. GSM8K moved by
−0.0004 (0.9486 → 0.9482), `NO_CHANGE_BEYOND_AA`. No re-tuned arm, so no attribution.

**4. On SGLang's pinned FlashInfer 0.7.0.post1: no usable verdict** (RECORD; SGLang main `5eca4d8c` plus the same
private runner).

- Run J6: SGLang's startup autotune rewrote the saved CuTe-DSL tactic table on every CuTe-DSL launch (8 of 21 entries
  in the first CuTe-DSL launch, 5 in the second), while baseline launches left it unchanged. A pre-registered gate exists for exactly
  this, so the run stopped; none of its timing or accuracy numbers may be quoted.
- Run J7 froze the table read-only and disabled startup autotune on both arms (not SGLang's out-of-box behaviour). The
  table held (74 hits, 0 misses per CuTe-DSL launch), but one prefill cell's A/A was too wide (`NOT_EVALUABLE`) and
  decode c = 1 was SLOWER (r = 1.035), so the pass gate was `NOT_EVALUABLE`.

**5. After SGLang moved to FlashInfer 0.7.1** (RECORD, run J9). Across three restarts onto a saved table, on SGLang
`130e11d5` + FlashInfer 0.7.1 no entry changed and all 21 tuning-time lookups hit; on SGLang `5eca4d8c` + FlashInfer
0.7.0.post1 each restart rewrote 10, 10 and 9 of 21 entries. This is consistent with
<https://github.com/flashinfer-ai/flashinfer/pull/5705>, which was merged before our runs and describes the same
behaviour; **we did not find this first**, and our two environments differ by 401 FlashInfer commits plus SGLang
changes, so the PR's effect is not isolated. The end-to-end A/B on the new stack again gave no pass verdict: on 30B the
server crashed (segfault) when the profiler stopped in one baseline launch, so the dispatch proof failed and every
cell was `INSTRUMENT`, so the band and the gate were `NOT_EVALUABLE`; and on 235B (J10) one prefill cell's A/A was 0.0855, so the band and the gate
were `NOT_EVALUABLE`.

## Limits

- Every end-to-end number that passed its gates (J2, J5, J8) was measured on FlashInfer 0.7.1rc4 with two non-default
  switches, not on the version SGLang pinned at the time.
- Decode at c = 1 was slower on 30B. WITHIN cells are not "the same speed".
- Not lossless: the CuTe-DSL arm is not run-to-run identical.
- No mechanism is claimed for any gain or loss (no profiler counters). Why the tactic table was rewritten is a source
  reading, consistent with the upstream PR, not tested in isolation.
- One node per run, one A/A pair per cell; the 0.03 floor is a chosen value. 235B ran at TP4, not the usual TP8.
- Op-level results use random routing and synthetic weights.
- Measured between 2026-10-08 and 2026-10-13.

## What I learned

_To be written by the author. Prompts:_

1. Why does an op-level win at large M show up as a prefill gain and not a decode gain?
2. What did the re-tuned Triton arm rule out, and what did it not rule out?
3. Why does startup autotuning make "the same configuration" not the same experiment from launch to launch?
4. Why does none of this support opening a pull request yet?

## Reproduce

Not reproducible from this repository: the integration runner is private and not published. The op-level comparison
uses FlashInfer's public `cute_dsl_fused_moe_bf16` (FlashInfer 0.7.1rc4) against SGLang's Triton fused MoE at the
shapes above.

## 給作者的備註（zh-TW）

這篇是私有草稿，標「等你決定」。這個案例只放 write-up，不放 patch；要不要公開由你決定（預設：不公開）。必須照實講的：好的數字都量在 0.7.1rc4、不是當時 SGLang 釘的版本；decode c=1 變慢；不能說 lossless；沒有機制證明；tactic 表被改寫的行為是上游 PR 5705 先描述的，不是我們先發現的。「What I learned」請用你自己的話寫。

---

Provenance: private research notes, case-02, 2026-10-13; measurements re-quoted verbatim from the independent scores of those runs.
