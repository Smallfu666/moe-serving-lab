# E4 preregistration — where decode-step time goes (nsys attribution of the E2/E3 results)

Status: **DRAFT** (not frozen). Frozen when an independent reviewer returns GO and the freeze commit is recorded here,
before any E4 job.

E4 is descriptive attribution. It splits measured GPU time into kernel classes and idle time. Ablations are the only
evidence for a cause, and E4 runs none, so no class is called the cause of any E2/E3 ratio. One SGLang nightly, one
model, the E2 configurations, short profiler-attached windows.

## 1. Questions

- **Q1.** For G2-FP8 (2 × GB200, TP2), G2-NV4 (2 × GB200, TP2, NVFP4) and H4-FP8 (4 × H200, TP4), how is decode-step
  GPU time per output token split across MoE GEMM, other MoE kernels, attention, dense GEMM, communication, other
  kernels and idle time? Which classes carry the per-GPU throughput differences behind E2's G2-FP8/H4 ratios
  (2.0080 / 2.1141 / 2.1756) and G2-NV4/H4 ratios (3.3202 / 3.9475 / 4.3252) at 50 / 75 / 100 tok/s/user?
- **Q2.** For H8-FP8 (8 × H200, TP8 + EP8) against H4-FP8, how large are communication (split into a transfer estimate
  and waiting for other ranks), idle time and kernel-busy GPU time per output token? These are the terms behind
  E2's H8/H4 ratios of 0.6158 / 0.5347 / 0.2827.
- **Q3.** Why is R16 (16 × GB200, TP16 + attention-DP16 + EP16, round-robin) at 0.02028 / 0.02781 of N4 per GPU?
  This is answered **on CPU only**, from the E3 logs and configuration (§8). No GPU job is registered for Q3.

## 2. Fixed inputs

As in E2 (`e2-prereg.md` §2, including erratum E-1). Image digests and SGLang `a6a8c56015b8`; FP8 and NVFP4
checkpoints; `--kv-cache-dtype bfloat16`; MoE and attention backends `auto`; `replay.jsonl` sha256 `baf4efba…d7ff`.
On GB200, the existing gate requires `flashinfer_trtllm` as the MoE runner (exit 16). The harness is
`main-study/harness/` and `main-study/analysis/` at the freeze commit, deployed as a separate clone (`harness-e4`).
Each job records the hashes of its wrapper, environment file and `run_arm.sh`.

**Profiler.** The server runs inside the container under the image's own nsys (2026.4.1, checked on the login node):
`nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --trace-fork-before-exec=true --capture-range=cudaProfilerApi
--capture-range-end=repeat --sample=none --cpuctxsw=none`. Recording happens only inside windows that the client opens
with `POST /start_profile {"activities": ["CUDA_PROFILER"], "num_steps": S}`. Each window becomes its own report, and
the report is written when the window closes, not at server shutdown. SGLang calls `cudaProfilerStart` from the
base-GPU rank only (`profiler_manager.py`, at the pinned commit). Whether nsys then records **every** rank's kernels
is not assumed; G-CAP tests it on each window.

## 3. Configurations and jobs (exact E2 server flags; only the nsys prefix is added)

| Job | Hardware | Arms, in order | Purpose |
|---|---|---|---|
| J1 | 1 × H200 (`dev`) | `e4-j1-30b`: Qwen3-30B-A3B-FP8, TP1, all five windows of §4 | rung 1: capture, clients, export, attribution end to end |
| J2 | 2 × GB200 (`gb200-dev`) | `e4-g2-30b` (30B FP8, TP2, one window `decodeprobe:4`; **gate arm**), then G2-FP8, then G2-NV4 | multi-rank capture on aarch64, then the two GB200 configs |
| J3 | 8 × H200 (`dev`) | H8-FP8 (TP8 + EP8, all 8 GPUs), then H4-FP8 (TP4 on GPUs 0–3 via `MSL_GPU_SUBSET`) | both H200 configs in one allocation |

- H4 inside an 8-GPU job is one change from E2: four GPUs of the node sit idle and other tenants cannot use them.
  It is recorded, not corrected.
- Subset mechanism: the container gets `CUDA_VISIBLE_DEVICES=0,1,2,3`, and the existing context check requires
  `torch.cuda.device_count() == 4`, so a subset that leaks fails with exit 14.
- Ladder: J2 and J3 are submitted only after J1 passes (§6). Each runs once. J3 does not wait for J2.

## 4. Capture windows (per arm)

1. `decodeprobe:B:16384:64` for B = 1, 4, 8, 16, in that order. B requests are sent at once. Each uses a real replay
   prompt (`seq[:16384]` of the first B sessions from row 200 with at least 16,384 tokens), with
   `max_new_tokens` 512, `ignore_eos` and greedy decoding. When every request has streamed ≥ 96 tokens, the client
   asks for 64 steps. L = 16,384 is close to E2's median replay prompt (19,306–20,397 tokens over c = 1…16). The
   largest window holds 16 × (16,384 + 512) tokens of KV, below every configuration's pool (G2-FP8: 514,048).
2. `replayprof:8:52:8:150:45:150`. The E2 c = 8 replay (sessions 52–59, closed loop, deadline 150 s) runs, and
   150 forward steps are recorded starting 45 s in. This window mixes prefill and decode and is only used to split
   time between decode and everything else (§5.4).

`/flush_cache` runs before each window. The client waits for the server log's `Stop profiling` count to increase.
The windows are short: 64 or 150 steps, never a whole run.

## 5. Attribution rules (frozen; implemented in `analysis/nsys_attrib.py`)

### 5.1 Steps and time bases (decode windows, per device)

- **Execution:** the kernels that share one graph launch's `correlationId` and carry a `graphNodeId`. A **full decode
  execution** has at least half the kernel count of the largest execution in the window.
- **Steady region:** from the start of the first full decode execution to the start of the last one. Steps =
  executions − 1. **Step time** = region ÷ steps; the median of start-to-start intervals is also reported.
- Over the region (kernels are clipped to it; memcpy/memset are not counted), three bases are reported side by side
  and never mixed:
  - **sum:** per-class kernel time;
  - **union:** busy time;
  - **idle** = region − union, split into time between executions and time inside them.

  **Overlap** = sum − union. The closure sum − overlap + idle = wall is an identity, so it checks the code and is
  not evidence.
- Device aggregation: the mean over the configuration's devices, with the per-device range reported beside it.

### 5.2 Kernel classes

The first matching regex in `CLASSES` decides, applied to the lower-cased demangled name. The order is:

1. `comm` (NCCL, custom/one-shot/Lamport all-reduce, all-gather, reduce-scatter, all-to-all).
2. `moe_gemm` (`fused_moe`, `bmm_`, grouped/cutlass MoE GEMM).
3. `moe_other` (routing, top-k gating, permute, finalize, align).
4. `attention` (flash, fmha, paged decode/prefill, KV-cache writes).
5. `dense_gemm` (gemm, nvjet, cublas, cutlass, xmma, matmul, w8a8/fp8 blockwise, gemv).
6. `other`: everything else (norms, activations, quantization, RoPE, sampling, elementwise).

Fused all-reduce + RMSNorm kernels count as `comm`. The Triton `fused_moe_kernel` on H200 includes its own
activation-quantize prologue, and this is not separated out.

**G-CLASS (classifier gap).** A single kernel name in `other` can hold ≥ 3 % of a window's kernel sum. When that
happens, the configuration's tables list that kernel on its own row, flagged `CLASSIFIER-GAP`, and it is not
reassigned. Changing the regex needs a dated erratum with independent review; both versions are then reported.

### 5.3 Per-token GPU time and ratios

For configuration X with N GPUs at batch B and step time T (µs):

- GPU time per output token: G = T × N ÷ B.
- Per class: G_c = class time per step × N ÷ B.
- Idle: G_idle = idle per step × N ÷ B.
- Decode interactivity: r = 10⁶ ÷ T (tok/s/user).
- Decode throughput per GPU: B × r ÷ N.

These definitions give:

- **Ratio at matched B:** throughput-per-GPU ratio X/Y = G_Y ÷ G_X.
- **Ratio at matched decode interactivity** for targets 50 / 75 / 100. Throughput per GPU is linearly interpolated in
  r between adjacent B points of the same configuration. No extrapolation: a target outside a configuration's
  r range is not evaluable.
- **Class contributions to a difference at matched B:** ΔG_c ÷ ΔG with ΔG = G_Y − G_X. Idle is counted as one class,
  and overlap enters with a negative sign. The contributions sum to 1 by construction, so they are a description,
  not evidence of cause.

### 5.4 Mixed window (replay c = 8)

Per device, the window wall is split into the span of full decode executions and the rest. Class sums are reported
inside and outside decode executions. On GB200, prefill uses breakable graphs, and their partial executions fall
outside the threshold, so they count as "rest". This window is description only. It enters no ratio.

### 5.5 Communication skew (Q2, TP ranks)

Steps are matched across devices by the nearest start time to device 0's executions. For each step:

- the minimum communication time over devices is the **transfer estimate**;
- each device's excess over that minimum is its **wait**.

The transfer estimate still contains whatever the fastest rank spent waiting, so it is an upper bound on pure
transfer. That limit is part of the claim.

## 6. Gates (each names its platforms; each can fail on each)

| Gate | Applies to | Passes when | On failure |
|---|---|---|---|
| G-CAP (instrument) | every window, H200 and GB200 | report exists; kernel table present; kernels on exactly TP devices; decode windows: full executions within 64 ± 4 on every device | window `INSTRUMENT`; a gate arm (J2 30B) failing stops the job (exit 40) |
| G-BATCH | decode windows, both | probe rc 0 (all B requests streamed through the stop, none finished, no errors); the last TP0 `Decode batch` line before `Stop profiling` shows `#running-req: B`; no `Prefill batch` line between `Profiling starts` and `Stop profiling` | window `NOT_STEADY`, excluded |
| G-DISTORT | decode windows, both | profiled median step time ÷ unprofiled step time ≤ 1.15. Unprofiled = B ÷ `gen throughput` of the last `Decode batch` line with `#running-req: B` before `Profiling starts` | absolute µs from that window not quoted; shares quoted, labelled `profiler-distorted` |
| G-BACKEND | GB200: exit 16 (existing). Both platforms: `moe_gemm` time > 0 in every valid decode window | MoE kernels classified | `CLASSIFIER-GAP` for that configuration |
| G-CLASS | every window, both | no single `other` kernel ≥ 3 % of kernel sum | `CLASSIFIER-GAP` flag (§5.2) |

Every gate can fail on both platforms. G-CAP fails on a single-device report from a TP2+ run. G-BATCH fails when the
B−1 admission trap appears (`TRAPS.md` §8) or a request ends early. G-DISTORT fails under heavy node-level graph
tracing. G-BACKEND and G-CLASS fail when a MoE GEMM name matches no class regex. The rung-1 smoke (J1) shows that
each record each gate reads is produced on H200, and the J2 gate arm shows the same on GB200.

**Ladder rule.** J1 passes when its five windows pass G-CAP, at least three of its four decode windows pass G-BATCH,
and `nsys_attrib.py` runs on every report. Otherwise there is a written analysis, then one re-run (an instrument VOID)
or a stop (a design defect, owner ruling).

**Stop rules.** A gate failure in a frozen design stops the job; it is not rescued. An instrument VOID gets one re-run
after a written analysis. No running or pending job is cancelled. If J2 or J3 leaves a configuration with fewer than
3 valid decode windows, that configuration is `NOT_EVALUABLE` for E4. It is not re-run unless the cause is an
instrument VOID.

## 7. Registered readings (falsifiers)

- **F-BRIDGE (Q1, Q2).** At each target where both sides are evaluable, compare the decode-only throughput-per-GPU
  ratio with E2's ratio at the same tok/s/user. E2 interactivity is also a per-request decode rate, but E2
  throughput includes prefill time. If decode-only ÷ E2 ∉ [0.77, 1.30], decode steps do not account for the E2 ratio
  at that target. The write-up must then say that the rest lies outside decode steps (prefill, interleaving,
  KV capacity), quote the §5.4 split as description, and attribute nothing further.
- **F-DOM (Q1).** A class may be called the largest contributor to a ratio at matched B only if its ΔG_c ÷ ΔG ≥ 0.40
  **and** the next class is ≤ half of it, at every B ∈ {4, 8, 16} with valid windows on both sides. Otherwise the
  reading is "no single dominant class".
- **F-COMM (Q2).** At every valid B ∈ {4, 8, 16}, compare (G_comm,H8 − G_comm,H4) with ΔG = G_H8 − G_H4:
  - ratio ≥ 0.50: "communication is the main component of H8's per-GPU deficit in decode steps";
  - ratio < 0.25 at every B: "communication is not the main component";
  - otherwise: "partial".

  The wait/transfer split is reported beside it, and the reading names the class or idle time that carries the rest.
  If F-BRIDGE fires for H8/H4, the reading covers decode steps only.
- **F-EMPTY.** A configuration with fewer than 3 valid decode windows is `NOT_EVALUABLE`, and the ratios that need
  it are not evaluable. This is not a result about the software.

Instrument buckets take precedence: a window that fails G-CAP or G-BATCH contributes to no reading.

## 8. Q3 (CPU only; exploratory, disclosed)

The lead examined the E3 R16 and N4 server logs and the pinned source **before** this file was written, so Q3 is
not preregistered. Its numbers must be recomputed by the independent scorer from `raw/e3`. The registered method is
the recomputation, using `q3_logs.py` in the lane folder (log windows from `phase_lines.txt` start/end lines, which
map to timestamps for worker logs). Per point it computes:

- prefill new and cached tokens summed over all ranks;
- the fraction of `Decode batch` lines with `cuda graph: False`;
- the median per-request decode rate in lines with and without a graph;
- the per-rank running-request count.

Source anchors are at `a6a8c56015b8`. Q3 is reported with the label `source+runtime, observational`. There is no
counterfactual run. The cheapest discriminating runs would be R16 with prefix-aware dispatch, or a 4-node replica
set, at ≥ 16 GB200 GPU-h each. Their worst-case cost (≥ 1,600 SU) exceeds the E4 budget, so they are a proposal,
not part of E4.

## 9. Claim ceiling

The strongest allowed sentence is: "In profiler-attached decode windows at batch B ∈ {1, 4, 8, 16} and 16,384-token
prompts, on SGLang `a6a8c56015b8` with the E2 configurations, class C accounted for x % of the per-token GPU-time
difference between X and Y (sum basis); idle time accounted for y %."

E4 does not cover:

- causes;
- other batch sizes, context lengths or workloads;
- prefill, except the §5.4 description;
- other software versions;
- unprofiled absolute timings, except where G-DISTORT passes;
- variation between nodes or jobs (each configuration runs one job);
- H4's behaviour on a node with no idle neighbours.

No cost, energy or ranking statement. Hardware facts used in the write-up cite `HARDWARE_INVENTORY.md` rows.

## 10. Budget (owner 2026-10-16)

- **Billing.** Only the project whose wallet name contains 「測試計畫」, balance 1,401.4171 SU at planning. Its ID is
  resolved at submit time and never written down.
- **Rates.** Worst case until measured: H200 60 SU/GPU-h, GB200 100 SU/GPU-h, whole hours per job. Every job is
  capped at 1 h of wall (`--time 01:00:00`; harness end time 50 min).
- **Worst-case cost:**
  - J1: 1 GPU-h → 60 SU;
  - J2: 2 GPU-h → 200 SU;
  - J3: 8 GPU-h → 480 SU;
  - total 740 SU, plus one 480 SU instrument re-run at most → 1,220 SU.
- **Rate check.** The real rate is measured from the wallet delta after J1 (and re-checked after some hours). The
  wallet and `sacct` billed GPU-h are logged before every batch. Before any job whose worst case could take the
  balance below zero, stop and report.

## 11. Scoring and write-up

- A fresh scorer that did not run the jobs reads the harvested JSON, logs and reports. It applies §5–§7 verbatim and
  writes `SCORE_E4.md`, recording every gate per window.
- The write-up `main-study/results/e4-results.md` quotes the score verbatim and is reviewed by a fresh reviewer.
- Raw reports stay on the cluster and in the private raw folder; only compact JSON summaries enter the repo, after
  a redaction scan.
