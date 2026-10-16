# E4 preregistration — where decode-step time goes (nsys attribution of the E2/E3 results)

Status: **FROZEN** 2026-10-16. Criteria as of commit `3f66fea` (independent review: round 1 NO-GO, round 2
GO-with-fixes, round 3 GO-with-fixes, round 4 GO on `3f66fea`); this status line is the only change in the freeze
commit, made before any E4 job.

E4 is descriptive attribution. It splits measured GPU time into kernel classes and idle time. Ablations are the only
evidence for a cause, and E4 runs none, so no class is called the cause of any E2/E3 ratio. One SGLang nightly, one
model, the E2 configurations, short profiler-attached windows.

## 1. Questions

- **Q1.** For G2-FP8 (2 × GB200, TP2), G2-NV4 (2 × GB200, TP2, NVFP4) and H4-FP8 (4 × H200, TP4), how is decode-step
  GPU time per output token split across MoE GEMM, other MoE kernels, attention, dense GEMM, communication, other
  kernels and idle time, and how does the per-token decode GPU-time difference split by class (description)? The
  split is bridged to E2's G2-FP8/H4 ratios (2.0080 / 2.1141 / 2.1756) and G2-NV4/H4 ratios
  (3.3202 / 3.9475 / 4.3252) at 50 / 75 / 100 tok/s/user only where F-BRIDGE holds.
- **Q2.** For H8-FP8 (8 × H200, TP8 + EP8) against H4-FP8, how large are communication (split into a transfer estimate
  and waiting for other ranks), idle time and kernel-busy GPU time per output token? The comparison is bridged to
  E2's H8/H4 ratios (0.6158 / 0.5347 / 0.2827) only where F-BRIDGE holds.
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
| J1 | 1 × H200 (`dev`) | `e4-j1-30b`: Qwen3-30B-A3B-FP8, TP1, all five windows of §4; then `e4-j1-30b-free`: the four decode windows with **no profiler attached** | rung 1: capture, clients, export, attribution end to end; nsys-attached startup time; profiler-free reference |
| J2 | 2 × GB200 (`gb200-dev`) | `e4-g2-30b` (30B FP8, TP2, one window; **gate arm**), then G2-NV4, then G2-FP8, then (if time remains) `e4-g2-30b-free` | multi-rank capture on aarch64, then the two GB200 configs (the slowest-loading one first), then the profiler-free reference |
| J3 | 8 × H200 (`dev`) | `e4-h-30b` (30B FP8, TP2 on GPUs 0–1; **gate arm**), then H8-FP8 (TP8 + EP8, all 8 GPUs), then H4-FP8 (TP4 on GPUs 0–3 via `MSL_GPU_SUBSET`), then (if time remains) `e4-h-30b-free` | multi-rank capture on x86, then both H200 configs in one allocation |

- H4 inside an 8-GPU job is one change from E2: four GPUs of the node sit idle and other tenants cannot use them.
  It is recorded, not corrected.
- Subset mechanism: the container gets `CUDA_VISIBLE_DEVICES=0,1,2,3`, and the existing context check requires
  `torch.cuda.device_count() == 4`, so a subset that leaks fails with exit 14.
- Ladder: J2 and J3 are submitted only after J1 passes (§6) **and** J1's `kernels_ge_3pct` lists (every kernel
  base name with ≥ 3 % of a window's kernel sum, with its class) have been checked by the reviewer of this file
  (diff-only). Each runs once. J3 does not wait for J2; each carries its own gate arm.

**Exact submissions** (login node, from `$WS`; arm = `tag|model|tp|server-args|gpu-subset|phases|gate|nsys|min-left-s`):

```
P4=decodeprobe:1:16384:64,decodeprobe:4:16384:64,decodeprobe:8:16384:64,decodeprobe:16:16384:64,replayprof:8:52:8:150:45:150
PD=decodeprobe:1:16384:64,decodeprobe:4:16384:64,decodeprobe:8:16384:64,decodeprobe:16:16384:64
PG=decodeprobe:4:16384:64
F8=Qwen3-235B-A22B-Instruct-2507-FP8 N4=Qwen3-235B-A22B-Instruct-2507-NVFP4 S=Qwen3-30B-A3B-FP8
J1: bash jobs/submit_e4.sh h200 1 01:00:00 -- H200 "e4-j1-30b|$S|1|||$P4|0|1|0" "e4-j1-30b-free|$S|1|||$PD|0|0|600"
J2: bash jobs/submit_e4.sh gb200 2 01:00:00 -- GB200 "e4-g2-30b|$S|2|||$PG|1|1|0" "e4-g2nv4|$N4|2|||$P4|0|1|1300" \
      "e4-g2fp8|$F8|2|||$P4|0|1|1000" "e4-g2-30b-free|$S|2|||$PD|0|0|500"
J3: bash jobs/submit_e4.sh h200 8 01:00:00 -- H200 "e4-h-30b|$S|2||0,1|$PG|1|1|0" "e4-h8|$F8|8|--ep-size 8||$P4|0|1|1000" \
      "e4-h4|$F8|4||0,1,2,3|$P4|0|1|900" "e4-h-30b-free|$S|2||0,1|$PD|0|0|500"
```

Each is first run as `submit_e4.sh <family> <n> <time> --test-only -- <arms>`. `min-left-s` values are E2 ready
times (G2-NV4 602–627 s, G2-FP8 407–422 s, H8 262–392 s) plus about 8 min of windows. J1 records the nsys-attached
ready time; if it exceeds the unprofiled 30B ready time by more than 50 %, the J2/J3 fit is re-checked (diff-only
review) before they are submitted.

## 4. Capture windows (per arm)

1. `decodeprobe:B:16384:64` for B = 1, 4, 8, 16, in that order. B requests are sent at once. Each uses a real replay
   prompt (`seq[:16384]` of the first B sessions from row 200 with at least 16,384 tokens), with
   `max_new_tokens` 512, `ignore_eos` and greedy decoding. When every request has streamed ≥ 96 tokens, the client
   asks for 64 steps. L = 16,384 is close to E2's median replay prompt (19,306–20,397 tokens over c = 1…16). The
   largest window holds 16 × (16,384 + 512) tokens of KV, below every configuration's pool (G2-FP8: 514,048).
2. `replayprof:8:52:8:150:45:150`. The E2 c = 8 replay (sessions 52–59, closed loop, deadline 150 s) runs, and
   150 forward steps are recorded starting 45 s in. This window mixes prefill and decode and is only used to split
   time between decode and everything else (§5.4).

`/flush_cache` runs before each window. The client waits for the server log's `Stop profiling` count to increase;
the harness then waits (≤ 300 s, capped by the end time) for a new report file whose size is stable for 10 s and maps
it to that window (`nsys/phase_reports.txt`); a window with no report is `missing`. A profiler window that fails
never ends the arm unless the server died. If nsys does not number reports in repeat mode, `--force-overwrite` would
overwrite earlier windows: a known J1 failure mode, caught by the mapping (one report for several windows).

**Deadlines that can end a window or an arm** (all registered): harness end time (`MSL_END_EPOCH` = job start +
50 min; a phase needs ≥ 60 s left; replay deadline capped by it); prof_client warm timeout min(600 s, left), window
timeout min(300 s, left), request timeout min(900 s, left); report wait min(300 s, left); arm entry gate
(`min-left-s`, compared with end time − 120 s margin, the same basis as the harness's phase gate); a profiler
window's `/flush_cache` is retried for ≤ min(120 s, left) and the window is skipped (not the arm) if it never
succeeds; a window whose client gave up before `Stop profiling` gets `POST /stop_profile`; no report wait for a window
whose `/start_profile` did not return 200; if the client wrote no JSON, the report wait still runs and the window is
marked `client_json_missing`; server stop ≤ 120 s; background exports run until end time + 420 s, then the marker is written and
unfinished exports are redone on CPU afterwards; Slurm wall 60 min. The server's scheduler watchdog (300 s) is far
above a 64-step window. Whether `cudaProfilerStop` blocks while nsys writes a report is unknown; J1 measures the time
from the `Stop profiling` line to a stable report.

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
2. `moe_gemm` (`fused_moe`, `bmm_`, grouped/cutlass/marlin MoE GEMM).
3. `moe_other` (router, routing, top-k gating/sum, permute, finalize, align).
4. `other_known` (norm, RoPE/rotary, sampling, act-and-mul/SiLU/GELU, quantization), matched on the **base name**
   only (template arguments stripped, so `deep_gemm::fp8_gemm_kernel<…GemmType::Normal…>` stays a GEMM). It comes
   before attention so that FlashInfer-namespaced norm/RoPE/sampling kernels are not filed as attention.
5. `attention` (specific tokens: `flash::`, `flash_fwd`, `flashattn`, fmha, attention/attn, paged decode/prefill,
   merge-state, KV-cache writes; a bare `flash` does not match).
6. `dense_gemm` (gemm, nvjet, cublas, cutlass, xmma, matmul, w8a8/fp8 blockwise, gemv).
7. `other`: everything else.

The unit tests classify real kernel names from earlier SGLang H200 traces (FlashInfer fused-add-RMSNorm, RoPE) plus
constructed FlashInfer sampling/activation/FP4-GEMM names.

Fused all-reduce + RMSNorm kernels count as `comm`. The Triton `fused_moe_kernel` on H200 includes its own
activation-quantize prologue, and this is not separated out.

**G-CLASS (classifier gap).** Kernel names are grouped by base name (template arguments and parameter lists
stripped; `(anonymous namespace)` is kept as part of the name). `nsys_attrib.py` lists every base name with ≥ 3 % of a window's kernel sum, in **any** class, with its
class (`kernels_ge_3pct`), and sets `gate_class_other` false if one of them is in `other`. A false gate, or a listed
kernel the scorer judges misfiled, makes the configuration's tables show that kernel on its own row, flagged
`CLASSIFIER-GAP`; it is not reassigned. Changing the regex needs a dated erratum with independent review; both
versions are then reported.

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
  not evidence of cause. If |ΔG| < 5 % of G_Y, the contributions at that B are `NOT_EVALUABLE` (they would blow up or
  flip sign).

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
| G-BATCH | decode windows, both | probe rc 0 (all B requests streamed through the stop, none finished, no errors); the last TP0 `Decode batch` line before `Stop profiling` shows `#running-req: B` **and** `cuda graph: True`; no `Prefill batch` line between `Profiling starts` and `Stop profiling` | `NOT_STEADY` (excluded); a window whose decode lines show `cuda graph: False` is `NO_GRAPH` (a real software state, excluded, reported) |
| G-DISTORT | decode windows, both | (a) profiled median step time ÷ **nsys-attached step time outside the capture range** (B ÷ `gen throughput` of the last `Decode batch` line with `#running-req: B` before `Profiling starts`) ≤ 1.15; (b) on the 30B model, the same ratio against the **profiler-free** arm's step time at the same B (B ÷ `gen throughput` of that arm's last `#running-req: B` line before `Profiling starts`) ≤ 1.15; also reported, not gated: the log-based attached ÷ log-based free ratio (one instrument on both sides). (b) is computed for each 30B pair in one job at one TP: J1 (TP1) and J3 (TP2, if its free arm ran) on H200, J2 (TP2, if its free arm ran) on GB200; the value transferred to 235B on a platform is the largest ratio available on that platform | absolute µs from that window not quoted; shares quoted, labelled `profiler-distorted`. (b) is measured on 30B only and is transferred to 235B as an assumption, stated as such |
| G-BACKEND | GB200: exit 16 (existing). Both platforms: `moe_gemm` time > 0 in every valid decode window | MoE kernels classified | `CLASSIFIER-GAP` for that configuration |
| G-CLASS | every window, both | `gate_class_other` true and no listed ≥ 3 % kernel judged misfiled | `CLASSIFIER-GAP` flag (§5.2) |

Every gate can fail on both platforms. G-CAP fails on a single-device report from a TP2+ run. G-BATCH fails when the
B−1 admission trap appears (`TRAPS.md` §8) or a request ends early. G-DISTORT fails under heavy node-level graph
tracing. G-BACKEND and G-CLASS fail when a MoE GEMM name matches no class regex. The rung-1 smoke (J1) shows that
each record each gate reads is produced on H200, and the J2 gate arm shows the same on GB200.

**Ladder rule.** J1 passes when its five windows pass G-CAP, at least three of its four decode windows pass G-BATCH,
`nsys_attrib.py` runs on every report, and the free arm logs a `Decode batch` line at each B. Otherwise there is a
written analysis, then the one re-run (an instrument VOID) or a stop (a design defect, owner ruling). The gate arms of
J2 and J3 must pass G-CAP on 2 devices, or the job stops (exit 40) before any 235B arm.

**Stop rules.** A gate failure in a frozen design stops the job; it is not rescued. An instrument VOID gets one re-run
after a written analysis. No running or pending job is cancelled. If J2 or J3 leaves a configuration with fewer than
3 valid decode windows, that configuration is `NOT_EVALUABLE` for E4. It is not re-run unless the cause is an
instrument VOID.

## 7. Registered readings (falsifiers)

- **F-BRIDGE (Q1, Q2).** At each target where both sides are evaluable, compare the decode-only throughput-per-GPU
  ratio with E2's ratio at the same tok/s/user. E2 interactivity is also a per-request decode rate, but its
  (out − 1) ÷ (e2e − TTFT) includes time a request spends stalled behind other requests' prefill, which E4's
  r = 10⁶ ÷ T does not; and E2 throughput includes prefill time. With B ≤ 16 at L = 16,384, some targets may fall
  outside a configuration's r range, so matched-interactivity ratios are expected to be partly `NOT_EVALUABLE`;
  the matched-B view is then the only one reported. If decode-only ÷ E2 ∉ [0.77, 1.30], decode steps do not account for the E2 ratio
  at that target. The write-up must then say that the rest lies outside decode steps (prefill, interleaving,
  KV capacity), quote the §5.4 split as description, and attribute nothing further.
- **F-DOM (Q1).** A class may be called the largest contributor to a ratio at matched B only if its ΔG_c ÷ ΔG ≥ 0.40
  **and** the next class is ≤ half of it, at every B ∈ {4, 8, 16} with valid windows on both sides. Otherwise the
  reading is "no single dominant class". F-DOM may name idle time inside executions only if G-DISTORT's ratios
  (a) and, where it exists, (b) are both ≤ 1.05 for both sides (node-level graph tracing adds gaps between graph
  nodes).
- **F-COMM (Q2).** At every valid B ∈ {4, 8, 16}, compare (G_comm,H8 − G_comm,H4) with ΔG = G_H8 − G_H4:
  - ratio ≥ 0.50 at every valid B: "communication is the main component of H8's per-GPU deficit in decode steps";
  - ratio < 0.25 at every B: "communication is not the main component";
  - otherwise: "partial".

  The wait/transfer split is reported beside it, and the reading names the class or idle time that carries the rest.
  If F-BRIDGE fires for H8/H4, the reading covers decode steps only.
- **F-EMPTY.** A configuration with fewer than 3 valid decode windows is `NOT_EVALUABLE`, and the ratios that need
  it are not evaluable. This is not a result about the software.

Instrument buckets take precedence: a window that fails G-CAP or G-BATCH contributes to no reading.

**Threshold rationale.** 1.15 (G-DISTORT): the order of the profiler-attached host shift seen before (TRAPS §8
Line B) and small next to the ≥ 2× ratios under study; 1.05 for idle claims because idle is the term the profiler
itself inflates. 3 % (G-CLASS): below it a misfiled kernel cannot move a class share by more than 3 points.
[0.77, 1.30] (F-BRIDGE): about ±25–30 %, wider than E2's launch spreads, so a miss is not noise. 0.40 / ½ (F-DOM):
"largest" must also be clearly ahead of the second. 0.50 / 0.25 (F-COMM): majority vs minor. "Half the largest
execution": a decode graph runs every layer, so prefill pieces and helper graphs fall well below it. 5 % of G_Y
(|ΔG| floor): below it the contributions are a ratio of small differences. None is derived from E4 data.

## 8. Q3 (CPU only; exploratory, disclosed)

The lead examined the E3 R16 and N4 server logs and the pinned source **before** this file was written, so Q3 is
not preregistered. Its numbers must be recomputed by the independent scorer from `raw/e3`. The registered method is
the recomputation, using `q3_logs.py` in the lane folder (log windows from `phase_lines.txt` start/end lines, which
map to timestamps for worker logs). Per point it computes:

- prefill new and cached tokens summed over all ranks;
- the fraction of `Decode batch` lines with `cuda graph: False`;
- the median per-request decode rate in lines with and without a graph;
- the per-rank running-request count.

`q3_logs.py` maps worker logs by timestamps of 1 s resolution, and its per-request decode rate is a per-line estimate
(`gen throughput` ÷ `#running-req`), not a per-request measurement, so it is not comparable to E2 interactivity.

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
- **Rates.** Every E4 decision uses the worst case, H200 60 SU/GPU-h and GB200 100 SU/GPU-h, whole hours per job
  (the wallet display can stay flat for hours and then drop, `serving-benchmark-study` lesson 14). Every job is
  capped at 1 h of wall (`--time 01:00:00`; harness end time 50 min).
- **Worst-case cost:**
  - J1: 1 GPU-h → 60 SU;
  - J2: 2 GPU-h → 200 SU;
  - J3: 8 GPU-h → 480 SU;
  - total 740 SU, plus **one** instrument re-run in all of E4 (not one per job), ≤ 480 SU → 1,220 SU.
- **Tracking.** Billed GPU-h come from `sacct` (whole hours per job), plus queued and running jobs counted as spent.
  The wallet is logged before every batch as a floor check only; an observed rate is recorded for information and
  changes no decision. Before any job whose worst case could take the balance below zero, stop and report.

## 11. Scoring and write-up

- A fresh scorer that did not run the jobs reads the harvested JSON, logs and reports. It applies §5–§7 verbatim and
  writes `SCORE_E4.md`, recording every gate per window.
- The write-up `main-study/results/e4-results.md` quotes the score verbatim and is reviewed by a fresh reviewer.
- Raw reports stay on the cluster and in the private raw folder; only compact JSON summaries enter the repo, after
  a redaction scan.

## Erratum E4-E1 (2026-10-16, after J1 556631; independent review GO on `75b5d4f`; accepted 2026-10-16)

Ruling (option 2): the J1 re-run under this erratum is E4's one instrument re-run (§10), so J2 and J3 have no
re-run; any rung that fails its gate or VOIDs ends E4 there. J1 556631's reports were exported (exploratory, CPU
only) before this erratum was written. The erratum changes only capture mechanics (window record, report mapping,
server stop, J1 cold caches), not the questions, gates or attribution rules of §5–§9.

J1 556631 (marker 0) failed the ladder rule: no window got a report, so `nsys_attrib.py` ran on none. nsys 2026.4.1
wrote all five reports (`e4-j1-30b.1` … `.5.nsys-rep`) only when the server exited, not when each window closed, so
every 300 s report wait in §4 ended `missing` (post rc 44). The §4 sentences on the report wait and on measuring
"the time from the `Stop profiling` line to a stable report" are replaced by:

- **Window record.** Each profiler window is recorded in `nsys/windows.txt` as `opened` (`/start_profile` returned
  200), `not_opened`, or `json_missing` (no client JSON). There is no report wait.
- **Mapping.** After the server has stopped, the server log decides which windows opened: a window opened if its
  server-log range (its `phase_lines.txt` start mark to its end mark, or to the next start mark) holds a
  `Profiling starts` line with the window's own profile id (`<tag>_b<B>` / `<tag>_rp<c>`; every TP rank logs one,
  so repeats are expected) and none without it. The k-th opened window maps to `<tag>.k.nsys-rep` (a single opened window may map to an
  unnumbered `<tag>.nsys-rep`). Every window of the arm is `missing` (G-CAP `INSTRUMENT`) if any of these holds: the
  number of opened windows differs from the number of reports; a window holds a `Profiling starts` line without its own id;
  the client state disagrees with the server log (`opened` with none, `not_opened` with one; `json_missing` takes the
  server log's answer); or the server was killed at stop (`killed=1`), since a killed nsys may leave a truncated
  report. G-CAP's other clauses (kernel table, TP devices, 64 ± 4 full executions) still apply to each mapped report.
  They catch a decode window mapped onto the mixed report (150 steps), but not a mixed window mapped onto a decode
  report or two decode windows swapped; those are prevented only by the checks above.
- **Server stop.** On nsys arms the server stop waits up to 300 s (`MSL_NSYS_STOP_WAIT_S`) instead of 120 s, and
  every arm records `server_stop killed=… stop_s=…`.
- **J1 cold caches.** J1 is submitted with `MSL_ARM_COLD_CACHE=1`, so both J1 arms start with empty JIT/compile
  caches and the attached ÷ profiler-free ready-time ratio compares like with like. (In 556631 the free arm ran second
  on the attached arm's warm caches: 241 s vs 116 s.) Whether every JIT cache (e.g. DeepGEMM's) lives under these
  directories is not established, so the ratio is quoted only if both arms' `Capture target prefill CUDA graph end.
  elapsed=` values differ by ≤ 25 %; otherwise the ratio is reported as not like-for-like. J2 and J3 keep one cache
  per job.
- **J1 submission:** `MSL_ARM_COLD_CACHE=1 bash jobs/submit_e4.sh h200 1 01:00:00 -- H200 "e4-j1-30b|$S|1|||$P4|0|1|0"
  "e4-j1-30b-free|$S|1|||$PD|0|0|600"` (otherwise unchanged).

Nothing in §5–§9 changes. J1 556631 itself is not scored: its reports were exported on CPU afterwards only for the
failure analysis (lane `E4_J1_ANALYSIS.md`), which is exploratory and enters no reading.
