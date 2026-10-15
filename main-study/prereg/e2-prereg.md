# E2 preregistration — throughput vs interactivity, each platform at its own configuration

Status: REVIEW CANDIDATE (not yet binding; becomes FROZEN, with the date, in the commit that follows an independent
review GO and precedes every E2 job).

## 1. Question

How much output throughput per GPU does each platform deliver at a given per-user decode speed, when each runs the
same model and the same agentic replay at a configuration reasonable for that platform? The answer is a set of
throughput–interactivity curves, their frontiers, and ratios of throughput per GPU at matched interactivity, each with
a launch-to-launch spread from three launches.

E2 is descriptive. It ranks nothing beyond the measured configurations and states no cause for any difference.

## 2. Configurations

| Config | Hardware | Parallelism | Checkpoint | Why this configuration |
|---|---|---|---|---|
| G2-FP8 | 2 × GB200 | TP2 | FP8 | single-node GB200 jobs on this cluster are limited to 1–2 GPUs (3–15 cannot be requested; ≥ 16 is multi-node and outside E2) — a **cluster constraint, not a design choice** |
| G2-NV4 | 2 × GB200 | TP2 | NVFP4 | as G2-FP8 |
| H4-FP8 | 4 × H200 | TP4 | FP8 | half a node |
| H8-FP8 | 8 × H200 | TP8 | FP8 | one full node |
| H100-8-FP8 | 8 × H100 (80 GB) | TP8 | FP8 | **conditional** on §2a. A different cluster, GPU driver (550 vs 580), node type and container runtime (`singularity` vs `apptainer`); **not a controlled comparison with the H200 configurations** and always labelled so |

Fixed as in the E1 preregistration (`e1-prereg.md` §2): the SGLang image digests and commit, the FP8 and NVFP4
checkpoint revisions, `--kv-cache-dtype bfloat16`, MoE and attention backends `auto`, `replay.jsonl` (sha256
`baf4efba580d0664387aa182a75ad6e61d44919b772421a26b8f3d6697f1d7ff`), the GSM8K and logprob set definitions, and the
client deadlines. The harness is `main-study/harness/` at the commit that freezes this file.

**Separation from E1.** E2 runs from its own checkout of this repository and its own copies of the job wrapper and
environment file; E1's checkout and wrappers are not touched until every E1 launch has its marker. Each E2 job
records the hashes of its own wrapper, environment file and `run_arm.sh`.

### 2a. H100 feasibility (decided before any H100-8-FP8 launch is submitted)

One 1-GPU job on the H100 cluster serves `Qwen/Qwen3-30B-A3B-FP8` (revision `d206ba73`) from the same amd64 image
digest, runs the GSM8K phase, and records GPU name, memory, driver version and the container's CUDA version.

- **Feasible** if the server reaches ready and GSM8K completes with zero request errors.
- **Infeasible** (no H100 configuration; reason recorded) only if the server itself fails: exit 15, 18 or 19, or
  GSM8K records with `error_kind` `server`.
- Any other exit or failure (including client-side errors and kill signals) is our instrument: the smoke is re-run
  once; a second such failure also means no H100 configuration, recorded as "not established".
- Before the first H100-8-FP8 launch, the H100 cluster's SIF size and sha256, the container runtime, and any
  workaround the smoke needed are recorded as fixed inputs. The replay file rebuilt there must reproduce the sha256
  above; if it does not, there is no H100 configuration.
- If the first H100-8-FP8 launch exits 15 or 18 (TP8 never served on that cluster), the whole configuration is
  `NOT_RUN`: no re-run and no further launches.

## 3. Load

Every configuration gets **three launches**, each one job with a fresh server, run in this interleaved order:
GB200 G2-FP8 L1, G2-NV4 L1, G2-NV4 L2, G2-FP8 L2, G2-FP8 L3, G2-NV4 L3; H200 H4 L1, H8 L1, H8 L2, H4 L2, H4 L3, H8 L3;
H100 L1, L2, L3. A launch runs, in order:

1. GSM8K, then the logprob set (as E1).
2. A concurrency sweep c ∈ {1, 4, 8, 16, 32, 64}, ascending. Point c replays sessions `[52, 52 + S_c)` of
   `replay.jsonl` with S_c = max(c, 4), closed loop, no think time, each session to its end; the sets are nested.
   `/flush_cache` before every point.

| c | sessions | turns (registered) | working set (Σ per-session largest context) |
|---|---|---|---|
| 1 | 4 | 142 | 120,815 |
| 4 | 4 | 142 | 120,815 |
| 8 | 8 | 273 | 249,366 |
| 16 | 16 | 543 | 496,549 |
| 32 | 32 | 974 | 1,008,319 |
| 64 | 64 | 2,021 | 2,013,679 |

Sessions `[52, 116)` are disjoint from E1's replay slices `[0, 52)` and logprob sessions `[512, 1024)`.

**Load shape.** This is closed-loop session replay: each point starts with c sessions in flight and concurrency
decays as shorter sessions finish, so a point mixes full-load and tail requests. It is not the fixed-concurrency
request load of public benchmarks. Each point therefore also reports the time-weighted mean number of requests in
flight and the highest `#running-req` in the server log.

**Wall time and the harness end time.** Each job has a 1 h Slurm wall; the harness end time is the job start plus
53 min (`MSL_END_EPOCH`), leaving room for requests already in flight and for server shutdown. A replay point gets
the time left before the end time minus 120 s as its deadline; a phase is not started unless at least 180 s remain
before the end time. When the end time stops a launch, it writes its markers with exit 30 (`wall_stop`). A launch
that hits the Slurm wall anyway is harvested from its `arm_status.txt` phase lines. Terms: a phase is **complete** only if it has an `arm_status.txt` line with rc 0 and, for replay,
`turns_done` equals the registered turns and `cut_by_deadline` is false; a phase that was started but not complete
is **cut**; later phases are **not started**. A missing high-concurrency point is reported as "did not complete
within the 1 h launch".

## 4. Metrics per (launch, concurrency point)

From the per-request records:

- **Throughput per GPU** = Σ output tokens of the point ÷ wall time of the point ÷ number of GPUs.
- **Interactivity** = median, over requests with ≥ 2 output tokens and end-to-end − TTFT > 0, of the per-request decode
  rate (output tokens − 1) ÷ (end-to-end − TTFT), in tokens per second per user. Median = value at index `int(n × 0.5)`
  of the sorted list.
- Secondary: TTFT p50 and p90, prefix hit rate (Σ cached ÷ Σ prompt), throughput of the whole configuration, KV pool
  size (`max_total_num_tokens`), time-weighted mean requests in flight, highest `#running-req`.

Per (configuration, point): the value is the launch-level median over the launches where that point is complete —
the middle value of three, the mean of two; with fewer than two the point is `NOT_EVALUABLE` — taken separately for
throughput per GPU and for interactivity. Its **spread** is (max − min) ÷ that median, for throughput per GPU.

## 5. Frontier and matched-interactivity comparison

- A configuration's **points** are its per-point (interactivity, throughput per GPU) medians over evaluable points;
  its **frontier** is the subset not dominated by another of its points (dominated = another point has both
  higher-or-equal interactivity and higher-or-equal throughput, one strictly). The same is built per launch.
- **Matched targets**: I* ∈ {25, 50, 75, 100} tokens/s/user.
- **Interpolation**: on the frontier sorted by interactivity, throughput at I* is the linear interpolation between the
  two adjacent points that bracket I*; a point exactly at I* is used as is; outside the frontier's range the
  configuration is `NOT_EVALUABLE` at I* (no extrapolation).
- **Registered ratios** at each I* (throughput per GPU, numerator / denominator): G2-FP8 / H4-FP8, G2-FP8 / H8-FP8,
  G2-NV4 / H4-FP8, G2-NV4 / H8-FP8, G2-NV4 / G2-FP8, H8-FP8 / H4-FP8, and, if it ran, H100-8-FP8 / H8-FP8 (labelled
  "different cluster, driver and runtime"). Also reported: an **H200 platform frontier** built from the union of H4 and
  H8 points, and the ratios G2-FP8 / H200-platform and G2-NV4 / H200-platform. Unregistered ratios are not reported.
- **Noise rule.** For one ratio at one I*:
  - r = interpolated numerator ÷ interpolated denominator on the median frontiers;
  - pair ratios = for every launch i of the numerator and launch j of the denominator whose own per-launch frontier
    covers I*, the ratio of their interpolated values (up to 9 pairs; a launch whose frontier does not cover I*, for
    example because its c = 64 point was cut, is left out of the pairs, not the medians);
  - s = the largest throughput spread among the median-frontier points used for I* in both configurations (the
    exact-match point, or the two bracketing points; up to four values).
  The ratio is `NOT_EVALUABLE` if r is undefined, fewer than 4 pairs exist, or s > 0.15; `SEPARATED` if
  |r − 1| ≥ max(0.03, 2 × s) and all pair ratios lie on the same side of 1; otherwise `NOT_SEPARATED`.
- **H200 platform.** Its median points are the union of H4's and H8's median points; its launch k is the union of H4
  launch k and H8 launch k; its spreads are the member points' spreads. The noise rule applies unchanged.
- These computations are implemented, with tests, in `main-study/analysis/e2_score.py` (`config_view`,
  `union_view`, `matched_ratio`); the independent scorer may recompute them by hand from this text, which governs.

## 6. Correctness (descriptive)

Per launch: GSM8K accuracy and invalid rate, and top-1 agreement of the logprob set with E1's REF launch (E1 §4).
Reference for a flag: for FP8 configurations the mean of E1's G-FP8 and H-FP8 launches, for G2-NV4 the mean of E1's
G-NV4 launches. A configuration whose mean GSM8K accuracy differs from its reference by more than 3 points is
"correctness flagged", and every ratio it is part of carries that label.

## 7. Gates and failure classes

| Gate | Condition |
|---|---|
| G-CLEAN, G-CTX, G-INPUT, G-COLD | as E1 |
| G-SAME | image digest and SIF size for the cluster, SGLang `0.0.0.dev1+ga6a8c5601`, the frozen harness commit with a clean tree, recorded wrapper hashes |
| G-BACKEND | GB200 configurations: the server log names `flashinfer_trtllm` as the selected MoE backend |
| G-LOAD | GSM8K 1,319 records and logprob 256 records with no server-side error; every complete point has every request completed with the requested output tokens and the replay prompt length |

Failures are scoped to phases: phases complete before a failure stay valid, whatever the exit code. A launch is
**VOID** and is re-run once, after a written analysis, if a correctness phase is not complete or fewer than two sweep
points are complete; otherwise it stands with its complete phases. A second VOID of the same launch leaves it missing.
Failure classes for the written analysis follow E1 §6a. A G-BACKEND failure makes that launch `NOT_EVALUABLE`; a
G-LOAD failure makes the affected phase `NOT_EVALUABLE`. There are no predictions, so no failure stops the other
configurations.

## 8. Claim ceiling

One SGLang nightly (`a6a8c560`); two public checkpoints of Qwen3-235B-A22B-Instruct-2507; one replay sample of SWE
agent trajectories with recorded assistant content; closed-loop session replay with decaying concurrency; bf16 KV;
three launches per configuration; the configurations in §2 only; shared, non-exclusive nodes; GB200, not B200; the
H100 configuration is on another cluster with another driver and runtime and is not a controlled comparison.

**Capacity.** KV pools from the smoke runs: G2-FP8 514,048 tokens and G2-NV4 1,019,072 tokens (H4, H8 and H100 pools
are recorded by E2 itself and added to this table in the score). Against the working sets of §3, G2-FP8 is below the
working set from c = 32 (0.51) and G2-NV4 from c = 64 (0.51). At points where a configuration's pool is below the
working set, its throughput reflects KV capacity at that GPU count — for GB200, the cluster's 2-GPU limit — and not
kernel or platform speed. Throughput per GPU leaves out host CPUs, networking and power and is not a cost statement.
Differences are described, not attributed.

## 9. Data seen before freezing

The targets I* were chosen from smoke runs only, all at TP2, single launches, c = 32 windows cut at 300 s: c = 1 TPOT
p50 8.0 ms (GB200 FP8), 6.6 ms (GB200 NVFP4), 9.9 ms (H200 FP8, TP2) and c = 32 TPOT p50 48.4, 28.9 and 155.5 ms. No
TP4, TP8 or H100 data and no E2 data existed when they were chosen. E1 launches that have finished before the
freeze are listed in the private run log with their markers only; no E1 metric had been read at the freeze. E2
uses no E1 number except as the correctness reference of §6, which is computed only after E1 is scored.

## 10. Method references (paraphrased)

- The framing — each platform at its own deployment, throughput per GPU plotted against interactivity and compared
  along the frontier — follows the InferenceX benchmark pages
  (<https://inferencex.semianalysis.com/glossary/throughput>, <https://inferencex.semianalysis.com/glossary/interactivity>).
  E2 makes its own choices within that framing: output tokens only, per GPU, median per-request interactivity.
- A public GB200-vs-Hopper serving comparison by per-GPU throughput, each at its own deployment:
  <https://lmsys.org/blog/2025-06-16-gb200-part-1>.

E2 does not reuse their numbers, and its configurations, workload and software differ from theirs.

## 11. Budget (whole GPU-hours) and priority

| Item | GPU-h |
|---|---|
| H100 feasibility job (one re-run allowed) | 1–2 |
| G2-FP8, G2-NV4: 6 launches × 2 GPUs × 1 h | 12 |
| H4-FP8: 3 × 4 × 1 h | 12 |
| H8-FP8: 3 × 8 × 1 h | 24 |
| H100-8-FP8: 3 × 8 × 1 h (conditional) | 24 |
| **E2 total** | **73–74** |

Project arithmetic: smoke runs 8 + E1 44 + E2 73–74 = 125–126 of a 160 cap, leaving 34–35; after the 10 reserve,
24–25 remain for E1 and E2 re-runs and anything after E2. Re-run priority when budget is short: E1 registered launches
first, then E2 GB200, H200, H100, in that order. Any later experiment is funded only from what remains after E2.

## Errata

None.
