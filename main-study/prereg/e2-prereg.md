# E2 preregistration — throughput vs interactivity, each platform at its own configuration

Status: REVIEW CANDIDATE (not yet binding; becomes FROZEN, with the date, in the commit that follows an independent
review GO and precedes every E2 job).

## 1. Question

How much output throughput per GPU does each platform deliver at a given per-user decode speed, when each runs the
same model and the same agentic replay at a configuration reasonable for that platform? The answer is a set of
throughput–interactivity curves, their Pareto frontiers, and ratios of throughput per GPU at matched interactivity.
This follows the way public inference benchmarks compare hardware (throughput per GPU against interactivity, compared
along the frontier); the method references are listed in §9.

E2 is descriptive. It ranks nothing beyond the measured configurations and states no cause for any difference.

## 2. Configurations

| Config | Hardware | Parallelism | Checkpoint | Why this configuration |
|---|---|---|---|---|
| G2-FP8 | 2 × GB200 | TP2 | FP8 | the largest GB200 job the cluster's development queue accepts (1–2 GPUs; 3–15 cannot be requested) — a **cluster constraint, not a design choice** |
| G2-NV4 | 2 × GB200 | TP2 | NVFP4 | as G2-FP8 |
| H4-FP8 | 4 × H200 | TP4 | FP8 | half a node |
| H8-FP8 | 8 × H200 | TP8 | FP8 | one full node |
| H100-8-FP8 | 8 × H100 (80 GB) | TP8 | FP8 | **conditional** on the feasibility check in §2a. A different cluster with a different GPU driver (550 vs 580) and node type; **not a controlled comparison with the H200 configurations** and always labelled so |

Everything else is fixed as in the E1 preregistration (`e1-prereg.md` §2): the same SGLang image digests and commit,
the same FP8 and NVFP4 checkpoint revisions, `--kv-cache-dtype bfloat16`, MoE and attention backends `auto`, the same
`replay.jsonl` (sha256 `baf4efba580d0664387aa182a75ad6e61d44919b772421a26b8f3d6697f1d7ff`), the same GSM8K set and
logprob set definitions, and the same client deadlines. The harness is `main-study/harness/` at the commit that freezes
this file.

### 2a. H100 feasibility (decided before the H100 configuration is submitted)

One 1-GPU job on the H100 cluster serves `Qwen/Qwen3-30B-A3B-FP8` from the same amd64 image digest and runs the GSM8K
phase. It records the GPU name, memory and driver version. The H100 configuration runs only if this job reaches the
server-ready state and the GSM8K phase completes with zero request errors; otherwise E2 has no H100 configuration and
the reason is recorded. The smoke's numbers are not E2 results. The replay file on the H100 cluster is rebuilt from the
same inputs and must reproduce the sha256 above.

## 3. Load

Each configuration is one launch (one job, a fresh server) that runs, in this order:

1. GSM8K (as E1), then the logprob set (as E1). These run first so that a wall-time cut can only remove the highest
   concurrency points.
2. A concurrency sweep c ∈ {1, 4, 8, 16, 32, 64}, ascending. Point c replays sessions `[52, 52 + S_c)` of
   `replay.jsonl` with S_c = max(c, 4) (S = 4, 4, 8, 16, 32, 64), closed loop, no think time, each session to its end;
   the session sets are nested, so a higher point contains every session of the lower points. Sessions `[52, 116)`
   are disjoint from E1's replay slices `[0, 52)` and logprob sessions `[512, 1024)`. `/flush_cache` before every point.

Wall time per job: 1 h. A configuration whose job is cut by the wall keeps every completed phase; the cut phase and
the phases after it are `NOT_RUN`.

## 4. Metrics per (configuration, concurrency point)

From the per-request records (same record format as E1):

- **Throughput per GPU** = Σ output tokens of the point ÷ wall time of the point ÷ number of GPUs.
- **Interactivity** = median over requests with ≥ 2 output tokens of the per-request decode rate
  (output tokens − 1) ÷ (end-to-end − TTFT), in tokens per second per user (the reciprocal of a request's TPOT).
  Median = the value at index `int(n × 0.5)` of the sorted list.
- Secondary: TTFT p50 and p90, prefix hit rate (Σ cached ÷ Σ prompt), throughput per GPU of the whole configuration
  (not divided), and the KV pool size (`max_total_num_tokens`).

## 5. Frontier and matched-interactivity comparison

- A configuration's **points** are its (interactivity, throughput per GPU) pairs over the completed concurrency points.
  Its **frontier** is the subset not dominated by another of its points (a point is dominated if another has both
  higher-or-equal interactivity and higher-or-equal throughput per GPU, one of them strictly).
- **Matched targets**: interactivity I* ∈ {25, 50, 75, 100} tokens/s/user. These were chosen from the single-launch
  smoke runs' c = 1 and c = 32 decode times so that most configurations span them; no other use of smoke data.
- **Interpolation**: sort the frontier by interactivity; throughput per GPU at I* is the linear interpolation between
  the two adjacent frontier points whose interactivities bracket I*. If I* lies outside the frontier's interactivity
  range, that configuration is `NOT_EVALUABLE` at I* (no extrapolation).
- **Comparisons** at each I*: G2-FP8 / H4-FP8, G2-FP8 / H8-FP8, G2-NV4 / G2-FP8, H8-FP8 / H4-FP8, and, if it ran,
  H100-8-FP8 / H8-FP8 (labelled "different cluster and driver"). Each ratio is reported with its two interpolated
  values and the bracketing points.
- **Separation**: E2 has one launch per configuration and therefore no measured run-to-run spread of its own. A ratio
  is reported as `SEPARATED` if |ratio − 1| ≥ S, otherwise `NOT_SEPARATED`, with
  S = max(0.05, 2 × the largest A/A of E1's G-FP8 and G-NV4 arms in output throughput at c = 32) — E1's two-launch
  spread on the same image and workload is the only spread measurement available. If E1 is not scored, S = 0.10.

## 6. Correctness (descriptive)

Per configuration: GSM8K accuracy and invalid rate, and top-1 agreement of the logprob set with E1's REF launch
(definitions of E1 §4). They are reported next to E1's two-launch values for the same checkpoint and are not used to
decide anything. A configuration whose GSM8K accuracy differs by more than 3 points from E1's mean for the same
checkpoint is flagged for inspection.

## 7. Gates (per launch; consequences as in E1 §6a, adapted)

| Gate | Condition |
|---|---|
| G-CLEAN, G-CTX, G-INPUT, G-COLD | as E1 |
| G-SAME | image digest, SIF size for the cluster, SGLang `0.0.0.dev1+ga6a8c5601`, the frozen harness commit with a clean tree; recorded wrapper hashes |
| G-BACKEND | GB200 configurations: the server log names `flashinfer_trtllm` as the selected MoE backend |
| G-LOAD | every completed point: every request completed with the requested output tokens and the replay prompt length; each session of the point replayed to its end |
| G-WALL | a wall cut voids only the phase it cut and the phases after it |

An instrument failure (E1 §6a classes) makes the launch VOID; it is re-run once after a written analysis. A second
VOID leaves that configuration `NOT_RUN`. A G-BACKEND or G-LOAD failure leaves that configuration `NOT_EVALUABLE`;
the other configurations are still reported. There are no predictions in E2, so no failure stops the other
configurations.

## 8. Claim ceiling

One SGLang nightly (`a6a8c560`); two public checkpoints of Qwen3-235B-A22B-Instruct-2507; one replay sample of SWE
agent trajectories with recorded assistant content; bf16 KV; one launch per configuration; the configurations in §2
only — GB200 is limited to 2 GPUs by the cluster's queues, so "GB200" here means 2 × GB200 TP2, not a GB200 node or
rack; shared, non-exclusive nodes; GB200, not B200; the H100 configuration is on another cluster with another driver
and is not a controlled comparison. Throughput per GPU does not account for host CPUs, networking or power, and is
not a cost statement. Differences are described, not attributed.

## 9. Method references (paraphrased, not copied)

- Throughput per GPU and interactivity definitions as used by the InferenceX dashboards:
  <https://inferencex.semianalysis.com/glossary/throughput>, <https://inferencex.semianalysis.com/glossary/interactivity>.
- Comparing GB200 and Hopper serving along throughput–latency trade-offs: <https://lmsys.org/blog/2025-06-16-gb200-part-1>.

These sources shape the framing (each platform at its own configuration, compared along the frontier). E2 does not
reuse their numbers, and its configurations, workload and software differ from theirs.

## 10. Budget (whole GPU-hours)

| Item | GPU-h |
|---|---|
| H100 feasibility job | 1 |
| G2-FP8, G2-NV4: 2 launches × 2 GPUs × 1 h | 4 |
| H4-FP8: 4 GPUs × 1 h | 4 |
| H8-FP8: 8 GPUs × 1 h | 8 |
| H100-8-FP8: 8 GPUs × 1 h (conditional) | 8 |
| **E2 total** | **25** |

Project: smoke runs 8 + E1 28 (after the pre-registered drop of E1's descriptive TP4 arm) + E2 25 = 61 of a 76 cap;
the remaining 15 cover re-runs (reserve ≥ 8). If a re-run would cross the cap, it is not run.

## Errata

None.
