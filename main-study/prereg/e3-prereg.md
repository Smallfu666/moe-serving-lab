# E3 preregistration — GB200 at node scale and across nodes

Status: **FROZEN** on 2026-10-16 (Asia/Taipei), after an independent review GO and before any E3 job.

## 1. Question

E2 compares GB200 only at 2 GPUs, the largest single-node GB200 job this cluster accepts. A full GB200 node (4 GPUs)
is only reachable here through a job of 16 or more GPUs (4 nodes). E3 asks: on the same throughput-per-GPU vs
interactivity axis as E2, where do (a) one GB200 node and (b) one server spanning four GB200 nodes land, relative to
E2's configurations? Descriptive only.

## 2. Configurations

| Config | Hardware | Server | Launches |
|---|---|---|---|
| N4-FP8 | 1 GB200 node (4 GPUs) | `--tp 4 --ep-size 4`, FP8 | 4 replicas, one per node, in one 16-GPU job |
| N4-NV4 | as N4-FP8 | as N4-FP8, NVFP4 | 4 replicas in one 16-GPU job |
| R16-FP8 | 4 GB200 nodes (16 GPUs), one server | `--tp 16 --attn-dp-size 16 --ep-size 16 --moe-a2a-backend flashinfer --moe-runner-backend flashinfer_trtllm`, FP8 | 3 jobs |

Everything else is fixed as in E2 §2 (image, checkpoints, bf16 KV, replay file and its sha256, GSM8K and logprob
definitions, client deadlines). E3 runs from its own checkout of this repository (`main-study/harness/` at the commit
that freezes this file) and its own environment file; every server has its own cache directories.

- **N4 setting.** SGLang's documentation at the pinned commit gives `--tp 4 --ep 4` as the GB200 setting for
  Qwen3-Coder-480B-A35B, a larger model of the same family; the Qwen3-235B page has no GB200 entry. N4 applies that
  setting to Qwen3-235B (`--ep-size` is the same option as `--ep`).
- **R16 setting.** The documentation at the pin has no multi-node setting for Qwen3 MoE models. R16 uses the pinned
  version's attention data parallelism and expert parallelism with its FlashInfer all-to-all. That all-to-all accepts
  five MoE runners; `flashinfer_trtllm` is used because it is the runner `auto` selects for N4 and in E2 (with the
  all-to-all on, `auto` does not resolve and the server refuses to start). The request dispatch policy is left at its
  default, which with attention data parallelism resolves to round-robin across the 16 attention ranks: consecutive
  turns of one session generally land on different ranks, so R16's prefix reuse and TTFT are expected to be worse than
  N4's for that reason alone. The resolved policy is recorded from the server log. R16 is "one plausible multi-node
  configuration", not a tuned or recommended one, and is labelled so.
- **Replicas.** The four N4 replicas of a job run at the same time on four different nodes, each with its own server,
  client and caches; they are treated as four launches. Their spread is **within-job, across concurrent nodes**, with no
  job-to-job variation, and every N4 spread and every ratio involving N4 carries that label.

## 3. Load and wall time

As E2 §3 (correctness phases first, then the concurrency sweep with nested session sets starting at session 52,
registered turns per point, harness end time 53 min, the same phase-completion terms), with these differences:

- N4 replicas run c ∈ {1, 4, 8, 16, 32, 64} (as E2).
- R16 runs c ∈ {1, 4, 8, 16, 32, 64, 128}. Point c = 128 replays sessions `[52, 180)`, disjoint from E1's replay
  `[0, 52)` and logprob `[512, 1024)` sets.
- **Expected cut.** From the smoke (4-node load about 6 min; c = 64 at about 2.9 turns/s), R16's c = 64 takes roughly
  12–14 min, and c = 128 (4,219 turns) needs more time than remains in one hour. **c = 128 is expected not to complete
  and is reported only if it completes**; c = 64 may also be cut.
- Order: N4-FP8 job, N4-NV4 job, then R16 L1, L2, L3, one after another.

### 3a. Registered turns and working sets

Computed from `replay.jsonl` (sha256 above) by summing, over the point's sessions, the number of turns and each
session's largest prompt plus capped answer:

| c | sessions | turns | working set (tokens) |
|---|---|---|---|
| 1, 4 | 4 | 142 | 120,815 |
| 8 | 8 | 273 | 249,366 |
| 16 | 16 | 543 | 496,549 |
| 32 | 32 | 974 | 1,008,319 |
| 64 | 64 | 2,021 | 2,013,679 |
| 128 | 128 | 4,219 | 4,046,797 |

## 4. Metrics, frontier and comparisons

Definitions are E2 §4–§5 unchanged (throughput per GPU, interactivity, launch-level median and spread, frontier,
targets I* ∈ {25, 50, 75, 100}, interpolation without extrapolation, noise rule). The GPU count is 4 for N4 and 16 for
R16. With four N4 replicas, the launch-level median is the mean of the two middle values. For R16, logged
`max_total_num_tokens` and `#running-req` are per attention rank, not for the whole server.

Registered ratios at each I*:

- N4-FP8 / G2-FP8 and N4-NV4 / G2-NV4 (GB200 node vs GB200 at 2 GPUs, same checkpoint);
- N4-FP8 / H8-FP8 and N4-NV4 / H8-FP8 (one GB200 node vs one H200 node);
- R16-FP8 / N4-FP8 (four nodes as one server vs one node).

E2 values come from E2's score (scored with E2's frozen harness commit). If E2's side of a ratio is `NOT_EVALUABLE`, so
is the ratio. Unregistered ratios are not reported.

Correctness (descriptive, E2 §6): GSM8K accuracy and logprob top-1 agreement with E1's REF per replica or launch,
flagged against the same references as E2.

## 5. Gates and failure classes

As E2 §7, per replica or launch, with:

- **G-BACKEND.** N4: the server log selects `flashinfer_trtllm`. R16: `server_cmd.txt` shows the registered arguments
  and rank 0's logged `server_args` show `moe_runner_backend='flashinfer_trtllm'` and `moe_a2a_backend='flashinfer'`.
- **N4.** A replica is VOID by E2 §7 (a correctness phase not complete, or fewer than two sweep points complete). A job
  is VOID if fewer than two replicas are non-VOID. If every replica of the N4-NV4 job fails before ready with a server
  error (exit 15, 18 or 19), N4-NV4 is NOT ESTABLISHED.
- **R16** is classified from all four ranks' `arm_rc.txt`; the earliest failure by time decides the class. A worker
  rank exiting 8–14 is an instrument VOID and never counts toward NOT ESTABLISHED. Rank 0 stops waiting for readiness
  with exit 31 as soon as a worker reports a failure. A worker exiting 19 makes the phase then in progress, and every
  later phase, cut. The scored directory is rank 0's, and only when every rank's `arm_rc.txt` exists. If R16 L1 is VOID
  because the server failed before ready (rank 0 exit 15 or 18, or a worker exit 19 before rank 0's ready line, with
  no worker exit 8–14) or because a correctness phase failed with a server-side or shape error, R16 is NOT ESTABLISHED
  and L2 and L3 are not run.
- **Re-runs** happen only if the budget check of §7 passes; otherwise a VOID job or launch is simply missing.

## 6. Data seen before freezing

The job request (4 nodes × 4 GPUs, 48 CPUs and 600 GB per node, 1 h) is the one the second feasibility job used.
From two feasibility jobs (FP8 only; short replays only; no correctness phases; NVFP4 not run): the four N4 replicas
were ready in 326–337 s with a KV pool of 2,235,840 tokens each; the R16 server was ready 352 s after its start (374 s into the job step, rank 0) with a
logged pool of 539,136 tokens per attention rank; replay turn counts in the short windows and zero bad requests. No
TTFT, TPOT or throughput summary of these jobs was read. N4's pool is above the c = 64 working set (2,235,840 ÷
2,013,679 = 1.11), so N4 is not capacity-limited at any point of its sweep, while G2-FP8 is from c = 32 (E2 §8).

## 7. Claim ceiling

As E2 §8, plus: N4 is one GB200 node running a setting SGLang documents for a larger model of the same family; its four
replicas ran at the same time on four nodes. R16 is one untuned multi-node configuration on 16 GPUs with round-robin
dispatch at concurrency ≤ 128; it is not a statement about large-scale expert-parallel serving in general. E3 jobs ran
on the cluster's large GB200 queue, whose nodes may differ from the development-queue nodes used by E2. The
inter-node transport used by R16 is whatever the pinned software selects and is recorded from the logs; no rack-scale
(NVL72) claim is made.

## 8. Budget and conditions

Billing is per job in whole hours with a one-hour minimum, so every E3 job costs 16 GPU-hours.

| Item | GPU-h |
|---|---|
| Billed before this freeze (all jobs of the project's runs, from the scheduler's accounting) | 62 |
| Remaining E1 (REF 4; H200 TP4 2 × 4) | 12 |
| Remaining E2 (GB200 5 × 2; H4 1 × 4; H8 3 × 8 incl. two re-runs of start failures; H100 1 + 3 × 8, conditional) | 63 |
| E3: N4-FP8, N4-NV4 (2 × 16); R16 L1–L3 (3 × 16) | 80 |
| **Projected total** | **217** of a 250 cap |

**Budget check (before every E3 submission):** billed so far (scheduler accounting, whole hours per job) + the
projected cost of every queued or running job + this job ≤ 240; otherwise the job is not submitted and is recorded
SKIPPED-BUDGET. Drop order if the check fails: R16 L3 first, then the rest of R16. The project balance is also
checked; a job is not submitted if the balance after it, net of queued and running jobs, would fall below about 1,000
units. The queue is checked; a job whose estimated start is more than about 6 h away is not queued (recorded
SKIPPED-UNSCHEDULABLE). R16 runs only after both N4 jobs have completed.

## Errata

None.
