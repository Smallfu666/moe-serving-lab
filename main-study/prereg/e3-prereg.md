# E3 preregistration — GB200 at node scale and across nodes

Status: REVIEW CANDIDATE (not yet binding; becomes FROZEN, with the date, in the commit that follows an independent
review GO and precedes every E3 job).

## 1. Question

E2 compares GB200 only at 2 GPUs, the largest single-node GB200 job this cluster accepts. Industry GB200 deployments
use at least a full node. The only way to get full GB200 nodes here is a job of 16 or more GPUs (4 nodes). E3 asks:
on the same throughput-per-GPU vs interactivity axis as E2, where do (a) one GB200 node running SGLang's GB200 recipe
and (b) one server spanning four GB200 nodes land, relative to E2's configurations? Descriptive only.

## 2. Configurations

| Config | Hardware | Server | Launches |
|---|---|---|---|
| N4-FP8 | 1 GB200 node (4 GPUs) | `--tp 4 --ep-size 4` (SGLang's documented GB200 setting for Qwen3 MoE models), FP8 | 4 replicas, one per node, in one 16-GPU job |
| N4-NV4 | as N4-FP8 | as N4-FP8, NVFP4 | 4 replicas in one 16-GPU job |
| R16-FP8 | 4 GB200 nodes (16 GPUs), one server | `--tp 16 --attn-dp-size 16 --ep-size 16 --moe-a2a-backend flashinfer --moe-runner-backend flashinfer_trtllm`, FP8 | 3 jobs |

Everything else is fixed as in E2 §2 (image, checkpoints, bf16 KV, replay file and its sha256, GSM8K and logprob
definitions, client deadlines). E3 runs from its own checkout of this repository (`main-study/harness/` at the commit
that freezes this file) and its own environment file; every server has its own cache directories.

- **Replicas.** The four N4 replicas of a job run at the same time on four different nodes, each with its own server,
  client and caches; they are treated as four launches. They are concurrent, not interleaved in time; this is stated
  wherever their spread is used.
- **R16 setting.** SGLang's documentation at the pinned commit has no multi-node setting for Qwen3 MoE models. The R16
  arguments are the pinned version's data-parallel-attention plus expert-parallel options with its all-to-all backend
  for this GPU family, plus the MoE runner that this backend requires (the server refuses to start without it). R16 is
  therefore "one plausible multi-node configuration", not a tuned or vendor-recommended one, and is labelled so.
- **Feasibility.** One smoke job (numbers not used) showed both servers reach ready on this cluster and complete short
  replays with zero request errors.

## 3. Load and wall time

As E2 §3 (correctness phases first, then the concurrency sweep with nested session sets from session 52, registered
turns per point, harness end time 53 min, the same phase-completion terms), with these differences:

- N4 replicas run c ∈ {1, 4, 8, 16, 32, 64} (as E2).
- R16 runs c ∈ {1, 4, 8, 16, 32, 64, 128}. Point c = 128 replays sessions `[52, 180)` (128 sessions) — still disjoint
  from E1's replay `[0, 52)` and logprob `[512, 1024)` sets — with its turn total registered from `replay.jsonl` before
  freezing (§3a).
- Order: N4-FP8 job, N4-NV4 job, then R16 L1, L2, L3, one after another.

### 3a. Registered turns

| c | sessions | turns |
|---|---|---|
| 1, 4 | 4 | 142 |
| 8 | 8 | 273 |
| 16 | 16 | 543 |
| 32 | 32 | 974 |
| 64 | 64 | 2,021 |
| 128 | 128 | 4,219 |

## 4. Metrics, frontier and comparisons

Definitions are E2 §4–§5 unchanged (throughput per GPU, interactivity, launch-level median and spread, frontier,
targets I* ∈ {25, 50, 75, 100}, interpolation without extrapolation, noise rule). The GPU count is 4 for N4 and 16
for R16.

Registered ratios at each I*:

- N4-FP8 / G2-FP8 and N4-NV4 / G2-NV4 (GB200 node vs GB200 at 2 GPUs, same checkpoint);
- N4-FP8 / H8-FP8 and N4-NV4 / H8-FP8 (one GB200 node vs one H200 node);
- R16-FP8 / N4-FP8 (four nodes as one server vs one node).

E2 values come from E2's score. If E2's side of a ratio is `NOT_EVALUABLE`, so is the ratio. Unregistered ratios are
not reported.

Correctness (descriptive, E2 §6): GSM8K accuracy and logprob top-1 agreement with E1's REF per replica or launch,
flagged against the same references as E2.

## 5. Gates and failure classes

As E2 §7, per replica or launch, with:

- **G-BACKEND** for N4: the server log selects `flashinfer_trtllm`. For R16 the runner is set explicitly; its
  `server_cmd.txt` must show the registered arguments.
- A job whose four N4 replicas all fail before any sweep point is VOID; a replica that fails alone is that replica's
  VOID and is not re-run separately (the other replicas stand). A VOID N4 job is re-run once after a written analysis.
- An R16 launch that exits 15, 18 or 19 before ready is VOID and is re-run once after a written analysis; if the
  re-run also fails before ready, R16 is NOT ESTABLISHED (no further launches).

## 6. Claim ceiling

As E2 §8, plus: N4 is one GB200 node running SGLang's documented GB200 setting; its four replicas ran at the same time
on four nodes. R16 is one untuned multi-node configuration on 16 GPUs at concurrency ≤ 128; it is not a statement
about large-scale expert-parallel serving in general. The N4 and R16 jobs ran on the cluster's large GB200 queue,
whose nodes may differ from the development queue's nodes used by E2. No rack-scale (NVL72) claim.

## 7. Budget and conditions

| Item | GPU-h |
|---|---|
| Feasibility smokes (done; one VOID + one re-run) | 32 |
| N4-FP8, N4-NV4: 2 jobs × 16 GPUs × 1 h | 32 |
| R16-FP8: 3 jobs × 16 GPUs × 1 h | 48 |
| **E3 total** | **112** |

Project: smokes 8 + E1 44 + E2 73–74 + E3 112 = 237–238 of 250, reserve ≥ 10 kept only if no further re-run is
needed; any E3 re-run that would cross 240 is not run. Before each E3 submission the project balance is checked and
the job is not submitted if the projected balance would fall below about 1,000 units; the queue is checked and a job
whose estimated start is more than about 6 h away is not queued (recorded SKIPPED-UNSCHEDULABLE). R16 runs only after
the N4 jobs have completed.

## Errata

None.
