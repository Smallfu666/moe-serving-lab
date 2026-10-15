# E1 preregistration

Status: REVIEW CANDIDATE (not yet binding; becomes FROZEN, with the date, in the commit that follows an independent
review GO and precedes every E1 job). Smoke runs before this date only set feasibility
inputs (§2a); none of their numbers is an E1 result and none moved a prediction band.

## 1. Question

With one SGLang commit, one model and one replay of recorded agent sessions, all on 2 GPUs: how do G-FP8 (GB200,
FP8), G-NV4 (GB200, NVFP4) and H-FP8 (H200, FP8) differ in TTFT, TPOT, throughput, prefix-cache hit rate and accuracy,
and which MoE backend runs? Descriptive comparison only: each arm differs from the others in more than one factor
(kernel path, HBM bandwidth, KV capacity, host CPU), so no difference is attributed to a single cause.

## 2. Fixed setup

| Item | Value |
|---|---|
| SGLang | image `lmsysorg/sglang:nightly-dev-cu13-20261015-a6a8c560` (SGLang `a6a8c56015b8`, reported as `0.0.0.dev1+ga6a8c5601`); amd64 `sha256:6b17a72a4c0e4fe8add1a23ea2ce203b8d3e709c41f6a3551f14502ea6be59a8`, arm64 `sha256:8c281367727d30a77c174a316610d3c022b66ededf44de26e7b0e659ada91512` |
| Checkpoints | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` @ `e156cb4e`; `nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4` @ `1c4ec358`; every file size- and sha256-checked against the hub after download |
| Reference (P5) | `Qwen/Qwen3-235B-A22B-Instruct-2507` (BF16) @ `ac9c66cc`, 4 × H200, TP4, logprob set only |
| Server flags | `--tp <tp> --kv-cache-dtype bfloat16`, everything else default (MoE / attention backend `auto`) |
| Harness | `main-study/harness/` at the commit that freezes this file; every job records the harness commit it ran |
| Workload | `nebius/SWE-rebench-openhands-trajectories` @ `35455389` (CC-BY-4.0); 1,024 ids sampled with seed 20261015 (`main-study/sample/e1-sessions.txt`); tokenized with the FP8 checkpoint's tokenizer and chat template into `replay.jsonl`: 1,024 sessions kept, 0 skipped by the prefix check, 34,596 turns, sha256 `baf4efba580d0664387aa182a75ad6e61d44919b772421a26b8f3d6697f1d7ff` |
| Replay | turn k sends the recorded conversation up to the k-th assistant message as token ids to `/generate`; output length = recorded answer length capped at 1024, `ignore_eos`, temperature 0; context capped at 32,768 tokens; each session is replayed to its end (no time cut) |
| Concurrency points | closed loop, no think time: c = 1 on sessions `[0, 4)` (121 turns), c = 8 on `[4, 20)` (525 turns), c = 32 on `[20, 52)` (1,059 turns) of `replay.jsonl`; the same slices in every launch. Sessions have different lengths, so concurrency falls below c at the end of each point; this is part of the registered load |
| Cache | `/flush_cache` before every concurrency point, GSM8K and the logprob set |
| GSM8K | 1,319 test questions, 5-shot from the train split (`openai/grade-school-math`, test sha256 `3730d312…9d14`, train `17f347dc…3465`), greedy, max 512 new tokens, prompts and extraction as in `harness/gsm8k_client.py` |
| Logprob set | 256 items from sessions `[512, 1024)`: per session the last turn with prompt ≤ 8,192 tokens and a recorded answer ≥ 64 tokens; 64 recorded answer tokens teacher-forced; top-20 logprobs per position |

### 2a. Design inputs from the smoke runs (feasibility only)

| Input | Value | Used for |
|---|---|---|
| KV pool, H-FP8 TP2 (`max_total_num_tokens`) | 160,140 tokens | §8 claim ceiling |
| KV pool, G-FP8 TP2 | 514,048 tokens | §8 |
| KV pool, G-NV4 TP2 | 1,019,072 tokens | §8 |
| c = 32 working set: Σ over the 32 sessions of each session's largest context (prompt + capped answer) | 1,028,203 tokens | §8 |
| Server load time | 6–11 min | one 2 h job per launch holds all phases on every arm |

The H-FP8 TP2 pool is below the c = 32 working set (and below G-FP8's and G-NV4's pools), so at c = 8 and c = 32 the
H200 arm's numbers include prefix-cache eviction and request queueing. A planning rule that would have moved H-FP8
to TP4 in this case is **dropped**: the registered comparison keeps the same GPU count (2) in every arm, and a TP4
H200 arm is added as description only (§3). One session alone fits in every pool, so c = 1 is not capacity-limited.

## 3. Arms, launches and order

| Arm | Hardware | Launches | Role |
|---|---|---|---|
| G-FP8 | 2 × GB200, TP2, arm64 image | 2 | registered |
| G-NV4 | 2 × GB200, TP2, arm64 image | 2 | registered |
| H-FP8 | 2 × H200, TP2, amd64 image | 2 | registered |
| H-FP8-TP4 | 4 × H200, TP4, amd64 image | 2 | **descriptive only; never used to decide any prediction** |
| REF | 4 × H200, TP4, BF16 | 1 | logprob reference for P5 only |

Each launch is a fresh server in its own job (2 h wall; REF 1 h) and runs, in order: c = 1, c = 8, c = 32, GSM8K,
logprob set (REF: logprob set only). GB200 launch order: G-FP8 L1, G-NV4 L1, G-NV4 L2, G-FP8 L2. H200 launch order:
H-FP8 L1, H-FP8 L2, H-FP8-TP4 L1, H-FP8-TP4 L2, REF.

## 4. Metrics (computed from the per-request records)

Per launch and concurrency point:

- TTFT p50 and p90 over all requests of the point; TPOT p50 over requests with ≥ 2 output tokens, where a request's
  TPOT = (end-to-end − TTFT) / (output tokens − 1). Percentile = the value at index `int(n × q)` of the sorted list.
- Output throughput = Σ output tokens / wall time of the point; also divided by the GPU count.
- Prefix hit rate = Σ `cached_tokens` / Σ `prompt_tokens` over all requests of the point.

Per arm: the mean of its two launches. A/A of an arm = |L2 / L1 − 1| for a ratio metric, |L2 − L1| in percentage
points for the hit rate. Comparing arm X to arm Y: r = X / Y; resolution T = max(0.03, 2 × max(A/A_X, A/A_Y));
label `LOWER` / `HIGHER` if |r − 1| ≥ T, else `WITHIN`. Hit rate: difference d = X − Y in points; resolution
max(2, 2 × max(A/A_X, A/A_Y)) points. A cell where either arm's A/A exceeds 0.10 (or 10 points for the hit rate) is
`NOT_EVALUABLE`.

GSM8K: accuracy per launch; arm accuracy = mean of two launches. A/A flips of an arm = questions whose correctness
differs between its two launches; floor F = max of the two arms' A/A flips. Cross flips(X, Y) = mean of
flips(X1, Y1) and flips(X2, Y2).

Logprob: top-1 agreement of a launch with REF = fraction of the 256 × 64 positions where the launch's top-1 token
equals REF's top-1 token; arm value = mean of its two launches. Also reported (not scored): mean |Δ logprob| of the
recorded token against REF, and each arm's own L1-vs-L2 top-1 agreement.

## 5. Predictions (registered arms only)

| ID | Prediction | Verdict rule |
|---|---|---|
| P1 | c = 1 TPOT p50: G-FP8 / H-FP8 ∈ [0.50, 0.85] | `HIT` if r is in the band and the cell is evaluable; `MISS` otherwise |
| P2 | c = 1 TPOT p50: G-NV4 / G-FP8 ∈ [0.55, 0.95] | as P1 |
| P3 | c = 32 prefix hit rate, G-FP8 − H-FP8 | `HIT` if ≥ 5 points and ≥ the hit-rate resolution; `CAPACITY_NOT_LIMITING` if < 2 points; otherwise `INCONCLUSIVE` |
| P4 | GSM8K, G-NV4 vs G-FP8 and H-FP8 vs G-FP8 | `NO_CHANGE_BEYOND_AA` if cross flips ≤ max(2 × F, 10) and \|Δacc\| ≤ 1.5 points, else `MOVED` |
| P5 | top-1 agreement with REF: G-FP8 ≥ G-NV4 | `HIT` / `MISS` |

P1's band comes from the vendor HBM bandwidth ratio of about 0.6 [SPEC]. P3 compares two arms that differ in KV
capacity and in several other factors; a `HIT` is descriptive, not proof that capacity causes the difference. The
H-FP8-TP4 arm is reported next to H-FP8 as description only; no verdict uses it.

## 6. Gates (a failed gate makes the affected launch VOID; it is not scored)

| Gate | Condition |
|---|---|
| G-CLEAN | max `memory.used` over the job's GPUs < 2000 MiB at start (exit 11) |
| G-CTX | a CUDA context and a 1 GiB allocation succeed on every GPU before the server starts (exit 14) |
| G-SAME | every launch reports the image digest for its architecture, SGLang `0.0.0.dev1+ga6a8c5601`, and one harness commit, the same in all launches |
| G-INPUT | `replay.jsonl` sha256 equals the value in §2 (exit 13) |
| G-BACKEND | GB200 launches: the server log contains `Use flashinfer_trtllm as MoE runner backend` (exit 16); the backend line of every launch is recorded |
| G-LOAD | every request of every phase completed: no error, output tokens = requested, prompt tokens = replay prompt length |
| G-COLD | the cache flush before each phase succeeded (exit 17) |
| G-BATCH | within each c = 32 point (server-log lines between the phase's start and end marks in `phase_lines.txt`), `#running-req` reaches ≥ 28 |
| G-WALL | no phase was cut by the job's wall time (every phase has an `arm_status.txt` line) |

Instrument VOID (G-CLEAN, G-CTX, exit codes 8–15, G-WALL): that launch is re-run once after a written analysis; a
second VOID of the same launch stops E1. Any other gate failure stops E1. No threshold changes after data.

## 7. NOT_EVALUABLE

- A prediction whose cells are `NOT_EVALUABLE`, or whose arm has a VOID launch that was not re-run successfully, is
  `NOT_EVALUABLE`, never `MISS`.
- If G-BACKEND fails on a GB200 launch, P1, P2, P3 and P5 for that arm are `NOT_EVALUABLE`.
- If REF does not complete, P5 is `NOT_EVALUABLE`; the arms' logprob records are still reported descriptively.
- The H-FP8-TP4 arm failing or being VOID changes no verdict.

## 8. Claim ceiling

One SGLang nightly (`a6a8c560`); two public checkpoints of Qwen3-235B-A22B-Instruct-2507; one replay sample of SWE
agent trajectories with recorded assistant content (so prefix hits are below those of real serving); bf16 KV; TP2 on 2
GPUs per registered arm on shared, non-exclusive nodes; GB200, not B200; descriptive, not causal; not a hardware
ranking; no cost statements. **At c = 8 and c = 32 the H200 arm is capacity-limited serving on 2 × H200**: its KV pool
(160,140 tokens) is about a sixth of the c = 32 working set (1,028,203 tokens), so its TTFT, TPOT, throughput and hit
rate there include eviction and queueing and are not a measure of H200 kernel or architecture speed. P3 describes that
difference; it does not attribute it.

## 9. Completion

A launch is complete only when its `arm_rc.txt` and `arm_status.txt` say so and the job marker exists. Scheduler state
and elapsed time are not used.

## 10. Budget (whole GPU-hours)

| Item | GPU-h |
|---|---|
| Smoke runs (done) | 8 |
| G-FP8, G-NV4: 4 launches × 2 GPUs × 2 h | 16 |
| H-FP8: 2 launches × 2 GPUs × 2 h | 8 |
| H-FP8-TP4: 2 launches × 4 GPUs × 2 h | 16 |
| REF: 4 GPUs × 1 h | 4 |
| **Planned total** | **52** |
| Reserve (one instrument-VOID re-run) | ≥ 8 |
| Project cap | 60 |

If the reserve is used up, the next VOID stops E1. H-FP8-TP4 is the first item dropped if the cap would be crossed.

## Errata

None.
