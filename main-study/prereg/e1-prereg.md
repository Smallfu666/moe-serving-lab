# E1 preregistration

Status: REVIEW CANDIDATE (not yet binding; becomes FROZEN, with the date, in the commit that follows an independent
review GO and precedes every E1 job). Smoke runs before this date only set feasibility
inputs (§2a); none of their numbers is an E1 result and none moved a prediction band.

## 1. Question

With one SGLang commit, one model and one replay of recorded agent sessions, all on 2 GPUs: how do G-FP8 (GB200,
FP8), G-NV4 (GB200, NVFP4) and H-FP8 (H200, FP8) differ in TTFT, TPOT, throughput, prefix-cache hit rate and accuracy,
and which MoE backend SGLang selects? Descriptive comparison only: each arm differs from the others in more than one factor
(kernel path, HBM bandwidth, KV capacity, host CPU), so no difference is attributed to a single cause.

## 2. Fixed setup

| Item | Value |
|---|---|
| SGLang | image `lmsysorg/sglang:nightly-dev-cu13-20261015-a6a8c560` (SGLang `a6a8c56015b8`, reported as `0.0.0.dev1+ga6a8c5601`); amd64 `sha256:6b17a72a4c0e4fe8add1a23ea2ce203b8d3e709c41f6a3551f14502ea6be59a8`, arm64 `sha256:8c281367727d30a77c174a316610d3c022b66ededf44de26e7b0e659ada91512` |
| Images, file level | SIF built from those digests: amd64 13,779,562,496 bytes, sha256 `38c7ac2b…8dff3`; arm64 13,116,723,200 bytes, sha256 `b26d8baf…61aa22`; every job checks the size |
| Checkpoints | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` @ `e156cb4e`; `nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4` @ `1c4ec358`; every file size- and sha256-checked against the hub after download |
| Reference (P5) | `Qwen/Qwen3-235B-A22B-Instruct-2507` (BF16) @ `ac9c66cc`, 4 × H200, TP4, logprob set only |
| Server flags | `--tp <tp> --kv-cache-dtype bfloat16`, everything else default (MoE / attention backend `auto`) |
| Harness | `main-study/harness/` at the commit that freezes this file; every job records the harness commit it ran |
| Workload | `nebius/SWE-rebench-openhands-trajectories` @ `35455389` (CC-BY-4.0); 1,024 ids sampled with seed 20261015 (`main-study/sample/e1-sessions.txt`); tokenized with the FP8 checkpoint's tokenizer and chat template into `replay.jsonl`: 1,024 sessions kept, 0 skipped by the prefix check, 34,596 turns, sha256 `baf4efba580d0664387aa182a75ad6e61d44919b772421a26b8f3d6697f1d7ff` |
| Replay | turn k sends the recorded conversation up to the k-th assistant message as token ids to `/generate`; output length = recorded answer length capped at 1024, `ignore_eos`, temperature 0; context capped at 32,768 tokens; each session is replayed to its end (no time cut) |
| Concurrency points | closed loop, no think time: c = 1 on sessions `[0, 4)` (121 turns), c = 8 on `[4, 20)` (525 turns), c = 32 on `[20, 52)` (1,059 turns) of `replay.jsonl`; the same slices in every launch. Sessions have different lengths, so concurrency falls below c at the end of each point; this is part of the registered load |
| Cache | `/flush_cache` before every concurrency point, GSM8K and the logprob set |
| GSM8K | 1,319 test questions, 5-shot from the train split (`openai/grade-school-math`, test sha256 `3730d312…9d14`, train `17f347dc…3465`), greedy, max 512 new tokens, prompts and extraction as in `harness/gsm8k_client.py` |
| Logprob set | 256 items from sessions `[512, 1024)`: per session the last turn with prompt ≤ 8,192 tokens and a recorded answer ≥ 64 tokens; the first 64 recorded answer tokens are teacher-forced and each of the 64 positions gets the recorded token's logprob and the top-20 (the request starts the logprob window one token earlier because SGLang returns `None` for the window's first position; a response that does not have exactly 65 entries, a `None` first entry and the forced token ids is an error) |
| Client deadlines | per request 900 s (replay, GSM8K, logprob); cache flush 120 s; readiness poll 600 s per probe, 2,400 s in total |

### 2a. Design inputs from the smoke runs, and what was seen before freezing

Five smoke jobs ran the same image, the same checkpoints (plus a 30B model on one GPU) and this harness, on the same
replay file. They were feasibility runs: c = 8 and c = 32 were cut after 300 s, and each arm ran once.

| Input | Value | Used for |
|---|---|---|
| KV pool (`max_total_num_tokens`), H-FP8 TP2 | 160,140 tokens | §8 |
| KV pool, G-FP8 TP2 | 514,048 tokens | §8 |
| KV pool, G-NV4 TP2 | 1,019,072 tokens | §8 |
| Working set (Σ over the point's sessions of each session's largest prompt + capped answer), c = 8 | 512,632 tokens | §8 |
| Working set, c = 32 | 1,028,203 tokens | §8 |
| Server load time | 6–11 min | one 2 h job per launch holds all phases on every arm |
| Highest `#running-req` in the c = 32 smoke window | 32 on H-FP8 TP2, G-FP8 and G-NV4 | G-BATCH threshold stays at 28 |
| Logprob response shape (smokes requested the window from the first answer token) | in all 6 smoke logprob files (960 items): entries = forced tokens, first entry `None`, entries 2…64 carry the forced ids, top-20 everywhere | the E1 window starts one token earlier, so 65 entries are expected |

Pool ÷ working set at c = 32: H-FP8 0.16, G-FP8 0.50, G-NV4 0.99. At c = 8: H-FP8 0.31, G-FP8 1.00, G-NV4 1.99. So
**every registered arm** is below the c = 32 working set and some prefix-cache eviction is expected in all of them;
the H200 arm is the most constrained, and it is also below the c = 8 working set. One session alone (≤ 33,792 tokens)
fits in every pool, so c = 1 is not capacity-limited.

A planning rule that would have moved H-FP8 to TP4 when its pool is below the c = 32 working set is **dropped**: the
condition holds for every arm, so it cannot pick out an unusable arm. The registered comparison keeps 2 GPUs in every
arm, and a TP4 H200 arm is added as description only (§3).

**Not blind.** Before freezing, the per-point summaries of every smoke job were looked at: TTFT p50/p90, TPOT p50,
output throughput, prefix hit rate, GSM8K accuracy and the logprob request count. In particular the smoke c = 1 TPOT
ratios (single launches, 4 sessions) fell inside the P1 and P2 bands, the smoke c = 32 hit rates (partial windows)
were ordered G-NV4 > G-FP8 > H-FP8 by more than 5 points, and the smoke GSM8K accuracies of the 235B arms were within
0.3 points of each other. All prediction bands were written in the planning document before any smoke ran and were not
changed. P1–P4 are therefore **confirmations of an observed direction**, not blind predictions; P5 is the only one
whose data (logprob agreement with REF) has not been looked at.

## 3. Arms, launches and order

| Arm | Hardware | Launches | Role |
|---|---|---|---|
| G-FP8 | 2 × GB200, TP2, arm64 image | 2 | registered |
| G-NV4 | 2 × GB200, TP2, arm64 image | 2 | registered |
| H-FP8 | 2 × H200, TP2, amd64 image | 2 | registered |
| H-FP8-TP4 | 4 × H200, TP4, amd64 image | 2 | **descriptive only; never used to decide any prediction** |
| REF | 4 × H200, TP4, BF16 | 1 | logprob reference for P5 only |

Each launch is a fresh server in its own job (2 h wall; REF 1 h) and runs, in order: c = 1, c = 8, c = 32, GSM8K,
logprob set (REF: logprob set only). Launch order, enforced with job dependencies: GB200 G-FP8 L1, G-NV4 L1, G-NV4 L2,
G-FP8 L2; H200 H-FP8 L1, H-FP8 L2, REF, H-FP8-TP4 L1, H-FP8-TP4 L2. A VOID H-FP8-TP4 launch is re-run only after every
registered launch and REF are complete. REF has not been smoke-tested.

## 4. Metrics (computed from the per-request records)

Per launch and concurrency point:

- TTFT p50 and p90 over all requests of the point; TPOT p50 over requests with ≥ 2 output tokens, where a request's
  TPOT = (end-to-end − TTFT) / (output tokens − 1). Percentile = the value at index `int(n × q)` of the sorted list.
- Output throughput = Σ output tokens / wall time of the point; also divided by the GPU count.
- Prefix hit rate = Σ `cached_tokens` / Σ `prompt_tokens` over all requests of the point.

A **cell** is one metric at one concurrency point for one pair of arms. Per arm: the mean of its two launches; a
launch re-run after a VOID replaces the VOID launch. A/A of an arm = |L2 / L1 − 1| for a ratio metric, |L2 − L1| in percentage
points for the hit rate. Comparing arm X to arm Y: r = X / Y; resolution T = max(0.03, 2 × max(A/A_X, A/A_Y));
label `LOWER` / `HIGHER` if |r − 1| ≥ T, else `WITHIN`. Hit rate: difference d = X − Y in points; resolution
max(2, 2 × max(A/A_X, A/A_Y)) points. A cell where either arm's A/A exceeds 0.10 (or 10 points for the hit rate) is
`NOT_EVALUABLE`. These labels describe cells; verdicts are decided only by §5.

GSM8K: accuracy per launch; arm accuracy = mean of two launches. A/A flips of an arm = questions whose correctness
differs between its two launches; floor F = max of the two arms' A/A flips. Cross flips(X, Y) = mean of
flips(X1, Y1) and flips(X2, Y2).

Logprob: top-1 agreement of a launch with REF = fraction of the 256 × 64 = 16,384 positions where the launch's top-1
token equals REF's top-1 token; arm value = mean of its two launches. Self-agreement of an arm = the same fraction
between its own two launches. Also reported (not scored): mean |Δ logprob| of the
recorded token against REF, and each arm's own L1-vs-L2 top-1 agreement.

## 5. Predictions (registered arms only)

| ID | Prediction | Verdict rule |
|---|---|---|
| P1 | c = 1 TPOT p50: r = G-FP8 / H-FP8 ∈ [0.50, 0.85] | on the point estimate r (no resolution): `HIT` in the band, `MISS` outside it, `NOT_EVALUABLE` per §7 |
| P2 | c = 1 TPOT p50: r = G-NV4 / G-FP8 ∈ [0.55, 0.95] | as P1 |
| P3 | c = 32 prefix hit rate, d = G-FP8 − H-FP8 (points), resolution R from §4 | `HIGHER` if d ≥ max(5, R); `NO_DIFFERENCE` if \|d\| < 2 and R ≤ 4; `REVERSED` if d ≤ −max(2, R); otherwise `INCONCLUSIVE` |
| P4a | GSM8K, G-NV4 vs G-FP8 | `NO_CHANGE_BEYOND_AA` if cross flips ≤ max(2 × F, 10) and \|Δacc\| ≤ 1.5 points, else `MOVED`; `NOT_EVALUABLE` if F > 40 |
| P4b | GSM8K, H-FP8 vs G-FP8 | as P4a |
| P5 | top-1 agreement with REF, d = G-FP8 − G-NV4, R = max(0.002, 2 × (1 − lower self-agreement of the two arms)) | `HIT` if d ≥ R; `MISS` if d ≤ −R; `WITHIN` otherwise; `NOT_EVALUABLE` if either arm's self-agreement < 0.95 |

P1's band comes from the vendor HBM bandwidth ratio of about 0.6 [SPEC]. P3 compares two arms that differ in KV
capacity and in several other factors; `HIGHER` is descriptive, not proof that capacity causes the difference. The
H-FP8-TP4 arm is reported next to H-FP8 as description only; no verdict uses it.

## 6. Gates (what a failure means is in §6a)

| Gate | Condition |
|---|---|
| G-CLEAN | max `memory.used` over the job's GPUs < 2000 MiB at start (exit 11) |
| G-CTX | a CUDA context and a 1 GiB allocation succeed on every GPU before the server starts (exit 14) |
| G-SAME | every launch reports: the image digest and SIF size for its architecture; SGLang `0.0.0.dev1+ga6a8c5601`; one harness commit, the same in all launches, with a clean tree; the same sha256 of the job wrapper, environment file and `run_arm.sh`; within an arm, identical logged `server_args` apart from ports, seeds and paths |
| G-INPUT | `replay.jsonl` sha256 equals the value in §2 (exit 13) |
| G-BACKEND | GB200 launches: the server log contains `Use flashinfer_trtllm as MoE runner backend` (exit 16); the backend line of every launch is recorded |
| G-LOAD | every request of every phase completed: no error, output tokens = requested, prompt tokens = replay prompt length; each replay point has `turns_done` = 121 / 525 / 1,059 and `cut_by_deadline` false; GSM8K 1,319 records; logprob 256 records |
| G-COLD | the cache flush before each phase succeeded (exit 17) |
| G-BATCH | within each c = 32 point (server-log lines between the phase's start and end marks in `phase_lines.txt`), `#running-req` reaches ≥ 28 |
| G-WALL | no phase was cut by the job's wall time (every phase has an `arm_status.txt` line) |

### 6a. Classifying a failed launch

| Outcome | Class |
|---|---|
| exit 7–14, 17, 18; G-WALL; any return code not in `run_arm.sh`'s table (for example 1, 137, 143) | instrument VOID |
| exit 15 (server exited before ready) or 19 (server died during a phase) | instrument VOID after a written analysis of `server.log`; if the re-run of that launch exits 15 or 19 again, E1 stops (overriding the general re-run rule below) and the repeated crash is recorded as an outcome for that arm |
| exit 20–22 where every bad record has `error_kind` `client_timeout` or `client_exception`, or where no record carries an error or a token mismatch (client crash, missing summary) | instrument VOID |
| exit 22 where every bad record has `error_kind` `client_shape` | instrument VOID after a written analysis; if the re-run repeats it, P5 is `NOT_EVALUABLE` and E1 continues |
| exit 20–22 with any record of `error_kind` `server`, or a token-count mismatch | G-LOAD failure |
| G-BACKEND (exit 16), G-SAME, G-LOAD | gate failure |
| G-BATCH | P3 `NOT_EVALUABLE` for that arm; not a stop |

An instrument VOID launch is re-run once after a written analysis; a second VOID of the same launch makes it
permanently VOID. A gate failure **stops E1**: no new E1 job is submitted, jobs already running finish, and every
prediction whose arms each have two complete, gate-clean launches is still scored. No threshold changes after data.

## 7. NOT_EVALUABLE

A prediction is `NOT_EVALUABLE`, never `MISS`, when any arm it uses lacks two complete, gate-clean launches, or when
its cell is `NOT_EVALUABLE` under §4 or §5.

| Prediction | Arms it uses |
|---|---|
| P1 | G-FP8, H-FP8 |
| P2 | G-NV4, G-FP8 |
| P3 | G-FP8, H-FP8 |
| P4a | G-NV4, G-FP8 |
| P4b | H-FP8, G-FP8 |
| P5 | G-FP8, G-NV4, REF (one complete launch) |

H-FP8-TP4 is used by none; its failure or VOID changes no verdict.

## 8. Claim ceiling

One SGLang nightly (`a6a8c560`); two public checkpoints of Qwen3-235B-A22B-Instruct-2507; one replay sample of SWE
agent trajectories with recorded assistant content (so prefix hits are below those of real serving); bf16 KV; TP2 on 2
GPUs per registered arm on shared, non-exclusive nodes; GB200, not B200; descriptive, not causal; not a hardware
ranking; no cost statements. **At c = 32 every arm serves from a KV pool below its working set** (pool ÷ working
set: H-FP8 0.16, G-FP8 0.50, G-NV4 0.99), and at c = 8 the H200 arm does too (0.31). Results at those points are
capacity-limited serving on 2 GPUs, including prefix-cache eviction and queueing; they are not a measure of kernel or
architecture speed, and the H200 arm is the most constrained. P3 describes the hit-rate difference; it does not
attribute it. The backend named for each arm is the one SGLang selected and logged; which kernels executed is not
traced in E1.

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
The planned profiler and GPQA study (E2) is not funded inside this cap.

## Errata

None.
