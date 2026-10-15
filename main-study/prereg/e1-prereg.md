# E1 preregistration — DRAFT (becomes binding when the status line says FROZEN)

Status: DRAFT. Values marked `‹S›` are set from the smoke runs before freezing; no prediction band depends on them.

## 1. Question

With one SGLang commit, one model and one replay of recorded agent sessions, all on 2 GPUs: how do G-FP8 (GB200,
FP8), G-NV4 (GB200, NVFP4) and H-FP8 (H200, FP8) differ in TTFT, TPOT, throughput, prefix-cache hit rate and accuracy,
and which MoE backend runs? Descriptive comparison only: each arm differs from the others in more than one factor
(kernel path, HBM bandwidth, KV capacity, host CPU), so no difference is attributed to a single cause.

## 2. Fixed setup

| Item | Value |
|---|---|
| SGLang | image `lmsysorg/sglang:nightly-dev-cu13-20261015-a6a8c560` (SGLang `a6a8c56015b8`); amd64 `sha256:6b17a72a…59a8`, arm64 `sha256:8c281367…1512` |
| Checkpoints | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` @ `e156cb4e`; `nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4` @ `1c4ec358` |
| Reference (P5) | `Qwen/Qwen3-235B-A22B-Instruct-2507` (BF16) @ `ac9c66cc`, H200 × 4, TP4, logprob set only |
| Server flags | `--tp 2 --kv-cache-dtype bfloat16`, everything else default (MoE / attention backend `auto`) |
| Harness | this repository at commit `‹hash›` (`main-study/harness/`) |
| Workload | `nebius/SWE-rebench-openhands-trajectories` @ `35455389` (CC-BY-4.0); 1,024 ids sampled with seed 20261015 (`main-study/sample/e1-sessions.txt`); tokenized with the FP8 checkpoint's tokenizer and chat template into `replay.jsonl` (sha256 `‹sha›`, `‹k›` sessions kept, `‹s›` skipped by the prefix check) |
| Replay | turn k sends the recorded conversation up to the k-th assistant message as token ids to `/generate`; output length = recorded answer length capped at 1024, `ignore_eos`, temperature 0; context capped at 32,768 tokens |
| Concurrency points | closed loop, no think time; c = 1 on sessions `[0, N1)`, c = 8 on `[N1, N1+N8)`, c = 32 on `[N1+N8, N1+N8+N32)` of `replay.jsonl`, with N1 = `‹S›`, N8 = `‹S›`, N32 = `‹S›`; the same slices in every launch |
| Cache | `/flush_cache` before every concurrency point, GSM8K and the logprob set |
| GSM8K | 1,319 test questions, 5-shot from the train split, greedy, max 512 new tokens, prompts and extraction as in `harness/gsm8k_client.py` |
| Logprob set | 256 items from sessions `[512, 1024)`: per session the last turn with prompt ≤ 8,192 tokens and a recorded answer ≥ 64 tokens; 64 recorded answer tokens teacher-forced; top-20 logprobs per position |

## 3. Arms, launches and order

| Arm | Hardware | Launches |
|---|---|---|
| G-FP8 | 2 × GB200, TP2, arm64 image | 2 |
| G-NV4 | 2 × GB200, TP2, arm64 image | 2 |
| H-FP8 | 2 × H200, TP2, amd64 image | 2 |
| REF | 4 × H200, TP4, BF16, logprob set only | 1 |

Each launch is a fresh server in its own job and runs, in order: c = 1, c = 8, c = 32, GSM8K, logprob set
(`‹S›`: or the GSM8K + logprob part as a second job of the same arm if the smoke shows a 2 h job cannot hold all
phases). GB200 launch order: G-FP8 L1, G-NV4 L1, G-NV4 L2, G-FP8 L2. H200 launches: H-FP8 L1, L2; REF once.

## 4. Metrics (computed from the per-request records)

Per launch and concurrency point:

- TTFT p50 and p90 over all requests of the point; TPOT p50 over requests with ≥ 2 output tokens, where a request's
  TPOT = (end-to-end − TTFT) / (output tokens − 1).
- Output throughput = Σ output tokens / wall time of the point; also divided by the GPU count.
- Prefix hit rate = Σ `cached_tokens` / Σ `prompt_tokens` over all requests of the point.

Per arm: the mean of its two launches. A/A of an arm = |L2 / L1 − 1| for a ratio metric, |L2 − L1| in percentage
points for the hit rate. Comparing arm X to arm Y: r = X / Y; resolution T = max(0.03, 2 × max(A/A_X, A/A_Y));
label `LOWER` / `HIGHER` if |r − 1| ≥ T, else `WITHIN`. A cell where either arm's A/A exceeds 0.10 (or 10 pt for the
hit rate) is `NOT_EVALUABLE`.

GSM8K: accuracy per launch; arm accuracy = mean of two launches. A/A flips of an arm = questions whose correctness
differs between its two launches; floor F = max of the two arms' A/A flips. Cross flips(X, Y) = mean of
flips(X1, Y1) and flips(X2, Y2).

Logprob: top-1 agreement of arm X with REF = fraction of the 256 × 64 positions where X's top-1 token equals REF's
top-1 token; per launch, arm value = mean of two launches. Also reported (not scored): mean |Δ logprob| of the
recorded token against REF, and the arm's own L1-vs-L2 top-1 agreement.

## 5. Predictions

| ID | Prediction | Verdict rule |
|---|---|---|
| P1 | c = 1 TPOT p50: G-FP8 / H-FP8 ∈ [0.50, 0.85] | `HIT` if r is in the band and the cell is evaluable; `MISS` otherwise |
| P2 | c = 1 TPOT p50: G-NV4 / G-FP8 ∈ [0.55, 0.95] | as P1 |
| P3 | c = 32 prefix hit rate, G-FP8 − H-FP8 | `HIT` if ≥ 5 pt and ≥ the hit-rate resolution; `CAPACITY_NOT_LIMITING` if < 2 pt; otherwise `INCONCLUSIVE` |
| P4 | GSM8K, G-NV4 vs G-FP8 and H-FP8 vs G-FP8 | `NO_CHANGE_BEYOND_AA` if cross flips ≤ max(2 × F, 10) and \|Δacc\| ≤ 1.5 pt, else `MOVED` |
| P5 | top-1 agreement with REF: G-FP8 ≥ G-NV4 | `HIT` / `MISS` |

P1's band comes from the vendor HBM bandwidth ratio of about 0.6 [SPEC]. P3 compares two arms that differ in KV
capacity and in several other factors; a `HIT` is descriptive, not proof that capacity causes the difference.

## 6. Gates (a failed gate makes the affected launch VOID; it is not scored)

| Gate | Condition |
|---|---|
| G-CLEAN | max `memory.used` over the job's GPUs < 2000 MiB at start (exit 11 otherwise) |
| G-CTX | a CUDA context and a 1 GiB allocation succeed on every GPU before the server starts (exit 14) |
| G-SAME | every launch reports the image digests and SGLang commit above, and the harness commit above |
| G-INPUT | `replay.jsonl` sha256 equals the value above (exit 13) |
| G-BACKEND | GB200 launches: the server log contains `Use flashinfer_trtllm as MoE runner backend` (exit 16); the backend line of every launch is recorded |
| G-LOAD | every request of every phase completed: no error, output tokens = requested, prompt tokens = replay prompt length |
| G-COLD | the cache flush before each phase succeeded (exit 17) |
| G-BATCH | in each c = 32 point, `#running-req` in the server log reaches ≥ 28 |
| G-WALL | no phase was cut by the job's wall time |

Instrument VOID (G-CLEAN, G-CTX, infrastructure exits, G-WALL): that launch is re-run once after a written analysis;
a second VOID of the same launch stops E1. Any other gate failure stops E1. No threshold changes after data.

## 7. Falsifiers and NOT_EVALUABLE

- A prediction whose cells are `NOT_EVALUABLE`, or whose arm has a VOID launch that was not re-run successfully, is
  `NOT_EVALUABLE`, never `MISS`.
- If G-BACKEND fails on a GB200 launch, P1, P2, P3 and P5 for that arm are `NOT_EVALUABLE` (the registered kernel path
  did not run).
- If REF does not complete, P5 is `NOT_EVALUABLE`; the arms' logprob records are still reported descriptively.

## 8. Claim ceiling

One SGLang nightly (`a6a8c560`); two public checkpoints of Qwen3-235B-A22B-Instruct-2507; one replay sample of SWE
agent trajectories with recorded assistant content (so prefix hits are below those of real serving); bf16 KV;
TP2 on 2 GPUs per arm on shared, non-exclusive nodes; GB200, not B200; descriptive, not causal; not a hardware
ranking; no cost statements.

## 9. Completion

A launch is complete only when its `arm_rc.txt` and `arm_status.txt` say so and the job marker exists. Scheduler state
and elapsed time are not used.

## 10. Budget

E1: 4 GB200 launches × 2 GPUs × 2 h + 2 H200 launches × 2 GPUs × 2 h + REF 4 GPUs × 1 h = 28 GPU-h, plus one VOID
re-run allowance; the project cap is 60 GPU-h in whole hours including the 8 GPU-h of smoke runs.

## Errata

None.
