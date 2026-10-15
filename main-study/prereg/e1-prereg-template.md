# E1 preregistration — TEMPLATE, NOT FROZEN

> Draft values below come from planning. They become binding only when this file is frozen (committed before the first
> measured job) and the frozen copy's commit hash is recorded. Smoke runs may set feasibility parameters (sessions per
> concurrency point, phase split, the TP rule); **they may not move any prediction band**.

Frozen at commit: `<hash>` on `<date>`. Harness commit: `<hash>`. Images: amd64 `<digest>`, arm64 `<digest>`.

## 1. Question

With one SGLang commit, one model and one agentic replay sample, all on 2 GPUs: how do G-FP8, G-NV4 and H-FP8 differ
in TTFT, TPOT, prefix-cache hit rate and accuracy; how much time goes to each kernel class; and which MoE backend runs?
Descriptive attribution only.

## 2. Fixed setup

| Item | Value |
|---|---|
| SGLang commit | `<hash>` (one commit with both amd64 and arm64 images) |
| Model | Qwen3-235B-A22B-Instruct-2507: FP8 `<HF revision>`, NVFP4 `<HF revision>` |
| Parallelism | TP2 in every arm (H-FP8 may switch to TP4, §6) |
| KV cache | bf16 |
| Backends | `auto`; the backend that ran is copied from the server log |
| Workload | `<N>` sessions per concurrency point, sampled with seed `<seed>`; id list committed at `<path>` |
| Concurrency | 1, 8, 32 concurrent sessions, closed loop, no think time; cache flushed before each point |
| Output length | recorded assistant length, capped at 1024, `ignore_eos`; context capped at 32k tokens |

## 3. Arms and order

Each arm: 2 fresh launches (A/A). One launch runs, in order: concurrency 1, 8, 32; GSM8K; the logprob set.
Launch order: `<order>`.

## 4. Metrics

- TTFT p50/p90, TPOT p50, output tok/s (total and per GPU) per concurrency point; TTFT by turn index.
- Prefix hit: per request `cached_tokens / prompt_tokens`, aggregated.
- Accuracy: GSM8K 1,319 questions, greedy, paired flips against the same arm's A/A flips.
- Logprob: 256 fixed prompts, 64 teacher-forced recorded tokens each, top-20 logprobs; top-1 agreement and truncated
  KL against the reference arm.

## 5. Predictions (draft bands)

| ID | Prediction | Band |
|---|---|---|
| P1 | c = 1 TPOT(G-FP8) ÷ TPOT(H-FP8) | [0.50, 0.85] |
| P2 | c = 1 TPOT(G-NV4) ÷ TPOT(G-FP8) | [0.55, 0.95] |
| P3 | c = 32 prefix hit, H-FP8 below G-FP8 | HIT ≥ 5 pt; < 2 pt = capacity not limiting; 2–5 pt INCONCLUSIVE |
| P4 | GSM8K paired flips G-NV4 vs G-FP8 (and H-FP8 vs G-FP8) | ≤ max(2 × A/A flips, 10) and \|Δacc\| ≤ 1.5 pt → `NO_CHANGE_BEYOND_AA`, else `MOVED` |
| P5 | top-1 agreement with the reference | G-FP8 ≥ G-NV4 |

P1's band starts from the vendor HBM bandwidth ratio (about 0.6) [SPEC].

## 6. TP rule for H-FP8 (fixed before data)

If the smoke run's reported `max_total_num_tokens` is below the working set of the highest concurrency point, H-FP8
runs at TP4 and per-GPU numbers are also reported. In that case P1 and P3 are `NOT_EVALUABLE`.

## 7. Gates (fail → VOID, not scored)

| Gate | Condition |
|---|---|
| G-CLEAN | max `memory.used` over the job's GPUs < 2000 MiB at start |
| G-SAME | one SGLang commit across arms; both image digests recorded |
| G-BACKEND | the SM100 arms' server log names the expected MoE backend (`flashinfer_trtllm`) |
| G-LOAD | every session completes, no errors or timeouts |
| G-COLD | cache flushed before every concurrency point |
| G-AA | resolution `T = max(0.03, 2 × \|A/A ratio − 1\|)` for TTFT/TPOT/throughput; `max(2, 2 × A/A)` pt for hit rate |
| G-WALL | any phase hits the wall-time limit → instrument VOID; one re-run only |

## 8. Falsifiers

`<one line per prediction: the observation that falsifies it>`

## 9. Claim ceiling

One SGLang commit; two public checkpoints; one replay sample (prefix hits below real serving); descriptive attribution;
2 GPUs per arm on shared nodes; GB200, not B200; not a hardware ranking; no cost statements.

## 10. Completion and stopping

- A job is complete only when the harness marker and per-arm return codes say so.
- An instrument VOID is re-run once; a second VOID stops the experiment for diagnosis.
- A gate failure stops the experiment. No rescue, no new threshold.
- Hard budget: `<GPU-hours>`.

## Errata

`<none yet>`
