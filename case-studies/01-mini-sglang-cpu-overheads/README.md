# Case 01: CPU-side serving overheads on a small engine

> Measured on mini-sglang (<https://github.com/sgl-project/mini-sglang>, MIT), a compact teaching
> implementation of an SGLang-style server. **mini-sglang is not SGLang**, and a 0.6B model on one workstation GPU is
> not a production setting. Numbers are quoted from the independent scores of the runs.

## Question

On a small model, the GPU finishes each step quickly, so CPU work around it (batch scheduling, tokenization,
detokenization) can become a visible share of the time. Three questions, each preregistered:

1. **E1.** Does overlap scheduling (prepare the next batch while the GPU runs the current one) raise throughput, and does
   the scheduler's CPU time actually hide under GPU work when it is on?
2. **E2.** How much does a warm prefix (radix) cache cut time-to-first-token for requests that share a long prompt?
3. **Tokenizer study.** E2 left a puzzle: with a warm cache, a 48-request burst still had a TTFT p50 of 234 ms against
   16.8 ms for one request at a time, although each request only computed about 37 new tokens. Is the burst waiting on
   the tokenizer?

## Setup

| | |
|---|---|
| GPU | 1 × RTX A4500 (SM86, 20 GB), driver 580.178.04, on a workstation with a 24-thread Xeon w5-3425 CPU |
| Engine | mini-sglang `9a91cfaf`, unmodified; logging-only wrappers around a few methods |
| Stack | Python 3.12, torch 2.9.1+cu128, flashinfer-python 0.7.0.post1, sgl-kernel 0.3.21 |
| Model | `Qwen/Qwen3-0.6B`, bf16, revision `c1899de`, TP = 1 |
| E1 workload | 128 sequences, random token ids, input length U[100, 1024], output U[100, 512], `ignore_eos`; 10 runs interleaved on/off/off/on…, 5 per arm, fresh process each; overlap off = `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` |
| E1 trace | one nsys trace per arm, 64 sequences; NVTX range around the batch-scheduling call |
| E2 workload | 2048-token shared system prompt + 32-token distinct suffixes, 32 output tokens; warm radix cache vs `--cache-type naive`; 48-request burst and 16 sequential requests; 3 runs per arm |
| Tokenizer study | as E2 with a warm radix cache; `--num-tokenizer` 0 (one process both tokenizes and detokenizes; N0), 1 (detokenizer split out; N1), 4 (four tokenizers plus a separate detokenizer; N4); 3 runs per arm, interleaved |

## Result

**E1, throughput** (RUNTIME). Turning overlap scheduling off lowered median throughput by 5.0 % (4169 → 3961 tok/s,
R1 = 1.052; n = 5 per arm, spread 2.0 % on / 1.4 % off, arms fully separated). The ratio is inside the predicted band
of 1.05–1.30 but **below the 1.15 point prediction**: the effect is real here, but smaller than expected.

**E1, mechanism** (TRACE, one trace per arm). The batch-scheduling CPU time overlapped GPU kernels 95.4 % of the time
with overlap on vs 1.7 % with overlap off (registered thresholds ≥ 50 % / ≤ 20 %). GPU kernel-busy fraction was 0.960
vs 0.864.

**E2, prefix cache** (RUNTIME). A warm radix cache cut 48-request burst TTFT p50 from 1157 ms to 234 ms (ratio 0.20;
p90 1940 → 245 ms), raised burst throughput from 17.2 to 73.7 req/s (ratio 4.29), and cut one-at-a-time TTFT p50 from
55.6 to 16.8 ms (ratio 0.30). This is the warm-cache best case.

**Tokenizer study** (RUNTIME).

| Arm | Burst TTFT p50 (median of 3 runs) | Ratio to N0 |
|---|---|---|
| N0: 1 process tokenizes and detokenizes | 263.15 ms | — |
| N1: detokenizer split out, still 1 tokenizer | 150.86 ms | R1 = 0.5733 |
| N4: 4 tokenizers + separate detokenizer | 117.18 ms | R4 = 0.4453 |

- The main hypothesis held: going from N0 to N4 cut burst TTFT p50 from 263.15 ms to 117.18 ms. In N0, the tokenizer
  queue plus tokenize stage was a median share of 0.3279 of the time from burst start to the first chunk.
- **A secondary prediction was falsified.** We predicted that splitting out only the detokenizer would matter little
  (R1 in [0.80, 1.05]). It gave R1 = 0.5733. So the N0 → N4 gain cannot be attributed to parallel tokenization alone.
- Two other predictions also missed: R4 = 0.4453 is just below the predicted 0.45–0.85 band, and the N0 tokenizer
  share of 0.3279 is below the predicted 0.45.
- The registered confound check stayed silent: one-at-a-time TTFT p50 ratios were 1.0351 (N4) and 0.9242 (N1), inside
  [0.80, 1.25], so no flag that the process change altered something besides burst queueing.

Unregistered stage timings from the same runs (descriptive, not verdicts):

| Median stage time (ms) | N0 | N1 | N4 |
|---|---|---|---|
| client send → frontend hands off the request | 90.96 | 63.13 | 80.14 |
| waiting in the tokenizer queue | 85.32 | 84.38 | 19.03 |
| first token computed → reaches detokenizer | 75.95 | 0.08 | 0.17 |

- N0 → N1 removed the detokenize wait (75.95 → 0.08 ms); N1 → N4 removed most of the tokenizer queue (84.38 →
  19.03 ms).
- The largest N0 stage was not the predicted tokenizer queue but "client send → frontend", about 91 ms. That stage
  covers the client, HTTP and frontend; it was not broken down further and its cause is not known.

## Limits

- One RTX A4500, one 0.6B model, synthetic workloads, mini-sglang at one commit. These are not SGLang numbers and do
  not transfer to H200-class GPUs or large models.
- E1's mechanism result rests on one trace per arm.
- E2 is the warm-cache best case. E2's per-request TTFT timestamps were not saved; its percentiles are the client's
  summaries.
- The tokenizer study changes two things between N0 and N4 (tokenizer count, and moving detokenization out). Only the
  combined effect is supported; the stage table is descriptive.
- The tokenizer study's first attempt produced no data because of a bug in our own harness (an argument was shifted
  away before use). It was recorded as an erratum, fixed, and the registered design was re-run exactly once.
- Measured on 2026-10-06 and 2026-10-07.

## What I learned

_To be written by the author. Prompts:_

1. In `overlap_loop`, what is scheduled before what, and what does the engine pay for it?
2. Why was the throughput gain only 5 % on this setup, although almost all scheduling time was hidden?
3. Why did the detokenizer, not the tokenizer, turn out to matter so much in the shared-process setup?
4. Which part of TTFT here is not GPU time at all, and how would you measure it in a real SGLang deployment?

## Reproduce

```bash
git clone https://github.com/sgl-project/mini-sglang.git && cd mini-sglang && git checkout 9a91cfaf
uv venv --python=3.12 && source .venv/bin/activate && uv pip install -e .
# flashinfer 0.7 JIT needs a CUDA 12.8 nvcc; an older system nvcc (12.2) failed to build it
python -m minisgl --model Qwen/Qwen3-0.6B --port 30111                 # smoke test
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl ...              # E1, overlap off
python -m minisgl ... --cache-type naive                                # E2, no prefix cache
python -m minisgl ... --num-tokenizer 4                                 # tokenizer study, N4
```

Where to read in mini-sglang at `9a91cfaf`:

- `python/minisgl/scheduler/scheduler.py`: `overlap_loop` (line 83) vs `normal_loop` (line 108); `_process_last_data`
  (line 138) waits on the copy event of the previous step.
- `python/minisgl/server/launch.py` lines 71–109 and `server/args.py` lines 18–23: the process layout, where
  `num_tokenizer = 0` means one shared tokenizer/detokenizer process.

The measurement harness (load clients, NVTX probes, stage timestamps) is in private notes and not published here.

---

Provenance: private research notes, case-01, 2026-10-07; measurements re-quoted verbatim from the independent scores of those runs.
