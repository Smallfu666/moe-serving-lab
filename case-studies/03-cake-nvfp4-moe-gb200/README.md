# Case 03: an NVFP4 MoE decode kernel, end to end on GB200

> Private draft. Every claim here is limited to what the public report
> <https://github.com/sgl-project/sglang/issues/43258> already states; this page adds no claim beyond it. Numbers are
> quoted from the independent score of the run (the issue rounds some of them).

## Question

FlashInfer's "Cake" NVFP4 warp-decode MoE kernel covers 1–32 tokens on SM100. The FlashInfer PR that extended it
(<https://github.com/flashinfer-ai/flashinfer/pull/5134>) reports it faster than the FlashInfer baseline when measured
alone; we did not re-measure that. A draft SGLang PR wires it into
the server behind an environment variable (<https://github.com/sgl-project/sglang/pull/42406>), and an SGLang tracking
issue asks for end-to-end reports of exactly this kind (<https://github.com/sgl-project/sglang/issues/42176>).

**When the route is switched on in a real server, does decode get faster?**

## Setup

| | |
|---|---|
| GPU | 1 × GB200 (SM100), TP1, on a shared (non-exclusive) node |
| Model | `nvidia/Qwen3-30B-A3B-NVFP4` at revision `2538ded` (H = 2048, I = 768, E = 128, top-k = 8) |
| SGLang | PR 42406 head `b89daf5` |
| FlashInfer | `ea5d2728` |
| MoE backend | `--moe-runner-backend flashinfer_trtllm` |
| KV cache | bf16 (the checkpoint's default is FP8 KV; not tested) |
| CUDA graph | on |
| Load | `bench_serving` random, 1024 tokens in / 512 out, concurrency 1, 8, 32 |
| Arms | route off vs `SGLANG_CAKE_ROUTES=moe_nvfp4_warp_decode` |
| Repeats | 3 interleaved server launches per arm |
| Accuracy | GSM8K (`sglang.test.few_shot_gsm8k` prompts), 5-shot, greedy, 1,314 questions (test lines 5–1318; lines 0–4 are the shots), two passes per arm |

Timing runs were unprofiled. A separate set of launches with the profiler attached was used to check which
kernels ran and for the descriptive glue figures below; those numbers are never mixed with the timing table.

## Result

**Median TPOT, route off vs on** (RUNTIME)

| Concurrency | Off | On | Change |
|---|---|---|---|
| 1 | 2.2406 ms | 3.0942 ms | +0.854 ms (+38.1 %) |
| 8 | 3.2056 ms | 4.1403 ms | +0.935 ms (+29.2 %) |
| 32 | 4.7108 ms | 5.5382 ms | +0.827 ms (+17.6 %) |

Launch-to-launch spread was 0.045 %, 0.52 % and 0.73 % at concurrency 1, 8 and 32, so every difference above is far
outside run-to-run noise.

**The route really ran** (TRACE). In profiled decode steps under CUDA-graph replay, at concurrency 1 and 32, every step
with the route on ran 144 `kernel_cake_warp_decode_*` kernels and no `bmm_*` MoE GEMMs. With the route off: 0 Cake
kernels and 96 `bmm_*` per step. The only fallback logged was during prefill-graph capture (`num_tokens=8192`, outside
the route's 1..32 admission range).

**Where the time may go — not resolved** (TRACE, profiler attached). With the route on, each MoE layer also runs 6
elementwise mask/fill kernels. At concurrency 1, their span is about 657 µs per decode step; their summed kernel time is
376–380 µs. The span is a time window that also contains memory copies, so it is not a sum of kernel times, and it
cannot be compared with the TPOT table. Whether the Cake kernel chain itself is faster or slower than the chain it
replaces could not be resolved: the comparison was `NOT_EVALUABLE` because the registered resolution threshold, set from
window-to-window variation in the profiled runs, was larger than the difference.

**Accuracy — exploratory** (RUNTIME). Mean of two passes per arm: 90.94 % (route off) vs 90.37 % (on). Off-vs-on pairs
differed on 49 and 52 questions; repeating the same arm differed on 10 (off) and 5 (on). The registered accuracy
falsifier fired on the flip count. This is an inconclusive signal from one session, not an established accuracy
regression: confirming it would need a second pair and a layer-level check, which were not run.

## Limits

- One GB200, TP1, one model, concurrency 1, 8 and 32, on a shared node without `--exclusive`.
- bf16 KV cache only. The checkpoint's default FP8 KV path was not tested.
- The frozen versions above only: draft PR head `b89daf5` and FlashInfer `ea5d2728`. Later revisions may behave
  differently.
- Not covered: prefill, TP > 1, FP8 KV cache, SM103, the kernel's stand-alone speed.
- No claim about why decode got slower. The extra glue kernels are where the time *may* go; that is not shown.
- Measured on 2026-10-08.

## What I learned

_To be written by the author. Prompts:_

1. In your own words: why can a kernel that is faster alone make decode slower in the server?
2. Why is the 656.8 µs glue figure a span and not a sum, and why does that matter when you explain it?
3. Why does the accuracy result count as "inconclusive" even though the falsifier fired?
4. What would you measure next to separate the Cake kernel chain from the glue around it?

## Reproduce

The flags below are the ones the harness passed; prompt counts and the warm-up sequence are left out.

```bash
# SGLang at the PR head b89daf5, FlashInfer at ea5d2728, one SM100 GPU; model downloaded at revision 2538ded
python -m sglang.launch_server --model-path <path/to/Qwen3-30B-A3B-NVFP4> --tp 1 \
  --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm \
  --attention-backend trtllm_mha --kv-cache-dtype bfloat16 \
  --max-running-requests 32 --cuda-graph-max-bs-decode 32 --context-length 4096 \
  --chunked-prefill-size 8192 --mem-fraction-static 0.80 --random-seed 0
# second arm: the same command with the route switched on
SGLANG_CAKE_ROUTES=moe_nvfp4_warp_decode python -m sglang.launch_server ...
# load, at --max-concurrency 1, 8 and 32
python -m sglang.bench_serving --backend sglang --dataset-name random --dataset-path <ShareGPT json> \
  --random-input-len 1024 --random-output-len 512 --random-range-ratio 1.0 \
  --request-rate inf --seed 20261008 --max-concurrency 1 --num-prompts <n>
```

The full harness (interleaving, clean-GPU check, profiler windows) is in private notes and not published here.

## 給作者的備註（zh-TW）

這篇的上限就是已公開的 issue 43258，repo 版不比 issue 多講任何主張。可以講的：開 Cake 之後 TPOT 在 c=1/8/32 都變慢，而且遠大於啟動間的雜訊；trace 證明 Cake kernel 真的有跑。不能講的：變慢的原因（glue 只是「可能」）、Cake kernel 本身快或慢（無法判定）、準確度有退步（只是不確定的訊號）。prereg 的預測區間和另外兩個臂不在 issue 裡，所以這裡也不放。「What I learned」請用你自己的話寫。

---

Provenance: private research notes, case-03, 2026-10-09; measurements re-quoted verbatim from the independent score of that run.
