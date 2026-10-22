# Attention-DP8 vs TP8 on 8 × H200, simulated before any GPU run

**Question.** On one 8 × H200 node serving `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` (94 layers, 64 query heads,
**4 KV heads**, head_dim 128, 128 experts, top-8; bf16 KV), does SGLang attention-DP8 + EP8 (`--tp 8 --attn-dp-size 8
--ep-size 8`) beat the measured TP8 + EP8 (`--tp 8 --ep-size 8`, E2/E4 "H8")? And is it worth GPU time to find out?

**Answer in one line.** The simulator predicts that DP8 is **slower** at the batches behind E2's matched-interactivity
targets: +4 % to +36 % per decode step at a global batch of 8 to 16 with 8k to 33k contexts, and worse below
B = 8, where some DP ranks sit idle (§4). It **wins** only with large
batches and long contexts (−14 % to −24 % at B = 64 with 16k to 33k contexts), and only if the load across DP ranks
is balanced. With the per-rank imbalance that round-robin dispatch produces on the study's replay, most of that win goes away.
A full measured DP8-vs-TP8 run (about 1,700 SU by an earlier estimate; SU is the cluster's billing unit, 60 per H200
GPU-hour) is not worth it. A one-job check of the simulator (480 SU) is optional (§6).

Evidence labels used below: **[measured]** = E4/E2 numbers re-quoted from the results files; **[sim]** = AIConfigurator
output; **[bytes]** = arithmetic from the model config; **[replay]** = derived on CPU from the E2 per-request records;
**[SPEC]** = vendor figure, not measured. Nothing here is a GPU measurement of DP attention.

## 1. Tools considered

| Tool | What it models | Fit for this question |
|---|---|---|
| **NVIDIA AIConfigurator** (`ai-dynamo/aiconfigurator`; now on PyPI as `aisimulate` 0.12.0, Apache-2.0) | Per-op latency from kernel timings measured on real GPUs, per system, backend and version; TP, attention-DP, MoE-TP/EP; static batches and in-flight batching | **Chosen.** Ships `h200_sxm` silicon data for the **SGLang 0.5.14** backend; its support matrix lists `Qwen/Qwen3-235B-A22B-FP8` on h200_sxm + sglang as PASS. It sets KV heads per GPU to ceil(4 / tp), so it models the KV-head replication at TP8 |
| BBuf `model-compute-simulation` skill (no license; read in place) | Op sequence, tensor shapes, FLOPs and MFU from a model config and TP/DP/EP | Not used. It counts FLOPs, and decode here is limited by memory traffic, not FLOPs. Its script directory was empty when it was read |
| Vidur, LLMServingSim, LLMCompass | Vidur: a scheduler simulator driven by profiled dense-model ops. LLMServingSim: a system-level serving simulator. LLMCompass: a hardware-design roofline | **Not installed or checked in this study.** None is known to carry SGLang attention-DP plus EP and H200 data, but that was not verified. AIConfigurator already had the right backend, GPU and model, so it was used |

Why the SGLang version matters: the E1–E4 server is a nightly (`a6a8c560`, reported as `0.0.0.dev1+ga6a8c5601`).
AIConfigurator's newest H200 SGLang tables are 0.5.14. The validation in §3 checks whether that mismatch matters for
these shapes.

## 2. Method

1. `run_aic.py` runs `aiconfigurator cli estimate --estimate-mode static_gen` for each point: a fixed batch decodes
   64 steps after an ISL-token prompt, which is the shape of the E4 decode windows (16,384-token prompts, at least 96
   warm tokens, 64 profiled steps; ISL = 16,480 here). Flags: `--backend sglang --kvcache-quant-mode bfloat16
   --fmha-quant-mode bfloat16`; MoE and GEMM quantization are inferred from the checkpoint as `fp8_block`. The
   model config is the local `model/…/config.json` of the E1–E4 revision `e156cb4e`. In AIConfigurator `--batch-size` is
   **per attention-DP rank**, so DP points pass B / dp.
2. `analyze.py` (standard library only):
   - **Validation.** Compares simulated TPOT with the E4 step time T (device-mean step; the median start-to-start
     step is within 1 % of it, per the independent E4 score) of the nsys-profiled decode windows for TP4 and TP8+EP8 at B = 1, 4, 8, 16
     [measured; the B = 1 values come from the independent E4 score, which is not in this repo]. It also compares the simulated attention op with the E4 attention class per GPU per step
     (G_attention × B ÷ GPUs). Sum basis: that class includes any overlap.
   - **Bytes.** KV bytes read per token per GPU, and FP8 attention-weight bytes per GPU, for each layout [bytes].
   - **Imbalance.** Replays the recorded E2 H8 requests (`t_send`, `e2e_s`, `prompt_tokens`) and assigns them
     round-robin in arrival order to 8 ranks, as SGLang's default dispatch does with attention-DP (E3 prereg). Every
     0.25 s it records the in-flight count, the mean context and I = max-rank context sum ÷ mean-rank context sum
     [replay]. A sample counts when at least one request is in flight, so the tail of each closed-loop point is
     included. Simplifications: "in flight" spans the whole request (queue + prefill + decode), context is prompt
     tokens only, and errored requests are dropped. I is measured at the replay's mean in-flight count (5.7 / 12 /
     22 / 46 for c = 8 / 16 / 32 / 64) and applied to the nearest static B.
   - **Prediction.** TP8+EP8 vs DP8+EP8 (and DP2 × TP4 + EP8) at global B ∈ {8, 16, 32, 64, 128} and context
     ∈ {8,192; 16,480; 32,768}, with two adjustments to the DP numbers:
     - *imbalanced:* the DP step plus (I − 1) × the simulated DP attention time. This assumes attention time scales
       with the most loaded rank's KV load. It is a **sensitivity estimate, not a bound**. Fixed per-kernel costs
       push it one way. The busiest rank's extra share of the GEMM and MoE work, which is ignored, pushes it the other
       way.
     - *comm-as-TP:* the DP step with its dispatch/combine ops replaced by TP8's. This is a sensitivity check on
       the least-validated simulator component (§4).

## 3. Validation against E4 (TP4 and TP8+EP8, 16k prompts)

| Config | B | measured T (µs) | sim (µs) | error |
|---|---|---|---|---|
| TP4 | 1 | 8301.8 | 7446.0 | −10.3 % |
| TP4 | 4 | 11247.1 | 10741.0 | −4.5 % |
| TP4 | 8 | 14172.3 | 13540.0 | −4.5 % |
| TP4 | 16 | 16573.2 | 17046.0 | +2.9 % |
| TP8+EP8 | 1 | 10030.8 | 8591.0 | −14.4 % |
| TP8+EP8 | 4 | 11462.2 | 10402.0 | −9.2 % |
| TP8+EP8 | 8 | 12927.6 | 12130.0 | −6.2 % |
| TP8+EP8 | 16 | 15567.4 | 15553.0 | −0.1 % |

- **Absolute step time:** within ±10 % at B ≥ 4 (mean |error| 4.6 %). At B = 1 it is too fast by 10–14 %.
- **TP8 ÷ TP4 step ratio** (the relative quantity this question needs): measured 1.208 / 1.019 / 0.912 / 0.939 at
  B = 1/4/8/16; simulated 1.154 / 0.968 / 0.896 / 0.912. Error −1.8 % to −5.0 %.
- **Attention per GPU per step:** measured TP4 1716 / 2300 / 3334 µs and TP8 1693 / 2297 / 3344 µs at B = 4/8/16.
  Simulated TP4 1629 / 2693 / 4050 and TP8 1612 / 2667 / 4059. Error −5 % at B = 4 and +16 % to +22 % at B = 8 and
  16. Both the measurement and the simulator show **no attention shrink from TP4 to TP8**. [bytes] At both TP sizes
  each GPU holds one KV head (ceil(4/4) = ceil(4/8) = 1), so each GPU reads the same 48,128 B of KV per token
  (94 layers × K and V × 1 head × 128 × 2 B). That KV-head replication rests on this arithmetic and on a
  simulator that encodes the same rule. It is still not a measured KV read. At the spec 4.8 TB/s HBM figure, the measured attention time corresponds to 39 %
  (B = 4), 58 % (B = 8) and 79 % (B = 16) of the bandwidth bound [bytes, SPEC].

The error is small enough that no separate roofline was fitted. The byte arithmetic above is the cross-check.
Neither the simulator nor E4 has a **DP-attention measurement on H200**, so the DP side of §4 is the simulator's
extrapolation from its own silicon tables.

## 4. Prediction: TP8+EP8 vs DP8+EP8 (static decode, per-step time)

Global batch B, all requests at the stated context. "DP ÷ TP" < 1 means DP8 is faster. Source:
`results/tables.md`, `results/summary.json`.

| B | ctx | TP8+EP8 ms | DP8+EP8 ms | DP ÷ TP | DP ÷ TP, comm-as-TP | DP ÷ TP, imbalanced (I) | DP2×TP4 ÷ TP |
|---|---|---|---|---|---|---|---|
| 8 | 16,480 | 12.13 | 15.25 | 1.257 | 1.115 | 1.566 (3.30) | 1.236 |
| 16 | 8,192 | 14.07 | 16.51 | 1.173 | 1.074 | 1.323 (2.37) | 1.150 |
| 16 | 16,480 | 15.55 | 17.64 | 1.134 | 1.045 | 1.368 (2.37) | 1.115 |
| 16 | 32,768 | 18.38 | 19.15 | 1.042 | 0.966 | 1.351 (2.37) | 1.025 |
| 32 | 16,480 | 21.87 | 22.45 | 1.026 | 0.965 | 1.190 (1.90) | 1.038 |
| 32 | 32,768 | 27.37 | 25.31 | 0.925 | 0.875 | 1.148 (1.90) | 0.932 |
| 64 | 8,192 | 23.21 | 21.77 | 0.938 | 0.954 | 1.058 (1.70) | 0.956 |
| 64 | 16,480 | 28.77 | 24.69 | 0.858 | 0.872 | 1.026 (1.70) | 0.873 |
| 64 | 32,768 | 40.26 | 30.57 | 0.760 | 0.769 | 0.980 (1.70) | 0.766 |
| 128 | 16,480 | 41.78 | 32.15 | 0.769 | 0.748 | 0.978 (1.70) | 0.778 |

(Full grid, including B = 8 at 8k/33k and B = 128 at 8k/33k, is in `results/tables.md`. B = 128 at 32,768 exceeds
both KV pools. For B ≥ 64 the imbalance uses the c = 64 value, the nearest available.)

**Where the time goes** (sim, B = 16, 16k context, ms per step, TP8 → DP8):

| Op | TP8+EP8 | DP8+EP8 |
|---|---|---|
| attention | 4.06 | 2.65 |
| QKV + O-proj GEMMs | 1.25 | 3.23 |
| MoE dispatch + combine | 1.24 | 2.64 |
| MoE experts | 7.46 | 7.46 |

- **What DP buys** [bytes]: no replicated KV head. Each GPU reads 4 heads for B/8 requests instead of 1 head for all
  B, so KV bytes per GPU per step halve. The KV pool also roughly doubles: the TP8 pool is 2,127,459 tokens
  [measured, E2], and the estimate for DP8 is about 4.0 M tokens after the extra attention weights are taken out [bytes].
- **What DP costs:**
  - Every GPU holds and reads the full attention weights: 6.7 GB of FP8 per step against 0.89 GB at TP8 [bytes].
    The simulator charges about +2.0 ms per step for the QKV and O-proj GEMMs [sim]. That is 1.6 × the
    1.21 ms byte floor of the extra 5.8 GB at 4.8 TB/s [bytes, SPEC].
  - Tokens must be gathered to and scattered from the experts [sim: +1.4 ms at B = 16].
  - The step waits for the slowest rank.
  - Attention cost does not halve at small per-rank batch: 2.65 ms rather than 2.03 ms [sim].
- **Crossover** [sim, balanced load]: DP8 wins at B ≥ 64 at every simulated context, and at B = 32 only with 33k
  context. B × ctx alone does not decide it: B = 64 at 8k (0.52 M tokens) wins, while B = 32 at 16k (0.53 M) and
  B = 16 at 33k (0.52 M) lose.
- **The dispatch cost is the least certain input.** Even if DP's dispatch cost equalled TP8's all-reduce cost, DP8
  would still be 4.5–20 % slower at B ≤ 16 with 8k–16k contexts ("comm-as-TP" column). So the low-batch
  conclusion does not depend on the dispatch cost. It does still depend on the simulator's DP GEMM and attention
  numbers, which are not validated.
- **DP2 × TP4 + EP8** (two attention groups of TP4, no KV replication, attention weights only duplicated) tracks
  DP8 within about 3 % in the simulator. It is not a way around the low-batch penalty.

**Conditions where DP loses** (each one can be checked):

1. **Small global batch** (B ≤ 16, or B ≤ 32 below about 20k context). The weight and communication costs are fixed per
   step, and the KV saving is small.
2. **Unbalanced ranks.** [replay] With round-robin dispatch over the recorded E2 H8 replay, the time-weighted
   imbalance I is 3.30 (c = 8, 5.7 requests in flight on average), 2.37 (c = 16, 12.0), 1.90 (c = 32, 22.0) and 1.70
   (c = 64, 46.1). Mean context is 22.6k–23.5k at c ≥ 4. With the c = 64 value applied at B ≥ 64, the
   estimate puts DP8 within ±7 % of TP8.
3. **Prefix-cache loss in multi-turn sessions** (expected, not measured for DP8). Round-robin can send a session's
   turns to different ranks, and each rank has its own radix cache. In E3, R16 (16 GB200 ranks across four nodes,
   FlashInfer all-to-all) cached 0.48–0.61 of prompt tokens against 0.95–0.96 for N4, and computed 9.4–13 × more
   new prefill tokens (`main-study/results/e4-results.md` Q3). Those are observational counts with no counterfactual,
   and E4 does not say which cause produces R16's ratios. Applying them to DP8 on one H200 node is an
   extrapolation. Static decode does not see this cost at all.
4. **Prefill on one rank may stall every rank** (expected from the per-layer MoE collective, not measured). Static
   decode does not model this either. E4 Q3 reports co-occurring R16 symptoms but attributes no cause.

**Where E2's targets sit.** 50/75/100 tok/s/user means a TPOT of 20/13.3/10 ms. The measured TP8 step at 16k
context is 10.03 / 12.93 / 15.57 ms at B = 1 / 8 / 16 [measured]. So 100 tok/s/user sits near B = 1, 75 near B ≤ 8,
and only 50 reaches B = 16 or somewhat beyond.
- **B < 8** is outside the simulated grid, because DP8 needs B/8 to be an integer. With fewer requests than ranks,
  some DP ranks sit idle. Each request's attention then runs on one GPU reading all 4 KV heads, instead of one
  head on each of 8 GPUs. DP8 also keeps its replicated-weight and gather costs, so it has no saving to offset
  them [bytes; reasoning, not simulated].
- **B = 8–16:** DP8 is predicted 4–36 % slower with balanced load (full grid), and more once imbalance is included. The win region (B ≥ 64 at 16k–33k context) has a TP8 TPOT of 28.8–40.3 ms, i.e. about 25–35 tok/s/user, the low-interactivity end of the H8 frontier (20.06–48.66 tok/s/user). The
replay's imbalance and cache loss work against DP there.

**GB200.** On 4-GPU GB200 (N4, TP4+EP4), KV heads are not replicated (4 heads over 4 ranks), so DP4 has no KV
saving to offset its weight and communication costs. It was not simulated.

## 5. What this does not establish

- No DP-attention number is measured. The DP side rests on AIConfigurator's SGLang 0.5.14 tables, not the study's nightly.
- Static decode only: no prefill, no chunked-prefill interleaving, no scheduler or CUDA-graph padding effects, no
  closed-loop dynamics.
- The imbalance adjustment is a sensitivity estimate built on a proportionality assumption, from one launch's request
  timing (L2).
- The [DeepInfer](https://github.com/deepseek-ai/DeepInfer) stack (DP attention + EP, no TP) is a different engine. Nothing here predicts DeepInfer's numbers.

## 6. Recommendation

**Do not spend the planned budget (about 1,700 SU by an earlier estimate) on a measured DP8 vs TP8 run for this workload.** The predicted answer in the E2 range
is "DP8 is slower". The only win is at high batch with balanced ranks. Round-robin and the multi-turn prefix cache
make that region unfavourable on the study's replay.

**Optional: a simulator check, one job, ≤ 1 h, 8 × H200 = 480 SU** (8 GPU × 1 h × 60 SU/GPU-h; billing is per second
of GPU time, so 480 is the cap). Use the same image and harness as E4, with no nsys. Measure step time from the server
log the same way as the G-DISTORT log basis:

1. Server A: `--tp 8 --ep-size 8` (H8 as in E2/E4). Run decode probes at global B = 16 and B = 64 with 16,384-token
   prompts. B = 64 has no TP8 measurement yet.
2. Server B: `--tp 8 --attn-dp-size 8 --ep-size 8`, same probes. B divisible by 8 gives every rank B/8 requests under
   round-robin, which matches the simulator's balanced case. Server B is also the new-configuration smoke test: if it
   does not reach ready, stop and analyse before anything else.
3. Falsifiers, registered before the run:
   - The simulator's DP model is wrong at low batch if measured DP ÷ TP at B = 16 is ≤ 0.95 (predicted 1.134).
   - It is wrong at high batch if measured DP ÷ TP at B = 64 is ≥ 1.05 (predicted 0.858).
   - Margin: the simulator's TP8 ÷ TP4 ratio error was 2–5 %, and its attention error was up to +22 %. A 5 %
     margin on either side of 1.00 is therefore the tightest that is defensible.

Run this only if a measured DP number is wanted for the write-up. It would not change the serving choice.

## 7. Reproduce (CPU only, about 3 min)

```bash
uv venv -p 3.12 /path/to/venv && VIRTUAL_ENV=/path/to/venv uv pip install -r requirements.txt
python3 run_aic.py --aic /path/to/venv/bin/aiconfigurator          # writes results/aic_points.json
python3 analyze.py --replay-dir <raw>/e2/555000/e2-h8fp8ep-l2 > results/tables.md   # writes results/summary.json
```

The replay records are in the private raw data, not in this repo.
Only the derived aggregates are stored here. Without `--replay-dir`, the imbalance columns are left empty.
