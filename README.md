# moe-serving-lab

Serving one large MoE model on Hopper and Grace-Blackwell under an agentic multi-turn workload:
measurement, attribution, correctness.

> Status: work in progress. The main study (E1–E4) ran on 2026-10-15/16 and is scored; start with
> [`main-study/SUMMARY.md`](main-study/SUMMARY.md). Three case studies are written up from earlier, independently
> scored runs.

## 1. Summary

_To be written by the author after the main study: the question, the hardware, one main finding._

## 2. Main study: same MoE, same agentic workload — H200 FP8 vs GB200 FP8 vs GB200 NVFP4

- Setup: Qwen3-235B-A22B-Instruct-2507 (public FP8 and NVFP4 checkpoints), one SGLang commit; 2 GPUs per arm in E1,
  each platform at its own configuration in E2–E4 (2 × GB200 TP2, 4 × H200 TP4, 8 × H200 TP8 + EP8, 4 and 16 × GB200).
- Workload: replay of public agent trajectories with multi-turn prefix reuse.
- Metrics: TTFT / TPOT / throughput at fixed concurrency, prefix-cache hit rate.
- Results: E1–E4, preregistered (except E4 Q3) and independently scored: [`main-study/results/`](main-study/results/), one-page
  overview in [`main-study/SUMMARY.md`](main-study/SUMMARY.md). Post-hoc CPU analyses of the same logs (not scored):
  [`main-study/results/posthoc-log-analyses.md`](main-study/results/posthoc-log-analyses.md).
- Attribution: where the time goes (nsys), and which kernels actually ran. Descriptive, not causal.
- Correctness: fixed eval with paired flips, logprob agreement against a reference.
- Limits: what these numbers do not show.

See [`main-study/`](main-study/).

## 3. Case studies

| # | Topic | Hardware | Evidence | Link |
|---|---|---|---|---|
| 01 | On a small engine, overlap scheduling hides almost all scheduler CPU time yet gains only 5 % throughput; splitting tokenize/detokenize work across processes cut burst TTFT p50 from 263 to 117 ms | 1 × RTX A4500 | preregistered, independently scored; one secondary prediction falsified | [01](case-studies/01-mini-sglang-cpu-overheads/) |
| 03 | An NVFP4 MoE decode kernel reported faster alone makes end-to-end TPOT 17.6–38.1 % slower | 1 × GB200 | preregistered, independently scored; public report | [03](case-studies/03-cake-nvfp4-moe-gb200/) |
| 04 | TensorRT-LLM's KV estimation probe ignores the host-memory budget and can be OOM-killed in a 16 GiB container | 1 × H200 | preregistered, independently scored; public report | [04](case-studies/04-trtllm-kv-probe-host-oom/) |

## 4. Method

Preregistration before running, an A/A noise floor, independent scoring, a claim ceiling per result.
See [`docs/method.md`](docs/method.md).

## 5. Hardware notes

H200 vs GB200: host CPU architecture, GPUs per job, what differs from B200.
See [`docs/hardware.md`](docs/hardware.md).

## 6. Reproducing

Main study: image digests, checkpoint revisions and the replay input's sha256 are in
[`main-study/prereg/e1-prereg.md`](main-study/prereg/e1-prereg.md) §2; the server flags, harness commit and job ids of
every configuration are in [`main-study/README.md`](main-study/README.md#configurations-and-server-flags). The data
(model weights, trajectories, raw logs) is not stored here, and the script that built `replay.jsonl` from the sampled
ids (`harness/load_trajectories.py` inside the pinned image) runs from a private job wrapper that is not in this
repository. Each case study has its own Reproduce section.

## 7. Prequel

Earlier deployment record: <https://github.com/Smallfu666/glm-5.2-sglang>

## 8. Acknowledgements and third-party licenses

Software, models and data used or referenced. Nothing below is redistributed in this repository.

| Item | Used in | License |
|---|---|---|
| [SGLang](https://github.com/sgl-project/sglang) | main study, 03 | see repository |
| [mini-sglang](https://github.com/sgl-project/mini-sglang) | 01 | MIT |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | 03 | see repository |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | 04 | see repository |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 01 | Apache-2.0 |
| [nvidia/Qwen3-30B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4) | 03 | Apache-2.0 |
| [Qwen/Qwen3-235B-A22B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8) | main study | Apache-2.0 |
| [nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4](https://huggingface.co/nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4) | main study | see model card |
| [Qwen/Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) | main study (BF16 reference) | Apache-2.0 |
| [Qwen/Qwen3-30B-A3B-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8) | main study (E4 profiler-overhead check) | Apache-2.0 |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 04 | see model card |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 03, main study | MIT |
| [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | main study | CC-BY-4.0; attribution text still to be completed (TODO, author) |

## 9. Notes on how this was made

_TODO (author): decide whether and how to describe AI assistance before this repository is made public._
