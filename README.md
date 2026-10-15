# moe-serving-lab

Serving one large MoE model on Hopper and Grace-Blackwell under an agentic multi-turn workload:
measurement, attribution, correctness.

> Status: outline only. Sections below are placeholders; no results yet.

## 1. Summary

_To be written by the author after the main study: the question, the hardware, one main finding._

## 2. Main study: same MoE, same agentic workload — H200 FP8 vs GB200 FP8 vs GB200 NVFP4

- Setup: model, checkpoints, engine version, GPU count per arm.
- Workload: replay of public agent trajectories with multi-turn prefix reuse.
- Metrics: TTFT / TPOT / throughput at fixed concurrency, prefix-cache hit rate.
- Results
- Attribution: where the time goes (nsys), and which kernels actually ran.
- Correctness: fixed eval with paired flips, logprob agreement against a reference.
- Limits: what these numbers do not show.

See [`main-study/`](main-study/).

## 3. Case studies

| # | Topic | Hardware | Evidence | Link |
|---|---|---|---|---|
| 01 | CPU-side serving overheads on a small engine | | | [01](case-studies/01-mini-sglang-cpu-overheads/) |
| 02 | A Hopper MoE kernel wired end to end into a serving engine | | | [02](case-studies/02-cutedsl-moe-e2e/) |
| 03 | An NVFP4 MoE decode kernel end to end on GB200 | | | [03](case-studies/03-cake-nvfp4-moe-gb200/) |
| 04 | A KV-cache estimation probe and host-memory OOM | | | [04](case-studies/04-trtllm-kv-probe-host-oom/) |

## 4. Method

Preregistration before running, A/A noise floor, independent scoring, a claim ceiling per result.
See [`docs/method.md`](docs/).

## 5. Hardware notes

H200 vs GB200: host CPU architecture, GPUs per job, what differs from B200.
See [`docs/hardware.md`](docs/).

## 6. Reproducing

Image digests, launch commands, data download scripts (data itself is not stored here).

## 7. Prequel

Earlier deployment record: <https://github.com/Smallfu666/glm-5.2-sglang>

## 8. Acknowledgements and third-party licenses

Models, datasets and their licenses and attributions.

## 9. Notes on how this was made

_Placeholder._
