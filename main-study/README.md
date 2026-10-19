# Main study: same MoE, same agentic workload — H200 FP8 vs GB200 FP8 vs GB200 NVFP4

> **Status: done.** E1–E4 ran on 2026-10-15/16 and each was independently scored and reviewed. Start with
> [`SUMMARY.md`](SUMMARY.md); the scored results are in [`results/`](results/), the figures in [`figures/`](figures/)
> and a 2-minute talk in [`TALK_2MIN.md`](TALK_2MIN.md). The sections below are the original E1 plan; E2 (per-GPU
> throughput at matched interactivity), E3 (GB200 scaling) and E4 (nsys attribution) extended it, as each
> `results/eN-results.md` describes.

## Question

With one SGLang commit, one MoE model and one replay of real agent trajectories, all on 2 GPUs per arm: how do
GB200 FP8, GB200 NVFP4 and H200 FP8 differ in TTFT, TPOT, prefix-cache hit rate and accuracy, how much time goes to each
kernel class, and which MoE backend actually runs? The attribution is descriptive, not causal: each arm changes several
things at once (kernel path, HBM bandwidth, KV capacity, host CPU).

## Arms (planned)

| Arm | GPUs | Checkpoint | Notes |
|---|---|---|---|
| G-FP8 | 2 × GB200, TP2 | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | |
| G-NV4 | 2 × GB200, TP2 | `nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4` | |
| H-FP8 | 2 × H200, TP2 | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | may move to TP4 under a rule fixed before any data |

KV cache is bf16 in every arm. MoE and attention backends are left on `auto`; the backend that actually runs is read
from the server log and is a gate.

The two checkpoints were quantized by different parties (block FP8 by the model authors, NVFP4 by NVIDIA ModelOpt), so
"FP8 vs NVFP4" compares two public checkpoints, not a controlled quantization experiment.

## Why GB200 and 2 GPUs

See [`../docs/hardware.md`](../docs/hardware.md): a GB200 job can ask for 1–2 GPUs or for 16 and more, nothing in
between, so the model has to fit in 2 × 185 GiB.

## Layout

| Folder | Contents |
|---|---|
| [`prereg/`](prereg/) | the frozen preregistrations (`e1`–`e4-prereg.md`) |
| [`figures/`](figures/) | figures for the summary, drawn by `make_figures.py` from the results files |
| [`harness/`](harness/) | replay client, launch wrapper, pre-flight checks; environment variables and placeholders only |
| [`results/`](results/) | small summary tables, quoted from the independent score; no raw logs |
| [`analysis/`](analysis/) | scripts that turn the scored results into the tables and plots of the write-up |

## Workload (planned)

Replay of `nebius/SWE-rebench-openhands-trajectories` (CC-BY-4.0): fixed-seed sample of sessions; turn k sends the
recorded first k−1 turns plus the k-th user/tool message as token ids, so every arm sees identical prompts. The data
itself is not stored here; only the sampled session ids and a download script will be.

Replaying recorded assistant text (instead of the server's own output) means server-generated KV is never reused, so
the measured prefix-hit rate is systematically lower than in real agentic serving. The analysis reports two upper bounds
computed from the sample alone (see `harness/replay_bounds.py`):

- **replay bound** = shared prefix of turn k and turn k−1 prompts ÷ turn k prompt length;
- **serving bound** = (turn k−1 prompt + recorded turn k−1 answer) ÷ turn k prompt length.

## Claim ceiling (draft)

One SGLang commit; two public checkpoints of Qwen3-235B-A22B-Instruct-2507; one replay sample of SWE agent
trajectories (prefix hits below real serving); descriptive attribution, not a causal split; 2 GPUs per arm on shared,
non-exclusive nodes; GB200, not B200; not a hardware ranking; no cost or price-per-token statements.
