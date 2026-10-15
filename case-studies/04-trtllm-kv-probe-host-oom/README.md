# Case 04: a KV-cache estimation probe that ignores the host-memory budget

> Private draft. The scope of this page is capped by the public report
> <https://github.com/NVIDIA/TensorRT-LLM/issues/20211>; nothing here goes beyond it. Numbers are quoted from the
> independent scores of the runs. No fix is claimed or included.

## Question

TensorRT-LLM's PyTorch backend runs a KV-cache estimation pass at startup: it builds throwaway "probe" cache managers to
measure memory, then builds the real ones. With KVCacheManagerV2 a user can set a host-memory tier budget,
`kv_cache_config.host_cache_size`.

**Does the estimation probe respect that budget, and what happens in a memory-limited container if it does not?**

## Setup

| | |
|---|---|
| GPU | 1 × H200 (143,771 MiB), x86_64 host |
| Software | container `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc29` (TensorRT-LLM 1.3.0rc29), `trtllm-serve` |
| Model | Llama-3.1-8B-Instruct, bf16, TP = 1 |
| Memory limit | Slurm job with cgroup v2 `memory.max` = 16 GiB (`--mem=16G`), `swap.max` = 0 |
| Config | `kv_cache_config: host_cache_size: 1073741824` (1 GiB) |
| Arms | default (estimation on) vs `TRTLLM_SKIP_KV_CACHE_ESTIMATION=1` (estimation skipped), same command otherwise |
| Repeats | 6 interleaved starts per arm (run R1) |

The skip arm removes the whole estimation pass, not only the probe's host tier. Pinning the OOM on the probe's tier rests
on the probe's logged tier size and on the startup phase that was open when the process was killed.

## Result

**R1: the probe sizes its own host tier, and startup is OOM-killed** (RUNTIME)

| Arm | Probe host tier (from the log) | Result |
|---|---|---|
| default (estimation on) | 17.37 GiB in every start, not the configured 1 GiB | 6 of 6 starts OOM-killed during `profiling_kv_cache_allocation` (cgroup `oom_kill` +1 each) |
| `TRTLLM_SKIP_KV_CACHE_ESTIMATION=1` | no probe; host tier 1.00 GiB | 6 of 6 starts ready in 72.5–73.5 s, no OOM |

The OOM count was checked against two sources: the scheduler's error log reported 6 `oom_kill` events for the job, and
the final cgroup `memory.events` showed `oom_kill` = 6.

**Why the probe ignores the budget** (SOURCE, read on main `fced9ff98`, the same code as rc29). The estimation pass
deliberately clears the configured budget for its probe managers (`_drop_explicit_offload_tier_budgets`, added in
<https://github.com/NVIDIA/TensorRT-LLM/pull/17850> so that probes stop pinning the full configured host tier). The
probe then falls back to the automatic host tier, sized as
`min(probe device quota, SC_AVPHYS_PAGES × page size / local ranks / 2, memlock × 0.8)`
(`kv_cache_manager_v2.py:425-432`, `:1656`; `_util.py:2829-2833`). Neither the configured budget nor the cgroup limit
enters that bound. `SC_AVPHYS_PAGES` is node-wide free memory (it equalled `MemFree` on our nodes), not the cgroup's
headroom.

In R1 TRT-LLM logged 141.6–156.3 GiB of available host memory, so the device-quota term set the tier (17.37 GiB); the
free-memory term was 70.8–78.2 GiB.

**R2: a larger budget is exceeded too.** In a separate run with `host_cache_size` = 8 GiB under `--mem=48G`, all six
starts came up, but every probe tier exceeded the 8 GiB budget: the tiers were 11.32–17.37 GiB, and the free-memory
term set the tier in two of them (11.32 and 16.84 GiB).

**R3: the OOM does not always happen.** In a later job with the same container, model, config and 16 GiB limit, a
default-arm start logged 14.22 GiB of available host memory, sized the probe tier at 7.11 GiB, and became ready in
88.9 s with no OOM kill.

**R4: tier size decides it** (RUNTIME). In one job on one node, only the `SC_AVPHYS_PAGES` value read by the tier
sizing was overridden, alternating starts: a 17.00 GiB probe tier was OOM-killed during `profiling_kv_cache_allocation`
in 3 of 3 starts, and a 4.00 GiB tier came up in 3 of 3. Since the tier is half of node free memory until it reaches
the device quota, whether a start is OOM-killed depends on how much memory the whole node has free at probe time.

## Limits

- "6 of 6" holds for the node state of R1 only. R3 shows a default-arm start under the same 16 GiB limit that was not killed.
- Only two tier sizes were tested in R4 (4.00 and 17.00 GiB). Where the threshold lies between them is not measured.
- Attribution is at the level of the startup phase: the kill happened while `profiling_kv_cache_allocation` was open.
  Which allocation inside that phase tipped the cgroup over is not isolated.
- One model, TP = 1, one GPU type, Slurm cgroup v2 without swap. Not covered: other models, TP > 1, Kubernetes or other
  cgroup setups, swap, the open upstream change <https://github.com/NVIDIA/TensorRT-LLM/pull/19650>.
- The runs were on rc29; main is the same code by source reading only.
- **No claim about any fix.** Nothing here shows that a particular change prevents the OOM.

## What I learned

_To be written by the author. Prompts:_

1. Why does the estimation pass drop the configured budget on purpose, and what did that design not account for?
2. Why can the same command be OOM-killed on one day and start fine on another?
3. Why is the skip arm a control for the whole estimation pass, not for the host tier alone, and why is that still
   enough here?
4. If you had to explain this to a maintainer in two minutes, which single table would you show?

## Reproduce

From the public report (rc29 container, 1 × H200, inside a 16 GiB cgroup v2 limit with no swap):

```yaml
# cfg.yaml
kv_cache_config:
  host_cache_size: 1073741824
```

```bash
# default arm (estimation on)
trtllm-serve serve <path/to/Llama-3.1-8B-Instruct> --host 127.0.0.1 --port <port> \
  --log_level info --no-telemetry --config cfg.yaml
# skip-estimation arm
TRTLLM_SKIP_KV_CACHE_ESTIMATION=1 trtllm-serve serve ...   # same arguments
```

Whether the default arm is killed depends on node free memory at probe time (see R3, R4).

## 給作者的備註（zh-TW）

這篇的上限是已公開的 issue 20211 的正文，repo 版不多講任何主張。重點是一條因果鏈：估算階段刻意丟掉 `host_cache_size` → 探測改用自動 tier → 自動 tier 看的是整台節點的空閒記憶體，不是 cgroup → 節點空閒多時 tier 太大，在 16 GiB cgroup 裡被 OOM 殺掉。R3 是反例（節點空閒少、tier 小、沒死），所以「6/6」只對 R1 那次的節點狀態成立。「What I learned」請用你自己的話寫。

---

Provenance: private research notes, case-04, 2026-10-13; measurements re-quoted verbatim from the independent scores of those runs.
