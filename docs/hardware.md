# Hardware notes

> Draft. Every value below is either one we read on the machine (`nvidia-smi`, `uname -m`, scheduler limits) or is
> tagged **[SPEC]**, meaning a vendor figure we have not measured.

The GPU work runs on a national HPC center's Slurm cluster, which has an H200 partition and a GB200 partition. One
earlier case study ran on a single-GPU workstation.

## 1. What we have

| | H200 nodes | GB200 nodes | Workstation |
|---|---|---|---|
| GPU | NVIDIA H200, 8 per node | NVIDIA GB200, 4 per node | NVIDIA RTX A4500, 1 |
| Compute capability | 9.0 (SM90), 132 SMs | 10.0 (SM100) | 8.6 (SM86) |
| Memory per GPU | 143,771 MiB HBM | 189,471 MiB ≈ 185 GiB HBM | 20 GB |
| L2 (measured) | 60 MiB | not read | not read |
| Driver | 580.65.06 | 580.105.08 | — |
| Host CPU | x86_64 | **Grace, aarch64**, 144 cores | x86_64 |
| Host RAM per node | 1,900 GB | 914 GB | — |

- The H200 and GB200 drivers are the same generation but different minor versions.
- The cluster login node is x86_64. The GB200 compute nodes are aarch64. An x86 container, wheel or virtualenv built on
  the login node does not run there; GB200 jobs need arm64 images.
- On the GB200 nodes, `nvcc` (13.0.88) is installed but not on `PATH`. The H200 nodes have a different `nvcc` minor
  version (13.0.48), so anything compiled is built on the node type it will run on.

## 2. How many GB200s a job can ask for

GPU counts are limited per queue, and the limits live in the scheduler's QoS settings, not in the partition description.

| Queue | Wall-time limit | GPUs per job |
|---|---|---|
| development queue | 2 h | 1–2 |
| large queue | 24 h | 16–32 |
| rack queue | 12 h | 32–72 |

- **3 to 15 GPUs cannot be requested at all.** This is why the main study runs every Blackwell arm on 2 GPUs (TP2) and
  needs a model that fits in 2 × 185 GiB.
- The 2-GPU cap of the development queue is per project account, shared with other users of the same project.
- A request above the cap pends forever, and the scheduler still prints an estimated start time.

## 3. Occupied GPUs on the development queue

In 3 of our first 4 placements on the development queue, at least one assigned GPU was already holding 175,113–177,503
MiB (92–94 % of its memory). We have not determined what held that memory.

What we do about it:

- Every GB200 job first reads `memory.used` on all assigned GPUs and takes the maximum. If it is ≥ 2000 MiB, the job
  stops with a distinct exit code before running any arm. Such a stop is an infrastructure event, not a result.
- Re-queuing the job did not help: it landed on the same node again. Excluding known-bad nodes at submit time was the
  only lever that worked.
- In one run the `--exclusive` flag was rejected on this queue, so nodes are shared. Results from it are labelled
  "shared, non-exclusive".

## 4. GB200 is not B200

| | GB200 (what we have) | B200 (HGX) | Effect on our numbers |
|---|---|---|---|
| GPU architecture | SM100 | SM100 [SPEC] | The same SM100 code paths run (for example SGLang's `is_sm100` branches) |
| Clocks and power limits | — | differ from GB200 [SPEC] | We label every number "GB200" and do not put it next to published B200 numbers |
| Host CPU | Grace, aarch64 | x86_64 [SPEC] | CPU-side work (scheduler, tokenizer) runs on a different CPU from the H200 arm, so low-concurrency latency mixes in host differences |
| CPU↔GPU link | NVLink-C2C [SPEC] | PCIe [SPEC] | Host-memory KV offload would behave differently; the main study does not test offload |
| GPU↔GPU link between the 2 GPUs we get | not measured | — | Each job records `nvidia-smi topo -m` |

HBM bandwidth: H200 4.8 TB/s and GB200 8 TB/s per GPU [SPEC]. The main study's latency predictions start from this
ratio (about 0.6); it is a vendor figure, not something we measured.
