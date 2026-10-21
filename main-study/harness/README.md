# Harness

One copy of the harness, developed here. Each E1–E4 job recorded the commit it ran (see the configuration table in
[`../README.md`](../README.md#configurations-and-server-flags)). Files contain environment variables and placeholders only: no project
account, partition, node name or user path. Real values are substituted at deployment time on the cluster side, and the
preregistration pins the commit of this folder that was deployed.

| File | Purpose | Status |
|---|---|---|
| `env.example.sh` | every variable the harness reads, with placeholder values | template; the E1–E4 jobs sourced private per-experiment copies with the real values |
| `job.sbatch.template` | Slurm wrapper; account, partition and excluded nodes come from the environment at submit time | template; E1–E4 used private wrappers of the same shape (source env, call `run_arm.sh`, write a marker) that also record `job_info.txt`; every E1–E4 job recorded `dirty=0` there |
| `preflight_gpu_clean.sh` | refuse to run on GPUs that already hold memory | ran in every E1–E4 launch (`gpu_clean.txt`) |
| `run_arm.sh` | one server launch of one arm: fail-closed checks, server start, backend gate, phases, per-arm status | ran every E1–E4 launch; early exits also tested on CPU |
| `sample_sessions.py` | fixed-seed sample of session ids (`../sample/e1-sessions.txt`: seed 20261015, 1,024 ids) | produced the E1–E4 sample |
| `load_trajectories.py` | sampled trajectories → replay turns of token ids, with a prefix-stability check | built `replay.jsonl` (sha256 in `../prereg/e1-prereg.md` §2) inside the pinned image |
| `replay_client.py` | closed-loop multi-turn replay against `/generate`; per-request TTFT, TPOT, cached tokens | used in E1–E3 and the E4 mixed windows |
| `gsm8k_client.py` | GSM8K 5-shot greedy, per-question records for paired flips | used in E1–E3 |
| `logprob_client.py` | teacher-forced top-20 logprobs on recorded answer tokens | used in E1–E3 |
| `prof_client.py` | E4 profiler windows: a steady decode batch, or a timed trigger beside a replay; opens one nsys capture range via `/start_profile` | used in E4 |
| `replay_bounds.py` | the two prefix-hit upper bounds | written; not used in the scored results |
| `tests/` | CPU-only unit tests: `python -m unittest discover -s tests` (client tests need `aiohttp`) | ready |

Exit codes are listed at the top of `run_arm.sh`; 11 = GPU not clean at start (an infrastructure event, not a result).
