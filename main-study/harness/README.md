# Harness

One copy of the harness, developed here. Files contain environment variables and placeholders only: no project
account, partition, node name or user path. Real values are substituted at deployment time on the cluster side, and the
preregistration pins the commit of this folder that was deployed.

| File | Purpose | Status |
|---|---|---|
| `env.example.sh` | every variable the harness reads, with placeholder values | scaffold |
| `job.sbatch.template` | Slurm wrapper; account, partition and excluded nodes come from the environment at submit time | scaffold |
| `preflight_gpu_clean.sh` | refuse to run on GPUs that already hold memory | ready, untested on GPU |
| `run_arm.sh` | one server launch of one arm: fail-closed checks, server start, backend gate, phases, per-arm status | ready, early exits tested on CPU |
| `sample_sessions.py` | fixed-seed sample of session ids (`../sample/e1-sessions.txt`: seed 20261015, 1,024 ids) | ready |
| `load_trajectories.py` | sampled trajectories → replay turns of token ids, with a prefix-stability check | ready, run on real rows with a stand-in template |
| `replay_client.py` | closed-loop multi-turn replay against `/generate`; per-request TTFT, TPOT, cached tokens | ready, tested against a fake server |
| `gsm8k_client.py` | GSM8K 5-shot greedy, per-question records for paired flips | ready, tested against a fake server |
| `logprob_client.py` | teacher-forced top-20 logprobs on recorded answer tokens | ready, tested against a fake server |
| `prof_client.py` | E4 profiler windows: a steady decode batch, or a timed trigger beside a replay; opens one nsys capture range via `/start_profile` | ready, tested against a fake server |
| `replay_bounds.py` | the two prefix-hit upper bounds | ready |
| `tests/` | CPU-only unit tests: `python -m unittest discover -s tests` (client tests need `aiohttp`) | ready |

Exit codes are listed at the top of `run_arm.sh`; 11 = GPU not clean at start (an infrastructure event, not a result).
