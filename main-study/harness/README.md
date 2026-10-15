# Harness

One copy of the harness, developed here. Files contain environment variables and placeholders only: no project
account, partition, node name or user path. Real values are substituted at deployment time on the cluster side, and the
preregistration pins the commit of this folder that was deployed.

| File | Purpose | Status |
|---|---|---|
| `env.example.sh` | every variable the harness reads, with placeholder values | scaffold |
| `preflight_gpu_clean.sh` | refuse to run on GPUs that already hold memory | ready, untested on GPU |
| `job.sbatch.template` | Slurm wrapper; account, partition and excluded nodes come from the environment at submit time | scaffold |
| `sample_sessions.py` | fixed-seed sample of session ids from a list of ids | ready |
| `replay_bounds.py` | per-turn prompt construction and the two prefix-hit upper bounds | ready |
| `tests/` | CPU-only unit tests: `python -m unittest discover -s tests` | ready |

Not written yet (needs the dataset schema, checked in week 1): the trajectory loader, the tokenizer step and the
replay client that sends `input_ids` to SGLang's native `/generate`.

Exit codes used by the shell pieces: 11 = GPU not clean at start (an infrastructure event, not a result).
