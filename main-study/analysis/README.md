# Analysis

- `summarize_launch.py`, `e2_score.py`: per-launch summaries and E2 scoring from raw records.
- `nsys_attrib.py`: E4 decode-step attribution from one nsys sqlite export (rules in `../prereg/e4-prereg.md` §5).
- Post-hoc, CPU only, standard library (inputs: the private raw launch folders; outputs committed in `derived/`):
  - `serverlog.py`: shared parsing of server logs, phase marks and replay records.
  - `server_flags.py <raw>`: flags, resolved settings, harness commit and jobs per configuration.
  - `e4_per_gpu_step.py`: E4 per-token table → µs per GPU per decode step (reads `../results/e4-results.md`).
  - `prefill_decode_coupling.py <raw> [--md]`: decode rate with and without prefill in the same log span (R16, N4).
  - `client_vs_engine.py <raw>`: client TPOT p50 against the server log's decode rate.
- `tests/`: `python -m unittest discover -s tests`.
