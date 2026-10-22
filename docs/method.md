# Method

How every number in this repository was produced, and how far each number may be read.

## 1. Preregister before running

Each experiment starts with a short written plan that is committed before the first job is submitted. The plan fixes:

- **Question.** One sentence, answerable by the run.
- **Arms.** What changes between arms, and everything that must stay the same (commit, image, model revision, flags).
- **Predictions with bands.** A number and a range for every quantity that will be read, written before any data
  exists. A result outside the band is reported as a miss, not reinterpreted.
- **Falsifiers.** For each prediction, the observation that would make it wrong.
- **Gates.** Checks that decide whether a run counts at all (clean GPU at start, the intended kernel actually ran, every
  request completed, no thermal throttling). A run that fails a gate is VOID and is not scored.
- **Claim ceiling.** The widest sentence the run could support if every prediction held: one GPU type, one model, one
  commit, one workload, measured on these dates.

Once data has been seen, the plan does not change. If a defect is found, it is written down as an erratum next to the
plan, with what it changes and what it does not. A failed gate stops the experiment; it is not rescued by a new
threshold or a different reading.

## 2. A/A noise floor

Every comparison carries its own noise floor: the same arm launched twice from scratch. A difference between arms is
reported as resolved only if it is larger than the floor (for example `T = max(0.03, 2-3 × |A/A ratio − 1|)`; the exact
rule is fixed in each plan). Concretely this means:

- Fresh server launches per arm, interleaved (for example `B C C B`) so drift does not line up with one arm.
- Labels per cell: `FASTER`, `SLOWER`, `WITHIN` (inside the floor) or `NOT_EVALUABLE` (the floor itself is too wide).
- `WITHIN` never means "the same". It only means the difference was not resolved.
- For accuracy, the floor is the number of answers that flip between two launches of the same arm. Two arms are compared
  on paired flips against that floor, not only on the accuracy difference.

## 3. Independent scoring

The person who designs and runs an experiment does not score it. A separate scorer, who did not write the plan or the
harness, recomputes every number from the raw records and applies the plan's rules as written. Numbers in this
repository are quoted from that score, without rounding them into a better story. Where the score says a reading is
"not supported", the write-up does not make it.

## 4. Kinds of evidence

Each claim names the kind of evidence behind it, and kinds are not mixed in one sentence:

| Kind | What it is | Example |
|---|---|---|
| RUNTIME | Timing or accuracy from unprofiled runs | median TPOT over interleaved launches |
| TRACE | What a profiler showed in a few captured steps | which kernels ran in a CUDA-graph replay |
| SOURCE | Reading the code at a pinned commit; nothing was run | "this function drops the configured budget" |
| SPEC | A vendor datasheet number, not measured here | HBM bandwidth |

Profiler-attached numbers are never compared with unprofiled timing. A summed kernel time is not called "the
bottleneck"; it can include waits and overlap.

## 5. Completion

A job counts as finished only when the harness writes its own completion marker and per-arm return codes. The scheduler's
job state and elapsed time are not used for this (a job can show `COMPLETED` in seconds without having run anything).

## 6. Engine configuration parity (main study)

Two rules carried over from an earlier three-engine comparison:

- The KV-cache pool must be defined the same way across arms (memory left after weights vs a fraction of total memory
  are not the same thing).
- Server time-to-ready is recorded separately from serving metrics.

The main study runs one engine, so the first rule reduces to recording what SGLang chose. The memory fraction was left
at SGLang's default for each platform and parallelism, not equalised; the flags passed and the settings the server
resolved (memory fraction, page size, backends) are listed per configuration in
[`../main-study/README.md`](../main-study/README.md#configurations-and-server-flags); the E2 and E3 results files give
each configuration's KV pool in tokens.

## 7. What is not published here

Raw logs, profiler reports, the independent score files and the cluster-specific job wrappers stay in private notes.
The harness (with placeholders for cluster values), the analysis scripts and their small derived outputs are
published. Each write-up ends with a provenance line that names the private source by a neutral label and date.
