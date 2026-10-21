#!/bin/bash
# Usage: REPLAY=<frozen replay.jsonl> RAW=<raw log folder> LAUNCH_MAP=<json: launch label -> directory under RAW> bash run_all.sh
set -euo pipefail
: "${REPLAY:?}" "${RAW:?}" "${LAUNCH_MAP:?}"
cd "$(dirname "$0")"
python3 calibrate.py --raw "$RAW" --out results/calibration.json
python3 validate.py --replay "$REPLAY" --raw "$RAW" --config validation_runs.json --out results/validation.json
python3 timing_check.py --replay "$REPLAY" --raw "$RAW" --calibration results/calibration.json --out results/timing_check.json
python3 routing_sim.py --replay "$REPLAY" --calibration results/calibration.json --out results/routing.json
python3 coupling.py --raw "$RAW" --calibration results/calibration.json --out results/coupling.json
python3 project_r16.py --replay "$REPLAY" --raw "$RAW" --calibration results/calibration.json \
  --coupling results/coupling.json --out results/r16_projection.json
python3 tables.py > results/tables.md
