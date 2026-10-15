#!/usr/bin/env bash
# Exit 11 when any visible GPU already holds >= MSL_CLEAN_MIB MiB: one dirty GPU makes a TP2 job unusable.
set -euo pipefail
limit="${MSL_CLEAN_MIB:-2000}"
out="${1:-gpu_clean.txt}"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
max=$(printf '%s\n' "$used" | sort -rn | head -1)
{
  echo "limit_mib=$limit"
  echo "max_used_mib=$max"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv 2>&1 || true
} > "$out"

if [ "$max" -ge "$limit" ]; then
  echo "GPU_NOT_CLEAN max_used_mib=$max" >&2
  exit 11
fi
echo "GPU_CLEAN max_used_mib=$max"
