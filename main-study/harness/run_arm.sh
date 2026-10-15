#!/usr/bin/env bash
# One server launch of one arm, run inside a Slurm allocation. All site values come from the deployed env file.
#
# Usage: run_arm.sh <arm-tag> <model-dir> <expect-gpu: GB200|H200> <phases> <out-dir>
#   phases: comma list of  replay:<concurrency>:<start>:<count>[:<deadline-s>] | gsm8k | logprob:<start>:<n>
#
# Exit codes (distinct per cause, so a 0-second failure is never read as a hardware result):
#    8 cache dir outside MSL_ROOT    9 arch / env leak            10 wrong GPU type or count
#   11 GPU not clean                12 SIF missing or wrong size 13 replay input sha mismatch
#   14 CUDA context check failed    15 server process exited before ready
#   16 expected MoE backend not in log                           17 cache flush refused
#   18 server not ready within MSL_READY_TIMEOUT_S               19 server process died during a phase
#   20+ a phase client reported bad requests (20 replay, 21 gsm8k, 22 logprob); see each record's error_kind
#   30 stopped at the harness end time (MSL_END_EPOCH): the phase in progress was cut or the next one could not start
#
# MSL_END_EPOCH (optional): epoch seconds by which the launch must stop. A replay phase gets the remaining time minus
# MSL_END_MARGIN_S as its deadline; a phase that would start with less than MSL_MIN_PHASE_S left is not started.
set -uo pipefail
TAG=$1 MODEL=$2 EXPECT=$3 PHASES=$4 OUT=$5
: "${MSL_ROOT:?}" "${MSL_SIF:?}" "${MSL_HARNESS:?}" "${MSL_REPLAY:?}" "${MSL_REPLAY_SHA:?}" "${MSL_GSM8K_DIR:?}"
: "${MSL_TP:?}" "${MSL_SIF_BYTES:?}" "${MSL_PORT:=30000}" "${MSL_READY_TIMEOUT_S:=2400}" "${MSL_CLEAN_MIB:=2000}"
mkdir -p "$OUT"
cd "$OUT" || exit 9
status () { echo "$TAG $1 rc=$2 elapsed_s=$SECONDS $(date -Is)" >> "$OUT/arm_status.txt"; }
finish () { echo "$TAG $1" >> "$OUT/arm_rc.txt"; status done "$1"; exit "$1"; }

for v in HF_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME TMPDIR FLASHINFER_WORKSPACE_BASE SGLANG_CACHE_DIR; do
  case "${!v:-}" in "$MSL_ROOT"/*) ;; *) echo "cache var $v=${!v:-unset} not under MSL_ROOT"; finish 8 ;; esac
done
unset CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
want_arch=$([ "$EXPECT" = GB200 ] && echo aarch64 || echo x86_64)
[ "$(uname -m)" = "$want_arch" ] || { echo "arch $(uname -m) != $want_arch"; finish 9; }
[ -s "$MSL_SIF" ] && [ "$(stat -c %s "$MSL_SIF")" = "$MSL_SIF_BYTES" ] || finish 12

nvidia-smi -L > gpus.txt
n=$(grep -c '^GPU ' gpus.txt)
[ "$n" -eq "$MSL_TP" ] && [ "$(grep -c "$EXPECT" gpus.txt)" -eq "$n" ] || { cat gpus.txt; finish 10; }
nvidia-smi topo -m > topo.txt 2>&1
bash "$MSL_HARNESS/preflight_gpu_clean.sh" gpu_clean.txt || finish 11
sha256sum "$MSL_REPLAY" | cut -d' ' -f1 > input_sha.txt
[ "$(cat input_sha.txt)" = "$MSL_REPLAY_SHA" ] || finish 13

ENVS=(--env PYTHONNOUSERSITE=1)
for v in HF_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME TMPDIR FLASHINFER_WORKSPACE_BASE SGLANG_CACHE_DIR MSL_TP; do
  ENVS+=(--env "$v=${!v}")
done
AX=("${MSL_CONTAINER_BIN:-apptainer}" exec --nv --cleanenv "${ENVS[@]}" --bind "$MSL_ROOT" "$MSL_SIF")
"${AX[@]}" python3 - > ctx.txt 2>&1 <<'EOF' || finish 14
import os, site, sys, torch, sglang
assert site.ENABLE_USER_SITE is False, "user site enabled"
assert torch.cuda.device_count() == int(os.environ["MSL_TP"]), torch.cuda.device_count()
assert not sglang.__file__.startswith("/home/"), sglang.__file__
for i in range(torch.cuda.device_count()):
    x = torch.ones(1 << 28, device=f"cuda:{i}")
    torch.cuda.synchronize(i)
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i), float(x.sum()))
print("sglang", sglang.__version__, "torch", torch.__version__)
EOF
status preflight 0

"${AX[@]}" python3 -m sglang.launch_server --model-path "$MODEL" --tp "$MSL_TP" --kv-cache-dtype bfloat16 \
  --host 127.0.0.1 --port "$MSL_PORT" --log-level info > server.log 2>&1 &
SERVER=$!
stop_server () { kill -INT "$SERVER" 2>/dev/null; for _ in $(seq 60); do kill -0 "$SERVER" 2>/dev/null || return; sleep 2; done; kill -9 "$SERVER" 2>/dev/null; }
trap stop_server EXIT
URL=http://127.0.0.1:$MSL_PORT
t0=$SECONDS
until curl -sf -m 600 "$URL/health_generate" > /dev/null; do
  kill -0 "$SERVER" 2>/dev/null || { echo "server exited"; finish 15; }
  [ $((SECONDS - t0)) -lt "$MSL_READY_TIMEOUT_S" ] || { echo "ready timeout"; finish 18; }
  sleep 5
done
echo "ready_s=$((SECONDS - t0))" > ready.txt
grep -o 'max_total_num_tokens=[0-9]*' server.log | head -1 >> ready.txt
grep -E 'as MoE runner backend|MoE runner backend:|moe_runner_backend=' server.log | head -5 > backend.txt
if [ "$EXPECT" = GB200 ] && [ -n "${MSL_EXPECT_BACKEND:-}" ]; then
  grep -q "Use $MSL_EXPECT_BACKEND as MoE runner backend" server.log || finish 16
fi
status ready 0

PY=("${AX[@]}" python3)
left () { [ -n "${MSL_END_EPOCH:-}" ] && echo $(( MSL_END_EPOCH - $(date +%s) - ${MSL_END_MARGIN_S:-120} )) || echo 999999; }
wall_stop () { echo "$TAG $1 wall_stop left_s=$(left) $(date -Is)" >> "$OUT/arm_status.txt"; finish 30; }
phase_fail () { kill -0 "$SERVER" 2>/dev/null || finish 19; finish "$1"; }
IFS=, read -ra steps <<< "$PHASES"
for step in "${steps[@]}"; do
  IFS=: read -r kind p1 p2 p3 p4 <<< "$step"
  [ "$(left)" -ge "${MSL_MIN_PHASE_S:-60}" ] || wall_stop "$step"
  if [ "$kind" = replay ] && [ -n "${MSL_END_EPOCH:-}" ]; then
    lim=$(left); [ -n "${p4:-}" ] && [ "$p4" -lt "$lim" ] && lim=$p4
    p4=$lim
  fi
  echo "$step start $(wc -l < server.log)" >> phase_lines.txt
  case "$kind" in
    replay)
      curl -sf -m 120 -X POST "$URL/flush_cache" > /dev/null || finish 17
      "${PY[@]}" "$MSL_HARNESS/replay_client.py" --url "$URL" --sessions "$MSL_REPLAY" --concurrency "$p1" \
        --start "$p2" --count "$p3" ${p4:+--deadline-s "$p4"} --out "replay_c$p1.jsonl" --summary "replay_c$p1.summary.json" > "replay_c$p1.log" 2>&1 \
        || phase_fail 20 ;;
    gsm8k)
      curl -sf -m 120 -X POST "$URL/flush_cache" > /dev/null || finish 17
      "${PY[@]}" "$MSL_HARNESS/gsm8k_client.py" --url "$URL" --train "$MSL_GSM8K_DIR/train.jsonl" \
        --test "$MSL_GSM8K_DIR/test.jsonl" --out gsm8k.jsonl --summary gsm8k.summary.json > gsm8k.log 2>&1 || phase_fail 21 ;;
    logprob)
      curl -sf -m 120 -X POST "$URL/flush_cache" > /dev/null || finish 17
      "${PY[@]}" "$MSL_HARNESS/logprob_client.py" --url "$URL" --sessions "$MSL_REPLAY" --start "$p1" --n "$p2" \
        --out logprob.jsonl > logprob.log 2>&1 || phase_fail 22 ;;
    *) echo "unknown phase $step"; finish 9 ;;
  esac
  echo "$step end $(wc -l < server.log)" >> phase_lines.txt
  if [ "$kind" = replay ] && grep -q '"cut_by_deadline": true' "replay_c$p1.summary.json" && [ -n "${MSL_END_EPOCH:-}" ]; then
    wall_stop "$step"
  fi
  status "$step" 0
done
finish 0
