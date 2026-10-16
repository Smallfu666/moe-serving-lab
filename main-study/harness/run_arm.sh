#!/usr/bin/env bash
# One server launch of one arm, run inside a Slurm allocation. All site values come from the deployed env file.
#
# Usage: run_arm.sh <arm-tag> <model-dir> <expect-gpu: GB200|H200> <phases> <out-dir>
#   phases: comma list of  replay:<concurrency>:<start>:<count>[:<deadline-s>] | gsm8k | logprob:<start>:<n>
#           | decodeprobe:<batch>:<prompt-len>:<steps> | replayprof:<concurrency>:<start>:<count>:<deadline-s>:<delay-s>:<steps>
#   (the two profiler phases need MSL_NSYS_ARGS; see prof_client.py)
#
# Exit codes (distinct per cause, so a 0-second failure is never read as a hardware result):
#    8 cache dir outside MSL_ROOT    9 arch / env leak            10 wrong GPU type or count
#   11 GPU not clean                12 SIF missing or wrong size 13 replay input sha mismatch
#   14 CUDA context check failed    15 server process exited before ready
#   16 expected MoE backend not in log                           17 cache flush refused
#   18 server not ready within MSL_READY_TIMEOUT_S               19 server process died during a phase
#   20+ a phase client reported bad requests (20 replay, 21 gsm8k, 22 logprob); see each record's error_kind
#   Profiler phases never end the arm unless the server died (19): each window's prof_client rc goes to prof_phases.txt
#   and its report (or "missing") to nsys/phase_reports.txt, after waiting for the report file to stop growing.
#   30 stopped at the harness end time (MSL_END_EPOCH): the phase in progress was cut or the next one could not start
#   31 multi-node rank 0: a peer rank reported a nonzero exit while rank 0 waited for the server to become ready
#
# Multi-node (optional): MSL_NNODES > 1 runs one server across nodes; every node runs this script with its own
# MSL_NODE_RANK, the same MSL_DIST_ADDR (host:port of rank 0) and MSL_DONE_FILE on shared storage. Only rank 0 serves
# HTTP and runs the phases; other ranks wait for rank 0's done file. MSL_PEER_RC_GLOB (optional, rank 0) points at
# the peers' arm_rc.txt files; rank 0 stops waiting for readiness as soon as one of them is nonzero. MSL_LOCAL_GPUS is the GPU count per node
# (default MSL_TP). MSL_SERVER_ARGS (optional) is appended to the server command and recorded in server_cmd.txt.
#
# MSL_GPU_SUBSET (optional): comma list of GPU indices; the server sees only those (CUDA_VISIBLE_DEVICES), the GPU
# count check is against the list, and the clean check still covers every GPU of the node.
# MSL_NSYS_ARGS (optional): run the server under `nsys profile $MSL_NSYS_ARGS -o nsys/<tag>` inside the container.
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
finish () {
  echo "$TAG $1" >> "$OUT/arm_rc.txt"; status done "$1"
  if [ "${MSL_NNODES:-1}" -gt 1 ] && [ "${MSL_NODE_RANK:-0}" = 0 ]; then echo "$1" > "$MSL_DONE_FILE"; fi
  exit "$1"
}

for v in HF_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME TMPDIR FLASHINFER_WORKSPACE_BASE SGLANG_CACHE_DIR; do
  case "${!v:-}" in "$MSL_ROOT"/*) ;; *) echo "cache var $v=${!v:-unset} not under MSL_ROOT"; finish 8 ;; esac
done
unset CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
want_arch=$([ "$EXPECT" = GB200 ] && echo aarch64 || echo x86_64)
[ "$(uname -m)" = "$want_arch" ] || { echo "arch $(uname -m) != $want_arch"; finish 9; }
[ -s "$MSL_SIF" ] && [ "$(stat -c %s "$MSL_SIF")" = "$MSL_SIF_BYTES" ] || finish 12

nvidia-smi -L > gpus.txt
n=$(grep -c '^GPU ' gpus.txt)
LOCAL=${MSL_LOCAL_GPUS:-$MSL_TP}
if [ -n "${MSL_GPU_SUBSET:-}" ]; then
  want=$(tr ',' '\n' <<< "$MSL_GPU_SUBSET" | grep -c .)
  [ "$want" -eq "$LOCAL" ] && [ "$want" -le "$n" ] && [ "$(grep -c "$EXPECT" gpus.txt)" -eq "$n" ] || { cat gpus.txt; finish 10; }
else
  [ "$n" -eq "$LOCAL" ] && [ "$(grep -c "$EXPECT" gpus.txt)" -eq "$n" ] || { cat gpus.txt; finish 10; }
fi
nvidia-smi topo -m > topo.txt 2>&1
bash "$MSL_HARNESS/preflight_gpu_clean.sh" gpu_clean.txt || finish 11
sha256sum "$MSL_REPLAY" | cut -d' ' -f1 > input_sha.txt
[ "$(cat input_sha.txt)" = "$MSL_REPLAY_SHA" ] || finish 13

ENVS=(--env PYTHONNOUSERSITE=1)
export MSL_LOCAL_GPUS=$LOCAL
for v in HF_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME TMPDIR FLASHINFER_WORKSPACE_BASE SGLANG_CACHE_DIR MSL_LOCAL_GPUS; do
  ENVS+=(--env "$v=${!v}")
done
[ -n "${MSL_GPU_SUBSET:-}" ] && ENVS+=(--env "CUDA_VISIBLE_DEVICES=$MSL_GPU_SUBSET")
AX=("${MSL_CONTAINER_BIN:-apptainer}" exec --nv --cleanenv "${ENVS[@]}" --bind "$MSL_ROOT" "$MSL_SIF")
"${AX[@]}" python3 - > ctx.txt 2>&1 <<'EOF' || finish 14
import os, site, sys, torch, sglang
assert site.ENABLE_USER_SITE is False, "user site enabled"
assert torch.cuda.device_count() == int(os.environ["MSL_LOCAL_GPUS"]), torch.cuda.device_count()
assert not sglang.__file__.startswith("/home/"), sglang.__file__
for i in range(torch.cuda.device_count()):
    x = torch.ones(1 << 28, device=f"cuda:{i}")
    torch.cuda.synchronize(i)
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i), float(x.sum()))
print("sglang", sglang.__version__, "torch", torch.__version__)
EOF
status preflight 0

CMD=(python3 -m sglang.launch_server --model-path "$MODEL" --tp "$MSL_TP" --kv-cache-dtype bfloat16
  --host 127.0.0.1 --port "$MSL_PORT" --log-level info)
if [ "${MSL_NNODES:-1}" -gt 1 ]; then
  CMD+=(--nnodes "$MSL_NNODES" --node-rank "$MSL_NODE_RANK" --dist-init-addr "$MSL_DIST_ADDR")
fi
read -ra EXTRA <<< "${MSL_SERVER_ARGS:-}"
CMD+=("${EXTRA[@]}")
if [ -n "${MSL_NSYS_ARGS:-}" ]; then
  mkdir -p "$OUT/nsys"
  read -ra NSYS <<< "$MSL_NSYS_ARGS"
  CMD=(nsys profile "${NSYS[@]}" -o "$OUT/nsys/$TAG" --force-overwrite=true "${CMD[@]}")
  : > "$OUT/nsys/claimed.txt"
fi
printf '%q ' "${CMD[@]}" > server_cmd.txt; echo >> server_cmd.txt
"${AX[@]}" "${CMD[@]}" > server.log 2>&1 &
SERVER=$!
stop_server () { kill -INT "$SERVER" 2>/dev/null; for _ in $(seq 60); do kill -0 "$SERVER" 2>/dev/null || return; sleep 2; done; kill -9 "$SERVER" 2>/dev/null; }
trap stop_server EXIT
if [ "${MSL_NNODES:-1}" -gt 1 ] && [ "${MSL_NODE_RANK:-0}" != 0 ]; then
  status worker_started 0
  until [ -s "$MSL_DONE_FILE" ]; do
    kill -0 "$SERVER" 2>/dev/null || finish 19
    sleep 10
  done
  finish 0
fi
URL=http://127.0.0.1:$MSL_PORT
t0=$SECONDS
until curl -sf -m 600 "$URL/health_generate" > /dev/null; do
  kill -0 "$SERVER" 2>/dev/null || { echo "server exited"; finish 15; }
  if [ -n "${MSL_PEER_RC_GLOB:-}" ] && cat $MSL_PEER_RC_GLOB 2>/dev/null | awk '$2 != 0 {f=1} END {exit !f}'; then
    echo "peer rank failed"; finish 31
  fi
  [ $((SECONDS - t0)) -lt "$MSL_READY_TIMEOUT_S" ] || { echo "ready timeout"; finish 18; }
  sleep 5
done
echo "ready_s=$((SECONDS - t0))" > ready.txt
grep -o 'max_total_num_tokens=[0-9]*' server.log | head -1 >> ready.txt
grep -oE "as MoE runner backend.*|MoE runner backend:.*|'moe_runner_backend': '[^']*'|moe_runner/triton_utils/configs/[^ ]*" server.log | head -5 > backend.txt
if [ "$EXPECT" = GB200 ] && [ -n "${MSL_EXPECT_BACKEND:-}" ]; then
  grep -q "Use $MSL_EXPECT_BACKEND as MoE runner backend" server.log || finish 16
fi
status ready 0

PY=("${AX[@]}" python3)
left () { [ -n "${MSL_END_EPOCH:-}" ] && echo $(( MSL_END_EPOCH - $(date +%s) - ${MSL_END_MARGIN_S:-120} )) || echo 999999; }
wall_stop () { echo "$TAG $1 wall_stop left_s=$(left) $(date -Is)" >> "$OUT/arm_status.txt"; finish 30; }
phase_fail () { kill -0 "$SERVER" 2>/dev/null || finish 19; finish "$1"; }
cap () { local l; l=$(left); [ "$l" -lt "$1" ] && echo "$l" || echo "$1"; }
flush_wait () {
  local t0=$SECONDS lim
  lim=$(cap 120)
  until curl -sf -m 30 -X POST "$URL/flush_cache" > /dev/null; do
    [ $((SECONDS - t0)) -lt "$lim" ] || return 1
    sleep 3
  done
}
prof_done () {  # <step> <rc> <client-json>
  echo "$1 rc=$2 $(date -Is)" >> "$OUT/prof_phases.txt"
  kill -0 "$SERVER" 2>/dev/null || finish 19
  case "$2" in 5*) curl -s -m 60 -X POST "$URL/stop_profile" > /dev/null ;; esac
  [ -n "${MSL_NSYS_ARGS:-}" ] || return 0
  # No client JSON means the client died at an unknown point, possibly after /start_profile succeeded, so the
  # report may still come and must be claimed here, not by the next window.
  local note=""
  if [ "$3" != none ] && [ ! -s "$3" ]; then
    note=" client_json_missing"
  elif ! grep -qE '"start_profile_status": *200[,}]' "$3" 2>/dev/null; then
    echo "$1 missing wait_s=0 no_window_opened" >> "$OUT/nsys/phase_reports.txt"
    return 0
  fi
  local t0=$SECONDS lim seen="" new s1
  lim=$(cap 300)
  while [ $((SECONDS - t0)) -lt "$lim" ]; do
    new=$(ls -1 "$OUT"/nsys/*.nsys-rep 2>/dev/null | grep -vxF -f "$OUT/nsys/claimed.txt" | head -1)
    if [ -n "$new" ]; then
      s1=$(stat -c %s "$new"); sleep 10
      [ "$(stat -c %s "$new")" = "$s1" ] && { seen=$new; break; }
    else
      sleep 5
    fi
  done
  echo "$1 ${seen:-missing} wait_s=$((SECONDS - t0))$note" >> "$OUT/nsys/phase_reports.txt"
  [ -z "$seen" ] || echo "$seen" >> "$OUT/nsys/claimed.txt"
}
IFS=, read -ra steps <<< "$PHASES"
for step in "${steps[@]}"; do
  IFS=: read -r kind p1 p2 p3 p4 p5 p6 <<< "$step"
  [ "$(left)" -ge "${MSL_MIN_PHASE_S:-60}" ] || wall_stop "$step"
  by_end=0
  if [ "$kind" = replay ] && [ -n "${MSL_END_EPOCH:-}" ]; then
    lim=$(left); by_end=1
    if [ -n "${p4:-}" ] && [ "$p4" -lt "$lim" ]; then lim=$p4; by_end=0; fi
    p4=$lim
  fi
  if [ "$kind" = replayprof ]; then
    lim=$(left)
    [ "$p4" -lt "$lim" ] || p4=$lim
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
    decodeprobe)
      flush_wait || { prof_done "$step" flush_failed none; continue; }
      lim=$(cap 900)
      "${PY[@]}" "$MSL_HARNESS/prof_client.py" decode --url "$URL" --server-log "$OUT/server.log" --tag "${TAG}_b$p1" \
        --steps "$p3" --sessions "$MSL_REPLAY" --start "${MSL_PROBE_START:-200}" --batch "$p1" --prompt-len "$p2" \
        --warm-timeout "$(cap 600)" --window-timeout "$(cap 300)" --request-timeout "$lim" \
        --out "probe_b$p1.json" > "probe_b$p1.log" 2>&1
      prof_done "$step" $? "probe_b$p1.json" ;;
    replayprof)
      flush_wait || { prof_done "$step" flush_failed none; continue; }
      "${PY[@]}" "$MSL_HARNESS/replay_client.py" --url "$URL" --sessions "$MSL_REPLAY" --concurrency "$p1" \
        --start "$p2" --count "$p3" --deadline-s "$p4" --out "replayprof_c$p1.jsonl" \
        --summary "replayprof_c$p1.summary.json" > "replayprof_c$p1.log" 2>&1 &
      rp=$!
      "${PY[@]}" "$MSL_HARNESS/prof_client.py" trigger --url "$URL" --server-log "$OUT/server.log" \
        --tag "${TAG}_rp$p1" --steps "$p6" --delay "$p5" --window-timeout "$(cap 300)" \
        --out "trigger_c$p1.json" > "trigger_c$p1.log" 2>&1
      trc=$?
      wait "$rp"; rrc=$?
      prof_done "$step" "$trc/replay=$rrc" "trigger_c$p1.json" ;;
    *) echo "unknown phase $step"; finish 9 ;;
  esac
  echo "$step end $(wc -l < server.log)" >> phase_lines.txt
  if [ "$by_end" = 1 ] && grep -q '"cut_by_deadline": true' "replay_c$p1.summary.json"; then
    wall_stop "$step"
  fi
  status "$step" 0
done
finish 0
