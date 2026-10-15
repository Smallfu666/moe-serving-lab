# Copy outside the repository and fill in at deployment time.
# Slurm reads SBATCH_ACCOUNT and SBATCH_PARTITION from the environment at submit time.
export SBATCH_ACCOUNT="<project-account>"
export SBATCH_PARTITION="<partition>"
export MSL_EXCLUDE_NODES="<comma-separated nodes, may be empty>"

export MSL_ROOT="<scratch-root>/msl"
export MSL_HARNESS="$MSL_ROOT/harness"
export MSL_SIF="$MSL_ROOT/containers/<sglang image for this node's architecture>.sif"
export MSL_REPLAY="$MSL_ROOT/inputs/replay.jsonl"
export MSL_REPLAY_SHA="<sha256 of replay.jsonl>"
export MSL_GSM8K_DIR="$MSL_ROOT/inputs/gsm8k"
export MSL_TP=2
export MSL_EXPECT_BACKEND="flashinfer_trtllm"   # SM100 arms only

# Every cache stays under MSL_ROOT, one set per job so concurrent jobs never share one.
export MSL_JOB_CACHE="$MSL_ROOT/cache/job_${SLURM_JOB_ID:-nojob}"
export HF_HOME="$MSL_ROOT/hf_home"
export TRITON_CACHE_DIR="$MSL_JOB_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$MSL_JOB_CACHE/inductor"
export XDG_CACHE_HOME="$MSL_JOB_CACHE/xdg"
export FLASHINFER_WORKSPACE_BASE="$MSL_JOB_CACHE/flashinfer"
export SGLANG_CACHE_DIR="$MSL_JOB_CACHE/sglang"
export TMPDIR="$MSL_JOB_CACHE/tmp"
