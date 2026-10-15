# Copy to env.sh outside the repository and fill in at deployment time.
# Slurm reads SBATCH_ACCOUNT and SBATCH_PARTITION from the environment at submit time.
export SBATCH_ACCOUNT="<project-account>"
export SBATCH_PARTITION="<partition>"
export MSL_EXCLUDE_NODES="<comma-separated nodes, may be empty>"

export MSL_ROOT="<scratch-root>/moe-serving-lab"
export MSL_IMAGE_AMD64="<image.sif or digest>"
export MSL_IMAGE_ARM64="<image.sif or digest>"
export MSL_MODEL_FP8="<path/to/Qwen3-235B-A22B-Instruct-2507-FP8>"
export MSL_MODEL_NVFP4="<path/to/Qwen3-235B-A22B-Instruct-2507-NVFP4>"
export HF_HOME="$MSL_ROOT/hf"

export MSL_CLEAN_MIB=2000
export MSL_SEED="<seed>"
