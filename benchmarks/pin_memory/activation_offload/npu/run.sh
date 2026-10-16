#!/bin/bash
# Activation checkpoint offload (recipe C) — Ascend 910B4 one-click run
# Parameters pinned for this card (29.5 GiB usable):
#   Qwen3-4B / batch 1 x seq 8192 / ZeRO-2 CPU AdamW / Liger fused linear CE
#   + TiledMLP / activation offload (CheckpointHiddenStatesOffload)
#   / --steps 10 --warmup 1
# Usage:
#   ./run.sh                          # model from the HF hub (config only, random weights)
#   MODEL=/path/to/Qwen3-4B ./run.sh  # local config
# Results and logs go to ./results/.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
# Pin OpenMP threads to the container CPU quota (42 cores on this host);
# comment out on non-container setups.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-42}"
# The torch backend measured as the faster pin path on this NPU host; this
# script selects it explicitly (bench.py uses setdefault, default native).
export DS_PIN_MEMORY_BACKEND="${DS_PIN_MEMORY_BACKEND:-torch}"
ulimit -l unlimited 2>/dev/null || true

# conda environment and CANN toolkit; skipped when already activated.
# Override CONDA_ENV / CONDA_SH / CANN_SET_ENV for different setups.
CONDA_ENV="${CONDA_ENV:-cann}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CANN_SET_ENV="${CANN_SET_ENV:-$HOME/Ascend/cann-8.5.1/set_env.sh}"
if [ -f "$CONDA_SH" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi
if [ -f "$CANN_SET_ENV" ]; then
    # shellcheck disable=SC1091
    source "$CANN_SET_ENV"
fi

mkdir -p results
TS="$(date +%F_%H%M%S)"

python ../bench.py --model "$MODEL" --batch 1 --seq 8192 \
    --offload-optimizer --act-mem-opt \
    --steps 10 --warmup 1 \
    2>&1 | tee "results/activation_offload_${TS}.log"
