#!/bin/bash
# Saved-activation offload — H200 one-click run
# Parameters pinned for this card (~139.8 GiB usable):
#   Qwen3-8B / batch 1 x seq 4096 / ZeRO-2 CPU AdamW optimizer offload
# Usage:
#   ./run.sh                          # model from the HF hub
#   MODEL=/path/to/Qwen3-8B ./run.sh  # local checkpoint
# Results and logs go to ./results/.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ulimit -l unlimited 2>/dev/null || true  # native pin mlocks host memory

# $REPO is the sfc_ds_pin_memory workspace root, five levels above this
# script; it must be on PYTHONPATH for the benchmark's own package imports.
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# conda environment (torch 2.14 + C++20 flash-attn, see ../README.md);
# skipped when already activated. Override CONDA_ENV / CONDA_SH.
CONDA_ENV="${CONDA_ENV:-torch214}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

mkdir -p results
TS="$(date +%F_%H%M%S)"

python ../bench.py --model "$MODEL" --batch 1 --seq 4096 \
    --offload-optimizer \
    --manifest-out "results/qwen3-8b-seq4096-manifest.json" \
    2>&1 | tee "results/saved_activation_${TS}.log"
