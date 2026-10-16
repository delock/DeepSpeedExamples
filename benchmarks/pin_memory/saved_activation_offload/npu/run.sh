#!/bin/bash
# Saved-activation offload — Ascend 910B4 一键启动
# 参数钉死（NPU 口径，H200 的 8B×seq65536 在本卡 29.5 GiB 上不可行）：
#   Qwen3-4B / batch 1 × seq 2048 / K=3 / --credits 1 / ZeRO-2 CPU AdamW
# 用法：
#   ./run.sh                          # 模型从 HF hub 下载
#   MODEL=/path/to/Qwen3-4B ./run.sh  # 本地权重
# 结果与日志写入 ./results/。
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
# 容器 CPU 配额（本机 42 核）显式钉 OpenMP 线程数；非容器环境可注释此行
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-42}"
ulimit -l unlimited 2>/dev/null || true

# conda 环境与 CANN 工具链（已激活的环境可跳过）
if [ -f ~/miniconda3/etc/profile.d/conda.sh ] && [ -z "${CONDA_PREFIX:-}" ]; then
    # shellcheck disable=SC1091
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate cann
fi
if [ -f ~/Ascend/cann-8.5.1/set_env.sh ]; then
    # shellcheck disable=SC1091
    source ~/Ascend/cann-8.5.1/set_env.sh
fi

mkdir -p results
TS="$(date +%F_%H%M%S)"

python ../bench.py --model "$MODEL" --batch 1 --seq 2048 \
    --offload-tensors 3 --offload-optimizer --act-mem-opt \
    --steps 20 --warmup 5 --credits 1 \
    --manifest-out results/qwen3-4b-seq2048-k3-manifest.json \
    2>&1 | tee "results/saved_activation_${TS}.log"
