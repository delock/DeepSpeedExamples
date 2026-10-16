#!/bin/bash
# Activation checkpoint offload (recipe C) — Ascend 910B4 一键启动
# 参数钉死（NPU 口径；H200 行用 8B × seq 8192-65536，本卡 29.5 GiB 装不下）：
#   Qwen3-4B / batch 1 × seq 8192 / ZeRO-2 CPU AdamW / liger FLCE + TiledMLP
#   / activation offload (CheckpointHiddenStatesOffload) / --steps 10 --warmup 1
# 用法：
#   ./run.sh                          # 模型从 HF hub 下载（只读 config，随机权重）
#   MODEL=/path/to/Qwen3-4B ./run.sh  # 本地 config
# 结果与日志写入 ./results/。
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
# 容器 CPU 配额（本机 42 核）显式钉 OpenMP 线程数；非容器环境可注释此行
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-42}"
# NPU 用 torch_npu 的 pin 路径：native 后端的设备注册（aclrtHostRegisterV2）在
# 每步 ~42MB 隐藏态张量（seq 8192）上塌方（pinned 臂 2-4× 慢于 pageable）；
# torch 后端实测正常且快于 pageable。脚本默认值仍是 native（H200 行为不变），
# 本 run.sh 显式覆盖。
export DS_PIN_MEMORY_BACKEND="${DS_PIN_MEMORY_BACKEND:-torch}"
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

python ../bench.py --model "$MODEL" --batch 1 --seq 8192 \
    --offload-optimizer --act-mem-opt \
    --steps 10 --warmup 1 \
    2>&1 | tee "results/activation_offload_${TS}.log"
