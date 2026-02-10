#!/bin/bash
# =============================================================================
# 节点 2 专用：Qwen3-4B MATH-Lighteval，mask 比例消融 (0.1, 0.2, 0.5, 0.9)
# 每任务 4 GPU。GPUs 0-3 与 4-7 并行，2 轮完成。nohup ./run_batch_4b_math_mask_sweep.sh > log_4b_sweep.log 2>&1 &
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export N_GPUS=4

# 共用配置
export USE_PRETRAINED_CRITIC=no
export FREEZE_CRITIC=no
export FLIP_ADV_WHEN_MASKED=false

run_one() {
    local GPUS=$1
    local RATIO=$2
    export CUDA_VISIBLE_DEVICES=$GPUS
    export RAY_TMPDIR="/tmp/hyl_ray_4b_${GPUS//,/_}"
    mkdir -p "$RAY_TMPDIR"
    export REWARD_MASK_RATIO=$RATIO
    bash ./run_qwen3-4b_ppo_math_lighteval_test.sh
}

echo "=========================================="
echo "Qwen3-4B MATH mask sweep 开始: $(date)"
echo "  0-3 与 4-7 并行，2 轮完成"
echo "  Mask: 0.1, 0.2, 0.5, 0.9"
echo "=========================================="

# 第 1 轮：0.1 (0-3) || 0.2 (4-7)
echo "=== 第 1 轮: mask 0.1 (GPU 0-3) | mask 0.2 (GPU 4-7) ==="
run_one "0,1,2,3" 0.1 &
PID1=$!
run_one "4,5,6,7" 0.2 &
PID2=$!
wait $PID1 $PID2

# 第 2 轮：0.5 (0-3) || 0.9 (4-7)
echo "=== 第 2 轮: mask 0.5 (GPU 0-3) | mask 0.9 (GPU 4-7) ==="
run_one "0,1,2,3" 0.5 &
PID1=$!
run_one "4,5,6,7" 0.9 &
PID2=$!
wait $PID1 $PID2

echo "=========================================="
echo "Qwen3-4B MATH mask sweep 全部完成: $(date)"
echo "=========================================="
