#!/bin/bash
# =============================================================================
# 阶段 2：控制变量对照（在阶段 1 基线之上，固定 Actor/数据等，只改 Critic 与 mask 等条件）
#
# 与阶段 1 同数据/SFT；差异：预训练 critic、冻结、mask 等。
#
# Actor：默认与阶段 1 相同起训，无需接阶段 1 的 actor checkpoint（除非你要显式续训 policy）。
#
# 前置条件：阶段 1 后已把 critic 转为 HF，默认加载项目根下 qwen3-4b-critic（与 run_ppo.sh 的 HF_CRITIC_DIR 一致）
#   也可用 CRITIC_MODEL_PATH= 指向其它 HF critic 目录
#
# 与直接调 run_ppo.sh 的区别：
#   USE_PRETRAINED_CRITIC = yes   ← 使用阶段 1 训练好的 critic
#   FREEZE_CRITIC         = true  ← 冻结 critic（只用不更新）
#   REWARD_MASK_RATIO     = 1.0  ← 默认 100% mask（选中轨迹整条 reward 置 0）；可改 0~1 做部分 mask
#   EXPERIMENT_NAME       未指定：qwen3_4b_RL2_m<mask>（默认 m1.0）
#
# 用法示例：
#   ./run_ppo_stage2.sh
#   REWARD_MASK_RATIO=0.5 ./run_ppo_stage2.sh
#   REWARD_MASK_RATIO=0 ./run_ppo_stage2.sh   # 与阶段 1 同：不 mask
#   CRITIC_MODEL_PATH=/path/to/my_critic_hf ./run_ppo_stage2.sh
#
# 所有 run_ppo.sh 支持的参数均可透传，详见 run_ppo.sh 头部参数表。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA="${DATA:-dapo_math}"
export USE_PRETRAINED_CRITIC="${USE_PRETRAINED_CRITIC:-yes}"
export FREEZE_CRITIC="${FREEZE_CRITIC:-true}"
export REWARD_MASK_TYPE="${REWARD_MASK_TYPE:-fixed_ratio}"
export REWARD_MASK_RATIO="${REWARD_MASK_RATIO:-1.0}"
export HYL_RUN_LOG_PREFIX="${HYL_RUN_LOG_PREFIX:-stage2}"

# 未指定 EXPERIMENT_NAME → qwen3_4b_RL2_m<mask>
if [[ -z "${EXPERIMENT_NAME:-}" ]]; then
  _hyl_mask="${REWARD_MASK_RATIO:-1.0}"
  [[ "$_hyl_mask" == "1" ]] && _hyl_mask="1.0"
  export EXPERIMENT_NAME="qwen3_4b_RL2_m${_hyl_mask}"
  unset _hyl_mask
else
  export EXPERIMENT_NAME
fi

echo "============================================"
echo "  PPO 阶段 2：控制变量对照（预训练 Critic + 冻结 + mask）"
echo "  USE_PRETRAINED_CRITIC=$USE_PRETRAINED_CRITIC"
echo "  FREEZE_CRITIC=$FREEZE_CRITIC  REWARD_MASK_RATIO=$REWARD_MASK_RATIO  REWARD_MASK_TYPE=$REWARD_MASK_TYPE"
echo "  EXPERIMENT_NAME=$EXPERIMENT_NAME  log_prefix=$HYL_RUN_LOG_PREFIX"
echo "============================================"

exec "${SCRIPT_DIR}/run_ppo.sh" "$@"
