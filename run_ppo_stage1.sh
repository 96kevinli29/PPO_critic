#!/bin/bash
# =============================================================================
# 阶段 1：基线 — 标准 PPO（Actor + Critic 同时训练，与直接 run_ppo.sh 逻辑一致）
#
# 默认 checkpoint / W&B 实验名 qwen3_4b_RL1 → $OUTPUTS/ppo/qwen3_4b_RL1/（SAVE_FREQ 默认 50）；阶段 2 为控制变量对照（默认 qwen3_4b_RL2）
# 每个 global_step 下含 actor/、critic/；终端会打印 model_slug（来自 MODEL_ROOT 目录名）便于区分模型
#
# 与 run_ppo.sh 的唯一区别：日志前缀 HYL_RUN_LOG_PREFIX=stage1（可覆盖）。
#
# 用法示例：
#   ./run_ppo_stage1.sh
#   MODEL_ROOT=/path/to/MyModel-HF SAVE_FREQ=100 ./run_ppo_stage1.sh
#
# Critic 供阶段 2 用时，自行转 HF（建议与 run_ppo.sh 中 HF_CRITIC_DIR 一致，默认项目根 qwen3-4b-critic）：
#   python scripts/convert_critic_to_hf.py --local_dir <.../global_step_N/critic> --target_dir "$HYLBASE/qwen3-4b-critic" --trust-remote-code
# Actor 若需 HuggingFace：verl.model_merger merge --backend fsdp --local_dir <.../global_step_N/actor> --target_dir ...
#
# 阶段 2：run_ppo_stage2.sh（预训练 HF critic + 冻结 + 稀疏 reward 等）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HYL_RUN_LOG_PREFIX="${HYL_RUN_LOG_PREFIX:-stage1}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_RL1}"

echo "============================================"
echo "  PPO 阶段 1：标准 PPO（Actor+Critic，checkpoint 含 actor/ 与 critic/）"
echo "  log_prefix=$HYL_RUN_LOG_PREFIX  （SAVE_FREQ 等见 run_ppo.sh，默认 50）"
echo "============================================"

exec "${SCRIPT_DIR}/run_ppo.sh" "$@"
