#!/bin/bash
# Qwen3-4B PPO with MATH-Lighteval (Rule-based Reward)
# 支持 4 或 8 GPU，通过 N_GPUS 和 CUDA_VISIBLE_DEVICES 配置
# 新增功能: Reward Mask + Freeze Critic
# 配置参数: REWARD_MASK_RATIO, FLIP_ADV_WHEN_MASKED, FREEZE_CRITIC,
#           USE_PRETRAINED_CRITIC (yes/no), CRITIC_MODEL_PATH, N_GPUS,
#           REWARD_SOURCE (rule_based | critic)

set -x

# GPU 数量（需与 CUDA_VISIBLE_DEVICES 中的卡数一致）
N_GPUS=${N_GPUS:-8}
[ "$N_GPUS" = "8" ] && TRAIN_BATCH=128 && PPO_MINI=32 || TRAIN_BATCH=64 && PPO_MINI=16

# vLLM 配置
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Ray 临时目录配置（使用短路径软链接，解决socket路径长度限制和磁盘空间问题）
export RAY_TMPDIR=/tmp/hyl_ray
mkdir -p /data_storage/lixiao/research_proj_xiao/hyl/ray_tmp

# ========== 训练时间记录 ==========
START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
echo "=========================================="
echo "训练开始时间: $START_TIME_STR"
echo "=========================================="

# 训练结束时记录时间的函数
log_training_time() {
    END_TIME=$(date +%s)
    END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))
    echo "=========================================="
    echo "训练结束时间: $END_TIME_STR"
    echo "总训练时长: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    echo "=========================================="
}
trap log_training_time EXIT

# 设置 Python 共享库路径
export LD_LIBRARY_PATH="/data_storage/lixiao/research_proj_xiao/hyl/miniconda3/envs/verl-ppo/lib:$LD_LIBRARY_PATH"

# 禁用 expandable_segments (与 vLLM memory pool 不兼容)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# 使用 SDPA 加速 (系统 GLIBC 2.31 不支持 flash-attn，改用 PyTorch 内置的 SDPA)
export VERL_ATTN_IMPLEMENTATION=sdpa

# 指定使用的 GPU（8 卡默认 0-7，需与 N_GPUS 一致）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# wandb 缓存放到数据盘，避免占满根分区
export WANDB_CACHE_DIR="${SCRIPT_DIR}/.wandb_cache"
mkdir -p "$WANDB_CACHE_DIR"
export WANDB_API_KEY="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"

# ========== 模型路径配置 ==========
ACTOR_MODEL_PATH="/data_storage/lixiao/research_proj_xiao/hyl/Qwen3-4B"

# Critic 模型（需先跑 run_qwen3-4b_ppo_math_lighteval_normal.sh 训练并转为 HF）
BASE_CRITIC_PATH="/data_storage/lixiao/research_proj_xiao/hyl/Qwen3-4B"
PRETRAINED_CRITIC_PATH="/data_storage/lixiao/research_proj_xiao/hyl/critic_qwen3_4b_ppo_math_lighteval_hf"
USE_PRETRAINED_CRITIC=${USE_PRETRAINED_CRITIC:-yes}
if [[ "$USE_PRETRAINED_CRITIC" == "yes" ]] && [[ -z "${CRITIC_MODEL_PATH+x}" ]]; then
    CRITIC_MODEL_PATH="$PRETRAINED_CRITIC_PATH"
else
    CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-$BASE_CRITIC_PATH}"
fi

# ========== 实验配置 ==========
# Reward 来源: rule_based (规则/reward_fn) | critic (critic value)
REWARD_SOURCE=${REWARD_SOURCE:-rule_based}
REWARD_MASK_RATIO=${REWARD_MASK_RATIO:-1.0}
FREEZE_CRITIC=${FREEZE_CRITIC:-true}
FLIP_ADV_WHEN_MASKED=${FLIP_ADV_WHEN_MASKED:-true}

# 实验名称后缀
CRITIC_BASENAME=$(basename "$CRITIC_MODEL_PATH")
if [[ "$CRITIC_BASENAME" == critic_* ]]; then
    C_TAG="cpt"
else
    C_TAG="cb"
fi
F_TAG=$([ "$FREEZE_CRITIC" == "true" ] && echo "fr" || echo "tr")
RS_TAG=$([ "$REWARD_SOURCE" == "critic" ] && echo "rs_critic" || echo "rs_rule")
EXPERIMENT_SUFFIX="_${RS_TAG}_m${REWARD_MASK_RATIO}_${C_TAG}_${F_TAG}"
[ "$REWARD_MASK_RATIO" != "0.0" ] && EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX}_$([ "$FLIP_ADV_WHEN_MASKED" == "true" ] && echo "f" || echo "nf")"

# MATH-Lighteval 数据集路径
DATA_DIR="/data_storage/lixiao/research_proj_xiao/hyl/data"
math_train_path="${DATA_DIR}/math_lighteval/train.parquet"
math_test_path="${DATA_DIR}/math_lighteval/test.parquet"

train_files="['$math_train_path']"
test_files="['$math_test_path']"

echo "=========================================="
echo "实验配置:"
echo "  Actor: $ACTOR_MODEL_PATH"
echo "  Critic: $CRITIC_MODEL_PATH | 预训练: $C_TAG | 冻结: $F_TAG"
echo "  RewardSource: $REWARD_SOURCE | Mask: $REWARD_MASK_RATIO | Flip: $FLIP_ADV_WHEN_MASKED"
echo "  实验名: qwen3_4b_ppo_math_lighteval${EXPERIMENT_SUFFIX}"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    +algorithm.reward_source=$REWARD_SOURCE \
    +algorithm.reward_mask_ratio=$REWARD_MASK_RATIO \
    +algorithm.reward_mask_flip_adv_when_masked=$FLIP_ADV_WHEN_MASKED \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    '+data.apply_chat_template_kwargs={enable_thinking: true}' \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.path="$CRITIC_MODEL_PATH" \
    +critic.model.override_config.attn_implementation=sdpa \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.freeze=$FREEZE_CRITIC \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.log_val_generations=50 \
    trainer.validation_data_dir='/data_storage/lixiao/research_proj_xiao/hyl/outputs/validation' \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='PPO-Project2026' \
    trainer.experiment_name="qwen3_4b_ppo_math_lighteval${EXPERIMENT_SUFFIX}" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 $@
