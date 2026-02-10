#!/bin/bash
# Qwen3-0.6B PPO with MATH-Lighteval (Rule-based Reward) - GPU 0,1,2,3
# 使用 4 GPU, rule-based reward
# 新增功能: Reward Mask + Freeze Critic（与 gsm8k_test 对齐）
#
# 配置参数: REWARD_MASK_RATIO, FLIP_ADV_WHEN_MASKED, FREEZE_CRITIC,
#           USE_PRETRAINED_CRITIC (yes/no), CRITIC_MODEL_PATH,
#           REWARD_SOURCE (rule_based | critic)

set -x

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

# 指定使用的 GPU (4张卡: 0,1,2,3)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# wandb 缓存放到数据盘，避免占满根分区
export WANDB_CACHE_DIR="${SCRIPT_DIR}/.wandb_cache"
mkdir -p "$WANDB_CACHE_DIR"
# 让 Ray worker（含副节点）也能使用 wandb
export WANDB_API_KEY="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"

# ========== 模型路径配置 ==========
# Actor 模型
ACTOR_MODEL_PATH="/data_storage/lixiao/research_proj_xiao/hyl/Qwen3-0.6B"

# Critic 模型（默认使用预训练 critic，可通过 USE_PRETRAINED_CRITIC 或 CRITIC_MODEL_PATH 覆盖）
BASE_CRITIC_PATH="/data_storage/lixiao/research_proj_xiao/hyl/Qwen3-0.6B"
PRETRAINED_CRITIC_PATH="/data_storage/lixiao/research_proj_xiao/hyl/critic_qwen3_0.6b_ppo_math_lighteval_hf"
USE_PRETRAINED_CRITIC=${USE_PRETRAINED_CRITIC:-yes}
if [[ "$USE_PRETRAINED_CRITIC" == "yes" ]] && [[ -z "${CRITIC_MODEL_PATH+x}" ]]; then
    CRITIC_MODEL_PATH="$PRETRAINED_CRITIC_PATH"
else
    CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-$BASE_CRITIC_PATH}"
fi

# ========== 实验配置 ==========
# Reward 来源: rule_based (规则/reward_fn) | critic (critic value)
REWARD_SOURCE=${REWARD_SOURCE:-rule_based}

# Reward Mask 比例 (0.0 = 不 mask, 0.5 = 50%，1.0 = 全部 mask)
REWARD_MASK_RATIO=${REWARD_MASK_RATIO:-1.0}

# 是否冻结 Critic
FREEZE_CRITIC=${FREEZE_CRITIC:-true}

# 是否对被 mask 轨迹翻转 advantage
FLIP_ADV_WHEN_MASKED=${FLIP_ADV_WHEN_MASKED:-true}

# 实验名称后缀 (便于 wandb 对比)
# m=mask比例(0~1) cpt=预训练critic cb=base critic fr=冻结 tr=可训练 f=翻转 nf=不翻转
# 例: _m1_cpt_fr_f = mask1.0 + 预训练 + 冻结 + 翻转; _m1_cpt_fr_nf = 不翻转; _m0_cpt_fr = 无mask
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

# MATH-Lighteval 数据集路径 (使用本地数据)
DATA_DIR="/data_storage/lixiao/research_proj_xiao/hyl/data"
math_train_path="${DATA_DIR}/math_lighteval/train.parquet"
math_test_path="${DATA_DIR}/math_lighteval/test.parquet"

train_files="['$math_train_path']"
test_files="['$math_test_path']"

echo "=========================================="
echo "实验配置 (便于对比):"
echo "  Actor: $ACTOR_MODEL_PATH"
echo "  Critic: $CRITIC_MODEL_PATH | 预训练: $C_TAG | 冻结: $F_TAG | USE_PRETRAINED_CRITIC: $USE_PRETRAINED_CRITIC"
echo "  RewardSource: $REWARD_SOURCE | Mask: $REWARD_MASK_RATIO | Flip: $FLIP_ADV_WHEN_MASKED"
echo "  实验名: qwen3_0.6b_ppo_math_lighteval${EXPERIMENT_SUFFIX}"
echo "  [m=mask cpt/cb=预训练/基座 fr/tr=冻结/可训 f/nf=翻转/不翻转]"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    +algorithm.reward_source=$REWARD_SOURCE \
    +algorithm.reward_mask_ratio=$REWARD_MASK_RATIO \
    +algorithm.reward_mask_flip_adv_when_masked=$FLIP_ADV_WHEN_MASKED \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    '+data.apply_chat_template_kwargs={enable_thinking: true}' \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.path="$CRITIC_MODEL_PATH" \
    +critic.model.override_config.attn_implementation=sdpa \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.freeze=$FREEZE_CRITIC \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.log_val_generations=50 \
    trainer.validation_data_dir='/data_storage/lixiao/research_proj_xiao/hyl/outputs/validation' \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="PPO-Project2026" \
    trainer.experiment_name="qwen3_0.6b_ppo_math_lighteval${EXPERIMENT_SUFFIX}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 $@
