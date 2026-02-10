#!/bin/bash
# Qwen3-1.7B PPO with GSM8K (Rule-based Reward)
# 使用 4 GPU, rule-based reward (不需要外部 Reward Model)

set -x

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

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 指定使用的 GPU (4张卡)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# wandb 缓存放到数据盘，避免占满根分区
export WANDB_CACHE_DIR="${SCRIPT_DIR}/.wandb_cache"
mkdir -p "$WANDB_CACHE_DIR"

# 模型路径
MODEL_PATH="/data_storage/lixiao/research_proj_xiao/hyl/Qwen3-1.7B"

# GSM8K 数据集路径 (使用本地数据)
DATA_DIR="/data_storage/lixiao/research_proj_xiao/hyl/data"
gsm8k_train_path="${DATA_DIR}/gsm8k/train.parquet"
gsm8k_test_path="${DATA_DIR}/gsm8k/test.parquet"

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    '+data.apply_chat_template_kwargs={enable_thinking: false}' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.path="$MODEL_PATH" \
    +critic.model.override_config.attn_implementation=sdpa \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.log_val_generations=10000 \
    trainer.validation_data_dir='/data_storage/lixiao/research_proj_xiao/hyl/outputs/validation' \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='PPO-Project2026' \
    trainer.experiment_name='qwen3_1.7b_ppo_gsm8k' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@
