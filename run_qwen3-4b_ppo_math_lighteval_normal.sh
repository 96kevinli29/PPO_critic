#!/bin/bash
# Qwen3-4B PPO with MATH-Lighteval (Rule-based Reward)
# 支持 4 或 8 GPU，通过 N_GPUS 和 CUDA_VISIBLE_DEVICES 配置
# 正常训练，输出 critic 供后续 freeze critic 等实验

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
cd "$SCRIPT_DIR"

# wandb 缓存放到数据盘，避免占满根分区
export WANDB_CACHE_DIR="${SCRIPT_DIR}/.wandb_cache"
mkdir -p "$WANDB_CACHE_DIR"
export WANDB_API_KEY="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"

# 模型路径 (Actor 和 Critic 都用 Qwen3-4B)
MODEL_PATH="/data_storage/lixiao/research_proj_xiao/hyl/Qwen3-4B"

# 训练结束后 Critic 导出路径（供后续 freeze critic / reward mask 等实验加载）
CRITIC_OUTPUT_PATH="${SCRIPT_DIR}/critic_qwen3_4b_ppo_math_lighteval"

# MATH-Lighteval 数据集路径 (使用本地数据)
DATA_DIR="/data_storage/lixiao/research_proj_xiao/hyl/data"
math_train_path="${DATA_DIR}/math_lighteval/train.parquet"
math_test_path="${DATA_DIR}/math_lighteval/test.parquet"

train_files="['$math_train_path']"
test_files="['$math_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    '+data.apply_chat_template_kwargs={enable_thinking: true}' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
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
    trainer.log_val_generations=50 \
    trainer.validation_data_dir='/data_storage/lixiao/research_proj_xiao/hyl/outputs/validation' \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='PPO-Project2026' \
    trainer.experiment_name='qwen3_4b_ppo_math_lighteval' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="${SCRIPT_DIR}/checkpoints/PPO-Project2026/qwen3_4b_ppo_math_lighteval" \
    +trainer.save_critic_model=True \
    +trainer.critic_model_output_path="$CRITIC_OUTPUT_PATH" $@

PYTHON_EXIT=$?

# 训练成功后，导出 Critic 并自动转为 HF 格式
if [ $PYTHON_EXIT -eq 0 ]; then
    CKPT_DIR="${SCRIPT_DIR}/checkpoints/PPO-Project2026/qwen3_4b_ppo_math_lighteval"
    if [ -d "$CKPT_DIR" ]; then
        LATEST=$(ls -d "${CKPT_DIR}"/global_step_* 2>/dev/null | sort -V | tail -1)
        if [ -n "$LATEST" ] && [ -d "${LATEST}/critic" ]; then
            echo "=========================================="
            echo "导出 Critic (checkpoint 格式) 到: $CRITIC_OUTPUT_PATH"
            rm -rf "$CRITIC_OUTPUT_PATH"
            cp -r "${LATEST}/critic" "$CRITIC_OUTPUT_PATH"
            if [ -f "${CRITIC_OUTPUT_PATH}/huggingface/config.json" ] && [ ! -f "${CRITIC_OUTPUT_PATH}/config.json" ]; then
                cp "${CRITIC_OUTPUT_PATH}/huggingface/config.json" "${CRITIC_OUTPUT_PATH}/config.json"
            fi
            echo "Critic 导出完成，正在转为 HF 格式..."
            python3 scripts/convert_critic_to_hf.py --local_dir "$CRITIC_OUTPUT_PATH" --target_dir "${CRITIC_OUTPUT_PATH}_hf" --trust-remote-code
            if [ $? -eq 0 ]; then
                echo "HF 格式已保存到: ${CRITIC_OUTPUT_PATH}_hf"
                echo "后续实验: critic.model.path=${CRITIC_OUTPUT_PATH}_hf"
            else
                echo "HF 转换失败，请手动执行: python3 scripts/convert_critic_to_hf.py --local_dir $CRITIC_OUTPUT_PATH --target_dir ${CRITIC_OUTPUT_PATH}_hf --trust-remote-code"
            fi
            echo "=========================================="
        fi
    fi
fi
