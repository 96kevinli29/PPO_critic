#!/bin/bash
# =============================================================================
# 统一 Qwen3 PPO 入口：改参数即可切换数据集、模型路径、RM、二值化、mask、冻结等
# 便于做对照试验（只改参数，不碰脚本）
# 消融实验指令模版见同目录：run_ppo_ablation_commands.md
# =============================================================================

# 重要：各项训练配置（batch/mini/micro/grad_ckpt/log_val 等）与以下脚本严格一致，
# 否则易导致加载失败或 GPU OOM/跑不通：
#   Normal（阶段1/基线）:
#     run_qwen3-0.6b_ppo_gsm8k_normal.sh
#     run_qwen3-0.6b_ppo_math_lighteval_normal.sh
#     run_qwen3-4b_ppo_gsm8k_normal.sh
#     run_qwen3-4b_ppo_math_lighteval_normal.sh
#   Test（mask/冻结等对照）: run_qwen3-0.6b_ppo_math_lighteval_test.sh, run_qwen3-4b_ppo_gsm8k_test.sh
# 卡数统一：0.6b 统一 4 卡，4b 统一 8 卡。
# =============================================================================
# ------------------------------ 参数一览（均可用环境变量覆盖）------------------
# | 参数                   | 说明                              | 默认值        |
# |------------------------|-----------------------------------|---------------|
# | DATA                   | 数据集 gsm8k | math               | gsm8k         |
# | MODEL_TAG              | 模型规模 0.6b | 4b（实验名+路径）  | 0.6b          |
# | ACTOR_MODEL_PATH       | Actor 模型路径                    | 按 MODEL_TAG  |
# | CRITIC_MODEL_PATH      | Critic 模型路径                   | 同 Actor 基座 |
# | USE_PRETRAINED_CRITIC  | yes=用预训练 critic（需已转 HF）  | no            |
# | REWARD_MODEL_ENABLE    | 是否用外来模型当 RM（含 critic）  | false         |
# | REWARD_MODEL_PATH      | RM 模型路径（critic 转 HF 等）    | 同 CRITIC     |
# | REWARD_BINARIZE        | 是否二值化 reward 为 0/1          | false         |
# | REWARD_THRESHOLD       | 二值化阈值 score>此值则为 1       | 0.88          |
# | REWARD_MASK_RATIO      | 轨迹级 mask 比例 0~1（0=不 mask） | 0.0           |
# | REWARD_MASK_TYPE       | mask 类型 bernoulli / fixed_ratio | bernoulli     |
# | FREEZE_CRITIC          | 是否冻结 critic（只算不更新）     | false         |
# | LAM_ACTOR              | GAE lambda for advantage (actor)  | 同 verl 默认不设，用 lam |
# | LAM_CRITIC             | GAE lambda for returns (critic)   | 同 verl 默认不设，用 lam |
# | ACTOR_LR               | Actor 学习率                       | 1e-6                   |
# | CRITIC_LR              | Critic 学习率（冻结时无效）        | 1e-5                   |
# | DIAGNOSE_REWARD_ADVANTAGE | 是否打 diag/reward、advantage、returns 等诊断指标 | false   |
# | RS_TAG                 | reward 来源标签，可手设覆盖自动推断 | 见下方        |
# | N_GPUS                 | GPU 数量（0.6b=4卡, 4b=8卡）      | 按 MODEL_TAG  |
# | CUDA_VISIBLE_DEVICES   | 可见 GPU                           | 0..N_GPUS-1   |
# | TOTAL_EPOCHS           | 训练 epoch                        | 3             |
# | SAVE_FREQ              | 保存间隔（-1 不保存）             | -1            |
# | TEST_FREQ              | 验证间隔步数                      | 5             |
# | DATA_DIR               | 数据根目录                        | $HYLBASE/data |
# | HYLBASE                | 项目根目录                        | 见下方        |
# -----------------------------------------------------------------------------
# ------------------------------ 实验名格式 -----------------------------------
# qwen3_{MODEL_TAG}_ppo_{DATA}_{RS}[_bin]_m{MASK}[_{MT}]_{C_TAG}_{F_TAG}[_la{X}_lc{Y}]
#
#   RS:  rb=rule-based | c06b/c4b=RM 为 critic 0.6b/4b（可手设 RS_TAG 覆盖）
#   bin: 仅二值化时出现
#   m:   mask 比例（0.0 表示不 mask）
#   MT:  mask>0 时的类型 bern/fix（mask=0 时省略）
#   C:   cpt=预训练 critic / cb=基座 critic
#   F:   fr=冻结 / tr=可训
#   la/lc: 仅当设置 LAM_ACTOR 或 LAM_CRITIC 时追加；未设的一侧在标题中显示为 1（默认）
# ------------------------------ 对照试验示例命令 ----------------------------
#
# 【阶段 1】正常 PPO，训练 critic 并保存（供阶段 2 使用）
#   FREEZE_CRITIC=false REWARD_MASK_RATIO=0 REWARD_MODEL_ENABLE=false \
#   SAVE_FREQ=99999 DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh
#
# 【阶段 2】稀疏/reward-free 实验：预训练 critic + 冻结 + rule-based reward + mask
#   对照 A（不 mask，基线）:
#     USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=0 \
#     DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh
#
#   对照 B（50% mask）:
#     USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=0.5 \
#     DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh
#
#   对照 C（100% mask，完全 reward-free）:
#     USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=1.0 \
#     DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh
#
#   对照 D（RM=critic，二值化，50% mask）:
#     USE_PRETRAINED_CRITIC=yes REWARD_MODEL_ENABLE=true REWARD_MODEL_PATH=/path/to/critic_hf \
#     REWARD_BINARIZE=true FREEZE_CRITIC=true REWARD_MASK_RATIO=0.5 \
#     DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh
#
#   对照 E（fixed_ratio mask）:
#     USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=0.5 \
#     REWARD_MASK_TYPE=fixed_ratio DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh
#
# 【mask 比例扫一遍】
#   for m in 0 0.3 0.5 0.7 1.0; do
#     REWARD_MASK_RATIO=$m USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true \
#     DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh
#   done
#
# 【GAE 对照：lam_actor=0.95 / lam_critic=1.0，会体现在实验名 _la0.95_lc1】
#   LAM_ACTOR=0.95 LAM_CRITIC=1.0 DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh
#
# =============================================================================

[[ -n "${DEBUG}" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 使用项目 conda 环境的 Python（含 verl），若不存在则用当前 python3
if [[ -x "${SCRIPT_DIR}/miniconda3/envs/verl-ppo/bin/python" ]]; then
  PYTHON_CMD="${SCRIPT_DIR}/miniconda3/envs/verl-ppo/bin/python"
else
  PYTHON_CMD="python3"
fi

# ========== 运行前配置（与 run_qwen3-0.6b_ppo_math_lighteval_normal.sh 对齐）==========
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Ray 需短路径（AF_UNIX socket 上限 107 字节），用 /tmp/ray_$USER 且由当前用户创建
export RAY_TMPDIR="/tmp/ray_${USER:-$(id -un)}"
mkdir -p "$RAY_TMPDIR"
# Hydra 与 validation 写入当前用户可写目录
OUTPUTS_RUN="${SCRIPT_DIR}/outputs_run"
mkdir -p "$OUTPUTS_RUN" "${OUTPUTS_RUN}/validation"

# ========== 训练时间记录 ==========
START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
echo "=========================================="
echo "训练开始时间: $START_TIME_STR"
echo "=========================================="
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

export LD_LIBRARY_PATH="${SCRIPT_DIR}/miniconda3/envs/verl-ppo/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VERL_ATTN_IMPLEMENTATION=sdpa
export WANDB_CACHE_DIR="${SCRIPT_DIR}/.wandb_cache"
mkdir -p "$WANDB_CACHE_DIR"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# ========== 可配置项 ==========
DATA="${DATA:-gsm8k}"
MODEL_TAG="${MODEL_TAG:-0.6b}"
case "$MODEL_TAG" in 0.6b) DEFAULT_N_GPUS=4 ;; 4b) DEFAULT_N_GPUS=8 ;; *) DEFAULT_N_GPUS=4 ;; esac
N_GPUS="${N_GPUS:-$DEFAULT_N_GPUS}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_GPUS-1)))}"

HYLBASE="${HYLBASE:-$SCRIPT_DIR}"
DATA_DIR="${DATA_DIR:-$HYLBASE/data}"

case "$DATA" in
  gsm8k)
    TRAIN_FILE="${DATA_DIR}/gsm8k/train.parquet"
    VAL_FILE="${DATA_DIR}/gsm8k/test.parquet"
    PRETRAINED_CRITIC_DEFAULT="${HYLBASE}/critic_qwen3_${MODEL_TAG}_ppo_gsm8k_hf"
    CRITIC_OUTPUT_PATH="${HYLBASE}/critic_qwen3_${MODEL_TAG}_ppo_gsm8k"
    ;;
  math)
    TRAIN_FILE="${DATA_DIR}/math_lighteval/train.parquet"
    VAL_FILE="${DATA_DIR}/math_lighteval/test.parquet"
    PRETRAINED_CRITIC_DEFAULT="${HYLBASE}/critic_qwen3_${MODEL_TAG}_ppo_math_lighteval_hf"
    CRITIC_OUTPUT_PATH="${HYLBASE}/critic_qwen3_${MODEL_TAG}_ppo_math_lighteval"
    ;;
  dapo_math)
    TRAIN_FILE="${DATA_DIR}/dapo_math/train.parquet"
    VAL_FILE="${DATA_DIR}/dapo_math/test.parquet"
    PRETRAINED_CRITIC_DEFAULT="${HYLBASE}/critic_qwen3_${MODEL_TAG}_ppo_dapo_math_hf"
    CRITIC_OUTPUT_PATH="${HYLBASE}/critic_qwen3_${MODEL_TAG}_ppo_dapo_math"
    ;;
  *)
    echo "Unknown DATA=$DATA, use gsm8k, math or dapo_math"
    exit 1
    ;;
esac

case "$MODEL_TAG" in
  0.6b) DEFAULT_BASE="$HYLBASE/Qwen3-0.6B" ;;
  4b)   DEFAULT_BASE="$HYLBASE/Qwen3-4B" ;;
  *)    DEFAULT_BASE="$HYLBASE/Qwen3-${MODEL_TAG}" ;;
esac
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$DEFAULT_BASE}"
BASE_CRITIC_PATH="${BASE_CRITIC_PATH:-$DEFAULT_BASE}"
USE_PRETRAINED_CRITIC="${USE_PRETRAINED_CRITIC:-no}"
if [[ "$USE_PRETRAINED_CRITIC" == "yes" ]] && [[ -z "${CRITIC_MODEL_PATH+x}" ]]; then
  CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-$PRETRAINED_CRITIC_DEFAULT}"
else
  CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-$BASE_CRITIC_PATH}"
fi

REWARD_MODEL_ENABLE="${REWARD_MODEL_ENABLE:-false}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$CRITIC_MODEL_PATH}"
REWARD_BINARIZE="${REWARD_BINARIZE:-false}"
REWARD_THRESHOLD="${REWARD_THRESHOLD:-0.88}"
REWARD_MASK_RATIO="${REWARD_MASK_RATIO:-0.0}"
REWARD_MASK_TYPE="${REWARD_MASK_TYPE:-bernoulli}"
FREEZE_CRITIC="${FREEZE_CRITIC:-false}"
# GAE：与 verl 原库一致，默认不覆盖 lam_actor/lam_critic，使用 ppo_trainer.yaml 默认（lam=1.0 用于两者）
LAM_ACTOR="${LAM_ACTOR:-}"
LAM_CRITIC="${LAM_CRITIC:-}"
LAM_EXTRA=""
if [[ -n "$LAM_ACTOR" ]]; then LAM_EXTRA="algorithm.lam_actor=$LAM_ACTOR"; fi
if [[ -n "$LAM_CRITIC" ]]; then LAM_EXTRA="${LAM_EXTRA:+$LAM_EXTRA }algorithm.lam_critic=$LAM_CRITIC"; fi
DIAGNOSE_REWARD_ADVANTAGE="${DIAGNOSE_REWARD_ADVANTAGE:-false}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-1e-5}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:-5}"

# ========== Reward 来源标签 ==========
if [[ -n "${RS_TAG+x}" && -n "$RS_TAG" ]]; then
  :
elif [[ "$REWARD_MODEL_ENABLE" != "true" ]]; then
  RS_TAG="rb"
else
  RM_BASE=$(basename "$REWARD_MODEL_PATH")
  if [[ "$RM_BASE" == *"0.6b"* || "$RM_BASE" == *"0_6b"* ]]; then
    RS_TAG="c06b"
  elif [[ "$RM_BASE" == *"4b"* || "$RM_BASE" == *"4_b"* ]]; then
    RS_TAG="c4b"
  else
    RS_TAG="c"
  fi
fi

# ========== 实验名：体现所有独立对照条件 ==========
# 格式: qwen3_{MODEL_TAG}_ppo_{DATA}_{RS}[_bin]_m{MASK}[_{MT}]_{C_TAG}_{F_TAG}[_la{X}_lc{Y}]
CRITIC_BASENAME=$(basename "$CRITIC_MODEL_PATH")
[[ "$CRITIC_BASENAME" == critic_* ]] && C_TAG="cpt" || C_TAG="cb"
F_TAG=$([ "$FREEZE_CRITIC" == "true" ] && echo "fr" || echo "tr")

# 二值化标签
BIN_TAG=""
[[ "$REWARD_BINARIZE" == "true" ]] && BIN_TAG="_bin"

# Mask 类型标签（仅 mask>0 时有意义）
MT_TAG=""
if [[ "$REWARD_MASK_RATIO" != "0" && "$REWARD_MASK_RATIO" != "0.0" ]]; then
  [[ "$REWARD_MASK_TYPE" == "fixed_ratio" ]] && MT_TAG="_fix" || MT_TAG="_bern"
fi

# GAE lambda 标签：仅当设置了 LAM_ACTOR 或 LAM_CRITIC 时追加到实验名，便于区分不同 GAE 对照
# 未设置的一侧显示为 1（与 verl 默认 lam=1.0 一致），避免标题出现 null
LAM_TAG=""
if [[ -n "$LAM_ACTOR" || -n "$LAM_CRITIC" ]]; then
  LAM_TAG="_la${LAM_ACTOR:-1}_lc${LAM_CRITIC:-1}"
fi

EXPERIMENT_NAME="qwen3_${MODEL_TAG}_ppo_${DATA}_${RS_TAG}${BIN_TAG}_m${REWARD_MASK_RATIO}${MT_TAG}_${C_TAG}_${F_TAG}${LAM_TAG}"

train_files="['$TRAIN_FILE']"
test_files="['$VAL_FILE']"

# ========== 按 DATA + MODEL_TAG 区分硬件配置 ==========
if [[ "$MODEL_TAG" == "4b" ]]; then
  if [[ "$N_GPUS" == "8" ]]; then
    TRAIN_BATCH=128; PPO_MINI=32
  else
    TRAIN_BATCH=64; PPO_MINI=16
  fi
  MICRO_PER_GPU=2; GRAD_CKPT=true; LOG_VAL_GENERATIONS=50
else
  TRAIN_BATCH=128; PPO_MINI=32; MICRO_PER_GPU=4; GRAD_CKPT=false
  [[ "$DATA" == "gsm8k" ]] && LOG_VAL_GENERATIONS=10000 || LOG_VAL_GENERATIONS=50
fi
[[ "$DATA" == "gsm8k" ]] && GPU_UTIL=0.5 || GPU_UTIL=0.4

# ========== RM 额外参数（reward_model.enable=true 时需 rollout 配置）==========
RM_EXTRA=""
[[ "$REWARD_MODEL_ENABLE" == "true" ]] && RM_EXTRA="reward_model.enable=True reward_model.model.path=$REWARD_MODEL_PATH reward_model.rollout.name=vllm reward_model.rollout.gpu_memory_utilization=$GPU_UTIL reward_model.rollout.tensor_model_parallel_size=1 reward_model.rollout.prompt_length=1024 reward_model.rollout.response_length=2048"

echo "=========================================="
echo "  DATA=$DATA  MODEL_TAG=$MODEL_TAG  N_GPUS=$N_GPUS"
echo "  Actor:  $ACTOR_MODEL_PATH"
echo "  Critic: $CRITIC_MODEL_PATH ($C_TAG) freeze=$FREEZE_CRITIC ($F_TAG)"
echo "  RM:     enable=$REWARD_MODEL_ENABLE  path=$REWARD_MODEL_PATH"
echo "  Reward: binarize=$REWARD_BINARIZE thresh=$REWARD_THRESHOLD"
echo "  Mask:   ratio=$REWARD_MASK_RATIO type=$REWARD_MASK_TYPE"
[[ -n "$LAM_EXTRA" ]] && echo "  GAE:    $LAM_EXTRA" || echo "  GAE:    同 verl 默认 (lam=1.0，不设 lam_actor/lam_critic)"
echo "  Actor LR: $ACTOR_LR   Critic LR: $CRITIC_LR"
echo "  Experiment: $EXPERIMENT_NAME"
echo "=========================================="

"$PYTHON_CMD" -m verl.trainer.main_ppo \
  hydra.run.dir="$OUTPUTS_RUN" \
  algorithm.adv_estimator=gae \
  algorithm.reward_mask_ratio=$REWARD_MASK_RATIO \
  algorithm.reward_mask_type=$REWARD_MASK_TYPE \
  algorithm.reward_binarize=$REWARD_BINARIZE \
  algorithm.reward_threshold=$REWARD_THRESHOLD \
  $LAM_EXTRA \
  algorithm.diagnose_reward_advantage=$DIAGNOSE_REWARD_ADVANTAGE \
  $RM_EXTRA \
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
  actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_PER_GPU \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=$GRAD_CKPT \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_PER_GPU \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_PER_GPU \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
  critic.optim.lr=$CRITIC_LR \
  critic.model.use_remove_padding=False \
  critic.model.path="$CRITIC_MODEL_PATH" \
  +critic.model.override_config.attn_implementation=sdpa \
  critic.model.enable_gradient_checkpointing=$GRAD_CKPT \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_PER_GPU \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  critic.freeze=$FREEZE_CRITIC \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.log_val_generations=$LOG_VAL_GENERATIONS \
  trainer.validation_data_dir="${OUTPUTS_RUN}/validation" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='PPO-Project2026-1' \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.default_local_dir="${SCRIPT_DIR}/checkpoints/PPO-Project2026/${EXPERIMENT_NAME}" \
  "$@"

TRAIN_EXIT=$?

# ========== 阶段 1 结束后：先备份 FSDP critic，再尝试转 HF ==========
CKPT_BASE="${SCRIPT_DIR}/checkpoints/PPO-Project2026/${EXPERIMENT_NAME}"
if [[ $TRAIN_EXIT -eq 0 ]] && [[ "$FREEZE_CRITIC" == "false" ]] && [[ "$USE_PRETRAINED_CRITIC" != "yes" ]] && [[ "$SAVE_FREQ" -gt 0 ]]; then
  LATEST_STEP=$(cat "${CKPT_BASE}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
  if [[ -n "$LATEST_STEP" ]]; then
    CRITIC_CKPT="${CKPT_BASE}/global_step_${LATEST_STEP}/critic"
    HF_OUTPUT="${PRETRAINED_CRITIC_DEFAULT}"

    # 1) 先备份 FSDP critic 到 critic_backups，便于 HF 转换失败时手动转换
    if [[ -d "$CRITIC_CKPT" ]]; then
      BACKUP_BASE="${SCRIPT_DIR}/critic_backups"
      BACKUP_DIR="${BACKUP_BASE}/${EXPERIMENT_NAME}_step${LATEST_STEP}_critic"
      mkdir -p "$BACKUP_BASE"
      if cp -r "$CRITIC_CKPT" "$BACKUP_DIR" 2>/dev/null; then
        echo "=========================================="
        echo "阶段 1 完成：已备份 FSDP critic（HF 转换失败时可手动转换）"
        echo "  备份路径: $BACKUP_DIR"
        echo "  手动转换示例: $PYTHON_CMD scripts/convert_critic_to_hf.py --local_dir \"$BACKUP_DIR\" --target_dir <目标HF路径> --trust-remote-code"
        echo "=========================================="
      else
        echo "WARNING: FSDP critic 备份失败，请手动复制: $CRITIC_CKPT -> $BACKUP_DIR"
      fi
    fi

    # 2) 再尝试自动转 HF（仅当目标目录不存在时执行，不覆盖已有）
    if [[ -d "$CRITIC_CKPT" ]] && [[ ! -d "$HF_OUTPUT" ]]; then
      echo "=========================================="
      echo "阶段 1 训练完成，自动转换 critic → HF 格式"
      echo "  源: $CRITIC_CKPT"
      echo "  目标: $HF_OUTPUT"
      echo "=========================================="
      "$PYTHON_CMD" "${SCRIPT_DIR}/scripts/convert_critic_to_hf.py" \
        --local_dir "$CRITIC_CKPT" \
        --target_dir "$HF_OUTPUT" \
        --trust-remote-code \
        && echo "Critic HF 转换完成: $HF_OUTPUT" \
        || echo "WARNING: Critic HF 转换失败，请用备份手动转换: $BACKUP_DIR"
    elif [[ -d "$CRITIC_CKPT" ]] && [[ -d "$HF_OUTPUT" ]]; then
      echo "已存在 HF 目录，跳过自动转换（未覆盖）: $HF_OUTPUT ；FSDP 备份见: $BACKUP_DIR"
    fi
  fi
fi

exit ${TRAIN_EXIT}
