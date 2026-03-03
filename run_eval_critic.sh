#!/bin/bash
# =============================================================================
# 评估预训练 critic 的 reward 预测能力
#
# 用法:
#   DATA=math MODEL_TAG=0.6b ./run_eval_critic.sh                  # 生成+评估
#   ROLLOUT_FILE=./eval_critic_results/.../rollouts.jsonl ./run_eval_critic.sh  # 跳过生成
#
# 参数 (环境变量):
#   DATA            数据集 gsm8k | math                  (默认 math)
#   MODEL_TAG       模型规模 0.6b | 4b                   (默认 0.6b)
#   MAX_SAMPLES     评估样本数                            (默认 1000)
#   THRESHOLD       二分类阈值                            (默认 0.5；建议用 THRESHOLD_SWEEP=1 先扫一遍再定)
#   THRESHOLD_SWEEP 是否做阈值扫描并输出推荐阈值          (默认 0；设为 1 时会打印各阈值准确率)
#   BATCH_SIZE      Critic 前向 batch size               (默认 8)
#   TEMPERATURE     生成温度                              (默认 0.6)
#   MAX_NEW_TOKENS  最大生成长度                          (默认 1024)
#   N_GPUS          使用 GPU 数量                          (默认 4)
#   CUDA_VISIBLE_DEVICES  可见 GPU（默认 0..N_GPUS-1）
#   ROLLOUT_FILE    已有 rollout 文件路径（跳过生成阶段） (默认 空)
# =============================================================================

set -euo pipefail

[[ -n "${DEBUG:-}" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ========== 运行前配置（与 run_qwen3_ppo.sh 对齐）==========
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_TMPDIR="/tmp/ray_${USER:-$(id -un)}"
mkdir -p "$RAY_TMPDIR"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/miniconda3/envs/verl-ppo/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VERL_ATTN_IMPLEMENTATION=sdpa
export WANDB_CACHE_DIR="${SCRIPT_DIR}/.wandb_cache"
mkdir -p "$WANDB_CACHE_DIR"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# ========== 运行时间记录 ==========
START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
echo "=========================================="
echo "评估开始时间: $START_TIME_STR"
echo "=========================================="
log_eval_time() {
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  HOURS=$((ELAPSED / 3600)); MINUTES=$(((ELAPSED % 3600) / 60)); SECS=$((ELAPSED % 60))
  echo "=========================================="
  echo "评估结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECS}秒"
  echo "=========================================="
}
trap log_eval_time EXIT

# ========== 可配置项 ==========
DATA="${DATA:-math}"
MODEL_TAG="${MODEL_TAG:-0.6b}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
THRESHOLD="${THRESHOLD:-0.5}"
THRESHOLD_SWEEP="${THRESHOLD_SWEEP:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
N_GPUS="${N_GPUS:-4}"
ROLLOUT_FILE="${ROLLOUT_FILE:-}"
WANDB_PROJECT="${WANDB_PROJECT:-PPO-Project2026-2}"

HYLBASE="${HYLBASE:-$SCRIPT_DIR}"

case "$MODEL_TAG" in
  0.6b) ACTOR_PATH="${ACTOR_PATH:-$HYLBASE/Qwen3-0.6B}" ;;
  4b)   ACTOR_PATH="${ACTOR_PATH:-$HYLBASE/Qwen3-4B}" ;;
  *)    ACTOR_PATH="${ACTOR_PATH:-$HYLBASE/Qwen3-${MODEL_TAG}}" ;;
esac

case "$DATA" in
  gsm8k)
    DATA_PATH="${DATA_PATH:-$HYLBASE/data/gsm8k/test.parquet}"
    DATA_SOURCE="openai/gsm8k"
    CRITIC_PATH="${CRITIC_PATH:-$HYLBASE/critic_qwen3_${MODEL_TAG}_ppo_gsm8k_hf}"
    ;;
  math)
    DATA_PATH="${DATA_PATH:-$HYLBASE/data/math_lighteval/test.parquet}"
    DATA_SOURCE="lighteval/MATH"
    CRITIC_PATH="${CRITIC_PATH:-$HYLBASE/critic_qwen3_${MODEL_TAG}_ppo_math_lighteval_hf}"
    ;;
  dapo_math)
    DATA_PATH="${DATA_PATH:-$HYLBASE/data/dapo_math/test.parquet}"
    DATA_SOURCE="math_dapo"
    CRITIC_PATH="${CRITIC_PATH:-$HYLBASE/critic_qwen3_${MODEL_TAG}_ppo_dapo_math_hf}"
    ;;
  *)
    echo "Unknown DATA=$DATA, use gsm8k, math or dapo_math"
    exit 1
    ;;
esac

EXPERIMENT_NAME="eval_critic_${DATA}_${MODEL_TAG}"
OUTPUT_DIR="${OUTPUT_DIR:-$HYLBASE/eval_critic_results/${DATA}_${MODEL_TAG}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_GPUS-1)))}"

echo "=========================================="
echo "Critic Prediction Evaluation"
echo "=========================================="
echo "  DATA:        $DATA"
echo "  MODEL_TAG:   $MODEL_TAG"
echo "  ACTOR:       $ACTOR_PATH"
echo "  CRITIC:      $CRITIC_PATH"
echo "  DATA_PATH:   $DATA_PATH"
echo "  MAX_SAMPLES: $MAX_SAMPLES"
echo "  THRESHOLD:   $THRESHOLD"
[[ "$THRESHOLD_SWEEP" = "1" ]] && echo "  THRESHOLD_SWEEP: yes (will print recommended threshold)"
echo "  N_GPUS:      $N_GPUS  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  WANDB:       $WANDB_PROJECT / $EXPERIMENT_NAME"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
[[ -n "$ROLLOUT_FILE" ]] && echo "  ROLLOUT:     $ROLLOUT_FILE (skip generation)"
echo "=========================================="

ROLLOUT_ARG=""
[[ -n "$ROLLOUT_FILE" ]] && ROLLOUT_ARG="--rollout_file $ROLLOUT_FILE"

THRESHOLD_SWEEP_ARG=""
[[ "$THRESHOLD_SWEEP" = "1" ]] && THRESHOLD_SWEEP_ARG="--threshold_sweep"

python3 "${SCRIPT_DIR}/scripts/eval_critic_prediction.py" \
    --actor_path "$ACTOR_PATH" \
    --critic_path "$CRITIC_PATH" \
    --data_path "$DATA_PATH" \
    --data_source "$DATA_SOURCE" \
    --max_samples "$MAX_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --threshold "$THRESHOLD" \
    --batch_size "$BATCH_SIZE" \
    --n_gpus "$N_GPUS" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$EXPERIMENT_NAME" \
    $THRESHOLD_SWEEP_ARG \
    $ROLLOUT_ARG

echo ""
echo "Done. Results in: $OUTPUT_DIR"
