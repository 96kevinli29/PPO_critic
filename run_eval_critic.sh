#!/bin/bash
# =============================================================================
# AIME2024 / AIME2025 / AMC2023 三数据集评估预训练 critic 的 reward 预测能力
#
# 用法:
#   ./run_eval_critic.sh                                    # 默认 critic=$HYLBASE/qwen3-4b-critic，4B actor，8 卡，3 数据集
#   DATA=math ./run_eval_critic.sh                          # 仅影响输出目录/wandb 名中的标签（默认 dapo_math）
#   CRITIC_PATH=/path/to/critic ./run_eval_critic.sh        # 直接指定 critic 路径
#   ROLLOUT_FILE=$HYLBASE/hyl_outputs/eval_critic/.../rollouts.jsonl ./run_eval_critic.sh  # 跳过生成
#
# 参数 (环境变量):
#   DATA            输出目录/wandb 名标签 gsm8k | math | dapo_math（默认 dapo_math）
#   MODEL_TAG       模型规模 0.8b | 4b | 8b（qwen3）        (默认 4b)
#   MODEL_FAMILY    模型族 qwen3 | qwen35                    (默认 qwen3)
#   HF_CRITIC_DIR   预训练 critic HF 目录（默认 $HYLBASE/qwen3-4b-critic）
#   CRITIC_PATH     直接指定 critic 路径（优先级高于 HF_CRITIC_DIR）
#   ACTOR_PATH      直接指定 actor 路径（优先级高于 MODEL_TAG 推断）
#   MAX_SAMPLES     每数据集评估样本数                        (默认 1000)
#   THRESHOLD       二分类阈值                                (默认 0.5；建议用 THRESHOLD_SWEEP=1 先扫一遍再定)
#   THRESHOLD_SWEEP 是否做阈值扫描并输出推荐阈值              (默认 0；设为 1 时会打印各阈值准确率)
#   BATCH_SIZE      Critic 前向 batch size                   (默认 8)
#   TEMPERATURE     生成温度                                  (默认 0.7)
#   MAX_NEW_TOKENS  最大生成长度                              (默认 8192)
#   N_GPUS          使用 GPU 数量                             (默认 8)
#   CUDA_VISIBLE_DEVICES  可见 GPU（默认 0..N_GPUS-1）
#   ROLLOUT_FILE    已有 rollout 文件路径（跳过生成阶段）     (默认 空；仅对单数据集有效)
#   USE_FLASHINFER_SAMPLER  默认自动；=1 强制开，=0 强制关
#   DRY_RUN         1=仅跑第一个数据集（验证流程）
# =============================================================================

set -euo pipefail

[[ -n "${DEBUG:-}" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/hyl_env.sh" eval

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/hyl_run_log.sh"
if [[ "${HYL_AUTO_LOG:-1}" != "0" ]]; then
  HYL_LOG_FILE=$(hyl_next_log_path "${HYL_RUN_LOG_PREFIX:-eval_critic}") || exit 1
  [[ -n "${HYL_LOG_FILE:-}" ]] || exit 1
  export HYL_LOG_FILE
  hyl_tee_stdout_stderr "$HYL_LOG_FILE"
fi

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
  if [[ -n "${HYL_LOG_FILE:-}" ]]; then
    echo "log_file:         ${HYL_LOG_FILE}"
  fi
  echo "=========================================="
}
trap log_eval_time EXIT

# ========== 可配置项 ==========
DATA="${DATA:-dapo_math}"
MODEL_TAG="${MODEL_TAG:-4b}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen3}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
THRESHOLD="${THRESHOLD:-0.5}"
THRESHOLD_SWEEP="${THRESHOLD_SWEEP:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
N_GPUS="${N_GPUS:-8}"
ROLLOUT_FILE="${ROLLOUT_FILE:-}"
WANDB_PROJECT="${WANDB_PROJECT:-Eval-Critic}"

DATA_DIR="${DATA_DIR:-$HYLBASE/data}"

# 按 MODEL_FAMILY+MODEL_TAG 得到模型目录名（与 run_ppo.sh 一致）
case "$MODEL_FAMILY" in
  qwen35)
    case "$MODEL_TAG" in
      0.8b) BASE_NAME="Qwen3.5-0.8B" ;;
      4b)   BASE_NAME="Qwen3.5-4B" ;;
      *)    BASE_NAME="Qwen3.5-${MODEL_TAG}" ;;
    esac ;;
  *)
    case "$MODEL_TAG" in
      4b)   BASE_NAME="Qwen3-4B-SFT" ;;
      8b)   BASE_NAME="Qwen3-8B" ;;
      *)    BASE_NAME="Qwen3-${MODEL_TAG}" ;;
    esac ;;
esac
ACTOR_PATH="${ACTOR_PATH:-$HYLBASE/${BASE_NAME}}"

case "$DATA" in
  gsm8k|math|dapo_math) ;;
  *)
    echo "Unknown DATA=$DATA, use gsm8k, math or dapo_math"
    exit 1
    ;;
esac

HF_CRITIC_DIR="${HF_CRITIC_DIR:-$HYLBASE/qwen3-4b-critic}"
CRITIC_PATH="${CRITIC_PATH:-$HF_CRITIC_DIR}"

# 与 run_eval_accuracy.sh / run_ppo.sh 验证集一致
BENCHMARK_NAMES=(aime2024 aime2025 amc2023)
BENCHMARK_PATHS=(
  "${DATA_DIR}/aime2024/test.parquet"
  "${DATA_DIR}/aime2025/test.parquet"
  "${DATA_DIR}/amc2023/test.parquet"
)

# 实验名 & 输出目录
EXPERIMENT_BASE="eval_critic_${MODEL_FAMILY}_${MODEL_TAG}_${DATA}"
# 自动检测：输出目录已存在时追加 _run2, _run3, ...
_exp_out="${OUTPUTS}/eval_critic/${EXPERIMENT_BASE}"
if [[ -d "$_exp_out" ]]; then
  _n=2
  while [[ -d "${_exp_out}_run${_n}" ]]; do ((_n++)); done
  EXPERIMENT_BASE="${EXPERIMENT_BASE}_run${_n}"
  echo "输出目录已存在，自动重命名: ${EXPERIMENT_BASE}"
fi
OUTPUT_BASE_DIR="${OUTPUTS}/eval_critic/${EXPERIMENT_BASE}"
if ! mkdir -p "$OUTPUT_BASE_DIR" 2>/dev/null || [[ ! -w "$OUTPUT_BASE_DIR" ]]; then
  echo "ERROR: 无法创建或写入 $OUTPUT_BASE_DIR （用户 $(id -un)）" >&2
  echo "  请检查 OUTPUTS 目录权限，或: export OUTPUTS=\"/path/其他可写目录\"" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_GPUS-1)))}"

# DRY_RUN: 仅跑第 1 个数据集（验证流程）
DRY_RUN="${DRY_RUN:-}"
if [[ -n "$DRY_RUN" ]] && [[ "$DRY_RUN" != "0" ]]; then
  BENCHMARK_NAMES=(aime2024)
  BENCHMARK_PATHS=("${DATA_DIR}/aime2024/test.parquet")
  MAX_SAMPLES=5
  echo "DRY_RUN=1: 仅评估第 1 个数据集（aime2024），max_samples=$MAX_SAMPLES"
fi

hyl_print_eval_critic_header

THRESHOLD_SWEEP_ARG=""
[[ "$THRESHOLD_SWEEP" = "1" ]] && THRESHOLD_SWEEP_ARG="--threshold_sweep"

EVAL_FAIL=0
for i in "${!BENCHMARK_NAMES[@]}"; do
  DS_NAME="${BENCHMARK_NAMES[$i]}"
  DS_PATH="${BENCHMARK_PATHS[$i]}"
  DS_OUTPUT="${OUTPUT_BASE_DIR}/${DS_NAME}"
  DS_WANDB_NAME="${EXPERIMENT_BASE}_${DS_NAME}"

  if [[ ! -f "$DS_PATH" ]]; then
    echo "WARNING: 跳过 ${DS_NAME}，数据文件不存在: $DS_PATH"
    continue
  fi

  echo ""
  echo "────────────────────────────────────────────"
  echo "  [$((i+1))/${#BENCHMARK_NAMES[@]}] 评估: $DS_NAME"
  echo "  数据: $DS_PATH"
  echo "  输出: $DS_OUTPUT"
  echo "────────────────────────────────────────────"

  ROLLOUT_ARG=""
  [[ -n "$ROLLOUT_FILE" ]] && ROLLOUT_ARG="--rollout_file $ROLLOUT_FILE"

  $PYTHON_CMD "${SCRIPT_DIR}/scripts/eval_critic_prediction.py" \
      --actor_path "$ACTOR_PATH" \
      --critic_path "$CRITIC_PATH" \
      --data_path "$DS_PATH" \
      --max_samples "$MAX_SAMPLES" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --threshold "$THRESHOLD" \
      --batch_size "$BATCH_SIZE" \
      --n_gpus "$N_GPUS" \
      --attn_implementation "$VERL_ATTN_IMPLEMENTATION" \
      --output_dir "$DS_OUTPUT" \
      --wandb_project "$WANDB_PROJECT" \
      --wandb_run_name "$DS_WANDB_NAME" \
      $THRESHOLD_SWEEP_ARG \
      $ROLLOUT_ARG \
    || { echo "ERROR: $DS_NAME 评估失败"; EVAL_FAIL=1; }
done

echo ""
echo "=========================================="
if [[ $EVAL_FAIL -eq 0 ]]; then
  echo "全部完成。结果目录: $OUTPUT_BASE_DIR"
else
  echo "部分数据集评估失败。结果目录: $OUTPUT_BASE_DIR"
fi
[[ -n "${HYL_LOG_FILE:-}" ]] && echo "log_file:         ${HYL_LOG_FILE}"
echo "=========================================="

exit $EVAL_FAIL
