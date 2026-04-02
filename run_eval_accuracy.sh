#!/bin/bash
# =============================================================================
# AIME2024 / AIME2025 / AMC2023 三数据集准确率评估（不分阶段，任意 checkpoint 均可），结果上报 wandb
#
# 用法:
#   ./run_eval_accuracy.sh Qwen3-4B-SFT          # 项目根下 HF 目录名（与 run_ppo.sh 默认 MODEL_ROOT 一致）
#   MODEL=Qwen3-4B-Base ./run_eval_accuracy.sh
#   ACTOR_PATH=/path/to/model ./run_eval_accuracy.sh
#
# 参数:
#   MODEL / ACTOR_PATH    模型绝对路径，或项目根下的 HF 权重目录名（必填其一）
#   DATA_DIR               benchmark 数据根目录（默认 $HYLBASE/data）
#   WANDB_PROJECT          wandb 项目名（默认 Eval-Accuracy）
#   WANDB_RUN_NAME         wandb run 名（默认 eval_<模型目录名>）
#   HYL_RUN_LOG_PREFIX     自动日志前缀（默认 eval_accuracy → $HYL_LOG_DIR/eval_accuracy_NNN.log）
#   HYL_LOG_DIR / HYL_AUTO_LOG  见 scripts/hyl_run_log.sh
#   N_GPUS                 vLLM tensor parallel（默认 8）
#   N_RUNS                 重复评估 N 次，上报准确率为 N 次平均值（默认 3）
#   DRY_RUN                1=小实验：N_RUNS=1 且每数据集仅 5 条样本（验证流程）
#   RUN_SUFFIX             结果目录后缀，避免覆盖（如 20250311_1）
#   MAX_NEW_TOKENS         单条「最多生成」的 token 数（默认 2084）
#   MAX_MODEL_LEN          vLLM 整条序列上限 = prompt + 生成（默认 4096）。须满足：
#                         max_model_len >= 单条 prompt 长度 + max_new_tokens，否则实际
#                         生成长度会被压成 max_model_len - prompt_len。两参数不必相等。
#   ENABLE_THINKING        与 SFT 一致（默认 1；设为 0 关闭 --enable_thinking）
#   TEMPERATURE            生成温度（默认 0.7）
#   GPU_MEMORY_UTILIZATION vLLM 单卡显存占用比例（默认 0.5）。0.5=保守安全；0.85~0.9=单任务时更充分利用 A100。
#   USE_FLASHINFER_SAMPLER 默认自动（有 cuda/functional 则用 FlashInfer，没有则用 PyTorch）；=1 强制开，=0 强制关。（环境总入口 scripts/hyl_env.sh）
#
# 上报 wandb：case 问答与打分（eval_accuracy/samples）、三数据集汇总表（eval_accuracy/summary）
# 本地结果保存到 $OUTPUTS/eval_accuracy/<model_name>/<dataset>/（默认 OUTPUTS=$HYLBASE/hyl_outputs）
# =============================================================================

[[ -n "${DEBUG:-}" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/hyl_env.sh" eval

# ---------- 参数：支持 MODEL= 或 ACTOR_PATH 或位置参数 $1 ----------
if [[ -n "${MODEL}" && -z "${ACTOR_PATH}" ]]; then
  ACTOR_PATH="${MODEL}"
fi
if [[ -n "$1" && -z "${ACTOR_PATH}" ]]; then
  ACTOR_PATH="$1"
fi
if [[ -z "$ACTOR_PATH" ]]; then
  echo "Usage: $0 <model_dir_under_HYLBASE>   OR   MODEL=... $0   OR   ACTOR_PATH=/abs/path $0"
  echo "  Example: ./run_eval_accuracy.sh Qwen3-4B-SFT"
  echo "  Evaluates model on 6 benchmarks and logs to wandb."
  exit 1
fi
# 非绝对路径时视为项目根下的模型目录名
if [[ "$ACTOR_PATH" != /* ]]; then
  ACTOR_PATH="${HYLBASE}/${ACTOR_PATH}"
fi
if [[ ! -d "$ACTOR_PATH" ]]; then
  echo "ERROR: Model path not found: $ACTOR_PATH"
  exit 1
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/hyl_run_log.sh"
if [[ "${HYL_AUTO_LOG:-1}" != "0" ]]; then
  HYL_LOG_FILE=$(hyl_next_log_path "${HYL_RUN_LOG_PREFIX:-eval_accuracy}") || exit 1
  [[ -n "${HYL_LOG_FILE:-}" ]] || exit 1
  export HYL_LOG_FILE
  hyl_tee_stdout_stderr "$HYL_LOG_FILE"
fi

DATA_DIR="${DATA_DIR:-$HYLBASE/data}"
WANDB_PROJECT="${WANDB_PROJECT:-Eval-Accuracy}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
N_GPUS="${N_GPUS:-8}"
N_RUNS="${N_RUNS:-3}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
# 小实验：N_RUNS=1，每数据集仅 5 条样本（验证流程）；结果目录与 wandb run 名带 _test
if [[ -n "${DRY_RUN:-}" ]] && [[ "$DRY_RUN" != "0" ]]; then
  N_RUNS=1
  EVAL_MAX_SAMPLES_DRY=5
  RUN_SUFFIX="${RUN_SUFFIX:-test}"
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-eval_$(basename "$ACTOR_PATH")_test}"
  echo "DRY_RUN=1: 小实验（N_RUNS=1，每数据集 max_samples=$EVAL_MAX_SAMPLES_DRY），输出带 _test"
fi
# 自动检测：输出目录已存在时追加 _run2, _run3, ...（第一次不加后缀）
_actor_base=$(basename "$ACTOR_PATH")
_eval_out_base="${OUTPUTS}/eval_accuracy/${_actor_base}${RUN_SUFFIX:+_${RUN_SUFFIX}}"
if [[ -z "${RUN_SUFFIX}" ]] && [[ -d "$_eval_out_base" ]]; then
  _n=2
  while [[ -d "${OUTPUTS}/eval_accuracy/${_actor_base}_run${_n}" ]]; do ((_n++)); done
  RUN_SUFFIX="run${_n}"
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-eval_${_actor_base}_run${_n}}"
  echo "输出目录已存在，自动重命名后缀: ${RUN_SUFFIX}"
fi
# ---------- 多次对比时建议固定以下变量，结果才可比 --------------
#   N_RUNS=3 TEMPERATURE ENABLE_THINKING MAX_MODEL_LEN 等与单次设置保持一致
#
# ---------- MAX_NEW_TOKENS 与 GPU_MEMORY_UTILIZATION 配套建议（8 卡 A100-80，4B 级模型）--------------
# 按生成长度选一组即可；显存紧张时选上一档或减小 GPU_MEMORY_UTILIZATION。
#
#   MAX_NEW_TOKENS  GPU_MEMORY_UTILIZATION  说明
#   --------------  ----------------------  -----
#   2084（默认）     0.5（默认）              省显存、速度快；GSM8k/math 等多数题够用，AIME/AMC 长推理可能偶有截断
#   4096            0.60 ~ 0.65              长推理更稳，显存仍可接受
#   8192            0.75 ~ 0.80              长链推理/thinking；单任务可拉高利用率
#   32768           0.85 ~ 0.90               与论文 32k 对齐、充分生成；需保证 MAX_MODEL_LEN≥32768
#
# 影响显存的其他因素：MAX_MODEL_LEN（越大 KV 越大）、N_GPUS（越多单卡越轻）、模型参数量、单次 generate 样本数。
# 不设 MAX_MODEL_LEN 时 Python 默认 32768；显存紧张时可减小 MAX_MODEL_LEN 或增大 N_GPUS。
#
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2084}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
EVAL_ENABLE_THINKING=""
[[ "$ENABLE_THINKING" == "1" || "$ENABLE_THINKING" == "true" ]] && EVAL_ENABLE_THINKING="--enable_thinking"
TEMPERATURE="${TEMPERATURE:-0.7}"

BENCHMARK_PATHS="${DATA_DIR}/aime2024/test.parquet,${DATA_DIR}/aime2025/test.parquet,${DATA_DIR}/amc2023/test.parquet"

EXTRA_ARGS=()
[[ -n "$RUN_SUFFIX" ]] && EXTRA_ARGS+=(--run_suffix "$RUN_SUFFIX")
# 小实验：每数据集最多 N 条
[[ -n "${EVAL_MAX_SAMPLES_DRY:-}" ]] && EXTRA_ARGS+=(--max_samples "$EVAL_MAX_SAMPLES_DRY")

# 输出根目录（hyl_env 已保证 OUTPUTS 可写；此处再保证 eval 子目录）
EVAL_OUT_ROOT="${OUTPUTS}/eval_accuracy"
if ! mkdir -p "$EVAL_OUT_ROOT" 2>/dev/null || [[ ! -w "$EVAL_OUT_ROOT" ]]; then
  echo "ERROR: 无法创建或写入 $EVAL_OUT_ROOT （用户 $(id -un)）" >&2
  echo "  请检查 OUTPUTS 目录权限，或: export OUTPUTS=\"/path/其他可写目录\"" >&2
  exit 1
fi

$PYTHON_CMD "$SCRIPT_DIR/scripts/eval_accuracy.py" \
  --actor_path "$ACTOR_PATH" \
  --data_paths "$BENCHMARK_PATHS" \
  $EVAL_ENABLE_THINKING \
  --temperature "$TEMPERATURE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --n_gpus "$N_GPUS" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --n_runs "$N_RUNS" \
  --attn_implementation "$VERL_ATTN_IMPLEMENTATION" \
  --wandb_project "$WANDB_PROJECT" \
  --output_dir "$_eval_out_base" \
  ${WANDB_RUN_NAME:+--wandb_run_name "$WANDB_RUN_NAME"} \
  "${EXTRA_ARGS[@]}"
