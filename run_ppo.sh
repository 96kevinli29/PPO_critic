#!/bin/bash
# =============================================================================
# 统一 Qwen3 PPO 入口：训练数据固定 dapo_math_17k；可改模型路径、RM、二值化、mask、冻结等
# 便于做对照试验（只改参数，不碰脚本）
#
# ===== PPO 流程（阶段 1 + 阶段 2）=====
# 阶段 1 — 标准 PPO（Actor+Critic 同训；基线；默认 HF 起点 $HYLBASE/Qwen3-4B-SFT，可用 MODEL_ROOT= 覆盖）
#   ./run_ppo_stage1.sh   → 默认实验名 qwen3_4b_RL1
# 阶段 2 — 控制变量对照：在 Actor/数据等与阶段 1 对齐的前提下，只改「Critic 冻结 + 阶段 1 预训练 critic」与 reward mask 等；
#   Critic 用阶段 1 转好的 HF、冻结；run_ppo_stage2 默认 REWARD_MASK_RATIO=1.0（100% mask）
#   ./run_ppo_stage2.sh   → 未设 EXPERIMENT_NAME 时为 qwen3_4b_RL2_m<mask>（默认 m1.0，见 run_ppo_stage2.sh）
#   REWARD_MASK_RATIO=0.5 ./run_ppo_stage2.sh   # 部分 mask
#   （若要从阶段 1 policy 续训，再显式 ACTOR_MODEL_PATH= 指向合并后的 HF / checkpoint）
#
# 也可直接运行本脚本（run_ppo.sh），用环境变量控制；run_ppo_stage*.sh 仅设默认日志前缀等后调用本脚本。
# =============================================================================
# 默认 8 卡（N_GPUS=4 可改）；HF 起点默认 $HYLBASE/Qwen3-4B-SFT（MODEL_ROOT= 覆盖）。
# =============================================================================
# ------------------------------ 参数一览（均可用环境变量覆盖）------------------
# | 参数                   | 说明                              | 默认值        |
# |------------------------|-----------------------------------|---------------|
# | TRAIN_MAX_SAMPLES      | 训练集最大样本数（dapo_math_17k） | 17917（-1=全量） |
# | MODEL_ROOT             | Actor/Critic 起点（HF 权重目录） | $HYLBASE/Qwen3-4B-SFT |
# | ACTOR_MODEL_PATH       | 覆盖 Actor                        | 同 MODEL_ROOT |
# | CRITIC_MODEL_PATH      | 覆盖 Critic                       | 同 MODEL_ROOT 或 HF_CRITIC_DIR |
# | HF_CRITIC_DIR          | 阶段1 critic→HF 建议目录；阶段2预训练默认路径 | $HYLBASE/qwen3-4b-critic |
# | USE_PRETRAINED_CRITIC  | yes=用预训练 critic（需已转 HF）  | no            |
# | REWARD_MODEL_ENABLE    | 是否用外来模型当 RM（含 critic）  | false         |
# | REWARD_MODEL_PATH      | RM 模型路径（critic 转 HF 等）    | 同 CRITIC     |
# | REWARD_BINARIZE        | 是否二值化 reward 为 0/1          | false         |
# | REWARD_THRESHOLD       | 二值化阈值 score>此值则为 1       | 0.88          |
# | REWARD_MASK_RATIO      | 轨迹级 reward mask 比例 0~1（0=不 mask）；只把被选中的轨迹的 reward 整条置 0，value 不 mask | 0.0（run_ppo_stage2 默认 1.0） |
# | REWARD_MASK_TYPE       | mask 类型 bernoulli / fixed_ratio | fixed_ratio   |
# | FREEZE_CRITIC          | 是否冻结 critic（只算不更新）     | false         |
# | LAM_ACTOR              | GAE lambda for advantage (actor)  | 0.95                     |
# | LAM_CRITIC             | GAE lambda for returns (critic)   | 不设（verl 默认 1）     |
# | MAX_PROMPT_LEN         | 最大 prompt 长度（所有数据集统一） | 16384         |
# | MAX_RESPONSE_LEN       | 最大 response 长度（所有数据集统一） | 12288         |
# | ROLLOUT_N              | 每 prompt 采样数                   | 1                      |
# | TOP_P                  | rollout 采样 top_p（nucleus sampling） | 1.0                  |
# | WEIGHT_DECAY           | 权重衰减                           | 0.1                    |
# | LR_WARMUP_STEPS        | 学习率预热步数                     | 10                     |
# | OVERLONG_BUFFER        | 超长 response 惩罚缓冲区长度       | 4096 |
# | ACTOR_LR               | Actor 学习率                       | 1e-6                   |
# | CRITIC_LR              | Critic 学习率（冻结时无效）        | 1e-5                   |
# | MICRO_PER_GPU          | 每 GPU 微批量大小                  | 2                       |
# | TRAIN_BATCH            | 全局训练 batch 大小                | 8卡=128, 其他=64        |
# | PPO_MINI               | PPO mini batch 大小                | 8卡=32,  其他=16        |
# | GRAD_CKPT              | 梯度检查点（省显存换时间）         | true                    |
# | GPU_UTIL               | vLLM GPU 显存占用比例              | 0.5                     |
# | PPO_PROJECT_NAME       | W&B / trainer.project_name（项目名）     | RL-Training |
# | EXPERIMENT_NAME        | checkpoint / wandb 实验名（stage1 默认 qwen3_4b_RL1；stage2 未设时 qwen3_4b_RL2_m*） | 见 run_ppo_stage*.sh |
# | N_GPUS                 | GPU 数量                          | 8             |
# | CUDA_VISIBLE_DEVICES   | 可见 GPU                           | 0..N_GPUS-1   |
# | TOTAL_EPOCHS           | 训练 epoch                        | 5             |
# | SAVE_FREQ              | checkpoint 间隔步数（每步目录含 actor/ 与 critic/） | 50            |
# | TEST_FREQ              | 验证间隔步数                      | 5             |
# | VAL_N                  | 验证时每题生成条数（val_kwargs.n；>1 需采样） | 2             |
# | VAL_DO_SAMPLE          | 验证是否随机采样（val_kwargs.do_sample） | true          |
# | VAL_TEMPERATURE        | 验证温度（val_kwargs；do_sample=true 时生效） | 1.0           |
# | VAL_TOP_P              | 验证 top_p（默认与 TOP_P 一致）   | 同 TOP_P      |
# | ENABLE_THINKING        | Qwen thinking 模板，与 SFT 对齐（0/false 关） | 1             |
# | RUN_ID                 | 可选，追加到实验名后便于区分同配置多次运行（如 RUN_ID=1） | 无           |
# | APPEND_RUN_ID          | 设为 1 时自动在实验名后追加时间戳，每次运行目录不同      | 无           |
# | DATA_DIR               | 数据根目录                        | $HYLBASE/data |
# | HYLBASE                | 项目根目录（请用普通用户如 hyl 运行，勿 root） | 默认 SCRIPT_DIR |
# | OUTPUTS                | 训练/评估产物根目录（ppo、hydra、eval_*、默认 W&B 本地目录） | $HYLBASE/hyl_outputs（旧名 OUTPUTS_BASE 仍可读） |
# | WANDB_DIR              | W&B 本地运行目录（见 scripts/hyl_env.sh，默认 \$OUTPUTS/wandb） | 默认在 OUTPUTS 下 |
# | WANDB_CACHE_DIR        | W&B 缓存目录（默认 \$OUTPUTS/.wandb_cache，不写仓库内） | 默认在 OUTPUTS 下 |
# | DRY_RUN                | 1=仅跑 1 步后退出（验证流程）     | 不设则正常训练 |
# | USE_FLASHINFER_SAMPLER | 默认自动（有 cuda/functional 则 FlashInfer）；=1 强制开，=0 强制关 | 自动 |
# | HYL_MULTI_NODE         | 设为 1 时不在单机默认 export NCCL_IB_DISABLE=1（多机 IB 训练需开） | 0           |
# | （环境）               | Conda/CUDA/PYTHONPATH/NCCL/Ray 等见 scripts/hyl_env.sh（由本脚本 source） |             |
# | HYL_RUN_LOG_PREFIX     | 自动日志文件名前缀：stage1/stage2/ppo（run_ppo_stage* 已设） | ppo |
# | HYL_LOG_DIR            | 日志目录（默认 $HYLBASE/hyl_logs） | 见 scripts/hyl_run_log.sh |
# | HYL_AUTO_LOG           | 0=关闭自动 tee 写日志             | 1 |
# 日志说明（不改 verl 源码）：scripts/hyl_env.sh 会设 RAY_DEDUP_LOGS、PYTHONWARNINGS、HF_HUB_*；本脚本通过 Hydra 再给 Ray worker 传 RAY_DEDUP_LOGS。
# 上游 verl 仍会 pprint 整份配置，属库行为；若要极简日志需 fork 或使用 tee+grep 自行过滤。
# 自动日志：默认面向本机多卡节点（如 8×GPU）直接运行；写入 $HYL_LOG_DIR/<prefix>_NNN.log，头部含 host / CUDA_VISIBLE_DEVICES（无 Slurm 也可用）。
# -----------------------------------------------------------------------------
# ------------------------------ 输出目录 / 实验名 -------------------------------
# 默认实验名：直接 run_ppo.sh 为 qwen3_4b_RL1；run_ppo_stage2 未指定时为 qwen3_4b_RL2_m<比例>（默认 m1.0）
# checkpoints 在 $OUTPUTS/ppo/<实验名>/；PPO_PROJECT_NAME=RL-Training 为 W&B 项目名；可设 EXPERIMENT_NAME 覆盖；DRY_RUN=1 加 _test；目录冲突 _run2…
# ------------------------------ 对照试验示例命令 ----------------------------
#
# 【换模型】MODEL_ROOT=/path/to/YourModel-HF ./run_ppo.sh
# 【阶段 1】./run_ppo_stage1.sh
# 【试跑】SAVE_FREQ=1 DRY_RUN=1 ./run_ppo_stage1.sh
# 【阶段 2】./run_ppo_stage2.sh（默认 mask 100%、预训练 critic、冻结、fixed_ratio）；部分 mask：REWARD_MASK_RATIO=0.5 ./run_ppo_stage2.sh
# 【预训练 Critic HF 默认路径】$HYLBASE/qwen3-4b-critic（阶段1 convert 的 --target_dir 建议与此一致；可 CRITIC_MODEL_PATH= 覆盖）
#
# =============================================================================

[[ -n "${DEBUG}" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/hyl_env.sh" ppo

# GPU 可见性须在日志头部之前设置，否则 hyl_print_ppo_header 会误报 cuda_visible=<unset>
N_GPUS="${N_GPUS:-8}"
if [[ "$N_GPUS" -eq 8 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_GPUS - 1)))}"
fi

# ========== 自动日志：stage1_NNN / stage2_NNN / ppo_NNN.log（stdout+stderr 合并）==========
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/hyl_run_log.sh"
if [[ "${HYL_AUTO_LOG:-1}" != "0" ]]; then
  HYL_LOG_FILE=$(hyl_next_log_path "${HYL_RUN_LOG_PREFIX:-ppo}") || exit 1
  [[ -n "${HYL_LOG_FILE:-}" ]] || exit 1
  export HYL_LOG_FILE
  hyl_tee_stdout_stderr "$HYL_LOG_FILE"
  hyl_print_ppo_header
fi

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
  if [[ -n "${HYL_LOG_FILE:-}" ]]; then
    echo "log_file:         ${HYL_LOG_FILE}"
  fi
  echo "exit_code:        ${TRAIN_EXIT:-$?}"
  echo "=========================================="
}
TRAIN_EXIT=""
trap log_training_time EXIT

# ========== 可配置项 ==========
DATA="dapo_math"
MODEL_ROOT="${MODEL_ROOT:-$HYLBASE/Qwen3-4B-SFT}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$MODEL_ROOT}"
BASE_CRITIC_PATH="${BASE_CRITIC_PATH:-$MODEL_ROOT}"
# 阶段 1：critic FSDP → HF 建议保存目录；阶段 2：USE_PRETRAINED_CRITIC=yes 且未设 CRITIC_MODEL_PATH 时默认加载
HF_CRITIC_DIR="${HF_CRITIC_DIR:-$HYLBASE/qwen3-4b-critic}"

DATA_DIR="${DATA_DIR:-$HYLBASE/data}"

# 训练集固定 dapo_math_17k；验证集仅 AIME2024 / AIME2025 / AMC2023（与 run_eval_*.sh 一致）
VAL_FILES_LIST=(
  "${DATA_DIR}/aime2024/test.parquet"
  "${DATA_DIR}/aime2025/test.parquet"
  "${DATA_DIR}/amc2023/test.parquet"
)
TRAIN_FILE="${DATA_DIR}/dapo_math_17k/train.parquet"
# 该 parquet 实际约 179 万行
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-17917}"

# ========== 统一训练超参（不因数据集而变，环境变量可覆盖）==========
ROLLOUT_N="${ROLLOUT_N:-1}"
TOP_P="${TOP_P:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
OVERLONG_BUFFER="${OVERLONG_BUFFER:-4096}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
if [[ "$ENABLE_THINKING" == "1" || "$ENABLE_THINKING" == "true" ]]; then
  PPO_THINKING_VAL="true"
else
  PPO_THINKING_VAL="false"
fi

# ========== 序列长度：与 verl data.max_prompt_length / data.max_response_length 对齐 ==========
DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-16384}"
DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-12288}"

# 容错：若误将后续环境变量写进路径（如 "path DATA=xxx"），只保留路径部分
[[ "$ACTOR_MODEL_PATH" == *" DATA="* ]] && ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH%% DATA=*}"

USE_PRETRAINED_CRITIC="${USE_PRETRAINED_CRITIC:-no}"
if [[ "$USE_PRETRAINED_CRITIC" == "yes" ]] && [[ -z "${CRITIC_MODEL_PATH+x}" ]]; then
  CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-$HF_CRITIC_DIR}"
else
  CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-$BASE_CRITIC_PATH}"
fi

REWARD_MODEL_ENABLE="${REWARD_MODEL_ENABLE:-false}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$CRITIC_MODEL_PATH}"
REWARD_BINARIZE="${REWARD_BINARIZE:-false}"
REWARD_THRESHOLD="${REWARD_THRESHOLD:-0.88}"
REWARD_MASK_RATIO="${REWARD_MASK_RATIO:-0.0}"
REWARD_MASK_TYPE="${REWARD_MASK_TYPE:-fixed_ratio}"
FREEZE_CRITIC="${FREEZE_CRITIC:-false}"
# GAE：lam_actor=0.95, lam_critic=1.0
LAM_ACTOR="${LAM_ACTOR:-0.95}"
LAM_CRITIC="${LAM_CRITIC:-1.0}"
LAM_EXTRA=""
if [[ -n "$LAM_ACTOR" ]]; then LAM_EXTRA="algorithm.lam_actor=$LAM_ACTOR"; fi
if [[ -n "$LAM_CRITIC" ]]; then LAM_EXTRA="${LAM_EXTRA:+$LAM_EXTRA }algorithm.lam_critic=$LAM_CRITIC"; fi
ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-1e-5}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-5}"
# 验证采样：高难度集常用每题多条（pass@2 等）；VAL_N=1 且 VAL_DO_SAMPLE=false 即贪心单次
VAL_N="${VAL_N:-2}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
VAL_TOP_P="${VAL_TOP_P:-$TOP_P}"

# ========== W&B 项目名 + 实验名（checkpoint / wandb / default_local_dir）==========
PPO_PROJECT_NAME="${PPO_PROJECT_NAME:-RL-Training}"
MODEL_SLUG=$(basename "$MODEL_ROOT" | tr '[:upper:]' '[:lower:]' | sed 's/-/_/g')
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_RL1}"
# 试跑/测试时实验名末尾加 _test，避免与正式训练输出混在一起
if [[ -n "${DRY_RUN:-}" ]] && [[ "$DRY_RUN" != "0" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_test"
fi
# 手动指定 RUN_ID 优先；否则自动检测：输出目录已存在时追加 _run2, _run3, ...
if [[ -n "${RUN_ID:-}" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_run${RUN_ID}"
elif [[ -n "${APPEND_RUN_ID:-}" ]] && [[ "$APPEND_RUN_ID" != "0" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_run$(date +%Y%m%d_%H%M%S)"
else
  _ppo_out="${OUTPUTS}/ppo/${EXPERIMENT_NAME}"
  if [[ -d "$_ppo_out" ]]; then
    _n=2
    while [[ -d "${_ppo_out}_run${_n}" ]]; do ((_n++)); done
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_run${_n}"
    echo "输出目录已存在，自动重命名: ${EXPERIMENT_NAME}"
  fi
fi

train_files="['$TRAIN_FILE']"
# 验证集（3 个 benchmark）
_vf_joined=""
for _vf in "${VAL_FILES_LIST[@]}"; do
  [[ -n "$_vf_joined" ]] && _vf_joined="${_vf_joined},"
  _vf_joined="${_vf_joined}'${_vf}'"
done
test_files="[${_vf_joined}]"

# ========== 统一硬件配置（与 normal 对齐：8 卡 128/32，否则 64/16，环境变量可覆盖）==========
[ "$N_GPUS" = "8" ] && { TRAIN_BATCH="${TRAIN_BATCH:-128}"; PPO_MINI="${PPO_MINI:-32}"; } || { TRAIN_BATCH="${TRAIN_BATCH:-64}"; PPO_MINI="${PPO_MINI:-16}"; }
MICRO_PER_GPU="${MICRO_PER_GPU:-2}"
GRAD_CKPT="${GRAD_CKPT:-true}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-50}"
# 与 normal 脚本对齐，vLLM 显存占比 0.5 有利于吞吐
GPU_UTIL="${GPU_UTIL:-0.5}"

# ========== RM 额外参数（reward_model.enable=true 时需 rollout 配置）==========
RM_EXTRA=""
[[ "$REWARD_MODEL_ENABLE" == "true" ]] && RM_EXTRA="reward_model.enable=True reward_model.model.path=$REWARD_MODEL_PATH reward_model.rollout.name=vllm reward_model.rollout.gpu_memory_utilization=$GPU_UTIL reward_model.rollout.tensor_model_parallel_size=1 reward_model.rollout.prompt_length=$DATA_MAX_PROMPT_LENGTH reward_model.rollout.response_length=$DATA_MAX_RESPONSE_LENGTH"

# Overlong buffer（仅 DAPORewardManager 支持；启用时自动切到 reward_manager=dapo，否则 NaiveRewardManager 会报错）
# 约束: data.max_response_length >= OVERLONG_BUFFER（否则 DAPORewardManager 会 assert 失败）
OVERLONG_EXTRA=""
if [[ "$OVERLONG_BUFFER" =~ ^[0-9]+$ ]] && [[ "$OVERLONG_BUFFER" -gt 0 ]]; then
  if [[ "$DATA_MAX_RESPONSE_LENGTH" -lt "$OVERLONG_BUFFER" ]]; then
    echo "ERROR: DATA_MAX_RESPONSE_LENGTH($DATA_MAX_RESPONSE_LENGTH) < OVERLONG_BUFFER($OVERLONG_BUFFER)，请增大 DATA_MAX_RESPONSE_LENGTH"
    exit 1
  fi
  OVERLONG_EXTRA="reward_model.reward_manager=dapo +reward_model.reward_kwargs.max_resp_len=$DATA_MAX_RESPONSE_LENGTH +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True +reward_model.reward_kwargs.overlong_buffer_cfg.len=$OVERLONG_BUFFER +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 +reward_model.reward_kwargs.overlong_buffer_cfg.log=False"
fi

# ========== 运行前路径校验（及早发现配置错误）==========
_path_err=0
[[ ! -f "$TRAIN_FILE" ]] && { echo "ERROR: 训练文件不存在: $TRAIN_FILE"; _path_err=1; }
[[ ! -d "$ACTOR_MODEL_PATH" ]] && { echo "ERROR: Actor 模型目录不存在: $ACTOR_MODEL_PATH"; _path_err=1; }
if [[ "$USE_PRETRAINED_CRITIC" == "yes" ]]; then
  [[ ! -d "$CRITIC_MODEL_PATH" ]] && { echo "ERROR: 预训练 Critic 目录不存在: $CRITIC_MODEL_PATH"; _path_err=1; }
  [[ -d "$CRITIC_MODEL_PATH" ]] && [[ ! -f "$CRITIC_MODEL_PATH/config.json" ]] && { echo "ERROR: Critic 目录缺少 config.json: $CRITIC_MODEL_PATH"; _path_err=1; }
fi
[[ $_path_err -eq 1 ]] && exit 1

# 可选：1 步试跑（仅跑 1 步后退出，用于验证流程；正式跑请勿设 DRY_RUN）
# 试跑时实验名已自动加 _test，checkpoint/wandb 写入 ..._test，不与正式训练混在一起
if [[ -n "${DRY_RUN:-}" ]] && [[ "$DRY_RUN" != "0" ]]; then
  TOTAL_TRAINING_STEPS=1
  echo "DRY_RUN=1: 将仅跑 1 步后退出（TOTAL_TRAINING_STEPS=1），输出目录带 _test"
fi

echo "=========================================="
echo "  【环境与产物】"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  N_GPUS=$N_GPUS"
echo "  OUTPUTS=$OUTPUTS  OUTPUTS_RUN(hydra)=$OUTPUTS_RUN"
echo "  checkpoint_root:  ${OUTPUTS}/ppo/${EXPERIMENT_NAME}"
echo "  RAY_TMPDIR=$RAY_TMPDIR  NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-}  HYL_MULTI_NODE=${HYL_MULTI_NODE:-0}"
echo "  VERL_ATTN_IMPLEMENTATION=$VERL_ATTN_IMPLEMENTATION (critic；+actor sdpa / +critic 见下方 hydra)"
echo "  【训练配置】"
echo "  DATA=$DATA  MODEL_ROOT=$MODEL_ROOT  model_slug=$MODEL_SLUG"
echo "  Train:  $TRAIN_FILE  (max_samples=$TRAIN_MAX_SAMPLES)"
echo "  Val:    $test_files"
echo "  Actor:  $ACTOR_MODEL_PATH"
echo "  Critic: $CRITIC_MODEL_PATH  (pretrained=$USE_PRETRAINED_CRITIC freeze=$FREEZE_CRITIC)"
echo "  HF_Critic 目录（阶段1→HF 建议输出 / 阶段2默认加载）: $HF_CRITIC_DIR"
echo "  RM:     enable=$REWARD_MODEL_ENABLE  path=$REWARD_MODEL_PATH"
echo "  Reward: binarize=$REWARD_BINARIZE thresh=$REWARD_THRESHOLD"
echo "  Mask:   ratio=$REWARD_MASK_RATIO type=$REWARD_MASK_TYPE"
echo "  Seq:    data.max_prompt_length=$DATA_MAX_PROMPT_LENGTH  data.max_response_length=$DATA_MAX_RESPONSE_LENGTH  overlong_buffer=$OVERLONG_BUFFER"
echo "  Batch:  train=$TRAIN_BATCH  mini=$PPO_MINI  micro/gpu=$MICRO_PER_GPU  rollout_n=$ROLLOUT_N"
echo "  GAE:    lam_actor=$LAM_ACTOR  lam_critic=$LAM_CRITIC"
echo "  PPO:    clip_ratio=$CLIP_RATIO  grad_clip=$GRAD_CLIP"
echo "  Actor LR: $ACTOR_LR   Critic LR: $CRITIC_LR   weight_decay=$WEIGHT_DECAY  warmup=$LR_WARMUP_STEPS"
echo "  Epochs: total_epochs=$TOTAL_EPOCHS  DRY_RUN=${DRY_RUN:-0}  total_training_steps=${TOTAL_TRAINING_STEPS:-由 epoch 与数据量决定}"
echo "  Checkpoint: save_freq=$SAVE_FREQ (每步含 actor/ 与 critic/)  test_freq=$TEST_FREQ"
echo "  Sampling: $([ "${VLLM_USE_FLASHINFER_SAMPLER:-}" = "0" ] && echo "PyTorch (无 cuda/functional)" || echo "FlashInfer")"
echo "  W&B:    project=$PPO_PROJECT_NAME  run=$EXPERIMENT_NAME  (本地 \$OUTPUTS/wandb)"
echo "=========================================="

"$PYTHON_CMD" -m verl.trainer.main_ppo \
  hydra.run.dir="$OUTPUTS_RUN" \
  +ray_kwargs.ray_init.runtime_env.env_vars.RAY_DEDUP_LOGS='"1"' \
  algorithm.adv_estimator=gae \
  algorithm.reward_mask_ratio=$REWARD_MASK_RATIO \
  algorithm.reward_mask_type=$REWARD_MASK_TYPE \
  algorithm.reward_binarize=$REWARD_BINARIZE \
  algorithm.reward_threshold=$REWARD_THRESHOLD \
  $LAM_EXTRA \
  $RM_EXTRA \
  data.train_files="$train_files" \
  data.val_files="$test_files" \
  data.train_batch_size=$TRAIN_BATCH \
  +data.gen_batch_size=$TRAIN_BATCH \
  data.train_max_samples=$TRAIN_MAX_SAMPLES \
  data.max_prompt_length=$DATA_MAX_PROMPT_LENGTH \
  data.max_response_length=$DATA_MAX_RESPONSE_LENGTH \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  '+data.apply_chat_template_kwargs={enable_thinking: '"$PPO_THINKING_VAL"'}' \
  actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
  actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
  actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_PER_GPU \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=$GRAD_CKPT \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_PER_GPU \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_PER_GPU \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.top_p=$TOP_P \
  actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
  actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
  actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
  actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((DATA_MAX_PROMPT_LENGTH + DATA_MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((DATA_MAX_PROMPT_LENGTH + DATA_MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((DATA_MAX_PROMPT_LENGTH + DATA_MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((DATA_MAX_PROMPT_LENGTH + DATA_MAX_RESPONSE_LENGTH)) \
  critic.ppo_max_token_len_per_gpu=$((DATA_MAX_PROMPT_LENGTH + DATA_MAX_RESPONSE_LENGTH)) \
  critic.optim.lr=$CRITIC_LR \
  critic.optim.weight_decay=$WEIGHT_DECAY \
  critic.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
  critic.grad_clip=$GRAD_CLIP \
  critic.model.use_remove_padding=False \
  critic.model.path="$CRITIC_MODEL_PATH" \
  +critic.model.override_config.attn_implementation=$VERL_ATTN_IMPLEMENTATION \
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
  trainer.project_name="$PPO_PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.total_epochs=$TOTAL_EPOCHS \
  ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps=$TOTAL_TRAINING_STEPS }\
  $OVERLONG_EXTRA \
  trainer.default_local_dir="${OUTPUTS}/ppo/${EXPERIMENT_NAME}" \
  "$@"

TRAIN_EXIT=$?

exit ${TRAIN_EXIT}
