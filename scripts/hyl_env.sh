#!/usr/bin/env bash
# HYL 项目统一运行环境（conda / CUDA / PyTorch / vLLM / Ray / W&B / PYTHONPATH）
# 由 run_ppo.sh、run_eval_accuracy.sh、run_eval_critic.sh source。
#
# PPO 模式默认：RAY_DEDUP_LOGS、HF_HUB 进度条/遥测、PYTHONWARNINGS（减轻控制台噪声；不修改 verl）。
# 默认 OUTPUTS=$HYLBASE/hyl_outputs、HYL_LOG_DIR=$HYLBASE/hyl_logs；
# Ray：AF_UNIX 全路径须 ≤107 字节，$HYLBASE/ray_tmp 真实路径会超长，故 RAY_TMPDIR 实际为短路径 /tmp/ray_$USER，
# 并在项目根创建符号链接 $HYLBASE/ray_tmp -> 该目录（便于在 hyl 树下定位；物理文件在 /tmp）。
# W&B 默认 WANDB_DIR/WANDB_CACHE_DIR 均在 OUTPUTS 下，避免在仓库根散落 .wandb_cache/。
#
# 用法:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   cd "$SCRIPT_DIR"
#   # shellcheck disable=SC1091
#   source "${SCRIPT_DIR}/scripts/hyl_env.sh" ppo    # 或 eval
#
# 参数 $1: ppo | eval
#
# 可在 source 前 export HYLBASE / OUTPUTS / OUTPUTS_BASE / HYL_LOG_DIR 等覆盖默认。

_HYL_MODE="${1:-ppo}"
HYL_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SCRIPT_DIR="${SCRIPT_DIR:-$HYL_PROJECT_ROOT}"
export HYLBASE="${HYLBASE:-$HYL_PROJECT_ROOT}"

if [[ "$(id -u)" -eq 0 ]]; then
  echo "WARNING: 当前为 root。建议用普通用户（非 root）跑训练/评估；若仍以 root 写默认目录，可能产生属主问题。" >&2
fi

VERL_ENV="verl-ppo-new"
PYTHON_CMD="${SCRIPT_DIR}/miniconda3/envs/${VERL_ENV}/bin/python"
if [[ ! -x "$PYTHON_CMD" ]]; then
  echo "ERROR: 未找到 Conda Python: $PYTHON_CMD （请先创建环境 ${VERL_ENV}）" >&2
  exit 1
fi
export VERL_ENV

export LD_LIBRARY_PATH="${SCRIPT_DIR}/miniconda3/envs/${VERL_ENV}/lib:${LD_LIBRARY_PATH:-}"

if [[ -x "${SCRIPT_DIR}/miniconda3/envs/${VERL_ENV}/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_HOME:-${SCRIPT_DIR}/miniconda3/envs/${VERL_ENV}}"
  export PATH="${SCRIPT_DIR}/miniconda3/envs/${VERL_ENV}/bin:${PATH}"
fi

export PYTHONPATH="${SCRIPT_DIR}/verl:${PYTHONPATH:-}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Ray：socket 路径不能过长。未显式设置 RAY_TMPDIR 时，使用短路径并链到项目根 ray_tmp/
if [[ -z "${RAY_TMPDIR:-}" ]]; then
  export RAY_TMPDIR="/tmp/ray_${USER:-$(id -un)}"
  mkdir -p "$RAY_TMPDIR"
  _HYL_RAY_LINK="${HYLBASE}/ray_tmp"
  if [[ -e "$_HYL_RAY_LINK" ]] && [[ ! -L "$_HYL_RAY_LINK" ]]; then
    echo "WARNING: ${_HYL_RAY_LINK} 已存在且为普通目录/文件，未改为符号链接。" >&2
    echo "  Ray 仍使用 RAY_TMPDIR=$RAY_TMPDIR。若需链接，请自行移走 ${_HYL_RAY_LINK} 后重新 source 本脚本。" >&2
  else
    ln -sfn "$RAY_TMPDIR" "$_HYL_RAY_LINK"
  fi
  unset _HYL_RAY_LINK
fi

if [[ "$_HYL_MODE" == "ppo" ]]; then
  rm -f /tmp/rl-colocate-zmq-*.sock 2>/dev/null || true
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
  # 日志（仅环境变量，不改 verl）：Ray 合并重复行；HF Hub；常见 Python 警告
  export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
  export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
  export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
  export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::UserWarning:verl.utils.tokenizer}"
  # 单机多卡：无可用 IB / 容器内 ibv 打不开时 NCCL 会对 ibv_open_device 刷 WARN；禁 IB 后仍走 NVLink/PCIe。
  # 多机且依赖 InfiniBand 时请 export HYL_MULTI_NODE=1（或自行 export NCCL_IB_DISABLE=0）。
  if [[ "${HYL_MULTI_NODE:-0}" != "1" ]] && ! [[ -v NCCL_IB_DISABLE ]]; then
    export NCCL_IB_DISABLE=1
  fi
else
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:False}"
fi

# Ray：消除「未来将不再在 num_gpus=0 时覆盖 CUDA_VISIBLE_DEVICES」的 FutureWarning（Ray 2.54+）
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"

# raylet 若报 libtinfo.so.6: no version information：多为 conda ncurses 与系统 bash 混用，一般可忽略。
# 若需消除：在 verl-ppo-new 内尝试 conda install -y ncurses 或使用未注入 conda lib 的 shell 启动训练。
# Worker 若刷「Flash Attention 2 only supports float16/bfloat16 but dtype is float32」：多为加载瞬间 dtype 提示，FSDP 常会再 cast；持续异常再考虑 critic 侧 attn 改为 sdpa。

if "$PYTHON_CMD" -c "import flash_attn" 2>/dev/null; then
  export VERL_ATTN_IMPLEMENTATION="${VERL_ATTN_IMPLEMENTATION:-flash_attention_2}"
else
  export VERL_ATTN_IMPLEMENTATION="${VERL_ATTN_IMPLEMENTATION:-sdpa}"
fi

# numpy 2.x：与 verl/requirements.txt 的 numpy<2 不一致；当前环境能 import 即可，遇数值/扩展问题再考虑降级
if [[ "$_HYL_MODE" == "ppo" ]]; then
  _hyl_np_major=$("$PYTHON_CMD" -c "import numpy as _n; print(int(_n.__version__.split('.')[0]))" 2>/dev/null) || _hyl_np_major=0
  if [[ "$_hyl_np_major" -ge 2 ]]; then
    echo "NOTE: numpy 主版本为 ${_hyl_np_major}（上游 verl 常写 numpy<2）；若训练异常可尝试: pip install 'numpy<2.0.0'" >&2
  fi
  unset _hyl_np_major
fi

if [[ -d "${SCRIPT_DIR}/local/cuda-12.2" ]]; then
  export CUDA_HOME="${SCRIPT_DIR}/local/cuda-12.2"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

export OUTPUTS="${OUTPUTS:-${OUTPUTS_BASE:-${HYLBASE}/hyl_outputs}}"

if [[ "$_HYL_MODE" == "ppo" ]]; then
  export OUTPUTS_RUN="${OUTPUTS}/hydra"
fi

export HYL_LOG_DIR="${HYL_LOG_DIR:-${HYLBASE}/hyl_logs}"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  if [[ ! -r "${SCRIPT_DIR}/.env" ]]; then
    echo "ERROR: 项目根 .env 存在但当前用户不可读: ${SCRIPT_DIR}/.env" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env"
  set +a
fi

# .env 常把 RAY_TMPDIR 设为 $HYLBASE/ray_tmp；该真实路径 + Ray 的 session/socket 后缀会超过 AF_UNIX 107 字节上限。
# 在加载 .env 之后统一校验：过长则强制改为短路径，并维护 $HYLBASE/ray_tmp -> 该路径的符号链接。
_HYL_RAY_REAL="/tmp/ray_${USER:-$(id -un)}"
_HYL_RAY_SESS="session_2099-12-31_23-59-59_1234567890_1234567890"
_RAY_SOCK_TEST="${RAY_TMPDIR%/}/ray/${_HYL_RAY_SESS}/sockets/plasma_store"
if [[ -z "${RAY_TMPDIR:-}" ]]; then
  export RAY_TMPDIR="$_HYL_RAY_REAL"
  mkdir -p "$RAY_TMPDIR"
elif [[ ${#_RAY_SOCK_TEST} -gt 107 ]]; then
  echo "WARNING: RAY_TMPDIR=${RAY_TMPDIR} 会导致 Ray socket 路径超过 107 字节（已改为 ${_HYL_RAY_REAL}）。" >&2
  echo "  请勿在 .env 中设置 RAY_TMPDIR=\$HYLBASE/ray_tmp；项目下 ray_tmp 仅作符号链接入口。" >&2
  export RAY_TMPDIR="$_HYL_RAY_REAL"
  mkdir -p "$RAY_TMPDIR"
fi
if [[ "$RAY_TMPDIR" == "$_HYL_RAY_REAL" ]]; then
  _HYL_RAY_LINK="${HYLBASE}/ray_tmp"
  if [[ -e "$_HYL_RAY_LINK" ]] && [[ ! -L "$_HYL_RAY_LINK" ]]; then
    echo "WARNING: ${_HYL_RAY_LINK} 已存在且非符号链接，未覆盖。" >&2
  else
    ln -sfn "$RAY_TMPDIR" "$_HYL_RAY_LINK"
  fi
  unset _HYL_RAY_LINK
fi
unset _HYL_RAY_REAL _HYL_RAY_SESS _RAY_SOCK_TEST

# W&B：默认全部落在 OUTPUTS 下（与 checkpoint/hydra 同源、同一套写权限），避免在仓库根写 .wandb_cache/、./wandb/ 导致多用户属主冲突
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${OUTPUTS}/.wandb_cache}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUTS}/wandb}"

if [[ "${USE_FLASHINFER_SAMPLER:-}" == "1" ]]; then
  :
elif [[ "${USE_FLASHINFER_SAMPLER:-}" == "0" ]]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
else
  _hyl_cuda_functional="${CUDA_HOME:-/usr/local/cuda}/include/cuda/functional"
  if [[ ! -f "$_hyl_cuda_functional" ]]; then
    export VLLM_USE_FLASHINFER_SAMPLER=0
  fi
  unset _hyl_cuda_functional
fi

# ---------- 写权限：失败立即退出，避免中途才报错 ----------
hyl_require_writable() {
  local dir="$1"
  local label="$2"
  if ! mkdir -p "$dir" 2>/dev/null; then
    echo "ERROR: 无法创建目录 $label: $dir" >&2
    echo "  当前用户: $(id -un) (uid=$(id -u))，请确认对父目录有写权限。" >&2
    exit 1
  fi
  if [[ ! -w "$dir" ]]; then
    echo "ERROR: 当前用户 $(id -un) 对 $label 无写权限: $dir" >&2
    echo "  可显式指定可写路径，例如:" >&2
    echo "    export OUTPUTS=\"/path/可写/hyl_outputs\" HYL_LOG_DIR=\"/path/可写/hyl_logs\"" >&2
    exit 1
  fi
}

hyl_require_writable "$OUTPUTS" "OUTPUTS（训练/评估产物根）"
hyl_require_writable "$WANDB_CACHE_DIR" "WANDB_CACHE_DIR（W&B 缓存，默认 \$OUTPUTS/.wandb_cache）"
hyl_require_writable "$WANDB_DIR" "WANDB_DIR（W&B 本地运行目录，默认 \$OUTPUTS/wandb）"
hyl_require_writable "$RAY_TMPDIR" "RAY_TMPDIR"
hyl_require_writable "$HYL_LOG_DIR" "HYL_LOG_DIR（日志）"

if [[ "$_HYL_MODE" == "ppo" ]]; then
  hyl_require_writable "$OUTPUTS_RUN" "OUTPUTS/hydra（OUTPUTS_RUN）"
  hyl_require_writable "${OUTPUTS_RUN}/validation" "Hydra validation 目录"
fi

unset -f hyl_require_writable
unset _HYL_MODE HYL_PROJECT_ROOT
