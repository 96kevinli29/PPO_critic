#!/usr/bin/env bash
# hyl_run_log.sh — 自动编号日志 + tee（stdout/stderr 合并到同一 .log）
# 典型场景：本机多卡节点直接 bash 跑（无 Slurm）；若在集群上提交作业，可选打印 SLURM_JOB_ID。

# 环境变量:
#   HYL_RUN_LOG_PREFIX   日志文件名前缀，如 stage1 / stage2 / ppo / eval_accuracy / eval_critic
#   HYL_LOG_DIR          日志目录（默认 $HYLBASE/hyl_logs；run 脚本会先 source hyl_env.sh 设置）
#   HYL_AUTO_LOG=0       关闭自动 tee
#
# 导出:
#   HYL_LOG_FILE         当前 run 的完整日志路径

hyl_next_log_path() {
  local prefix="${1:-ppo}"
  local log_dir="${HYL_LOG_DIR:-${HYLBASE:-$SCRIPT_DIR}/hyl_logs}"
  if ! mkdir -p "$log_dir" 2>/dev/null; then
    echo "ERROR: 无法创建日志目录: $log_dir （当前用户 $(id -un)）" >&2
    return 1
  fi
  if [[ ! -w "$log_dir" ]]; then
    echo "ERROR: 日志目录不可写: $log_dir （当前用户 $(id -un)）" >&2
    return 1
  fi
  local n=1
  local f
  while true; do
    f="${log_dir}/${prefix}_$(printf '%03d' $n).log"
    if [[ ! -e "$f" ]]; then
      echo "$f"
      return 0
    fi
    ((n++)) || true
  done
}

hyl_tee_stdout_stderr() {
  local log_file="$1"
  exec > >(tee -a "$log_file") 2>&1
}

hyl_print_ppo_header() {
  local _base="${HYLBASE:-${SCRIPT_DIR}}"
  local _py="${SCRIPT_DIR}/miniconda3/envs/${VERL_ENV:-verl-ppo-new}/bin/python"
  local _stack=""
  if [[ -x "$_py" ]]; then
    _stack=$("$_py" -c "import torch, transformers, vllm, numpy, ray; print(f'torch={torch.__version__} transformers={transformers.__version__} vllm={vllm.__version__} numpy={numpy.__version__} ray={ray.__version__}')" 2>/dev/null) || _stack="(未能导入 torch/transformers/vllm/numpy/ray)"
  else
    _stack="(未找到 $_py)"
  fi
  echo "=============================================="
  echo "HYL PPO [${HYL_RUN_LOG_PREFIX:-ppo}]"
  echo "=============================================="
  echo "log_file:         ${HYL_LOG_FILE}"
  echo "user:             $(id -un) uid=$(id -u)"
  echo "host:             $(hostname 2>/dev/null || echo unknown)"
  echo "workdir:          ${SCRIPT_DIR}"
  echo "cuda_visible:     ${CUDA_VISIBLE_DEVICES:-<unset>}"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "slurm_job_id:     ${SLURM_JOB_ID}"
  fi
  echo "conda_env:        ${VERL_ENV:-verl-ppo-new}"
  echo "python:           ${_py}"
  echo "stack:            ${_stack}"
  echo "outputs:          ${OUTPUTS:-<unset>}  hydra: ${OUTPUTS_RUN:-<unset>}"
  echo "ray_tmp:          ${RAY_TMPDIR:-<unset>}"
  echo "nccl_ib:          ${NCCL_IB_DISABLE:-<unset>}  HYL_MULTI_NODE=${HYL_MULTI_NODE:-0}"
  echo "verl_attn:        ${VERL_ATTN_IMPLEMENTATION:-<unset>}"
  echo "started_at:       $(date -Iseconds)"
  echo "=============================================="
}

hyl_print_eval_accuracy_header() {
  echo "=============================================="
  echo "Eval-Accuracy (AIME2024 / AIME2025 / AMC2023)"
  echo "=============================================="
  echo "log_file:         ${HYL_LOG_FILE:-N/A}"
  echo "host:             $(hostname 2>/dev/null || echo unknown)"
  echo "cuda_visible:     ${CUDA_VISIBLE_DEVICES:-<unset>}"
  [[ -n "${SLURM_JOB_ID:-}" ]] && echo "slurm_job_id:     ${SLURM_JOB_ID}"
  echo "actor_path:       ${ACTOR_PATH}"
  echo "data_dir:         ${DATA_DIR}"
  echo "wandb_project:    ${WANDB_PROJECT}"
  echo "n_gpus:           ${N_GPUS}"
  echo "n_runs:           ${N_RUNS}"
  echo "max_new_tokens:   ${MAX_NEW_TOKENS}"
  echo "temperature:      ${TEMPERATURE}"
  echo "enable_thinking:  ${ENABLE_THINKING}"
  echo "output_dir:       ${_eval_out_base:-}"
  echo "sampling:         $([ "${VLLM_USE_FLASHINFER_SAMPLER:-}" = "0" ] && echo "PyTorch (无 cuda/functional)" || echo "FlashInfer")"
  echo "started_at:       $(date -Iseconds)"
  echo "=============================================="
}

hyl_print_eval_critic_header() {
  echo "=============================================="
  echo "Critic Prediction Eval — AIME / AMC"
  echo "=============================================="
  echo "log_file:         ${HYL_LOG_FILE:-N/A}"
  echo "host:             $(hostname 2>/dev/null || echo unknown)"
  echo "cuda_visible:     ${CUDA_VISIBLE_DEVICES:-<unset>}"
  [[ -n "${SLURM_JOB_ID:-}" ]] && echo "slurm_job_id:     ${SLURM_JOB_ID}"
  echo "model_family:     ${MODEL_FAMILY}  tag: ${MODEL_TAG}"
  echo "actor_path:       ${ACTOR_PATH}"
  echo "critic_path:      ${CRITIC_PATH}"
  echo "experiment_tag:   ${DATA}"
  echo "benchmarks:       ${BENCHMARK_NAMES[*]}"
  echo "max_samples:      ${MAX_SAMPLES}"
  echo "threshold:        ${THRESHOLD}"
  [[ "${THRESHOLD_SWEEP:-0}" == "1" ]] && echo "threshold_sweep:  yes"
  echo "n_gpus:           ${N_GPUS}"
  echo "wandb_project:    ${WANDB_PROJECT}"
  echo "output_base_dir:  ${OUTPUT_BASE_DIR}"
  [[ -n "${ROLLOUT_FILE:-}" ]] && echo "rollout_file:     ${ROLLOUT_FILE}"
  echo "started_at:       $(date -Iseconds)"
  echo "=============================================="
}
