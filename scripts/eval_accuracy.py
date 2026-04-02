#!/usr/bin/env python3
"""
6 数据集模型准确率评估（不分阶段，任意 checkpoint 均可；rule-based，不依赖 critic）。

流程：加载 verl 格式 parquet → 用 vLLM 生成 → 用 default_compute_score 判对错 → 输出准确率并可选上报 wandb。

用法:
  # 单数据集
  python scripts/eval_accuracy.py \
    --actor_path /path/to/model \
    --data_path data/aime2024/test.parquet \
    --max_samples 0

  # 启用 Qwen3 thinking 模式（与 PPO 一致）
  python scripts/eval_accuracy.py \
    --actor_path /path/to/model \
    --data_path data/aime2024/test.parquet \
    --enable_thinking

  # 6 数据集 + wandb（推荐用 run_eval_accuracy.sh）
  python scripts/eval_accuracy.py --actor_path ... --data_paths ... --wandb_project Eval-Accuracy
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
# Do not add verl to sys.path here: scoring uses _get_default_compute_score() to avoid importing verl (tensordict/libpython.so)


def load_dataset(data_path: str, max_samples: int = 0):
    """加载 parquet，返回 prompts, ground_truths, data_sources。max_samples=0 表示不限制。"""
    df = pd.read_parquet(data_path)
    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    prompts, ground_truths, data_sources = [], [], []
    for _, row in df.iterrows():
        prompts.append(row["prompt"])
        ground_truths.append(row["reward_model"]["ground_truth"])
        data_sources.append(row["data_source"])

    print(f"[Data] Loaded {len(prompts)} samples from {data_path}")
    return prompts, ground_truths, data_sources


# 与论文设置对齐：Sequence cutoff length ≈ 32k
# - max_model_len: vLLM 总上下文长度（prompt + think + answer）
# - max_new_tokens: 生成长度上限，同样给到 32k（benchmarks 的 prompt 都很短，vLLM 会自动 cap 到 max_model_len - prompt_len）
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_MAX_NEW_TOKENS = 32768


def _truncate(text: str, max_len: int = 500) -> str:
    """表格展示用截断。"""
    if not text or len(text) <= max_len:
        return text or ""
    return text[:max_len].rstrip() + "…"


def _dataset_display_name(data_path: str) -> str:
    """从路径得到清晰数据集名，如 aime2024 / amc2023 / gpqa_diamond。"""
    parent = os.path.basename(os.path.dirname(os.path.abspath(data_path)))
    if parent and parent not in (".", "data"):
        return parent
    return os.path.splitext(os.path.basename(data_path))[0] or "unknown"


def build_prompt_texts(tokenizer, prompts: list, enable_thinking: bool = False):
    """根据 tokenizer 和 enable_thinking 将 prompts 转为 vLLM 输入字符串。"""
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking:
        kwargs["enable_thinking"] = True  # Qwen3 think 模式
    try:
        prompt_texts = [tokenizer.apply_chat_template(p, **kwargs) for p in prompts]
    except TypeError:
        kwargs.pop("enable_thinking", None)
        prompt_texts = [tokenizer.apply_chat_template(p, **kwargs) for p in prompts]
    return prompt_texts


def create_llm(
    actor_path: str,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.5,
    tensor_parallel_size: int = 1,
    attn_implementation: str | None = None,
):
    """创建并返回 vLLM LLM 实例，供多次 generate 复用。attn_implementation 仅 vLLM >= 0.12 支持。"""
    from vllm import LLM

    kwargs = dict(
        model=actor_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )
    # vLLM 0.11.x 的 EngineArgs 不支持 attn_implementation，传入会报 TypeError
    vllm_version = getattr(__import__("vllm"), "__version__", "0.0.0")
    if attn_implementation and vllm_version >= "0.12.0":
        kwargs["attn_implementation"] = attn_implementation
    return LLM(**kwargs)


def generate_with_llm(llm, prompt_texts: list, max_new_tokens: int, temperature: float = 0.6, top_p: float = 0.95):
    """用已创建的 LLM 做一次生成，返回 responses。"""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompt_texts, sampling_params)
    return [o.outputs[0].text for o in outputs]


def generate_rollouts(
    actor_path: str,
    tokenizer,
    prompts: list,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    temperature: float = 0.6,
    top_p: float = 0.95,
    gpu_memory_utilization: float = 0.5,
    tensor_parallel_size: int = 1,
    enable_thinking: bool = False,
):
    """用 vLLM 生成，返回 (prompt_texts, responses)。单次调用：内部创建并销毁 LLM。"""
    prompt_texts = build_prompt_texts(tokenizer, prompts, enable_thinking)
    llm = create_llm(actor_path, max_model_len, gpu_memory_utilization, tensor_parallel_size)
    responses = generate_with_llm(llm, prompt_texts, max_new_tokens, temperature, top_p)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Rollout] Generated {len(responses)} responses")
    return prompt_texts, responses


def _get_default_compute_score():
    """Load default_compute_score without importing verl package (avoids tensordict/libpython3.10.so)."""
    import importlib.util
    import types

    verl_verl = os.path.join(PROJECT_DIR, "verl", "verl")
    if verl_verl not in sys.path:
        sys.path.insert(0, verl_verl)

    # Build stub hierarchy: verl -> verl.utils -> verl.utils.import_utils
    verl_mod = types.ModuleType("verl")
    verl_mod.__path__ = [verl_verl]
    utils_mod = types.ModuleType("verl.utils")
    utils_mod.__path__ = [os.path.join(verl_verl, "utils")]
    import_utils_mod = types.ModuleType("verl.utils.import_utils")
    import_utils_mod.deprecated = lambda replacement="": lambda fn: fn
    verl_mod.utils = utils_mod
    utils_mod.import_utils = import_utils_mod
    sys.modules["verl"] = verl_mod
    sys.modules["verl.utils"] = utils_mod
    sys.modules["verl.utils.import_utils"] = import_utils_mod

    # Load reward_score package
    rs_dir = os.path.join(verl_verl, "utils", "reward_score")
    rs_init = os.path.join(rs_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "verl.utils.reward_score", rs_init, submodule_search_locations=[rs_dir]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verl.utils.reward_score"] = mod
    utils_mod.reward_score = mod
    spec.loader.exec_module(mod)
    return mod.default_compute_score


def compute_rewards(responses, ground_truths, data_sources):
    """Rule-based 正确率。"""
    default_compute_score = _get_default_compute_score()

    rewards = []
    for resp, gt, ds in zip(responses, ground_truths, data_sources):
        result = default_compute_score(data_source=ds, solution_str=resp, ground_truth=gt)
        if isinstance(result, dict):
            score = float(result.get("score", result.get("acc", 0)))
        else:
            score = float(result)
        rewards.append(max(score, 0.0))

    return np.array(rewards)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy on benchmarks (any checkpoint, rule-based only)")
    parser.add_argument("--actor_path", type=str, required=True, help="Model path (HF)")
    parser.add_argument("--data_path", type=str, default=None, help="Single test parquet (ignored if --data_paths set)")
    parser.add_argument("--data_source", type=str, default=None,
                        help="Override data_source for all rows (default: use column)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to evaluate (0 = all)")
    parser.add_argument("--math_lighteval_max_samples", type=int, default=0,
                        help="Max samples for math_lighteval only (0 = no limit). Used to speed up eval.")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Max generation length (default 32768; vLLM auto-caps to max_model_len - prompt_len)")
    parser.add_argument("--max_model_len", type=int, default=DEFAULT_MAX_MODEL_LEN,
                        help="vLLM total context length including prompt + generation")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable Qwen3 thinking mode (same as PPO)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--n_gpus", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--attn_implementation", type=str, default=None,
                        help="Attention backend: flash_attention_2, sdpa, eager, etc. (default: None = vLLM default)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Save rollouts and results here (default: outputs/eval_accuracy/<actor_name>/<dataset>)")
    parser.add_argument("--run_suffix", type=str, default=None,
                        help="If set, save under outputs/eval_accuracy/<actor_name>_<run_suffix>/ to avoid overwriting (e.g. --run_suffix 20250310_1)")
    parser.add_argument("--save_rollouts", action="store_true", default=True,
                        help="Save rollouts.jsonl")
    parser.add_argument("--data_paths", type=str, default=None,
                        help="Comma-separated list of parquet paths (eval each and log one wandb table)")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (if set, log results and one summary table)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (default: eval_<actor_name>)")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Run N times and average accuracy (for stochastic sampling)")
    args = parser.parse_args()

    # 多数据集：解析 data_paths，逐个评估后统一打表
    if args.data_paths:
        data_path_list = [p.strip() for p in args.data_paths.split(",") if p.strip()]
    elif args.data_path:
        data_path_list = [args.data_path]
    else:
        raise ValueError("Specify --data_path or --data_paths")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.actor_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    table_rows = []  # [(dataset_name, accuracy, num_correct, num_total), ...]
    # case 表：每条记录包含完整的 prompt/response 文本，便于在 wandb 中逐题查看
    sample_rows = []  # [(dataset, correct, ground_truth, prompt_full, response_full), ...]
    actor_name = os.path.basename(os.path.normpath(args.actor_path))
    output_name = f"{actor_name}_{args.run_suffix}" if getattr(args, "run_suffix", None) else actor_name
    n_samples_per_type = 20   # 每个数据集默认最多 20 条正确 + 20 条错误样本

    llm = create_llm(
        args.actor_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.n_gpus,
        attn_implementation=args.attn_implementation,
    )

    for data_path in data_path_list:
        if not os.path.isfile(data_path):
            print(f"[Skip] File not found: {data_path}")
            continue

        dataset_name = _dataset_display_name(data_path)
        max_here = args.max_samples
        if dataset_name == "math_lighteval" and getattr(args, "math_lighteval_max_samples", 0) > 0:
            max_here = args.math_lighteval_max_samples
        prompts, ground_truths, data_sources = load_dataset(data_path, max_here)
        if args.data_source:
            data_sources = [args.data_source] * len(prompts)

        n_total = len(prompts)
        acc_list = []
        prompt_texts = build_prompt_texts(tokenizer, prompts, args.enable_thinking)

        for run_i in range(args.n_runs):
            run_label = f" (run {run_i + 1}/{args.n_runs})" if args.n_runs > 1 else ""
            print(f"  [{dataset_name}]{run_label} generating...")
            t0 = time.time()
            responses = generate_with_llm(
                llm,
                prompt_texts,
                args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"  Generation took {time.time() - t0:.1f}s")

            rewards = compute_rewards(responses, ground_truths, data_sources)
            acc_list.append(float(rewards.mean()))

        acc = float(np.mean(acc_list))
        acc_std = float(np.std(acc_list)) if args.n_runs > 1 else 0.0
        n_correct = int(acc * n_total)  # 近似，用于表格展示
        table_rows.append((dataset_name, acc, n_correct, n_total))

        # 每个数据集采若干样本写入 wandb 表格（对/错各若干条，含问答与打分；用最后一轮）
        correct_idx = [i for i in range(n_total) if rewards[i] == 1.0]
        wrong_idx = [i for i in range(n_total) if rewards[i] == 0.0]
        for idx in correct_idx[:n_samples_per_type] + wrong_idx[:n_samples_per_type]:
            sample_rows.append((
                dataset_name,
                int(rewards[idx]),
                float(rewards[idx]),
                str(ground_truths[idx]),
                prompt_texts[idx] or "",
                responses[idx] or "",
            ))

        print("\n" + "=" * 60)
        print("模型准确率 (rule-based)")
        print("=" * 60)
        print(f"  Model:     {args.actor_path}")
        print(f"  Data:      {data_path}")
        if args.n_runs > 1:
            print(f"  Runs:      {args.n_runs} (mean ± std)")
            print(f"  Accuracy:  {acc:.4f} ± {acc_std:.4f}")
        else:
            print(f"  Correct:   {n_correct} / {n_total}")
            print(f"  Accuracy:  {acc:.4f}")
        print("=" * 60)

        if args.output_dir is None:
            out_dir = os.path.join(PROJECT_DIR, "outputs", "eval_accuracy", output_name, dataset_name)
        else:
            out_dir = args.output_dir if len(data_path_list) == 1 else os.path.join(
                args.output_dir, dataset_name
            )
        os.makedirs(out_dir, exist_ok=True)

        results = {
            "actor_path": args.actor_path,
            "data_path": data_path,
            "num_samples": n_total,
            "num_correct": n_correct,
            "accuracy": acc,
            "n_runs": args.n_runs,
            "accuracy_std": acc_std if args.n_runs > 1 else None,
            "enable_thinking": args.enable_thinking,
            "eval_time": datetime.now().isoformat(),
        }
        if getattr(args, "run_suffix", None):
            results["run_suffix"] = args.run_suffix
        results_path = os.path.join(out_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[Save] Results -> {results_path}")

        if args.save_rollouts and responses is not None:
            rollout_path = os.path.join(out_dir, "rollouts.jsonl")
            with open(rollout_path, "w") as f:
                for i in range(len(responses)):
                    f.write(json.dumps({
                        "idx": i,
                        "prompt_text": prompt_texts[i],
                        "response": responses[i],
                        "ground_truth": ground_truths[i],
                        "data_source": data_sources[i],
                        "reward": float(rewards[i]),
                    }, ensure_ascii=False) + "\n")
            print(f"[Save] Rollouts -> {rollout_path}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # 写入 wandb：一个 run、汇总表 + 样本表
    if args.wandb_project and table_rows:
        try:
            import wandb
            run_name = args.wandb_run_name or f"eval_{actor_name}"
            wandb.init(project=args.wandb_project, name=run_name, config={
                "actor_path": args.actor_path,
                "enable_thinking": args.enable_thinking,
                "max_samples": args.max_samples,
                "max_model_len": args.max_model_len,
                "max_new_tokens": args.max_new_tokens,
                "n_runs": args.n_runs,
                "math_lighteval_max_samples": getattr(args, "math_lighteval_max_samples", 0),
            })
            # 表1：各数据集准确率（dataset 列为 aime2024 / aime2025 / amc2023 等）
            summary_cols = ["dataset", "accuracy", "num_correct", "num_total"]
            summary_table = wandb.Table(columns=summary_cols, data=[
                [r[0], round(r[1], 4), r[2], r[3]] for r in table_rows
            ])
            mean_acc = sum(r[1] for r in table_rows) / len(table_rows)
            wandb.log({
                "eval_accuracy/summary": summary_table,
                # 与训练时 val-core/avg_acc/mean_across_benchmarks 对齐：多测试集 acc 的简单平均
                "eval_accuracy/mean_across_benchmarks": float(mean_acc),
            })
            wandb.summary["accuracy_mean"] = round(mean_acc, 4)
            # 表2：case 问答与打分（便于查看对/错例题，含完整 prompt/response 与 score）
            if sample_rows:
                sample_cols = ["dataset", "correct", "score", "ground_truth", "prompt", "response"]
                sample_table = wandb.Table(columns=sample_cols, data=sample_rows)
                wandb.log({"eval_accuracy/samples": sample_table})
            wandb.finish()
            print(f"\n[W&B] Logged to project={args.wandb_project} run={run_name} "
                  f"(summary + {len(sample_rows)} sample cases)")
        except Exception as e:
            print(f"\n[W&B] Log failed: {e}")


if __name__ == "__main__":
    main()
