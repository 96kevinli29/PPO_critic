#!/usr/bin/env python3
"""
评估预训练 critic 模型的 reward 预测能力。

流程:
  1. 用 vLLM 加载 actor 生成 rollout（或从文件加载已有 rollout）
  2. 用 rule-based 方法计算 ground-truth reward (0/1)
  3. 用预训练 critic 前向计算，取最后 token 的 value 作为预测分数
  4. 以 0.5 为阈值二分类，计算 accuracy、AUC、相关性等指标

用法:
  python scripts/eval_critic_prediction.py \
      --actor_path  ./Qwen3-0.6B \
      --critic_path ./critic_qwen3_0.6b_ppo_math_lighteval_hf \
      --data_path   ./data/math_lighteval/test.parquet \
      --data_source lighteval/MATH \
      --max_samples 1000 \
      --output_dir  ./eval_critic_results
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "verl"))


# ─────────────────────────── 1. 数据加载 ───────────────────────────

def load_dataset(data_path: str, max_samples: int = 1000):
    """加载 parquet 数据集，返回 prompts, ground_truths, data_sources。"""
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


# ─────────────────────────── 2. Rollout 生成 ───────────────────────

def generate_rollouts(
    actor_path: str,
    tokenizer,
    prompts: list,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.95,
    gpu_memory_utilization: float = 0.5,
    tensor_parallel_size: int = 1,
):
    """用 vLLM 生成 rollout，返回 (prompt_texts, responses)。"""
    from vllm import LLM, SamplingParams

    prompt_texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    llm = LLM(
        model=actor_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_new_tokens + 2048,
        tensor_parallel_size=tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    outputs = llm.generate(prompt_texts, sampling_params)
    responses = [o.outputs[0].text for o in outputs]

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[Rollout] Generated {len(responses)} responses")
    return prompt_texts, responses


# ─────────────────────────── 3. Ground-truth Reward ────────────────

def compute_rewards(responses, ground_truths, data_sources):
    """计算 rule-based ground-truth reward (0/1)。"""
    from verl.utils.reward_score import default_compute_score

    rewards = []
    for resp, gt, ds in zip(responses, ground_truths, data_sources):
        result = default_compute_score(data_source=ds, solution_str=resp, ground_truth=gt)
        if isinstance(result, dict):
            score = float(result.get("score", result.get("acc", 0)))
        else:
            score = float(result)
        rewards.append(max(score, 0.0))

    rewards = np.array(rewards)
    print(f"[Reward] Accuracy (ground-truth): {rewards.mean():.4f}  "
          f"({int(rewards.sum())}/{len(rewards)})")
    return rewards


# ─────────────────────────── 4. Critic Value 计算 ──────────────────

def compute_critic_values(
    critic_path: str,
    tokenizer,
    prompt_texts: list[str],
    responses: list[str],
    batch_size: int = 8,
    max_length: int = 2048,
    n_gpus: int = 1,
):
    """加载 critic，对 (prompt+response) 做前向，取最后有效 token 的 value。"""
    from transformers import AutoModelForTokenClassification

    if n_gpus > 1:
        model = AutoModelForTokenClassification.from_pretrained(
            critic_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map="auto",
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            critic_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        model.cuda()
    model.eval()

    input_device = next(model.parameters()).device

    orig_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"

    all_values = []
    n = len(prompt_texts)

    for i in range(0, n, batch_size):
        batch_full = [p + r for p, r in zip(prompt_texts[i : i + batch_size],
                                             responses[i : i + batch_size])]
        enc = tokenizer(
            batch_full,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(input_device)
        attention_mask = enc["attention_mask"].to(input_device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits = logits.float().squeeze(-1)  # (batch, seq_len)

        for j in range(logits.size(0)):
            seq_len = int(attention_mask[j].sum().item())
            last_val = logits[j, seq_len - 1].item()
            all_values.append(last_val)

        if (i // batch_size) % 20 == 0:
            print(f"  Critic forward: {min(i + batch_size, n)}/{n}")

    tokenizer.truncation_side = orig_truncation_side

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[Critic] Computed values for {len(all_values)} samples")
    return np.array(all_values)


# ─────────────────────────── 5. 评估指标 ───────────────────────────

def _auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC-ROC：随机抽一个正/负样本，critic 给正样本更高分的概率。"""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    tpr = np.concatenate([[0], tp / n_pos])
    fpr = np.concatenate([[0], fp / n_neg])
    return float(np.trapz(tpr, fpr))


def evaluate_metrics(values: np.ndarray, rewards: np.ndarray, threshold: float = 0.5):
    """
    核心指标（无 sklearn/scipy 依赖）：
      1. AUC-ROC — critic 是否倾向于给正确答案更高 value（>0.5 有区分能力）
      2. Value 分离度 — 正确/错误样本的 value 均值差
    """
    has_both = len(np.unique(rewards)) > 1
    auc = _auc_roc(rewards, values) if has_both else float("nan")

    correct_mask = rewards == 1.0
    wrong_mask = rewards == 0.0
    gt_acc = float(correct_mask.mean())
    majority_baseline = max(gt_acc, 1.0 - gt_acc)

    val_correct = float(values[correct_mask].mean()) if correct_mask.any() else None
    val_wrong = float(values[wrong_mask].mean()) if wrong_mask.any() else None
    val_separation = (val_correct - val_wrong) if (val_correct is not None and val_wrong is not None) else None

    preds = (values > threshold).astype(float)
    acc = float((rewards == preds).mean())

    results = {
        "num_samples": int(len(values)),
        "num_correct": int(correct_mask.sum()),
        "num_wrong": int(wrong_mask.sum()),
        "ground_truth_accuracy": gt_acc,
        "majority_baseline": majority_baseline,
        "auc_roc": float(auc) if not np.isnan(auc) else None,
        "value_mean_correct": val_correct,
        "value_std_correct": float(values[correct_mask].std()) if correct_mask.any() else None,
        "value_mean_wrong": val_wrong,
        "value_std_wrong": float(values[wrong_mask].std()) if wrong_mask.any() else None,
        "value_separation": val_separation,
        "threshold": threshold,
        "prediction_accuracy": acc,
        "accuracy_vs_majority": acc - majority_baseline,
    }

    print("\n" + "=" * 60)
    print("Critic Prediction Evaluation Results")
    print("=" * 60)
    print(f"  Samples:        {results['num_samples']}  "
          f"(correct={results['num_correct']}, wrong={results['num_wrong']})")
    print(f"  GT accuracy:    {gt_acc:.4f}  "
          f"(majority baseline={majority_baseline:.4f})")
    print()
    print(f"  [Core] AUC-ROC:         {results['auc_roc']}")
    print(f"         (>0.5=有区分能力, >0.7=有实用价值, >0.8=强)")
    print()
    if val_correct is not None and val_wrong is not None:
        print(f"  [Core] Value 分离度:    {val_separation:+.4f}")
        print(f"         correct mean:    {val_correct:.4f} +/- {results['value_std_correct']:.4f}")
        print(f"         wrong   mean:    {val_wrong:.4f} +/- {results['value_std_wrong']:.4f}")
        print(f"         (correct > wrong = 方向正确)")
    print()
    print(f"  [Ref]  Accuracy@{threshold}: {acc:.4f}  "
          f"(vs majority {majority_baseline:.4f}, diff={acc - majority_baseline:+.4f})")
    print("=" * 60)

    if results['auc_roc'] is not None:
        auc_val = results['auc_roc']
        if auc_val > 0.7:
            verdict = "Critic 具备较强的 reward 预测能力，可以替代 reward 信号"
        elif auc_val > 0.5:
            verdict = "Critic 有一定区分能力，但信号较弱"
        else:
            verdict = "Critic 无法区分正确/错误答案，不具备替代 reward 的能力"
        print(f"\n  结论: {verdict} (AUC={auc_val:.4f})")

    return results


def threshold_sweep(values: np.ndarray, rewards: np.ndarray, steps: np.ndarray = None):
    """
    扫描阈值，返回使 (value > t) 与 reward 一致率最高的阈值及对应准确率。
    用于推荐「用 value 作 reward 二分类」时的最佳门槛。
    """
    if steps is None:
        steps = np.concatenate([
            np.arange(0.45, 0.55, 0.05),
            np.arange(0.55, 0.90, 0.05),
        ])
    best_t, best_acc = 0.5, 0.0
    sweep = []
    for t in steps:
        preds = (values > t).astype(float)
        acc = float((rewards == preds).mean())
        sweep.append((float(t), acc))
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t, best_acc, sweep


# ─────────────────────────── 6. 保存 / 加载中间结果 ────────────────

def save_rollouts(path, prompt_texts, responses, ground_truths, data_sources, rewards):
    """保存 rollout 到 jsonl。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i in range(len(responses)):
            f.write(json.dumps({
                "idx": i,
                "prompt_text": prompt_texts[i],
                "response": responses[i],
                "ground_truth": ground_truths[i],
                "data_source": data_sources[i],
                "reward": rewards[i],
            }, ensure_ascii=False) + "\n")
    print(f"[Save] Rollouts saved to {path}")


def load_rollouts(path):
    """从 jsonl 加载已有 rollout。"""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    prompt_texts = [r["prompt_text"] for r in records]
    responses = [r["response"] for r in records]
    ground_truths = [r["ground_truth"] for r in records]
    data_sources = [r["data_source"] for r in records]
    rewards = np.array([r["reward"] for r in records])
    print(f"[Load] Loaded {len(records)} rollouts from {path}")
    return prompt_texts, responses, ground_truths, data_sources, rewards


# ─────────────────────────── Main ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate critic reward-prediction ability")
    parser.add_argument("--actor_path", type=str, required=True, help="Actor model path (for vLLM generation)")
    parser.add_argument("--critic_path", type=str, required=True, help="Pretrained critic HF path")
    parser.add_argument("--data_path", type=str, required=True, help="Test data parquet path")
    parser.add_argument("--data_source", type=str, default=None,
                        help="Override data_source (e.g. openai/gsm8k, lighteval/MATH)")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max generation length")
    parser.add_argument("--max_seq_length", type=int, default=3072, help="Max total seq length for critic (prompt+response)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary classification threshold")
    parser.add_argument("--threshold_sweep", action="store_true",
                        help="Run threshold sweep and print recommended threshold for max accuracy")
    parser.add_argument("--batch_size", type=int, default=8, help="Critic forward batch size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5, help="vLLM GPU memory fraction")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs (vLLM tensor parallel + critic auto shard)")
    parser.add_argument("--output_dir", type=str, default="./eval_critic_results", help="Output directory")
    parser.add_argument("--rollout_file", type=str, default=None,
                        help="Load existing rollouts from this jsonl file (skip generation)")
    parser.add_argument("--save_rollouts", action="store_true", default=True,
                        help="Save rollouts to output_dir/rollouts.jsonl")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (None=disable wandb)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── W&B init ──
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                tags=["eval_critic", args.data_source or "unknown"],
            )
            print(f"[W&B] Initialized: {args.wandb_project} / {args.wandb_run_name}")
        except Exception as e:
            print(f"[W&B] Failed to init: {e}, continuing without wandb")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.actor_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Step 1 & 2: Rollout + Reward ──
    if args.rollout_file and os.path.exists(args.rollout_file):
        prompt_texts, responses, ground_truths, data_sources, rewards = load_rollouts(args.rollout_file)
    else:
        prompts, ground_truths, data_sources = load_dataset(args.data_path, args.max_samples)

        if args.data_source:
            data_sources = [args.data_source] * len(prompts)

        t0 = time.time()
        prompt_texts, responses = generate_rollouts(
            args.actor_path, tokenizer, prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.n_gpus,
        )
        print(f"  Generation took {time.time() - t0:.1f}s")

        rewards = compute_rewards(responses, ground_truths, data_sources)

        if args.save_rollouts:
            rollout_path = os.path.join(args.output_dir, "rollouts.jsonl")
            save_rollouts(rollout_path, prompt_texts, responses, ground_truths, data_sources, rewards.tolist())

    # ── Step 3: Critic values ──
    t0 = time.time()
    values = compute_critic_values(
        args.critic_path, tokenizer, prompt_texts, responses,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        n_gpus=args.n_gpus,
    )
    print(f"  Critic forward took {time.time() - t0:.1f}s")

    # ── Step 4: Evaluate ──
    results = evaluate_metrics(values, rewards, threshold=args.threshold)
    results["config"] = vars(args)

    # ── Optional: threshold sweep（推荐 value 作 reward 时的最佳门槛）
    if getattr(args, "threshold_sweep", False):
        best_t, best_acc, sweep = threshold_sweep(values, rewards)
        print("\n" + "=" * 60)
        print("Threshold sweep (value 作 reward 二分类时推荐门槛)")
        print("=" * 60)
        for t, acc in sweep:
            mark = " <-- best" if t == best_t else ""
            print(f"  threshold {t:.2f} -> accuracy {acc:.4f}{mark}")
        print(f"  推荐阈值: {best_t:.2f} (accuracy={best_acc:.4f})")
        print("=" * 60)
        results["threshold_sweep"] = {"best_threshold": best_t, "best_accuracy": best_acc, "sweep": sweep}

    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Save] Results saved to {results_path}")

    # Save per-sample details
    detail_path = os.path.join(args.output_dir, "per_sample.jsonl")
    with open(detail_path, "w") as f:
        for i in range(len(values)):
            f.write(json.dumps({
                "idx": i,
                "reward": float(rewards[i]),
                "value": float(values[i]),
                "pred": int(values[i] > args.threshold),
                "correct_pred": int((values[i] > args.threshold) == rewards[i]),
            }) + "\n")
    print(f"[Save] Per-sample details saved to {detail_path}")

    # ── W&B log ──
    if wandb_run is not None:
        try:
            import wandb

            log_payload = {}

            # 1) 标量写入 summary（显示在 Overview，不依赖 step）
            for k, v in results.items():
                if k != "config" and v is not None:
                    wandb.summary[f"eval/{k}"] = v

            # 2) 每样本明细表
            sample_rows = []
            for i in range(len(values)):
                pred = int(values[i] > args.threshold)
                sample_rows.append([
                    i, int(rewards[i]), float(values[i]), pred,
                    int(pred == int(rewards[i])),
                ])
            log_payload["eval/per_sample"] = wandb.Table(
                columns=["idx", "reward", "value", "pred", "correct_pred"],
                data=sample_rows,
            )

            # 3) 指标汇总表
            auc_str = f"{results['auc_roc']:.4f}" if results.get("auc_roc") is not None else "N/A"
            sep_str = f"{results['value_separation']:+.4f}" if results.get("value_separation") is not None else "N/A"
            summary_rows = [
                ["AUC-ROC",              auc_str],
                ["Value 分离度",         sep_str],
                ["Value Mean (correct)", f"{results['value_mean_correct']:.4f}" if results.get("value_mean_correct") is not None else "N/A"],
                ["Value Mean (wrong)",   f"{results['value_mean_wrong']:.4f}"   if results.get("value_mean_wrong")   is not None else "N/A"],
                ["GT Accuracy",          f"{results['ground_truth_accuracy']:.4f}"],
                ["Majority Baseline",    f"{results['majority_baseline']:.4f}"],
                ["Accuracy@threshold",   f"{results['prediction_accuracy']:.4f}"],
                ["样本数",               str(results["num_samples"])],
            ]
            log_payload["eval/summary"] = wandb.Table(
                columns=["指标", "值"], data=summary_rows,
            )

            # 4) ROC 曲线（纯 numpy）
            n_pos = int(rewards.sum())
            n_neg = len(rewards) - n_pos
            if n_pos > 0 and n_neg > 0:
                auc_val = results.get("auc_roc") or 0.0
                order = np.argsort(-values)
                labels_sorted = rewards[order]
                tp = np.cumsum(labels_sorted)
                fp = np.cumsum(1 - labels_sorted)
                tpr_all = np.concatenate([[0.0], tp / n_pos, [1.0]])
                fpr_all = np.concatenate([[0.0], fp / n_neg, [1.0]])
                # 最多取 300 个点，避免 wandb table 太大
                idx = np.round(np.linspace(0, len(tpr_all) - 1, min(300, len(tpr_all)))).astype(int)
                roc_rows = [[float(fpr_all[i]), float(tpr_all[i])] for i in idx]
                roc_table = wandb.Table(columns=["FPR", "TPR"], data=roc_rows)
                log_payload["eval/roc_curve"] = wandb.plot.line(
                    table=roc_table, x="FPR", y="TPR",
                    title=f"ROC Curve (AUC={auc_val:.4f})",
                )

            # 5) value 分布（correct / wrong 分别直方图）
            try:
                correct_vals = values[rewards == 1.0].tolist()
                wrong_vals = values[rewards == 0.0].tolist()
                log_payload["eval/value_hist_correct"] = wandb.Histogram(correct_vals)
                log_payload["eval/value_hist_wrong"] = wandb.Histogram(wrong_vals)
            except Exception as he:
                print(f"[W&B] Histogram skipped: {he}")

            # 一次性提交所有 log，避免 step 覆盖问题
            wandb.log(log_payload)
            wandb.finish()
            print("[W&B] All charts and tables logged. Run finished.")
        except Exception as e:
            import traceback
            print(f"[W&B] Logging failed: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
