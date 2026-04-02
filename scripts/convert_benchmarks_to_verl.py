#!/usr/bin/env python3
"""
将 AIME2024, AIME2025, AMC2023, GPQA-Diamond 原始 parquet 转为 verl 训练格式。

verl 格式要求 5 列:
  data_source  (str)   → 决定 reward 函数路由
  prompt       (list)  → [{"role": "user", "content": ...}]
  ability      (str)   → 类别标签
  reward_model (dict)  → {"ground_truth": str, "style": str}
  extra_info   (dict)  → 附加信息

用法:
  python scripts/convert_benchmarks_to_verl.py [--data_dir data/]
"""

import argparse
import os

import datasets
import pyarrow.parquet as pq

MATH_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: $Answer "
    "(without quotes) where $Answer is the answer to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)

GPQA_INSTRUCTION = (
    "Answer the following multiple-choice question step by step. "
    "The last line of your response should be of the form Answer: $Answer "
    "(without quotes) where $Answer is one of A, B, C, D.\n\n"
    "{question}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


def convert_aime2024(data_dir: str):
    src = os.path.join(data_dir, "aime2024", "data", "train-00000-of-00001.parquet")
    dst = os.path.join(data_dir, "aime2024", "test.parquet")
    ds = datasets.load_dataset("parquet", data_files=src)["train"]

    def process(example, idx):
        return {
            "data_source": "aime_2024",
            "prompt": [{"role": "user", "content": MATH_INSTRUCTION.format(problem=example["problem"])}],
            "ability": "math",
            "reward_model": {"ground_truth": str(example["answer"]), "style": "rule"},
            "extra_info": {"index": idx, "id": example.get("id"), "year": example.get("year", "2024")},
        }

    ds = ds.map(process, with_indices=True, remove_columns=ds.column_names)
    ds.to_parquet(dst)
    print(f"  aime2024: {len(ds)} rows -> {dst}")


def convert_aime2025(data_dir: str):
    src = os.path.join(data_dir, "aime2025", "data", "train-00000-of-00001.parquet")
    dst = os.path.join(data_dir, "aime2025", "test.parquet")
    ds = datasets.load_dataset("parquet", data_files=src)["train"]

    def process(example, idx):
        return {
            "data_source": "aime_2025",
            "prompt": [{"role": "user", "content": MATH_INSTRUCTION.format(problem=example["problem"])}],
            "ability": "math",
            "reward_model": {"ground_truth": str(example["answer"]), "style": "rule"},
            "extra_info": {"index": idx, "problem_idx": example.get("problem_idx")},
        }

    ds = ds.map(process, with_indices=True, remove_columns=ds.column_names)
    ds.to_parquet(dst)
    print(f"  aime2025: {len(ds)} rows -> {dst}")


def convert_amc2023(data_dir: str):
    src = os.path.join(data_dir, "amc2023", "test-00000-of-00001.parquet")
    dst = os.path.join(data_dir, "amc2023", "test.parquet")
    ds = datasets.load_dataset("parquet", data_files=src)["train"]

    def process(example, idx):
        answer = example["answer"]
        gt = str(int(answer)) if float(answer) == int(answer) else str(answer)
        return {
            "data_source": "amc_2023",
            "prompt": [{"role": "user", "content": MATH_INSTRUCTION.format(problem=example["question"])}],
            "ability": "math",
            "reward_model": {"ground_truth": gt, "style": "rule"},
            "extra_info": {"index": idx, "id": example.get("id"), "url": example.get("url", "")},
        }

    ds = ds.map(process, with_indices=True, remove_columns=ds.column_names)
    ds.to_parquet(dst)
    print(f"  amc2023:  {len(ds)} rows -> {dst}")


def convert_gpqa_diamond(data_dir: str):
    src = os.path.join(data_dir, "gpqa_diamond", "test", "gpqa_diamond.parquet")
    dst = os.path.join(data_dir, "gpqa_diamond", "test.parquet")
    ds = datasets.load_dataset("parquet", data_files=src)["train"]

    def process(example, idx):
        return {
            "data_source": "gpqa_diamond",
            "prompt": [{"role": "user", "content": GPQA_INSTRUCTION.format(question=example["question"])}],
            "ability": "science",
            "reward_model": {"ground_truth": str(example["answer"]), "style": "rule"},
            "extra_info": {"index": idx},
        }

    ds = ds.map(process, with_indices=True, remove_columns=ds.column_names)
    ds.to_parquet(dst)
    print(f"  gpqa_diamond: {len(ds)} rows -> {dst}")


def create_train_symlink(data_dir: str):
    """为 dapo_math_17k 创建简洁的 train.parquet 软链接"""
    src_parquet = os.path.join(data_dir, "dapo_math_17k", "data", "dapo-math-17k.parquet")
    dst_link = os.path.join(data_dir, "dapo_math_17k", "train.parquet")
    if os.path.exists(src_parquet) and not os.path.exists(dst_link):
        os.symlink(os.path.abspath(src_parquet), dst_link)
        print(f"  dapo_math_17k: symlink {dst_link} -> {src_parquet}")
    elif os.path.exists(dst_link):
        print(f"  dapo_math_17k: train.parquet already exists, skipped")
    else:
        print(f"  WARNING: source not found: {src_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, help="data root (default: <project>/data)")
    args = parser.parse_args()

    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data_dir = os.path.join(os.path.dirname(script_dir), "data")

    print(f"Data dir: {args.data_dir}")
    print("Converting benchmark datasets to verl format...")

    convert_aime2024(args.data_dir)
    convert_aime2025(args.data_dir)
    convert_amc2023(args.data_dir)
    convert_gpqa_diamond(args.data_dir)
    create_train_symlink(args.data_dir)

    print("Done.")
