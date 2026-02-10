#!/usr/bin/env python3
"""
Rewrite GSM8K parquet prompts to enforce strict #### format.

This avoids downloading from Hugging Face by operating directly on
existing parquet files (train.parquet/test.parquet).
"""

import argparse
import os

import pandas as pd


DEFAULT_INSTRUCTION = (
    "Solve this math problem step by step. After your reasoning, you MUST output your final "
    "numerical answer in this exact format:\n"
    "#### <number>\n"
    "Do NOT use any other format like \\boxed{} or \"Answer:\". Only use \"#### \" followed by the number.\n"
    "Example:\n#### 42"
)


def build_prompt(question: str, instruction: str):
    question = question.strip()
    return [{"role": "user", "content": f"{question}\n\n{instruction}"}]


def rewrite_split(input_path: str, output_path: str, instruction: str):
    df = pd.read_parquet(input_path)
    if "extra_info" not in df.columns:
        raise ValueError(f"Missing 'extra_info' column in {input_path}")

    def to_prompt(extra_info):
        question = extra_info.get("question")
        if not question:
            raise ValueError("Missing extra_info.question in dataset row")
        return build_prompt(question, instruction)

    df["prompt"] = df["extra_info"].apply(to_prompt)
    df.to_parquet(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/data_storage/lixiao/research_proj_xiao/hyl/data/gsm8k",
        help="Directory containing train.parquet and test.parquet",
    )
    parser.add_argument(
        "--output_dir",
        default="/data_storage/lixiao/research_proj_xiao/hyl/data/gsm8k_v2",
        help="Directory to write rewritten parquet files",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Instruction appended to each question prompt",
    )
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rewrite_split(
        os.path.join(input_dir, "train.parquet"),
        os.path.join(output_dir, "train.parquet"),
        args.instruction,
    )
    rewrite_split(
        os.path.join(input_dir, "test.parquet"),
        os.path.join(output_dir, "test.parquet"),
        args.instruction,
    )

    print(f"Rewritten dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
