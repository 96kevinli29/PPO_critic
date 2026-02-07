#!/usr/bin/env python3
"""
将 VeRL/FSDP 格式的 Critic checkpoint 转换为 HuggingFace 标准格式。

用法:
    python scripts/convert_critic_to_hf.py \\
        --local_dir checkpoints/PPO-Project2026/qwen3_0.6b_ppo_gsm8k/global_step_116/critic \\
        --target_dir critic_qwen3_0.6b_hf \\
        [--trust-remote-code]

不依赖 verl 包，仅需: torch, transformers, accelerate, safetensors, tqdm
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor
from torch.distributed._tensor import Placement, Shard


def parse_args():
    parser = argparse.ArgumentParser(description="将 Critic FSDP checkpoint 转换为 HuggingFace 格式")
    parser.add_argument("--local_dir", type=str, required=True, help="Critic checkpoint 目录路径")
    parser.add_argument("--target_dir", type=str, required=True, help="输出 HuggingFace 格式模型目录")
    parser.add_argument("--trust-remote-code", action="store_true", help="是否信任远程代码 (Qwen3 等需要)")
    return parser.parse_args()


def get_world_size(local_dir: Path) -> int:
    config_path = local_dir / "fsdp_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到 fsdp_config.json: {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    world_size = config.get("world_size")
    if world_size is None:
        raise ValueError("fsdp_config.json 中缺少 world_size")
    return world_size


def load_and_merge_fsdp_shards(local_dir: Path, world_size: int) -> dict:
    """加载并合并 FSDP 分片"""
    model_state_dict_lst = [None] * world_size

    def load_shard(rank: int):
        path = local_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        return torch.load(path, map_location="cpu", weights_only=False)

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as executor:
        futures = [executor.submit(load_shard, r) for r in range(world_size)]
        for i, future in enumerate(tqdm(futures, desc="加载 FSDP 分片")):
            model_state_dict_lst[i] = future.result()

    # 检查是否为 DTensor
    pivot_key = sorted(model_state_dict_lst[0].keys())[0]
    sample = model_state_dict_lst[0][pivot_key]
    is_dtensor = isinstance(sample, DTensor)

    mesh_dim_names = ("fsdp",)
    param_placements = {}

    merged = {}
    all_keys = list(model_state_dict_lst[0].keys())
    for key in all_keys:
        shards = []
        for rank in range(world_size):
            tensor = model_state_dict_lst[rank].pop(key)
            if is_dtensor and isinstance(tensor, DTensor):
                shards.append(tensor._local_tensor.bfloat16())
                if key not in param_placements:
                    param_placements[key] = tensor.placements[0]
            else:
                shards.append(tensor.bfloat16() if tensor.dtype != torch.bfloat16 else tensor)

        if is_dtensor and key in param_placements:
            placement = param_placements[key]
            if placement.is_replicate():
                merged[key] = shards[0]
            elif placement.is_shard():
                merged[key] = torch.cat(shards, dim=placement.dim).contiguous()
            else:
                merged[key] = torch.cat(shards, dim=0)
        else:
            # 非 DTensor: 通常 replicated，取第一个；或按 dim=0 concat
            if all(s.shape == shards[0].shape for s in shards):
                merged[key] = shards[0]
            else:
                merged[key] = torch.cat(shards, dim=0)

    del model_state_dict_lst
    return merged


def save_hf_model(
    state_dict: dict,
    hf_config_path: Path,
    target_dir: Path,
    trust_remote_code: bool = False,
):
    """保存为 HuggingFace 格式，使用 Qwen3ForTokenClassification 以保留 value head (score 层)"""
    from transformers import AutoModelForTokenClassification

    config = AutoConfig.from_pretrained(str(hf_config_path), trust_remote_code=trust_remote_code)

    # Critic 使用 ForTokenClassification 架构，score 层即 value head，供 VeRL PPO 加载
    model_type = getattr(config, "model_type", "qwen3")
    # 如 qwen3 -> Qwen3ForTokenClassification, llama -> LlamaForTokenClassification
    model_name = "".join(w.capitalize() for w in model_type.replace("-", "_").split("_"))
    token_cls_arch = f"{model_name}ForTokenClassification"

    config.architectures = [token_cls_arch]
    config.num_labels = 1
    config.id2label = {"0": "LABEL_0"}
    config.label2id = {"LABEL_0": 0}
    config.classifier_dropout = 0.0

    with init_empty_weights():
        model = AutoModelForTokenClassification.from_config(
            config, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code
        )
    model.to_empty(device="cpu")

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"保存模型到 {target_dir}")
    model.save_pretrained(target_dir, state_dict=state_dict, safe_serialization=True)
    del state_dict
    del model

    # 保存 tokenizer
    if (hf_config_path / "tokenizer.json").exists() or (hf_config_path / "tokenizer_config.json").exists():
        print("保存 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(hf_config_path), trust_remote_code=trust_remote_code)
        tokenizer.save_pretrained(target_dir)
    else:
        # 复制 tokenizer 相关文件
        for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]:
            src = hf_config_path / f
            if src.exists():
                shutil.copy(src, target_dir / f)
                print(f"复制 {f}")

    # 确保 config 完整
    config.save_pretrained(target_dir)
    print(f"✓ HuggingFace 格式模型已保存到: {target_dir}")


def main():
    args = parse_args()
    local_dir = Path(args.local_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    hf_config_path = local_dir / "huggingface"

    if not local_dir.exists():
        raise FileNotFoundError(f"Checkpoint 目录不存在: {local_dir}")
    if not hf_config_path.exists():
        raise FileNotFoundError(f"缺少 huggingface 配置目录: {hf_config_path}")

    world_size = get_world_size(local_dir)
    print(f"World size: {world_size}")

    merged_state_dict = load_and_merge_fsdp_shards(local_dir, world_size)
    print(f"合并完成，共 {len(merged_state_dict)} 个参数")

    save_hf_model(
        merged_state_dict,
        hf_config_path,
        target_dir,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
