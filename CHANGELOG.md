# 修改日志

记录项目中的关键修改，便于追溯和回滚。

---

## 2026-02-01: Critic 转 HuggingFace 格式

### 背景

VeRL PPO 训练保存的 Critic 模型为 FSDP 分片格式（`model_world_size_4_rank_0.pt` 等），不符合 HuggingFace 标准，无法直接作为 `critic.model.path` 用于后续 PPO 训练或其它框架。需先转换为 HuggingFace 格式（含 value head）。

### 转换脚本

**文件路径**: `scripts/convert_critic_to_hf.py`

将 FSDP 格式的 Critic checkpoint 转换为 HuggingFace 标准格式，输出为 `Qwen3ForTokenClassification` 架构（score 层即 value head），可直接被 VeRL `load_valuehead_model` 加载。

### 用法

```bash
python scripts/convert_critic_to_hf.py \
    --local_dir /path/to/fsdp/critic/checkpoint \
    --target_dir /path/to/output_hf_model \
    [--trust-remote-code]
```

**参数说明**:
- `--local_dir`: Critic FSDP checkpoint 目录，需包含 `model_world_size_*_rank_*.pt`、`fsdp_config.json`、`huggingface/` 子目录
- `--target_dir`: 输出 HuggingFace 格式模型目录
- `--trust-remote-code`: 是否信任远程代码（Qwen3 等模型建议加上）

### 示例

```bash
# 转换 PPO 训练后保存的 critic (如 global_step_116)
python scripts/convert_critic_to_hf.py \
    --local_dir checkpoints/PPO-Project2026/qwen3_0.6b_ppo_gsm8k/global_step_116/critic \
    --target_dir critic_qwen3_0.6b_hf \
    --trust-remote-code
```

### 输出内容

转换后的目录包含 HuggingFace 标准文件：
- `model.safetensors`: 模型权重（含 value head）
- `config.json`: 模型配置（`architectures: ["Qwen3ForTokenClassification"]`, `num_labels: 1`）
- tokenizer 相关: `tokenizer.json`, `tokenizer_config.json`, `vocab.json` 等

### 依赖

不依赖 verl 包，仅需: `torch`, `transformers`, `accelerate`, `safetensors`, `tqdm`。

### 在 PPO 中使用

转换后的模型可直接作为 `critic.model.path` 用于 VeRL PPO：

```bash
CRITIC_MODEL_PATH="/path/to/critic_qwen3_0.6b_hf" bash run_qwen3-0.6b_ppo_gsm8k_reward_mask_freeze_critic.sh
```

---

## 2026-02-01: Reward Mask + Freeze Critic 功能

### 功能描述

1. **Reward Mask（轨迹级别随机 mask）**
   - 在 GAE 计算之前，对整条轨迹的 reward 做 Bernoulli 随机 mask
   - 被 mask 的轨迹整体 reward 置 0（不是 token-level mask）
   - 不改变 batch 结构，mask 后的 reward 仍参与 return/advantage/loss 计算

2. **Freeze Critic（冻结 critic 参数）**
   - Critic 参数 `requires_grad=False`，不创建 optimizer
   - Critic 仍参与前向计算（提供 value baseline 用于 GAE）
   - Actor 仍正常更新

### 修改文件

#### 1. `verl/verl/trainer/ppo/ray_trainer.py`
- **Reward Mask**: 约第 1565-1583 行，搜索 `"========== Reward Mask"`
- **Freeze Critic 跳过更新**: 约第 1621-1630 行，搜索 `"# update critic (skip if frozen)"`

#### 2. `verl/verl/workers/fsdp_workers.py`
- **冻结参数 + 不创建 optimizer**: 约第 1483-1493 行，搜索 `"========== Freeze Critic 支持 =========="`
- **标记冻结状态**: 约第 1536-1538 行，搜索 `"# 标记是否冻结 critic"`
- **跳过更新逻辑**: 约第 1588-1593 行，搜索 `"========== Frozen Critic: 跳过更新 =========="`

#### 3. `verl/verl/workers/critic/dp_critic.py`
- **处理 None optimizer**: 约第 195-199 行，搜索 `"========== Frozen Critic: 跳过更新 =========="`

#### 4. `verl/verl/workers/config/critic.py`
- **CriticConfig 基类**: 约第 78 行，添加 `freeze: bool = False` 字段
- **FSDPCriticConfig 子类**: 约第 212 行，添加 `freeze: bool = False` 字段（与基类一致）
- **文件头注释**: 添加 Custom Modifications 备注块

#### 5. `verl/verl/trainer/config/critic/critic.yaml`
- **freeze 配置项**: 在 `enable` 之后添加 `freeze: false`，使 Hydra 可正确解析 `critic.freeze` 覆盖

#### 6. 新增脚本 `run_qwen3-0.6b_ppo_gsm8k_reward_mask_freeze_critic.sh`
- 支持通过环境变量配置功能

### 配置项

| 配置项 | 说明 | 默认值 |
|-------|------|-------|
| `algorithm.reward_mask_ratio` | Reward mask 比例 (0.0-1.0) | 0.0 (不 mask) |
| `critic.freeze` | 是否冻结 critic | false |

### 使用方式

```bash
# 50% reward mask + 冻结 critic
REWARD_MASK_RATIO=0.5 FREEZE_CRITIC=true ./run_qwen3-0.6b_ppo_gsm8k_reward_mask_freeze_critic.sh

# 只使用 30% reward mask
REWARD_MASK_RATIO=0.3 ./run_qwen3-0.6b_ppo_gsm8k_reward_mask_freeze_critic.sh

# 只冻结 critic
FREEZE_CRITIC=true ./run_qwen3-0.6b_ppo_gsm8k_reward_mask_freeze_critic.sh
```

### 设计约束

- ✅ 不改 PPO 数学形式（GAE/clip/loss 公式不变）
- ✅ 不改 `core_algos.py`
- ✅ 改动局限在 "reward → GAE → loss" 链路
- ✅ 最小侵入（总共约 50 行代码）

---

## 2026-01-31: 修复 FlashAttention2 兼容性问题

### 问题描述
- 系统 GLIBC 版本为 2.31（Ubuntu 20.04），而 `flash-attn` 预编译包需要 GLIBC 2.32+
- verl 框架中 `load_valuehead_model` 函数硬编码了 `attn_implementation="flash_attention_2"`，导致在不支持的环境中报错

### 修改文件

#### 1. `verl/verl/utils/model.py`
修改 `load_valuehead_model()` 函数，从硬编码改为动态获取：
- 优先读取环境变量 `VERL_ATTN_IMPLEMENTATION` 或 `HF_ATTN_IMPLEMENTATION`
- 其次读取模型配置中的 `_attn_implementation`
- 默认使用 `sdpa`（PyTorch 2.0+ 内置加速，兼容无 flash_attn 环境）

#### 2. `run_qwen3-1.7b_ppo_math_lighteval.sh`
- 添加环境变量：`export VERL_ATTN_IMPLEMENTATION=sdpa`
- 添加配置覆盖：
  - `+actor_rollout_ref.model.override_config.attn_implementation=sdpa`
  - `+critic.model.override_config.attn_implementation=sdpa`

### 如何在其他脚本中应用
在脚本开头添加：
```bash
export VERL_ATTN_IMPLEMENTATION=sdpa
```

并在训练命令中添加：
```bash
+actor_rollout_ref.model.override_config.attn_implementation=sdpa \
+critic.model.override_config.attn_implementation=sdpa \
```

---
