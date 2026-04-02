# 修改日志

记录项目中的关键修改，便于追溯和回滚。  
**上游仓库**: [verl-project/verl](https://github.com/verl-project/verl)

---

## 2026-03-07: 全面代码审查与 Bug 修复

### 审查范围

对照原版 verl 与实验设计，全面审查 SFT/PPO 脚本及调用代码，排查导致"acc 上不去、reward 长期负值"的根因。

### 修复的致命问题

#### 1. `math_dapo.compute_score` 对错误答案返回 -1.0（根因 #1）

**文件**: `verl/verl/utils/reward_score/math_dapo.py`

**问题**: 原版 `math_dapo.compute_score` 对错误答案返回 `-1.0`（第 265 行 `reward = 1.0 if correct else -1.0`），而标准 PPO 通常使用 0/1 reward。训练初期模型准确率约 5%，平均 reward = `0.05×1 + 0.95×(-1) = -0.9`，导致：
- 训练 reward 和验证 score 长期负值
- Critic 初始化预测 ~0 但真实 baseline ~-0.9，收敛极慢
- DAPO-Math 训练集、AIME2024/2025、AMC2023、GPQA-Diamond 验证集全部受影响（均走此函数）

**修复**: 错误答案 reward 从 `-1.0` 改为 `0.0`；同时修复 `is_correct_strict_box` 中的 `-1` → `0`。

#### 2. DAPO-Math 训练集 TRAIN_MAX_SAMPLES 只覆盖约 200 个唯一 prompt（根因 #2）

**文件**: `run_qwen3_ppo.sh`

**问题**: DAPO-Math-17k 数据集共 1,791,700 行（17,398 唯一 prompt，每 prompt 重复约 103 次）。原设置 `TRAIN_MAX_SAMPLES=20000` 取前 20000 行，只覆盖约 200 个唯一题目。verl 的 `RLHFDataset` 不做去重，模型在 200 道题上反复训练，完全无法泛化。

**修复**: `TRAIN_MAX_SAMPLES` 从 `20000` 改为 `-1`（全量加载）。

#### 3. MATH 验证集使用了错误的 reward 函数

**文件**: `verl/verl/utils/reward_score/__init__.py`

**问题**: MATH 验证集（`data_source=DigitalLearningGmbH/MATH-lighteval`）原来走 `math_reward.compute_score`，只识别 `\boxed{}` 格式答案。但模型被 SFT 训练成输出 `Answer: X` 格式（与 DAPO-Math prompt 一致），导致 MATH 验证 acc 永远为 0。

**修复**: 将 `lighteval/MATH`、`DigitalLearningGmbH/MATH-lighteval`、`HuggingFaceH4/MATH-500` 合并到 `math_dapo.compute_score` 分支（支持 `Answer:` 格式解析）。

#### 4. GSM8K 验证集 reward 函数格式不匹配（acc/score/reward 全为 0）

**文件**: `verl/verl/utils/reward_score/__init__.py`

**问题**: GSM8K 验证集（`data_source=openai/gsm8k`）走 `gsm8k.compute_score`，用正则 `#### (\-?[0-9\.\,]+)` 提取答案。但模型经过 NuminaMath-CoT SFT 后输出 `Answer: X` 格式，`gsm8k.compute_score` 找不到 `####` 标记，所有样本 reward=0，导致 wandb 上 GSM8K 的 acc/score/reward 全部为 0。

**修复**: 将 `openai/gsm8k` 也合并到 `math_dapo.compute_score` 分支。`math_dapo` 用 `r"(?i)Answer\s*:\s*([^\n]+)"` 提取答案，与模型输出格式匹配。已验证 GSM8K 纯整数答案（如 "18"）在 `normalize_final_answer` 后能正确匹配。

### 修复的其他问题

#### 4. Freeze Critic 时 checkpoint 保存崩溃

**文件**: `verl/verl/workers/fsdp_workers.py`

**问题**: `freeze=True` 时 `critic_optimizer=None`，但 checkpoint 配置默认 `save_contents: ['model', 'optimizer', 'extra']`。调用 `save_checkpoint` 时触发 `assert self.optimizer is not None`。

**修复**: 在 `init_model()` 中，当 `freeze=True` 时动态将 checkpoint 的 `save_contents`/`load_contents` 改为 `['model', 'extra']`。

#### 5. SFT cosine scheduler 不支持 min_lr

**文件**: `verl/verl/trainer/fsdp_sft_trainer.py`、`verl/verl/trainer/config/sft_trainer.yaml`、`run_qwen3_sft.sh`

**问题**: 实验设计要求 SFT 用 cosine scheduler（lr=2e-5, min_lr=2e-6），但 `fsdp_sft_trainer.py` 调用 `get_cosine_schedule_with_warmup` 时未传 `min_lr_ratio` 参数，cosine 衰减一路降到 0。

**修复**:
- `fsdp_sft_trainer.py`: 从 `config.optim.min_lr_ratio` 读取并传入
- `sft_trainer.yaml`: 新增 `min_lr_ratio: 0.0` 默认值
- `run_qwen3_sft.sh`: 新增 `MIN_LR_RATIO` 环境变量（默认 0.1），传入 `optim.min_lr_ratio=$MIN_LR_RATIO`

#### 6. PPO 脚本中的无效配置

**文件**: `run_qwen3_ppo.sh`

**问题**: `+trainer.save_critic_model=True` 和 `+trainer.critic_model_output_path` 在 verl trainer 代码中无任何读取，为死配置。Critic 保存实际通过 verl 内置 checkpoint 机制（`save_freq` 控制）完成。

**修复**: 移除两行无效 Hydra 配置。

#### 7. SFT 脚本环境变量名不一致

**文件**: `run_qwen3_sft.sh`

**问题**: SFT 脚本用 `PYTORCH_CUDA_ALLOC_CONF`（旧名），PPO 脚本用 `PYTORCH_ALLOC_CONF`（新名）。

**修复**: 统一为 `PYTORCH_ALLOC_CONF`。

### 修复后 reward 函数对照表

| 数据集 | data_source | reward 函数 | 正确 | 错误 |
|--------|-------------|-------------|------|------|
| DAPO-Math（训练） | `math_dapo` | `math_dapo` | 1.0 | 0.0 |
| AIME2024/2025（验证） | `aime_2024`/`aime_2025` | `math_dapo` | 1.0 | 0.0 |
| AMC2023（验证） | `amc_2023` | `math_dapo` | 1.0 | 0.0 |
| GPQA-Diamond（验证） | `gpqa_diamond` | `math_dapo` | 1.0 | 0.0 |
| MATH（验证） | `MATH-lighteval` | `math_dapo`（已改） | 1.0 | 0.0 |
| GSM8K（验证/训练） | `openai/gsm8k` | `gsm8k` | 1.0 | 0.0 |

### 本次修改文件列表

| 文件 | 修改内容 |
|------|----------|
| `verl/verl/utils/reward_score/math_dapo.py` | 错误答案 reward `-1.0` → `0.0` |
| `verl/verl/utils/reward_score/__init__.py` | GSM8K + MATH-lighteval 统一走 `math_dapo` reward；返回值标准化 |
| `verl/verl/workers/fsdp_workers.py` | freeze critic 时 checkpoint 排除 optimizer |
| `verl/verl/trainer/fsdp_sft_trainer.py` | cosine scheduler 支持 `min_lr_ratio` |
| `verl/verl/trainer/config/sft_trainer.yaml` | 新增 `min_lr_ratio: 0.0` |
| `run_qwen3_ppo.sh` | 移除无效配置；DAPO-Math `TRAIN_MAX_SAMPLES` → `-1` |
| `run_qwen3_sft.sh` | 新增 `MIN_LR_RATIO`；`PYTORCH_CUDA_ALLOC_CONF` → `PYTORCH_ALLOC_CONF` |

---

## 2026-02-23: 代码审查与相对原版的修改记录核对

### 审查结论

- **原版参考**: [verl-project/verl](https://github.com/verl-project/verl)（PPO 流程、GAE、reward 来源、critic 更新）。
- **verl 子目录修改**: 与设计一致，可用于两阶段实验（阶段 1 正常 PPO 保存 critic；阶段 2 冻结 critic + 可配置 reward/mask/二值化）。  
- **实验主脚本 `run_qwen3_ppo.sh`**: 正确；通过环境变量切换 DATA、MODEL_TAG、CRITIC 路径、FREEZE_CRITIC、REWARD_*、LAM_ACTOR/LAM_CRITIC 等，实验名与对照条件一致；阶段 1 结束后自动转换 critic 为 HF 的逻辑正确。

### 与原版差异摘要（仅列本仓库修改点）

| 模块 | 原版行为 | 本仓库修改 |
|------|----------|------------|
| **algorithm 配置** | 仅 `lam`、无 reward mask/二值化 | 新增 `lam_actor`/`lam_critic`、`reward_mask_ratio`/`reward_mask_type`、`reward_binarize`/`reward_threshold`（见 `algorithm.py`、`ppo_trainer.yaml`） |
| **GAE** | 单一 `lam` | 支持 `lam_actor`/`lam_critic` 分离，在 `compute_advantage()` 中分别算 advantage 与 returns（`ray_trainer.py`、`core_algos` 调用方式不变） |
| **Reward** | rule-based / RM 产出 → 直接进 GAE | 在 GAE 前增加：二值化（score>threshold→1 否则 0）、轨迹级 mask（bernoulli 或 fixed_ratio） |
| **Critic** | 必训、有 optimizer | 支持 `critic.freeze=true`：参数冻结、不建 optimizer、仅前向算 value，`update_critic` 跳过（`fsdp_workers.py`、`dp_critic.py`、`critic.py` 配置） |
| **Reward 来源** | rule-based 或 reward_model | 不变；外部 RM（含「已有 critic 转 HF」）通过 `reward_model.enable` + `reward_model.model.path` 配置（由 `run_qwen3_ppo.sh` 的 REWARD_MODEL_ENABLE/REWARD_MODEL_PATH 传入） |
| **Attention** | 硬编码 flash_attention_2 | `load_valuehead_model` 改为从环境变量/配置读取 attn_implementation，默认 sdpa（`verl/utils/model.py`） |

### 实验可调 GAE 说明

- 脚本中 GAE 默认不传 `lam_actor`/`lam_critic`，与 verl 原版一致（使用 `algorithm.lam=1.0`）。  
- 若需 **lam_actor=0.95、lam_critic=1.0**，可在运行前设置：  
  `LAM_ACTOR=0.95 LAM_CRITIC=1.0 DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh`  
  或命令行覆盖：`algorithm.lam_actor=0.95 algorithm.lam_critic=1.0`。

### 配置与脚本对应关系

- `run_qwen3_ppo.sh` 传入的 Hydra 覆盖项与 `ppo_trainer.yaml` / `algorithm.py` / `critic` 配置一致：  
  `algorithm.reward_mask_ratio`、`algorithm.reward_mask_type`、`algorithm.reward_binarize`、`algorithm.reward_threshold`、`critic.freeze`、以及可选的 `algorithm.lam_actor`/`algorithm.lam_critic`。

---

## 相对于原版 verl 的修改总览（PPO reward/critic 实验）

为验证**稀疏 reward** 或 **reward-free** 设定下策略优化的可行性，在保持标准 PPO 数学形式（GAE/clip/loss 不变）的前提下，对 reward 与 critic 机制做了可配置扩展。实验分为两阶段：**阶段 1** 按标准 PPO 在 GSM8K/MATH 上训练并保存 critic 作为固定 baseline；**阶段 2** 以 Qwen3 小规模 base 为 actor 运行 PPO，critic 从初始化起**冻结**（仅前向算 value，不更新参数），reward 来源可配置（rule-based 或外部 RM/已有 critic），并支持 **reward 二值化** 与 **轨迹级 reward mask**（比例 0–100%，bernoulli 或 fixed_ratio），以模拟稀疏/缺失 reward。GAE 默认与**原 verl 一致**（不覆盖 `lam_actor`/`lam_critic`，使用 `ppo_trainer.yaml` 的 `lam=1.0` 用于两者）；若需 0.95/1.0 可在 `run_qwen3_ppo.sh` 中通过环境变量 `LAM_ACTOR`/`LAM_CRITIC` 或命令行参数覆盖。

### 修改文件列表（路径相对于本仓库根目录；与官方 diff 可参考项目根目录 `verl_diff_vs_official.patch`）

| 文件 | 修改内容 |
|------|----------|
| `verl/verl/trainer/config/algorithm.py` | 新增 `lam_actor` / `lam_critic`、`reward_mask_ratio` / `reward_mask_type`、`reward_binarize` / `reward_threshold` |
| `verl/verl/trainer/config/ppo_trainer.yaml` | 上述 algorithm 项的默认值与注释 |
| `verl/verl/trainer/config/critic/critic.yaml` | 新增 `freeze: false` |
| `verl/verl/workers/config/critic.py` | 新增 `freeze: bool = False` 及字符串解析（`true`/`1`/`yes` → True） |
| `verl/verl/trainer/ppo/ray_trainer.py` | Reward 二值化、Reward Mask（bernoulli/fixed_ratio）、GAE 双 lambda、freeze 时跳过 critic 更新；reward 来自 reward_fn 或 rm_scores |
| `verl/verl/workers/fsdp_workers.py` | freeze 时 critic 参数冻结、不创建 optimizer、update_critic 直接返回 |
| `verl/verl/workers/critic/dp_critic.py` | Frozen Critic：当 `critic_optimizer is None` 时跳过更新并返回 metrics |
| `verl/verl/utils/model.py` | `load_valuehead_model` 的 attn_implementation 从环境变量/配置读取，默认 sdpa |

**配套入口脚本**（在项目根目录，非 verl 子目录）：`run_qwen3_ppo.sh` — 通过环境变量切换 DATA、MODEL_TAG、CRITIC 路径、FREEZE_CRITIC、REWARD_* 等；GAE 的 `lam_actor`/`lam_critic` 默认不传（与 verl 一致），可选通过 `LAM_ACTOR`/`LAM_CRITIC` 覆盖，并生成统一实验名。

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

## 2026-02-01: Reward Mask + Freeze Critic + GAE 双 lambda + Reward 二值化

### 功能描述

1. **Reward Mask（轨迹级别 mask）**
   - 在 GAE 计算之前，对整条轨迹的 reward 做 mask，被 mask 的轨迹整体 reward 置 0（轨迹级，非 token-level）
   - **类型**: `bernoulli`（每条轨迹独立以概率 `reward_mask_ratio` 被 mask）/ `fixed_ratio`（每步恰好 `floor(batch_size * ratio)` 条被 mask）
   - 不改变 batch 结构，mask 后的 reward 仍参与 return/advantage/loss 计算

2. **Freeze Critic（冻结 critic 参数）**
   - Critic 参数 `requires_grad=False`，不创建 optimizer（返回 `None`）
   - Critic 仍参与前向计算（提供 value baseline 用于 GAE）
   - Actor 仍正常更新

3. **GAE lambda 分离（lam_actor / lam_critic）**
   - 默认与**原 verl 一致**：不覆盖 `lam_actor`/`lam_critic`（yaml 中为 null），二者均使用 `algorithm.lam`（默认 1.0）。
   - 若需分离：可在 `run_qwen3_ppo.sh` 中设置环境变量 `LAM_ACTOR=0.95`、`LAM_CRITIC=1.0`，或命令行传入 `algorithm.lam_actor=0.95 algorithm.lam_critic=1.0`；Actor 用 `lam_actor` 算 advantage，Critic 用 `lam_critic` 算 returns。

4. **Reward 二值化**
   - `algorithm.reward_binarize=true` 时：`reward > reward_threshold` 置 1，否则 0
   - 在 KL penalty 之前应用（先二值化 outcome，再加 per-token KL）

### 修改文件

#### 1. `verl/verl/trainer/ppo/ray_trainer.py`
- **Reward 二值化**: 约第 1608-1624 行，搜索 `"========= Reward"`
- **Reward Mask**: 约第 1625-1656 行，搜索 `"========== Reward Mask"`
- **GAE 双 lambda**: `compute_advantage()` 及调用处传入 `lam_actor`/`lam_critic`（约第 155-220、1677-1687 行）
- **Freeze Critic 跳过更新**: 约第 1690-1699 行，搜索 `"# update critic (skip if frozen)"`

#### 2. `verl/verl/workers/fsdp_workers.py`
- **冻结参数 + 不创建 optimizer**: 约第 1496-1505 行，搜索 `"========== Freeze Critic 支持 =========="`
- **标记冻结状态**: 约第 1564 行，`self._freeze_critic = self.config.get("freeze", False)`
- **跳过更新逻辑**: 约第 1597-1602 行，搜索 `"========== Frozen Critic: 跳过更新 =========="`

#### 3. `verl/verl/workers/critic/dp_critic.py`
- **处理 None optimizer**: 约第 206-210 行，搜索 `"========== Frozen Critic: 跳过更新 =========="`

#### 4. `verl/verl/workers/config/critic.py`
- **CriticConfig**: 添加 `freeze: bool = False` 及字符串解析（`"true"/"1"/"yes"` → True）

#### 5. `verl/verl/trainer/config/algorithm.py`
- **AlgoConfig**: 新增 `lam_actor`/`lam_critic`、`reward_mask_ratio`/`reward_mask_type`、`reward_binarize`/`reward_threshold` 及 `__post_init__` 类型转换

#### 6. `verl/verl/trainer/config/ppo_trainer.yaml`
- **algorithm**: `lam_actor`/`lam_critic`、`reward_mask_ratio`/`reward_mask_type`、`reward_binarize`/`reward_threshold` 默认值与注释

#### 7. `verl/verl/trainer/config/critic/critic.yaml`
- **freeze**: 添加 `freeze: false`，便于 Hydra 覆盖 `critic.freeze`

### 配置项

| 配置项 | 说明 | 默认值 |
|-------|------|-------|
| `algorithm.reward_mask_ratio` | 轨迹级 reward mask 比例 (0.0–1.0) | 0.0 |
| `algorithm.reward_mask_type` | mask 类型：`bernoulli` / `fixed_ratio` | bernoulli |
| `algorithm.reward_binarize` | 是否将 reward 二值化为 0/1 | false |
| `algorithm.reward_threshold` | 二值化阈值（score>此值为 1） | 0.5 |
| `algorithm.lam_actor` | GAE lambda（advantage，actor） | null → 用 lam（与 verl 原库一致；脚本默认不传，可用 LAM_ACTOR 覆盖） |
| `algorithm.lam_critic` | GAE lambda（returns，critic） | null → 用 lam（与 verl 原库一致；脚本默认不传，可用 LAM_CRITIC 覆盖） |
| `critic.freeze` | 是否冻结 critic（仅前向，不更新） | false |

### 使用方式（统一入口脚本：项目根目录 `run_qwen3_ppo.sh`）

```bash
# 阶段 1：正常 PPO，训练并保存 critic
FREEZE_CRITIC=false REWARD_MASK_RATIO=0 REWARD_MODEL_ENABLE=false \
SAVE_FREQ=99999 DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh

# 阶段 2：预训练 critic + 冻结 + 50% mask（bernoulli）
USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=0.5 \
DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh

# 100% mask（完全 reward-free）
USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=1.0 \
DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh

# RM=critic + 二值化 + 50% mask + fixed_ratio
USE_PRETRAINED_CRITIC=yes REWARD_MODEL_ENABLE=true REWARD_BINARIZE=true \
FREEZE_CRITIC=true REWARD_MASK_RATIO=0.5 REWARD_MASK_TYPE=fixed_ratio \
DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh
```

### 设计约束

- ✅ 不改 PPO 数学形式（GAE/clip/loss 公式不变）
- ✅ 不改 `core_algos.py`（仅调用现有 `compute_gae_advantage_return`，用不同 lam 调用两次实现双 lambda）
- ✅ 改动局限在 "reward → GAE → loss" 与 critic 更新分支
- ✅ 最小侵入

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
