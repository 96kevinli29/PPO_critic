# 代码审查报告：SFT/PPO 修改 vs 原版 verl

**审查日期**: 2026-03-09  
**对照原版**: [verl-project/verl](https://github.com/verl-project/verl)

---

## 一、SFT 阶段审查

### 1.1 超参数与实验要求对照

| 实验要求 | 脚本配置 | 状态 |
|----------|----------|------|
| Learning rate 2e-5 | `LR="${LR:-2e-5}"` | ✅ |
| Cosine scheduler | `optim.lr_scheduler=cosine` | ✅ |
| Min LR = 2e-6 (ratio=0.1) | `MIN_LR_RATIO=0.1` → `optim.min_lr_ratio=0.1` | ✅ |
| Weight decay 0.1 | `optim.weight_decay=0.1` | ✅ |
| Warmup ratio 0.01 | `optim.lr_warmup_steps_ratio=0.01` | ✅ |

### 1.2 核心逻辑验证

- **`fsdp_sft_trainer.py`**: 已正确支持 `min_lr_ratio`，`get_cosine_schedule_with_warmup` 会传入该参数，cosine 衰减不会降到 0
- **`sft_trainer.yaml`**: 已添加 `min_lr_ratio: 0.0` 默认值
- **数据流**: `model.partial_pretrain` → base 模型，`data.train_files` → NuminaMath-CoT
- **输出**: 训练后创建 `Qwen3-{TAG}-SFT` 软链接，供 PPO 使用

### 1.3 结论

SFT 阶段配置正确，符合实验要求，最小侵入（仅扩展 cosine 的 min_lr 支持）。

---

## 二、PPO 阶段审查

### 2.1 超参数与实验要求对照

| 实验要求 | 脚本/配置 | 状态 |
|----------|-----------|------|
| Clip ratio 0.2 | `actor/actor.yaml`: `clip_ratio: 0.2` | ✅ |
| Actor LR 1e-6 | `ACTOR_LR="${ACTOR_LR:-1e-6}"` | ✅ |
| Critic LR 1e-5 | `CRITIC_LR="${CRITIC_LR:-1e-5}"` | ✅ |
| Sampling temperature 1.0 | `rollout/rollout.yaml`: `temperature: 1.0` | ✅ |
| Top_p 1.0 | `rollout/rollout.yaml`: `top_p: 1` | ✅ |
| Weight decay 0.1 | `actor/critic` 均传入 `weight_decay=0.1` | ✅ |
| Gradient clip 1.0 | `optim/fsdp.yaml`: `clip_grad: 1.0` | ✅ |
| LR warmup steps 10 | `LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"` | ✅ |

### 2.2 建议注意项

| 参数 | 脚本默认 | 说明 |
|------|----------|------|
| `ROLLOUT_N` | 1 | 参数表写 4，代码默认 1。标准 PPO 常用 4，可显式设 `ROLLOUT_N=4` |
| `MAX_RESPONSE_LEN` | 12288 | 数学题常用 2048；12288 适合长 CoT/thinking，可按需调整 |

---

## 三、阶段 2/3 实验逻辑审查

### 3.1 阶段 1（正常 PPO）

- **流程**: SFT 模型 → Actor，同路径初始化 Critic → 标准 PPO 训练
- **Reward**: rule-based（math_dapo 等）
- **验证集**: AIME2024/2025, AMC2023, GPQA-Diamond, gsm8k, math
- **Critic 保存**: 通过 `save_freq` checkpoint，训练结束后自动转 HF 到 `critic_qwen3_*_hf`

### 3.2 阶段 2（稀疏/reward-free）

- **预训练 Critic**: `USE_PRETRAINED_CRITIC=yes` → 从 `CRITIC_MODEL_PATH` 加载
- **冻结 Critic**: `FREEZE_CRITIC=true` → `critic.freeze=True`
  - `fsdp_workers.py`: 参数 `requires_grad=False`，不建 optimizer，`update_critic` 直接返回
  - `ray_trainer.py`: `freeze=True` 时跳过 `_update_critic`
- **Reward 来源可配置**:
  - rule-based: `REWARD_MODEL_ENABLE=false`（默认）
  - 外部 RM（含 critic 转 HF）: `REWARD_MODEL_ENABLE=true` + `REWARD_MODEL_PATH`
- **二值化**: `REWARD_BINARIZE=true` → score>threshold 为 1，否则 0
- **Reward Mask**:
  - `bernoulli`: 每条轨迹以概率 `reward_mask_ratio` 被 mask
  - `fixed_ratio`: 每步恰好 `floor(batch_size * ratio)` 条被 mask
  - 被 mask 轨迹的 reward 整条置 0，value 不被 mask ✅

### 3.3 逻辑正确性

- 二值化在 KL penalty 之前应用 ✅
- Reward Mask 在 GAE 之前应用 ✅
- GAE 双 lambda：`lam_actor`/`lam_critic` 与 `lam` 的 fallback 逻辑正确 ✅
- freeze 时 checkpoint 不保存 optimizer，避免崩溃 ✅

---

## 四、最小侵入原则核查

### 4.1 默认行为与原版一致

| 配置 | 默认值 | 原版等效 |
|------|--------|----------|
| `reward_mask_ratio` | 0.0 | 不 mask，等价原版 |
| `reward_binarize` | false | 不二值化 |
| `critic.freeze` | false | 正常训练 critic |
| `lam_actor`/`lam_critic` | null | 使用 `lam`，与原版一致 |

### 4.2 扩展点均为可选

- 不设置上述参数时，行为与原版 PPO 一致
- 通过环境变量或 Hydra 覆盖即可启用实验功能

---

## 五、Reward 函数与数据集

### 5.1 已修复项（CHANGELOG 已记录）

- **math_dapo**: 错误答案 reward 从 -1.0 改为 0.0
- **GSM8K / MATH-lighteval**: 统一走 `math_dapo`，支持 `Answer:` 格式
- **DAPO-Math**: `TRAIN_MAX_SAMPLES` 按数据集正确设置（如 dapo_math 约 17k）

### 5.2 data_source 与 reward 函数映射

| 数据集 | data_source | reward 函数 | 正确/错误 |
|--------|-------------|-------------|-----------|
| DAPO-Math | math_dapo | math_dapo | 1.0 / 0.0 |
| AIME2024/2025 | aime_* | math_dapo | 1.0 / 0.0 |
| AMC2023 | amc_* | math_dapo | 1.0 / 0.0 |
| GPQA-Diamond | gpqa_diamond | math_dapo | 1.0 / 0.0 |
| MATH / GSM8K | ... | math_dapo | 1.0 / 0.0 |

---

## 六、发现的问题与建议

### 6.1 建议修复

1. **`ROLLOUT_N` 默认值与文档不一致**
   - 参数表写 4，代码默认 1
   - 建议：若标准 PPO 需 4 采样，将默认改为 4；或统一文档说明

### 6.2 建议确认

1. **`MAX_RESPONSE_LEN=12288`**  
   数学题常见为 2048；若为 CoT/thinking 可保留，否则建议改为 2048 或可配置。

2. **SFT 的 `model.partial_pretrain`**  
   脚本用 `model.partial_pretrain`，sft_trainer 中对应为 `config.model.partial_pretrain`，Hydra 会正确解析。

---

## 七、总结

| 维度 | 结论 |
|------|------|
| SFT 超参数 | 符合要求，逻辑正确 |
| PPO 超参数 | 符合要求 |
| Reward/Critic 扩展 | 逻辑正确，默认不改变原版行为 |
| 最小侵入 | 扩展均为可选，叠加调用 |
| 建议 | 统一 `ROLLOUT_N` 默认与文档；按需调整 `MAX_RESPONSE_LEN` |

整体实现与实验设计一致，可用于两阶段 PPO（正常训练 critic → 冻结 critic + 可配置 reward/mask）实验。
