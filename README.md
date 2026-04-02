# HYL：基于 VeRL 的 Qwen3 PPO 与稀疏 / 无 Reward 实验

本仓库在 [VeRL](https://github.com/verl-project/verl) 上扩展 PPO 的 **reward 链路**与 **critic 行为**，用于两阶段实验：先标准 PPO 得到可复用的 value 模型，再在「冻结 critic + 可配置 reward / mask」下观察策略是否仍能优化。**环境与依赖**以 [`env.md`](env.md) 为准；**逐条变更历史**见 [`CHANGELOG.md`](CHANGELOG.md)。

---

## 1. 实验设计（与你的目标对齐）

| 阶段 | Actor 起点 | Critic | Reward | 验证集 |
|------|------------|--------|--------|--------|
| **1** | Qwen3-4B-SFT（`MODEL_ROOT`） | 与 actor 同初始化，**联合训练** | 默认 **rule-based**（`NaiveRewardManager` → `default_compute_score`） | 训练中周期性跑 **AIME2024 / AIME2025 / AMC2023** |
| **2** | 仍为同一 SFT（控制变量；若续训 policy 可设 `ACTOR_MODEL_PATH`） | **阶段 1 转 HF 的 critic**，自始 **冻结**（仅前向 value） | 可选 rule-based 或 **外部 RM**（`REWARD_MODEL_ENABLE=true`，例如再用 critic HF 作 RM）；支持 **二值化**与 **轨迹级 mask**（0–100%） | 同上 |

阶段 1 结束后：从 checkpoint 的 `global_step_*/critic/` 用 [`scripts/convert_critic_to_hf.py`](scripts/convert_critic_to_hf.py) 转为 HuggingFace，建议输出到与 `HF_CRITIC_DIR` 一致的路径（默认 `$HYLBASE/qwen3-4b-critic`），供阶段 2 的 `critic.model.path` 加载。

---

## 2. 入口脚本

| 脚本 | 作用 |
|------|------|
| [`run_ppo.sh`](run_ppo.sh) | 统一 Hydra 命令与环境变量；阶段 1/2 均最终调用它 |
| [`run_ppo_stage1.sh`](run_ppo_stage1.sh) | 设 `EXPERIMENT_NAME=qwen3_4b_RL1`、`HYL_RUN_LOG_PREFIX=stage1`，其余与直接跑 `run_ppo.sh` 一致：**标准 PPO**（`USE_PRETRAINED_CRITIC=no`、`FREEZE_CRITIC=false`、`REWARD_MASK_RATIO=0`） |
| [`run_ppo_stage2.sh`](run_ppo_stage2.sh) | `USE_PRETRAINED_CRITIC=yes`、`FREEZE_CRITIC=true`、默认 `REWARD_MASK_RATIO=1.0`（100% mask）、未指定实验名时为 `qwen3_4b_RL2_m<比例>` |

**结论（两阶段脚本是否正确）**：逻辑与上述实验设定一致。阶段 1 负责「dapo_math 上正常 PPO + 三榜验证 + 保存含 critic 的 checkpoint」；阶段 2 在**同一数据与默认同一 actor 起点**下，只改「预训练 critic + 冻结 + mask（及可选 RM）」，便于对照。

使用前请自备：`$DATA_DIR/dapo_math_17k/train.parquet`，以及 `aime2024|aime2025|amc2023` 的 `test.parquet`（路径与 `run_ppo.sh` 中 `VAL_FILES_LIST` 一致）。

---

## 3. 相对官方 VeRL 的主要改动（改了什么、为什么）

以下均为**最小侵入**：不改 PPO clip / GAE 核心公式（`core_algos` 中 `compute_gae_advantage_return` 等保持不变），只在 **token_level_scores → token_level_rewards → GAE** 前后增加可配置变换，并扩展 critic 生命周期。

### 3.1 训练循环（`verl/verl/trainer/ppo/ray_trainer.py`）

- **Reward 二值化**：`algorithm.reward_binarize` + `algorithm.reward_threshold`；在 **KL-in-reward 之前**对 outcome score 做 `> threshold`→1/0，避免先把 KL 混入再二值化导致信号被破坏。
- **轨迹级 reward mask**：`algorithm.reward_mask_ratio`、`algorithm.reward_mask_type`（`bernoulli` / `fixed_ratio`）。在 GAE 前将选中整条轨迹的 `token_level_rewards` 置 0，**不 mask value**；用于模拟稀疏或缺失外部 reward（含 100% mask 的 reward-free 对照）。
- **冻结 critic 时跳过更新**：`critic.freeze=true` 时不调用 `_update_critic`，仅依赖固定 value 网络算 advantage。
- **GAE 双 λ**（可选）：`algorithm.lam_actor` / `algorithm.lam_critic`；未设时回退到 `algorithm.lam`，与官方单 λ 行为兼容。
- **Bug 修复**：`use_pf_ppo` 分支里 `pf_ppo.get("reweight_method")` 等为 `None` 时不应覆盖函数默认值（与 CHANGELOG 一致）。

### 3.2 配置（`verl/verl/trainer/config/algorithm.py`、`ppo_trainer.yaml`）

- 为上述算法项提供 **AlgoConfig / YAML 正式字段**（避免只能靠 Hydra `+` 注入、便于类型与默认值统一）。
- 本地默认倾向实验设定：`reward_mask_type=fixed_ratio`、`reward_threshold=0.88` 等（官方无这些项）。

### 3.3 Critic Worker（`verl/verl/workers/fsdp_workers.py`、`critic/dp_critic.py`、critic 配置）

- **`critic.freeze`**：`requires_grad=False`、不创建 optimizer；`update_critic` 短路；冻结时 checkpoint **不写 optimizer**，避免 `save_checkpoint` 断言失败。
- **`CriticConfig.freeze`** 在 dataclass `__post_init__` 中对字符串 `"true"/"1"/"yes"` 做规范化（与 shell/Hydra 混用时的常见坑一致）。

### 3.4 其它与实验相关的仓库内改动

- **`verl/verl/utils/reward_score/__init__.py`**：数学类数据源统一走 **`math-verify`** 风格打分；`aime_*` / `amc_*` / `math_dapo` 等与竞赛和 DAPO 训练对齐；GPQA 等保留专用分支。动机见 CHANGELOG（避免全走单一 `math_dapo` 提取逻辑导致与社区评测不一致）。
- **`verl/verl/utils/model.py`**：`VERL_ATTN_IMPLEMENTATION` 控制 value head 注意力实现，无 flash-attn 环境时可 **sdpa** 回退。
- **根目录脚本**：`run_ppo.sh` 聚合数据路径、RM、`OVERLONG`/`dapo` reward manager、W&B 目录、`OUTPUTS` 等；与 `scripts/hyl_env.sh` 配合。

官方主线（如 [verl-project/verl](https://github.com/verl-project/verl) 当前 `ray_trainer`）已演进为另一套特性集合（如更多 advantage 类型、GDPO 等），**本 fork 未逐行跟踪最新上游**；若合并上游，需重点解决 `ray_trainer.py`、`algorithm.py` 与 `compute_advantage` 签名的冲突。

---

## 4. 使用注意（审查结论摘要）

- **`fixed_ratio` + 小 batch**：`num_masked = int(batch_size * ratio)`，极小的 `batch_size×ratio` 可能为 0，等价于当步不 mask。
- **布尔配置**：推荐通过 Hydra 传 `true/false`；若某路径把 `freeze` / `reward_binarize` 以**非规范字符串**注入 DictConfig，`.get()` 可能出现 Python 真值语义问题；当前 `run_ppo.sh` 与 dataclass 路径在常规用法下是安全的。
- **阶段 2 默认 100% mask**：advantage 主要来自 **critic bootstrap**，与「弱/无外部 reward」假设一致；若要与阶段 1 公平对比，可设 `REWARD_MASK_RATIO=0` 跑「预训练 critic + 冻结但全 reward」基线。

---

## 5. 快速命令

```bash
source scripts/hyl_env.sh ppo

# 阶段 1
./run_ppo_stage1.sh

# 阶段 1 结束后（示例）
python scripts/convert_critic_to_hf.py \
  --local_dir "$OUTPUTS/ppo/qwen3_4b_RL1/global_step_<N>/critic" \
  --target_dir "$HYLBASE/qwen3-4b-critic" \
  --trust-remote-code

# 阶段 2（默认 100% mask）
./run_ppo_stage2.sh

# 阶段 2：外部 RM（例如 critic HF）+ 可自行配 mask / 二值化
# REWARD_MODEL_ENABLE=true REWARD_MODEL_PATH=... ./run_ppo_stage2.sh
```

更细的参数表见 `run_ppo.sh` 文件头注释。
