# 修改日志

记录相对 VeRL 原版的关键修改，便于追溯与回滚。算法与代码细节见 [docs/PPO_MODIFICATIONS_SUMMARY.md](docs/PPO_MODIFICATIONS_SUMMARY.md)。

---

## 2026-03：dapo_math 支持与评估脚本同步 PPO_critic

- **run_qwen3_ppo.sh**：支持 `DATA=dapo_math`（数据路径 `data/dapo_math/train|test.parquet`，critic 输出 `critic_qwen3_*_ppo_dapo_math*`）；阶段 1 结束后自动备份并转 HF 需 `SAVE_FREQ>0` 且训练正常退出。
- **run_eval_critic.sh**：新增 `dapo_math` 分支；实验名与输出目录去掉阈值后缀（`eval_critic_${DATA}_${MODEL_TAG}`）。
- **scripts/eval_critic_prediction.py**：兼容 `math_dapo` 的 `compute_score` 返回 dict 及 -1/1 reward（取 `score`、`max(score,0)` 转 0/1）。
- **scripts/convert_critic_to_hf.py**：已存在，与阶段 1 保存的 FSDP critic 转 HF 流程配套。
- 同步至 [PPO_critic](https://github.com/96kevinli29/PPO_critic) 的详细说明见 [docs/UPDATE_2026_03_PPO_CRITIC.md](docs/UPDATE_2026_03_PPO_CRITIC.md)。

---

## 算法修改目的

在 **GSM8K / MATH** 上验证**稀疏 reward / reward-free** 下的 PPO 表现：控制「无 reward」轨迹比例，观察在 critic 仅作 baseline 时 actor 是否仍能学习。

- **阶段 1**：正常 PPO，训练并保存 critic。
- **阶段 2**：Base 作 actor，PPO + 可配置 reward（rule-based 或 critic 转 HF 当 RM）+ **reward mask**（0–100%）+ **冻结 critic**。

实现上仅在「reward → GAE → loss」链路上增加可配置的 mask、二值化、critic 冻结；**不改变 PPO 数学形式**（GAE/clip/loss 不变），不修改 `core_algos.py`，保持最小侵入。

---

## 2026-02-21：Bug 修复 + 脚本优化

### 修改 1：`compute_pf_ppo_reweight_data` 调用缺少默认值（Bug 修复）

**文件**：`verl/verl/trainer/ppo/ray_trainer.py` 第 226–227 行

**原因**：`compute_pf_ppo_reweight_data(data, reweight_method="pow", weight_pow=2.0)` 函数签名有默认参数，但调用处使用 `config.pf_ppo.get("reweight_method")` 和 `config.pf_ppo.get("weight_pow")` 时没有提供 `.get()` 的默认值。当 YAML 配置中未设置这两项时，`.get()` 返回 `None`，会**覆盖**函数签名中的默认参数（Python 语义：显式传入 `None` 不等于不传），导致后续用 `None` 做字符串比较时报错。

**修复**：
```python
# 修复前
config.pf_ppo.get("reweight_method"),
config.pf_ppo.get("weight_pow"),
# 修复后
config.pf_ppo.get("reweight_method", "pow"),
config.pf_ppo.get("weight_pow", 2.0),
```

---

### 修改 2：`AlgoConfig` 补齐缺失字段定义（类型安全）

**文件**：`verl/verl/trainer/config/algorithm.py`、`verl/verl/trainer/config/ppo_trainer.yaml`

**原因**：`reward_mask_ratio`、`reward_binarize`、`reward_threshold` 三个配置项在 `ray_trainer.py` 中通过 `.get()` 使用，但未在 `AlgoConfig` dataclass 中声明。这意味着：
1. 没有类型检查——传入错误类型不会在启动时报错，只在运行时崩溃；
2. IDE 无法自动补全，容易拼写错误；
3. 启动脚本必须用 `+` 前缀（Hydra 的"注入新键"语法）来传递这些值，语义上不正确（这些不是"新增键"，而是算法的正式参数）。

**修复**：在 `AlgoConfig` 中添加字段定义，在 `ppo_trainer.yaml` 中添加对应默认值：
```python
# algorithm.py
reward_mask_ratio: float = 0.0
reward_binarize: bool = False
reward_threshold: float = 0.5
```
```yaml
# ppo_trainer.yaml
reward_mask_ratio: 0.0
reward_binarize: false
reward_threshold: 0.5
```

---

### 修改 3：脚本新增 `LAM_ACTOR` / `LAM_CRITIC` 环境变量

**文件**：`run_qwen3_ppo.sh`

**原因**：代码中已实现 dual GAE lambda（`lam_actor` 算 advantage、`lam_critic` 算 returns），`ppo_trainer.yaml` 中已有默认值 0.95 / 1.0，但启动脚本未提供对应的环境变量接口，也未将其传递到训练命令中。这意味着：用户无法通过环境变量覆盖 GAE lambda 做消融实验，且实验名中无法体现该条件。

**修复**：
- 新增环境变量 `LAM_ACTOR`（默认 0.95）、`LAM_CRITIC`（默认 1.0）
- 在训练命令中传递 `algorithm.lam_actor=$LAM_ACTOR algorithm.lam_critic=$LAM_CRITIC`

---

### 修改 4：移除已定义字段的 `+` 前缀

**文件**：`run_qwen3_ppo.sh`

**原因**：修改 2 将 `reward_mask_ratio`、`reward_binarize`、`reward_threshold` 添加到 `AlgoConfig` dataclass 后，它们已是配置 schema 的一部分。Hydra 中 `+key=value` 的语义是"注入一个 schema 中不存在的新键"，对于已定义的字段应使用 `key=value`（无 `+`）。保留 `+` 虽不报错，但语义不准确，且在 Hydra 严格模式下可能产生警告。

**修复**：
```bash
# 修复前
+algorithm.reward_mask_ratio=$REWARD_MASK_RATIO
+algorithm.reward_binarize=$REWARD_BINARIZE
+algorithm.reward_threshold=$REWARD_THRESHOLD
# 修复后
algorithm.reward_mask_ratio=$REWARD_MASK_RATIO
algorithm.reward_binarize=$REWARD_BINARIZE
algorithm.reward_threshold=$REWARD_THRESHOLD
```

---

### 修改 5：实验名称体现所有独立对照条件

**文件**：`run_qwen3_ppo.sh`

**原因**：原实验名格式 `qwen3_{MODEL}_ppo_{DATA}_{RS}_m{MASK}_{C_TAG}_{F_TAG}` 缺少以下独立实验条件：
- **二值化**（`reward_binarize`）：是否对 reward 做 0/1 转换，直接影响 advantage 的尺度；
- **Mask 类型**（`reward_mask_type`）：bernoulli vs fixed_ratio 是不同的稀疏化策略，实验结论不同；
- **GAE lambda**（`lam_actor`/`lam_critic`）：非默认 lambda 会改变 bias-variance 权衡。

这些条件在 wandb 中无法区分，导致同名实验覆盖或混淆。

**修复**：新实验名格式：
```
qwen3_{MODEL}_ppo_{DATA}_{RS}[_bin]_m{MASK}[_{MT}]_{C_TAG}_{F_TAG}[_la{X}_lc{Y}]
```
- `_bin`：仅 `REWARD_BINARIZE=true` 时出现
- `_bern` / `_fix`：仅 mask > 0 时出现，区分 mask 策略
- `_la{X}_lc{Y}`：仅 GAE lambda 非默认值时出现

---

## 2026-02：PPO 扩展（Reward 二值化 / Mask / 冻结 Critic）

### 新增功能

| 功能 | 配置 | 说明 |
|------|------|------|
| **Reward 二值化** | `+algorithm.reward_binarize`、`+algorithm.reward_threshold` | RM 连续分数转为 0/1（如 critic 当 RM 时） |
| **Reward Mask** | `+algorithm.reward_mask_ratio` (0.0–1.0)、`algorithm.reward_mask_type` | 轨迹级 mask：`bernoulli`（每条独立概率，默认）/ `fixed_ratio`（每步恰好固定条数被 mask，更贴近真实有标签比例） |
| **Freeze Critic** | `critic.freeze` | 参数不更新、不建 optimizer，仅提供 value baseline |
| **GAE lambda 分离** | `algorithm.lam_actor`、`algorithm.lam_critic` | Actor 用 `lam_actor`（默认 0.95）算 advantage，Critic 用 `lam_critic`（默认 1.0）算 returns；不设则用 `algorithm.lam`。设计理由：advantage 用较小 λ 降方差，returns 用 1.0 保持无偏。 |

### 涉及文件（概要）

- `verl/verl/trainer/ppo/ray_trainer.py`：二值化、mask（bernoulli / fixed_ratio）、跳过 frozen critic 更新、GAE 双 lambda（lam_actor/lam_critic）；`reward_binarize` 从环境变量传入时做字符串解析
- `verl/verl/trainer/config/algorithm.py`：`lam_actor`、`lam_critic`、`reward_mask_type`、`reward_mask_ratio`、`reward_binarize`、`reward_threshold` 配置项
- `verl/verl/trainer/config/ppo_trainer.yaml`：默认 `lam_actor: 0.95`、`lam_critic: 1.0`、`reward_mask_type: bernoulli`、`reward_mask_ratio: 0.0`、`reward_binarize: false`、`reward_threshold: 0.5`
- `verl/verl/workers/fsdp_workers.py`：freeze 时冻结参数并返回 `None` optimizer
- `verl/verl/workers/critic/dp_critic.py`：`critic_optimizer is None` 时跳过更新
- `verl/verl/workers/config/critic.py`、`verl/verl/trainer/config/critic/critic.yaml`：`freeze` 配置项
- `run_ablation.sh`：消融入口（stage1 / baseline_a|b / mask_0.3～1.0 / mask_sweep / rm_critic）；`rm_critic` 预设开启二值化
- `scripts/convert_critic_to_hf.py`：阶段 1 保存的 FSDP critic 转 HuggingFace，供阶段 2 加载

### 修复：布尔/数值从 Shell 传入时的字符串解析

**含义**：通过环境变量传参（如 `FREEZE_CRITIC=false`、`REWARD_BINARIZE=true`）时，Hydra 有时会把值当成**字符串**写进 config（如 `"false"`、`"true"`）。在 Python 里非空字符串是 truthy，若直接写 `if freeze_critic:`，则 `freeze_critic="false"` 也会被当成 True，导致「本意不冻结却变成冻结」等错误。字符串解析即：**仅当字符串为 `"true"` / `"1"` / `"yes"`（不区分大小写）时才视为 True，其它一律视为 False**；数值型（如 `reward_threshold`）则在使用前转为 `float`。

**正确性**：上述规则在三处一致使用，行为正确。

| 配置项 | 字符串解析位置 | 规则 |
|--------|----------------|------|
| `critic.freeze` | `fsdp_workers.py`（`_build_critic_model_optimizer`、`init_model`）、`ray_trainer.py`（是否调用 `_update_critic`） | 仅 `"true"`/`"1"`/`"yes"` → True；`"false"`/`"0"`/其它 → False |
| `algorithm.reward_binarize` | `ray_trainer.py`（是否二值化） | 同上 |
| `algorithm.reward_threshold` | `ray_trainer.py`（二值化阈值） | 若为字符串则 `float(threshold)`，再与 tensor 比较 |

**修改概要**：

- `verl/verl/workers/fsdp_workers.py`：`_build_critic_model_optimizer` 与 `init_model` 中对 `freeze` 做上述布尔字符串解析。
- `verl/verl/trainer/ppo/ray_trainer.py`：对 `freeze_critic`、`reward_binarize` 做布尔字符串解析；对 `reward_threshold` 若为字符串则转 `float`。

### 入口脚本

统一入口 **`run_qwen3_ppo.sh`**，通过环境变量切换 DATA、MODEL_TAG、RM、二值化、mask、冻结等。

### 逻辑检查结论（简要）

- **Reward 来源**：与 VeRL 一致；`use_rm` 时由 `compute_rm_score` 得到 `rm_scores`，否则用 `reward_fn`；`_compute_or_extract_reward` 有则提取、无则用 reward_fn；`token_level_scores` → 二值化（可选）→ `token_level_rewards` → mask（可选）→ GAE。未改 `core_algos.py`。
- **二值化**：`reward_binarize` 为真时 `token_level_rewards = (scores > threshold).float()`，含 Shell 传入的字符串解析。
- **Reward Mask**：轨迹级；`bernoulli` 为每条独立以 `1-ratio` 概率保留，`fixed_ratio` 为每步恰好 `floor(batch_size*ratio)` 条被置 0；`keep_mask` 与 `token_level_rewards` 逐轨迹相乘，形状正确。
- **Freeze Critic**：`critic.freeze=true` 时 FSDP 侧不建 optimizer、参数 `requires_grad=False`，worker 与 trainer 均做字符串解析；`update_critic` 在 freeze 时跳过，DP critic 在 `optimizer is None` 时直接返回。
- **GAE 双 lambda**：`lam_actor`/`lam_critic` 不一致时分别用两者调用 `compute_gae_advantage_return` 得到 advantages 与 returns，逻辑正确。

---

## 消融 / 对照实验指令（复制即跑）

**推荐**：用 **`run_ablation.sh`** 一条命令跑一组实验，见下方「便捷入口」；也可直接复制下面「完整单行命令」到终端运行。

**前提**：先完成阶段 1 并得到预训练 critic 的 HF 路径（如 `$HYLBASE/critic_qwen3_0.6b_ppo_gsm8k_hf`）。未做阶段 1 时，用「基线 A」即可（基座 critic、可训、无 mask），与阶段 2 的「预训练 + 冻结」做对比需先有 critic HF。

**实验名格式**：`qwen3_{MODEL}_ppo_{DATA}_{RS}[_bin]_m{MASK}[_{MT}]_{cpt|cb}_{fr|tr}[_la{X}_lc{Y}]`  
（RS=rb/c06b/c4b，bin=二值化，m=mask 比例，MT=bern/fix，cpt=预训练 critic / cb=基座，fr=冻结 / tr=可训，la/lc=非默认 GAE lambda）

### 便捷入口（run_ablation.sh）

在项目根目录执行（可加 `DATA=math` 或 `MODEL_TAG=4b` 覆盖默认）：

```bash
./run_ablation.sh stage1          # 阶段1，保存 critic
./run_ablation.sh baseline_a     # 基线A：基座 critic、可训、无 mask
./run_ablation.sh baseline_b     # 基线B：预训练 critic + 冻结、无 mask
./run_ablation.sh mask_0.3        # 30% mask
./run_ablation.sh mask_0.5        # 50% mask
./run_ablation.sh mask_0.7        # 70% mask
./run_ablation.sh mask_1.0        # 100% mask（完全 reward-free）
./run_ablation.sh mask_sweep     # 批量跑 0/0.3/0.5/0.7/1.0 共 5 组
./run_ablation.sh rm_critic      # Reward=critic（二值化）+ 50% mask，用默认 critic 路径
```

### 完整单行命令（复制到终端即跑）

在项目根目录执行；若用脚本则无需复制，执行上面 `./run_ablation.sh <实验名>` 即可。

```bash
# 阶段1（只跑一次）
FREEZE_CRITIC=false REWARD_MASK_RATIO=0 REWARD_MODEL_ENABLE=false SAVE_FREQ=99999 DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh

# 基线A
DATA=gsm8k MODEL_TAG=0.6b FREEZE_CRITIC=false REWARD_MASK_RATIO=0 REWARD_MODEL_ENABLE=false ./run_qwen3_ppo.sh

# 基线B
DATA=gsm8k MODEL_TAG=0.6b USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=0 REWARD_MODEL_ENABLE=false ./run_qwen3_ppo.sh

# mask 30% / 50% / 70% / 100%
DATA=gsm8k MODEL_TAG=0.6b USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=0.3 REWARD_MODEL_ENABLE=false ./run_qwen3_ppo.sh
DATA=gsm8k MODEL_TAG=0.6b USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=0.5 REWARD_MODEL_ENABLE=false ./run_qwen3_ppo.sh
DATA=gsm8k MODEL_TAG=0.6b USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=0.7 REWARD_MODEL_ENABLE=false ./run_qwen3_ppo.sh
DATA=gsm8k MODEL_TAG=0.6b USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=1.0 REWARD_MODEL_ENABLE=false ./run_qwen3_ppo.sh

# RM=critic + 50% mask（用默认 critic 路径，无需填 REWARD_MODEL_PATH）
DATA=gsm8k MODEL_TAG=0.6b USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MODEL_ENABLE=true REWARD_MASK_RATIO=0.5 ./run_qwen3_ppo.sh

# 批量扫 mask（5 组）
for m in 0 0.3 0.5 0.7 1.0; do REWARD_MASK_RATIO=$m USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true DATA=gsm8k MODEL_TAG=0.6b REWARD_MODEL_ENABLE=false ./run_qwen3_ppo.sh; done
```

MATH 或 4B：将上面命令中的 `DATA=gsm8k` 改为 `DATA=math`，或 `MODEL_TAG=0.6b` 改为 `MODEL_TAG=4b`（4B 建议 `N_GPUS=8`）。

### 1. 阶段 1（只跑一次，得到 critic）

```bash
FREEZE_CRITIC=false REWARD_MASK_RATIO=0 REWARD_MODEL_ENABLE=false \
SAVE_FREQ=99999 DATA=gsm8k MODEL_TAG=0.6b ./run_qwen3_ppo.sh
```

训练结束后，从 `checkpoints/PPO-Project2026/qwen3_0.6b_ppo_gsm8k/global_step_*/critic` 拷贝到本地，再转 HF：

```bash
python scripts/convert_critic_to_hf.py \
  --local_dir checkpoints/PPO-Project2026/qwen3_0.6b_ppo_gsm8k/global_step_XXXX/critic \
  --target_dir critic_qwen3_0.6b_ppo_gsm8k_hf \
  --trust-remote-code
```

将 `critic_qwen3_0.6b_ppo_gsm8k_hf` 放在项目下或记下绝对路径，供下面 `REWARD_MODEL_PATH` / 预训练 critic 使用。

---

### 2. 消融维度与对照表

| 维度 | 选项 | 环境变量 |
|------|------|----------|
| Reward 来源 | rule-based | `REWARD_MODEL_ENABLE=false`（默认） |
| | critic 当 RM | `REWARD_MODEL_ENABLE=true REWARD_MODEL_PATH=/path/to/critic_hf` |
| Mask 比例 | 0 / 0.3 / 0.5 / 0.7 / 1.0 | `REWARD_MASK_RATIO=0` 等 |
| Critic | 预训练 + 冻结 | `USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true` |
| | 基座 + 可训 | 默认（不设 USE_PRETRAINED_CRITIC 或 FREEZE_CRITIC=false） |

---

### 3. 单条对照命令（阶段 2，GSM8K 0.6b）

**基线 A**：基座 critic、可训、无 mask（与 VeRL 正常 PPO 一致）

```bash
DATA=gsm8k MODEL_TAG=0.6b \
FREEZE_CRITIC=false REWARD_MASK_RATIO=0 REWARD_MODEL_ENABLE=false \
./run_qwen3_ppo.sh
```

**基线 B**：预训练 critic + 冻结、无 mask（阶段 2 的 0% mask 对照）

```bash
DATA=gsm8k MODEL_TAG=0.6b \
USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=0 REWARD_MODEL_ENABLE=false \
./run_qwen3_ppo.sh
```

**消融：不同 mask 比例**（预训练 critic + 冻结，rule-based reward）

```bash
# 30% mask
DATA=gsm8k MODEL_TAG=0.6b USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=0.3 REWARD_MODEL_ENABLE=false ./run_qwen3_ppo.sh

# 50% mask
DATA=gsm8k MODEL_TAG=0.6b USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=0.5 REWARD_MODEL_ENABLE=false ./run_qwen3_ppo.sh

# 70% mask
DATA=gsm8k MODEL_TAG=0.6b USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=0.7 REWARD_MODEL_ENABLE=false ./run_qwen3_ppo.sh

# 100% mask（完全 reward-free，仅 critic value 作 baseline）
DATA=gsm8k MODEL_TAG=0.6b USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true REWARD_MASK_RATIO=1.0 REWARD_MODEL_ENABLE=false ./run_qwen3_ppo.sh
```

**消融：Reward 来源 = critic 当 RM**（二值化自动开，预训练 critic + 冻结 + 50% mask）

```bash
# 需把 /path/to/critic_hf 换成实际路径，如 $HYLBASE/critic_qwen3_0.6b_ppo_gsm8k_hf
DATA=gsm8k MODEL_TAG=0.6b USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true \
REWARD_MODEL_ENABLE=true REWARD_MODEL_PATH=/path/to/critic_hf \
REWARD_MASK_RATIO=0.5 ./run_qwen3_ppo.sh
```

---

### 4. 批量扫 mask 比例（一键跑完消融）

```bash
# GSM8K 0.6b：mask 0 / 0.3 / 0.5 / 0.7 / 1.0 共 5 组
for m in 0 0.3 0.5 0.7 1.0; do
  REWARD_MASK_RATIO=$m USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true \
  DATA=gsm8k MODEL_TAG=0.6b REWARD_MODEL_ENABLE=false \
  ./run_qwen3_ppo.sh
done
```

**MATH 数据集**：把上面命令里的 `DATA=gsm8k` 改为 `DATA=math`，并确保阶段 1 用 math 训练过 critic、转 HF 后路径为 `critic_qwen3_0.6b_ppo_math_lighteval_hf`（或自行设 `CRITIC_MODEL_PATH` / `REWARD_MODEL_PATH`）。

**4B 模型**（8 卡）：

```bash
for m in 0 0.3 0.5 0.7 1.0; do
  REWARD_MASK_RATIO=$m USE_PRETRAINED_CRITIC=yes FREEZE_CRITIC=true \
  DATA=gsm8k MODEL_TAG=4b N_GPUS=8 REWARD_MODEL_ENABLE=false \
  ./run_qwen3_ppo.sh
done
```

---

### 5. 对比时看什么

- **WandB**：实验名含 `m0` / `m0.5` / `m1.0`、`cpt`/`cb`、`fr`/`tr`，按 project 筛选后对比 val 准确率或 reward。
- **本地**：`trainer.default_local_dir` 下按实验名分子目录，checkpoint 与日志在同名目录中。

---

## 2026-02：Critic 转 HuggingFace 格式

阶段 1 保存的 Critic 为 FSDP 分片格式，需转为 HuggingFace 后才能作为 `critic.model.path` 或 RM 用于阶段 2。

- **脚本**：`scripts/convert_critic_to_hf.py`
- **用法**：`python scripts/convert_critic_to_hf.py --local_dir <fsdp_critic_dir> --target_dir <output_hf_dir> [--trust-remote-code]`
- 输出为 `Qwen3ForTokenClassification`（score 层即 value head），可直接被 VeRL 加载。

---

## 2026-01：注意力实现兼容（无 FlashAttention 环境）

系统 GLIBC 与 `flash-attn` 不兼容时，避免硬编码 `flash_attention_2`。

- **修改**：`verl/verl/utils/model.py` 中 `load_valuehead_model()` 改为从环境变量 `VERL_ATTN_IMPLEMENTATION`（或 `HF_ATTN_IMPLEMENTATION`）读取，默认 `sdpa`。
- **使用**：在训练脚本中设置 `export VERL_ATTN_IMPLEMENTATION=sdpa`，并在 Hydra 中覆盖 `+actor_rollout_ref.model.override_config.attn_implementation=sdpa`、`+critic.model.override_config.attn_implementation=sdpa`。`run_qwen3_ppo.sh` 已包含上述设置。
