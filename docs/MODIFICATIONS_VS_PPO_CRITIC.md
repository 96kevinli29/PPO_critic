# 当前 verl 代码与 PPO_critic 仓库的修改说明

本文档标记当前工作区 (hyl) 中相对于 [PPO_critic](https://github.com/96kevinli29/PPO_critic) 的**自定义修改**、实现功能及逻辑正确性检查。  
（PPO_critic 源码来自 [verl-project/verl](https://github.com/verl-project/verl/tree/main)，以下“原始”指未改动的 verl 行为。）

---

## 一、修改总览

| 功能 | 配置项 | 默认值 | 说明 |
|------|--------|--------|------|
| **Reward 来源** | `algorithm.reward_source` | `rule_based` | `rule_based`：规则/reward_fn；`critic`：critic 的 value 作为 reward |
| **Reward Mask** | `algorithm.reward_mask_ratio` | `0.0` | 轨迹级 Bernoulli mask 比例 (0.0~1.0) |
| **Advantage 翻转** | `algorithm.reward_mask_flip_adv_when_masked` | `true` | 被 mask 轨迹是否翻转 advantage（仅 rule_based 时生效） |
| **Critic 冻结** | `critic.freeze` | `false` | 是否冻结 critic 参数（不更新、不建 optimizer） |

四项可独立组合使用。

---

## 二、修改位置与标记

### 1. Reward 来源 (rule_based | critic)

**文件**: `verl/verl/trainer/ppo/ray_trainer.py`

- **约 21~27 行**（文件头注释）  
  - 标记: `[Custom Modifications - 2026-02-01]` → Reward Source 说明  
- **约 1581~1613 行**  
  - 标记: `# ========== Reward Source (rule_based | critic) ==========`  
  - 逻辑:
    - `reward_source == "critic"`: 要求 `use_critic=True`；用每条轨迹**最后一个有效 token 的 value** 作为 reward，写入 `token_level_rewards` 与 `token_level_scores`；打点 `reward_source/critic=1.0`。
    - `reward_source == "rule_based"`: 使用 reward_fn/rm 的 `token_level_scores`，可选 KL penalty 后写入 `token_level_rewards`；打点 `reward_source/rule_based=1.0`。

**实现功能**: Reward 可选来自规则 (reward_fn/rm) 或来自 critic 模型打分 (value)。

---

### 2. Reward Mask（轨迹级随机 mask）

**文件**: `verl/verl/trainer/ppo/ray_trainer.py`

- **约 29~34 行**（文件头）  
  - Reward Mask 与 Advantage 翻转的说明  
- **约 1615~1636 行**  
  - 标记: `# ========== Reward Mask (轨迹级别随机 mask) ==========`  
  - 逻辑:
    - 若 `reward_mask_ratio > 0`：对 batch 做 Bernoulli(1 - ratio) 得到 `keep_mask`（1=保留，0=mask）。
    - 被 mask 的轨迹：`token_level_rewards *= keep_mask.unsqueeze(-1)`（整条轨迹 reward 置 0）。
    - 记录 `reward_mask/num_masked`、`reward_mask/mask_ratio_actual`，并保存 `_reward_keep_mask` 供后续 advantage 翻转使用。

**实现功能**: 在 GAE 之前对整条轨迹的 reward 做随机 mask，便于做“部分轨迹无 reward、仅靠 value”的实验。

---

### 3. Advantage 计算与翻转

**文件**: `verl/verl/trainer/ppo/ray_trainer.py`

- **约 1657~1688 行**  
  - 标记: `# ========== Reward Source Critic: 使用 advantage = V(last) - V(t) ==========`  
  - 逻辑:
    - `reward_source == "critic"`: 不用 GAE；直接  
      `advantage(t) = V(last) - V(t)`，再 `masked_whiten`，`returns = advantages + values`。  
      避免“r_last = V(last) 时 GAE 的 delta_last=0”导致学习信号弱。
    - `reward_source == "rule_based"`: 仍走原有 `compute_advantage`（GAE 等）。

- **约 1690~1705 行**  
  - 标记: `# ========== Reward Mask: 翻转被 mask 轨迹的 advantage ==========`  
  - 逻辑:
    - 仅当存在 `_reward_keep_mask` **且** `reward_source != "critic"` 时执行。
    - 若 `reward_mask_flip_adv_when_masked == True`：  
      `advantages *= (2 * keep_mask - 1)`（被 mask 轨迹乘 -1，未 mask 乘 +1）。  
      原因：GAE 在 reward=0 时得到的 advantage 符号与“用 value 作代理”相反，翻转后高 V 被鼓励。

**实现功能**:  
- Critic 作 reward 时用 V(last)-V(t) 替代 GAE，保证有效学习信号。  
- Rule-based + mask 时对被 mask 轨迹做 advantage 翻转，使 critic value 成为合理代理奖励。

---

### 4. Critic 冻结（跳过更新）

**文件**: `verl/verl/trainer/ppo/ray_trainer.py`

- **约 36~39 行**（文件头）  
  - Freeze Critic 说明  
- **约 1707~1717 行**  
  - 标记: `# update critic (skip if frozen)`  
  - 逻辑: `freeze_critic = self.config.critic.get("freeze", False)`；若为 True 则不调用 `_update_critic`，只打点 `critic/frozen=1.0`。

**文件**: `verl/verl/workers/config/critic.py`

- **约 16~25 行**（文件头）  
  - 标记: `[Custom Modifications - 2026-02-01]`，Freeze Critic 配置说明  
- **约 78 行**（CriticConfig）  
  - `freeze: bool = False`  
- **约 213 行**（FSDPCriticConfig）  
  - `freeze: bool = False`  

**文件**: `verl/verl/trainer/config/critic/critic.yaml`

- **约 15~17 行**  
  - `freeze: false`（Hydra 可覆盖为 `critic.freeze=true`）

**文件**: `verl/verl/workers/fsdp_workers.py`

- **约 20~37 行**（文件头）  
  - Freeze Critic 支持说明  
- **约 1496~1506 行**  
  - 标记: `# ========== Freeze Critic 支持 ==========`  
  - 逻辑: 若 `config.get("freeze", False)`：  
    - 对 critic 所有参数 `requires_grad = False`，`eval()`；  
    - 不创建 optimizer/scheduler，返回 `(critic_module, None, None)`。  
- **约 1556~1566 行**  
  - 标记: `# 标记是否冻结 critic`  
  - `self._freeze_critic = self.config.get("freeze", False)`  
- **约 1599~1604 行**  
  - 标记: `# ========== Frozen Critic: 跳过更新 ==========`  
  - 若 `_freeze_critic` 为 True，`update_critic` 直接返回带 `critic/frozen=1.0` 的 output。

**文件**: `verl/verl/workers/critic/dp_critic.py`

- **约 206~210 行**  
  - 标记: `# ========== Frozen Critic: 跳过更新 ==========`  
  - 若 `self.critic_optimizer is None`（freeze 时传入），直接返回 `{"critic/frozen": 1.0}`。

**实现功能**: Critic 可完全冻结（不建 optimizer、不更新），仅做前向提供 value baseline。

---

## 三、逻辑正确性检查

### 1. Reward 来源

- **rule_based**  
  - 先有 `batch.batch["token_level_scores"] = reward_tensor`（来自 reward_fn），再根据 KL 等得到 `token_level_rewards`。  
  - 与原始 verl 行为一致，逻辑正确。

- **critic**  
  - 在已计算 `values` 的前提下，用每条轨迹最后一个有效 response 位置的 value 作为该轨迹的 reward，并展开为 token_level（仅最后有效位置非零）。  
  - 满足“reward 来自 critic 打分”的语义；且先取 value 再参与后续 mask/GAE 分支，顺序正确。

### 2. Reward Mask

- Mask 作用在 `token_level_rewards` 上，且是轨迹级（同一轨迹全 0 或全保留），与注释一致。  
- `_reward_keep_mask` 只在 `reward_mask_ratio > 0` 时写入，且仅在 rule_based 的 advantage 翻转中使用，用完后删除，无泄漏，正确。

### 3. Advantage 与翻转

- **reward_source=critic**  
  - 使用 `V(last) - V(t)`，不经过 GAE，避免 r_last=V(last) 导致 delta_last=0 的问题，逻辑正确。  
  - 注释明确说明此时不做 flip，符合设计（critic 模式下 advantage 已按 value 差定义）。

- **reward_source=rule_based + mask**  
  - 仅在此分支且存在 `_reward_keep_mask` 时翻转；  
  - GAE 在 reward=0 时符号与“高 V 应被鼓励”相反，乘以 `(2*keep_mask-1)` 修正，逻辑正确。

### 4. Critic 冻结

- Trainer 层：`freeze_critic=True` 时不调用 `_update_critic`，只记 metric，正确。  
- FSDP 层：freeze 时 `requires_grad=False`、不建 optimizer、返回 None；  
  - `CriticWorker` 用 `critic_optimizer is None` / `_freeze_critic` 跳过 step；  
  - `dp_critic.update_critic` 对 `critic_optimizer is None` 直接返回。  
  整条链路一致，逻辑正确。

### 5. 配置与默认值

- `reward_source`、`reward_mask_ratio`、`reward_mask_flip_adv_when_masked` 通过 `algorithm.get(..., default)` 读取，未在默认 yaml 中显式列出时由脚本用 `+algorithm.*` 覆盖，行为符合预期。  
- `critic.freeze` 在 `critic.yaml` 和 dataclass 中均有定义，Hydra 覆盖有效。

---

## 四、与 PPO_critic 的差异说明

因无法直接拉取 [PPO_critic](https://github.com/96kevinli29/PPO_critic) 做逐行 diff，上述内容基于**当前 hyl 工作区**中带注释和 CHANGELOG 的修改整理。  

若 PPO_critic 与上游 verl 一致，则本仓库相对 PPO_critic 的差异即为：

1. **Reward 来源可选**：`algorithm.reward_source`（rule_based | critic）。  
2. **Reward Mask**：`algorithm.reward_mask_ratio` + `algorithm.reward_mask_flip_adv_when_masked`。  
3. **Critic 作为 reward 时的 advantage**：使用 `V(last) - V(t)` 而非 GAE。  
4. **Critic 冻结**：`critic.freeze` + 不创建 optimizer、不执行 update。

若 PPO_critic 已包含部分上述修改，则实际差异会小于上述列表；建议在本地对 PPO_critic 执行一次 `git diff` 或 `git log` 对照本仓库以确认最终差异集合。

---

## 五、Shell 与文档

- 以下脚本均支持上述四项参数（如 `REWARD_SOURCE`、`REWARD_MASK_RATIO`、`FLIP_ADV_WHEN_MASKED`、`FREEZE_CRITIC`）：  
  - `run_qwen3-0.6b_ppo_math_lighteval_test.sh`  
  - `run_qwen3-4b_ppo_math_lighteval_test.sh`  
  - `run_qwen3-0.6b_ppo_gsm8k_test.sh`  
  - `run_qwen3-4b_ppo_gsm8k_test.sh`  
  - `verl/run_qwen3-0.6b_ppo_gsm8k_reward_mask_freeze_critic.sh`  
- 更细的使用示例见项目根目录 `CHANGELOG.md`（如 2026-02-05、2026-02-01 条目）。

---

**结论**：当前实现的 reward 来源、reward mask、advantage 翻转与 critic 冻结逻辑自洽，与注释和 CHANGELOG 一致，未发现逻辑错误；与 PPO_critic 的精确差异建议用本地 git 对比确认。
