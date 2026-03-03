# PPO_critic

> **目的 / Purpose**  
> 在 [verl](https://github.com/verl-project/verl) 上做 **带 Critic 的 PPO** 实验：用 Critic 的 value 作为 reward、支持 reward 轨迹级 mask、advantage 翻转与 critic 冻结等，用于探索「用价值函数替代/辅助规则 reward」的 RL 训练效果。  
>  
> *This repo experiments with **PPO + Critic** on verl: using critic value as reward, trajectory-level reward mask, advantage flipping, and critic freeze, to explore RL training with value-based reward instead of (or alongside) rule-based reward.*

**修改说明**：[docs/MODIFICATIONS_VS_PPO_CRITIC.md](docs/MODIFICATIONS_VS_PPO_CRITIC.md) · **Upstream**：基于 [verl-project/verl](https://github.com/verl-project/verl)。

---

## 原始库说明 / About verl (upstream)

Hi, everyone! **verl** is a RL training library initiated by **ByteDance Seed team** and maintained by the verl community.

**verl: Volcano Engine Reinforcement Learning for LLMs**

verl is a flexible, efficient and production-ready RL training library for large language models (LLMs).

verl is the open-source version of **HybridFlow: A Flexible and Efficient RLHF Framework** paper.

**verl is flexible and easy to use with:**

- **Easy extension of diverse RL algorithms:** The hybrid-controller programming model enables flexible representation and efficient execution of complex post-training dataflows. Build RL dataflows such as GRPO, PPO in a few lines of code.
- **Seamless integration of existing LLM infra with modular APIs:** Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as FSDP, Megatron-LM, vLLM, SGLang, etc.
- **Flexible device mapping:** Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.
- Ready integration with popular HuggingFace models.

**verl is fast with:**

- **State-of-the-art throughput:** SOTA LLM training and inference engine integrations and SOTA RL throughput.
- **Efficient actor model resharding with 3D-HybridEngine:** Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

---

## 本仓库结构 / Repo structure

| 路径 | 说明 |
|------|------|
| `verl/` | verl 源码（含上述扩展修改） |
| `scripts/` | 转换、数据处理等脚本 |
| `configs/` | PPO 训练配置 |
| `docs/MODIFICATIONS_VS_PPO_CRITIC.md` | 与上游 PPO_critic/verl 的修改说明 |
| `run_*.sh` | 各场景启动脚本 |

### 主要脚本 / Main scripts

| 脚本 | 说明 |
|------|------|
| **`run_qwen3_ppo.sh`** | 统一 Qwen3 PPO 入口：通过环境变量切换数据集（gsm8k/math）、模型规模（0.6b/4b）、预训练 critic、reward 二值化、轨迹级 mask、critic 冻结、GAE lambda 等；消融实验命令见同目录 `run_ppo_ablation_commands.md`。 |
| **`run_eval_critic.sh`** | 评估预训练 critic 的 reward 预测能力：用 vLLM 生成 rollout，rule-based 算 ground-truth，critic 预测并与阈值二分类，输出 accuracy/AUC 等；支持 `DATA=math MODEL_TAG=0.6b` 等环境变量。 |
| **`scripts/eval_critic_prediction.py`** | 上述评估的 Python 实现：加载 parquet 数据、生成 rollout、用 critic 取最后 token value 作预测、计算指标；可单独调用或由 `run_eval_critic.sh` 调用。 |
| **`scripts/convert_critic_to_hf.py`** | 将 VeRL/FSDP 格式的 Critic checkpoint 转为 HuggingFace 格式；仅需 `torch`、`transformers`、`accelerate`、`safetensors`，不依赖 verl 包。 |

更多使用方式请参考 `verl/` 内文档与 [verl 官方文档](https://verl.readthedocs.io/)。
