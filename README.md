# PPO_critic

> **Purpose**  
> This repo runs **PPO with Critic** on [verl](https://github.com/verl-project/verl): using critic value as reward, trajectory-level reward mask, and critic freeze, to explore value-based (or hybrid) reward for RL training.  
>  
> **目的**：在 verl 上做带 Critic 的 PPO 实验：用 Critic 的 value 作为 reward、轨迹级 mask、critic 冻结等，探索价值函数替代/辅助规则 reward 的 RL 效果。

**Modifications:** [docs/MODIFICATIONS_VS_PPO_CRITIC.md](docs/MODIFICATIONS_VS_PPO_CRITIC.md) · **Upstream:** [verl-project/verl](https://github.com/verl-project/verl)

---

## About verl (原始库说明)

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

## Repo structure (本仓库结构)

| Path | Description |
|------|--------------|
| `verl/` | verl source with PPO+Critic extensions（verl 源码及扩展） |
| `scripts/` | Conversion and data-processing scripts（转换与数据处理脚本） |
| `configs/` | PPO training configs（PPO 训练配置） |
| `docs/MODIFICATIONS_VS_PPO_CRITIC.md` | Modifications vs upstream（相对上游的修改说明） |
| `run_*.sh` | Launch scripts per scenario（各场景启动脚本） |

### Main scripts (主要脚本)

| Script | Description (English) |
|--------|----------------------|
| **`run_qwen3_ppo.sh`** | Unified Qwen3 PPO entry. Use env vars to switch dataset (gsm8k/math), model size (0.6b/4b), pretrained critic, reward binarization, trajectory-level mask, critic freeze, GAE lambdas. Ablation commands: `run_ppo_ablation_commands.md`. |
| **`run_eval_critic.sh`** | Evaluate pretrained critic’s reward prediction: vLLM generates rollouts, rule-based ground-truth, critic predicts with threshold; outputs accuracy/AUC. Example: `DATA=math MODEL_TAG=0.6b ./run_eval_critic.sh`. |
| **`scripts/eval_critic_prediction.py`** | Python evaluation: load parquet, generate rollouts, use critic’s last-token value as prediction, compute metrics. Run standalone or via `run_eval_critic.sh`. |
| **`scripts/convert_critic_to_hf.py`** | Convert VeRL/FSDP Critic checkpoint to HuggingFace format. Requires only torch, transformers, accelerate, safetensors (no verl). |

For more usage, see the `verl/` docs and [verl documentation](https://verl.readthedocs.io/).
