# Critic模型保存和加载指南

## 📋 概述

本指南说明如何：
1. **保存**训练完PPO后的Critic模型
2. **加载**保存的Critic模型作为新PPO训练的初始Critic模型

---

## 💾 第一部分：保存Critic模型

### 1.1 自动保存机制

VeRL在PPO训练过程中**自动保存Critic模型**，无需额外配置。

#### 保存位置

Critic模型会保存在以下路径：

```
{checkpoint_dir}/global_step_{step}/Critic/
```

**示例**：
```
checkpoints/gsm8k-ppo/experiment-name/global_step_1000/Critic/
```

#### 保存时机

- 根据`trainer.save_freq`配置自动保存
- 训练结束时也会保存

#### 保存内容

根据`critic.checkpoint.save_contents`配置，默认保存：
- `model`：Critic模型参数
- `optimizer`：优化器状态
- `extra`：额外信息（如学习率调度器状态）

### 1.2 配置保存参数

在PPO训练配置中：

```yaml
trainer:
  # Checkpoint保存目录
  default_local_dir: checkpoints/gsm8k-ppo/experiment-name
  
  # 保存频率（每N步保存一次）
  save_freq: 100
  
  # 最多保留多少个Critic checkpoint（避免占用太多空间）
  max_critic_ckpt_to_keep: 3

critic:
  checkpoint:
    # 保存内容
    save_contents: ['model', 'optimizer', 'extra']
```

### 1.3 手动保存最后一个Checkpoint

训练完成后，最后一个checkpoint通常保存在：

```bash
# 查看最新的checkpoint
ls -lt checkpoints/gsm8k-ppo/experiment-name/ | head -5

# 假设最新的是 global_step_1000
# Critic模型在：
checkpoints/gsm8k-ppo/experiment-name/global_step_1000/Critic/
```

**建议**：将最后一个checkpoint的Critic模型复制到专门的位置：

```bash
# 创建专门的critic模型目录
mkdir -p saved_critics/experiment-final

# 复制Critic模型
cp -r checkpoints/gsm8k-ppo/experiment-name/global_step_1000/Critic/* \
      saved_critics/experiment-final/
```

---

## 🔄 第二部分：加载Critic模型

### 2.1 方法一：使用Resume模式（推荐）

这是最简单的方法，适用于**只加载Critic模型，Actor使用新的初始模型**。

#### 步骤1：准备Critic Checkpoint路径

假设你的Critic模型保存在：
```
checkpoints/previous-ppo/global_step_1000/Critic/
```

#### 步骤2：在新PPO训练配置中设置

```yaml
trainer:
  # 设置为 "resume_path" 模式
  resume_mode: resume_path
  
  # 指定包含Critic的checkpoint路径（必须包含 "global_step_"）
  resume_from_path: checkpoints/previous-ppo/global_step_1000
  
  # 但是Actor使用新的初始模型
  # Actor配置保持不变，使用新的模型路径
```

#### 步骤3：在训练脚本中覆盖配置

```bash
python -m verl.trainer.main_ppo \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=checkpoints/previous-ppo/global_step_1000 \
    # Actor使用新模型
    actor_rollout_ref.model.path=~/models/new-actor-model \
    # 其他配置...
```

**注意**：使用`resume_mode=resume_path`会同时加载Actor和Critic。如果你只想加载Critic，需要使用下面的方法。

### 2.2 方法二：只加载Critic模型（需要修改代码）

如果你**只想加载Critic模型，Actor使用全新的模型**，需要修改代码逻辑。

#### 方案A：修改`ray_trainer.py`（不推荐，但可行）

在`_load_checkpoint`方法中，只加载Critic：

```python
# 在 verl/trainer/ppo/ray_trainer.py 的 _load_checkpoint 方法中
def _load_checkpoint(self):
    # ... 前面的代码 ...
    
    critic_path = os.path.join(global_step_folder, str(Role.Critic))
    
    # 只加载Critic，不加载Actor
    if self.use_critic:
        self.critic_wg.load_checkpoint(
            critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
    
    # 跳过Actor的加载
    # actor_path = os.path.join(global_step_folder, "actor")
    # self.actor_rollout_wg.load_checkpoint(...)  # 注释掉这行
```

#### 方案B：使用配置参数控制（推荐）

更好的方法是添加一个配置参数来控制是否只加载Critic。

**修改配置**：

```yaml
trainer:
  resume_mode: resume_path
  resume_from_path: checkpoints/previous-ppo/global_step_1000
  
  # 新增：是否只加载Critic模型
  load_critic_only: true
```

**修改代码**（在`ray_trainer.py`中）：

```python
def _load_checkpoint(self):
    # ... 前面的代码 ...
    
    actor_path = os.path.join(global_step_folder, "actor")
    critic_path = os.path.join(global_step_folder, str(Role.Critic))
    
    # 根据配置决定是否加载Actor
    if not self.config.trainer.get("load_critic_only", False):
        # 加载Actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
    
    # 加载Critic
    if self.use_critic:
        self.critic_wg.load_checkpoint(
            critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
```

### 2.3 方法三：直接指定Critic Checkpoint路径（需要代码支持）

如果VeRL支持直接指定Critic的checkpoint路径，可以这样配置：

```yaml
critic:
  # 直接指定Critic checkpoint路径
  checkpoint_path: checkpoints/previous-ppo/global_step_1000/Critic
```

但目前VeRL可能不支持这个功能，需要查看代码确认。

---

## 🎯 推荐方案

### 场景1：继续训练（Resume整个训练）

如果你要**继续之前的训练**（包括Actor和Critic都从checkpoint加载）：

```bash
python -m verl.trainer.main_ppo \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=checkpoints/previous-ppo/global_step_1000 \
    # 其他配置保持不变
```

### 场景2：只使用之前的Critic，Actor用新模型

**推荐方案**：使用`resume_mode=resume_path`，但修改代码只加载Critic（见方法二方案B）。

**临时方案**：
1. 使用`resume_mode=resume_path`加载整个checkpoint
2. 然后手动重新初始化Actor模型

---

## 📝 完整示例

### 示例1：保存最后一个Critic模型

```bash
# 训练完成后，找到最后一个checkpoint
LAST_STEP=$(ls -t checkpoints/gsm8k-ppo/exp1/ | grep global_step | head -1 | sed 's/global_step_//')

# 复制Critic模型
mkdir -p saved_critics/final-critic
cp -r checkpoints/gsm8k-ppo/exp1/global_step_${LAST_STEP}/Critic/* \
      saved_critics/final-critic/
```

### 示例2：在新训练中使用保存的Critic

```bash
# 假设Critic保存在：saved_critics/final-critic/
# 需要先将其放到一个包含 global_step_ 的路径中

# 创建临时checkpoint结构
mkdir -p temp_checkpoint/global_step_0/Critic
cp -r saved_critics/final-critic/* temp_checkpoint/global_step_0/Critic/

# 在新训练中使用
python -m verl.trainer.main_ppo \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=temp_checkpoint/global_step_0 \
    trainer.load_critic_only=true \  # 如果实现了这个功能
    actor_rollout_ref.model.path=~/models/new-actor \
    # 其他配置...
```

---

## ⚠️ 注意事项

### 1. Checkpoint路径格式

- 路径必须包含`global_step_`字符串
- VeRL会根据这个字符串提取step编号

### 2. 模型架构兼容性

- **确保新训练的Critic架构与保存的Critic架构一致**
- 如果模型大小、配置不同，加载会失败

### 3. 分布式训练

- 确保新训练的GPU数量和分布式配置与保存时兼容
- FSDP/Megatron的配置需要匹配

### 4. 优化器状态

- 如果只加载模型参数（`save_contents: ['model']`），优化器会重新初始化
- 如果想继续训练，建议也保存优化器状态

### 5. 检查Checkpoint内容

```bash
# 查看checkpoint目录结构
ls -R checkpoints/gsm8k-ppo/exp1/global_step_1000/

# 应该看到：
# global_step_1000/
#   ├── actor/
#   ├── Critic/  # 或 critic/（取决于Role枚举）
#   └── data.pt
```

---

## 🔍 验证加载是否成功

### 1. 查看训练日志

训练开始时应该看到：

```
Load from checkpoint folder: checkpoints/previous-ppo/global_step_1000
Setting global step to 1000
Resuming from checkpoints/previous-ppo/global_step_1000
```

### 2. 检查Critic参数

可以在训练代码中添加验证：

```python
# 在加载后检查Critic参数
if self.use_critic:
    # 打印Critic的第一个参数（用于验证）
    first_param = next(iter(self.critic_wg.critic_module.parameters()))
    print(f"Critic first param (sample): {first_param.data[0][:5]}")
```

### 3. 检查训练指标

- 如果Critic加载成功，训练应该能正常进行
- 如果加载失败，训练会在初始化阶段报错

---

## 🛠️ 故障排查

### 问题1：找不到checkpoint路径

**错误**：
```
AssertionError: resume ckpt must specify the global_steps
```

**解决**：
- 确保路径包含`global_step_`字符串
- 使用绝对路径或相对于工作目录的正确路径

### 问题2：Critic架构不匹配

**错误**：
```
RuntimeError: Error(s) in loading state_dict
```

**解决**：
- 检查新训练的Critic配置是否与保存时一致
- 确保模型大小、hidden_size等参数相同

### 问题3：分布式配置不匹配

**错误**：
```
RuntimeError: Number of processes does not match
```

**解决**：
- 确保新训练的GPU数量、FSDP配置与保存时兼容
- 如果使用Megatron，确保tensor/pipeline parallel配置一致

---

## 📚 相关配置说明

### Critic Checkpoint配置

```yaml
critic:
  checkpoint:
    # 保存内容
    save_contents: ['model', 'optimizer', 'extra']
    
    # 加载内容（默认与save_contents相同）
    load_contents: ${.save_contents}
    
    # 异步保存（仅Megatron）
    async_save: False
```

### Trainer Checkpoint配置

```yaml
trainer:
  # 保存目录
  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
  
  # 保存频率
  save_freq: 100
  
  # 最多保留的checkpoint数量
  max_critic_ckpt_to_keep: 3
  max_actor_ckpt_to_keep: 3
  
  # Resume模式
  resume_mode: auto  # auto, disable, resume_path
  
  # Resume路径（当resume_mode=resume_path时使用）
  resume_from_path: null
```

---

## 🎓 总结

1. **保存Critic**：训练过程中自动保存，位置在`{checkpoint_dir}/global_step_{step}/Critic/`
2. **加载Critic**：
   - 最简单：使用`resume_mode=resume_path`（但会同时加载Actor）
   - 只加载Critic：需要修改代码添加`load_critic_only`选项
3. **注意事项**：确保模型架构和分布式配置兼容

按照这个指南，你应该能够成功保存和加载Critic模型！🚀
