# 查找Checkpoint位置指南

## 📍 Checkpoint默认保存位置

根据VeRL的配置，checkpoint默认保存在：

```
checkpoints/${trainer.project_name}/${trainer.experiment_name}/global_step_{step}/
```

### 具体路径结构

```
checkpoints/
  └── {project_name}/           # 例如: gsm8k-ppo
      └── {experiment_name}/    # 例如: test-experiment
          └── global_step_100/  # 每个训练步的checkpoint
              ├── actor/        # Actor模型
              ├── Critic/       # Critic模型（注意大小写）
              └── data.pt        # Dataloader状态
```

## 🔍 如何查找你的Checkpoint

### 方法1：从训练脚本的工作目录查找

Checkpoint保存在**运行训练脚本时的工作目录**下。

#### 步骤1：找到你运行训练脚本的目录

```bash
# 如果你在 verl 目录下运行的训练
cd /data_storage/lixiao/research_proj_xiao/hyl/rl/verl

# 查找checkpoints目录
find . -type d -name "checkpoints" 2>/dev/null
```

#### 步骤2：查看checkpoints目录

```bash
# 如果找到了checkpoints目录
ls -la checkpoints/

# 查看具体的项目目录
ls -la checkpoints/*/

# 查看实验目录
ls -la checkpoints/*/*/

# 查看最新的checkpoint
ls -lt checkpoints/*/*/global_step_* | head -5
```

### 方法2：从配置文件查找

查看你的训练配置中的 `trainer.default_local_dir`：

```bash
# 查看PPO训练配置
cat verl/trainer/config/ppo_trainer.yaml | grep default_local_dir

# 或者查看你使用的具体配置文件
grep -r "default_local_dir" verl/trainer/config/
```

默认值通常是：
```yaml
default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
```

### 方法3：从训练日志查找

训练时会打印checkpoint保存路径：

```bash
# 查看训练日志
grep -i "checkpoint\|global_step" ppo.log | head -20

# 或者查看最近的日志
tail -100 ppo.log | grep -i checkpoint
```

日志中会显示类似：
```
local_global_step_folder: checkpoints/gsm8k-ppo/experiment-name/global_step_100
```

### 方法4：全局搜索

```bash
# 从当前目录向上查找
cd /data_storage/lixiao/research_proj_xiao/hyl/rl/verl
find .. -type d -name "checkpoints" 2>/dev/null

# 或者查找global_step目录
find .. -type d -name "global_step_*" 2>/dev/null | head -10

# 查找Critic目录
find .. -type d -name "Critic" -o -name "critic" 2>/dev/null
```

## 📂 常见位置

根据你的目录结构，checkpoint可能在以下位置：

### 1. 在verl目录下（运行脚本的目录）

```bash
cd /data_storage/lixiao/research_proj_xiao/hyl/rl/verl
ls -la checkpoints/
```

### 2. 在hyl/rl目录下

```bash
cd /data_storage/lixiao/research_proj_xiao/hyl/rl
ls -la checkpoints/
```

### 3. 在outputs目录下（如果配置了）

```bash
cd /data_storage/lixiao/research_proj_xiao/hyl/rl/verl
ls -la outputs/
```

## 🎯 快速查找命令

### 一键查找所有checkpoint

```bash
# 从verl目录开始查找
cd /data_storage/lixiao/research_proj_xiao/hyl/rl/verl

# 查找所有checkpoint目录
find . -type d -path "*/checkpoints/*/global_step_*" 2>/dev/null

# 查找最新的checkpoint
find . -type d -path "*/checkpoints/*/global_step_*" 2>/dev/null | sort -V | tail -1

# 查看checkpoint内容
find . -type d -path "*/checkpoints/*/global_step_*/Critic" 2>/dev/null
```

### 从日志中查找PPO checkpoint路径

```bash
# 查找PPO训练日志中的checkpoint保存路径
grep "local_global_step_folder" ppo.log

# 或者更详细的查找
grep -E "local_global_step_folder|Saving checkpoint to" ppo.log | tail -20

# 如果日志中有保存信息，会显示类似：
# local_global_step_folder: checkpoints/gsm8k-ppo/exp1/global_step_100
# 或
# Saving checkpoint to: checkpoints/gsm8k-ppo/exp1/global_step_100
```

### 查找Critic模型

```bash
# 查找所有Critic目录
find . -type d \( -name "Critic" -o -name "critic" \) 2>/dev/null

# 查看Critic目录内容
find . -type d -name "Critic" -exec ls -la {} \; 2>/dev/null
```

## 📝 检查Checkpoint是否完整

找到checkpoint后，检查内容：

```bash
# 假设checkpoint在：checkpoints/gsm8k-ppo/exp1/global_step_1000/
CHECKPOINT_DIR="checkpoints/gsm8k-ppo/exp1/global_step_1000"

# 查看目录结构
ls -la $CHECKPOINT_DIR/

# 检查Actor
ls -la $CHECKPOINT_DIR/actor/

# 检查Critic（注意大小写）
ls -la $CHECKPOINT_DIR/Critic/  # 或 critic/

# 检查是否有data.pt
ls -la $CHECKPOINT_DIR/data.pt
```

## ⚠️ 区分SFT和PPO的Checkpoint

从你的日志中看到：
```
/data_storage/lixiao/research_proj_xiao/jcl/verl_outputs/qwen25_15b_sft_gsm8k_peft/hf_global_step_29
```

这是**SFT的checkpoint**，不是PPO的。

### SFT Checkpoint特征：
- 路径通常包含 `sft`、`hf_global_step_` 等
- 保存在 `verl_outputs/` 或类似目录
- 只有模型checkpoint，没有 `actor/` 和 `Critic/` 子目录

### PPO Checkpoint特征：
- 路径通常是 `checkpoints/{project}/{experiment}/global_step_{step}/`
- 包含 `actor/` 和 `Critic/` 子目录
- 日志中会打印 `local_global_step_folder: ...`

### 查找PPO Checkpoint的正确方法：

```bash
# 1. 从日志中查找（最准确）
grep "local_global_step_folder" ppo.log

# 2. 查找checkpoints目录（PPO专用）
find . -type d -name "checkpoints" 2>/dev/null

# 3. 查找包含actor和Critic的目录（PPO特征）
find . -type d -path "*/checkpoints/*/global_step_*/actor" 2>/dev/null
find . -type d -path "*/checkpoints/*/global_step_*/Critic" 2>/dev/null
```

## 🔧 如果找不到Checkpoint

### 可能的原因：

1. **Checkpoint还没保存**：训练刚开始，还没到保存频率
2. **保存路径配置不同**：检查训练脚本中的配置
3. **保存在其他位置**：检查 `trainer.default_local_dir` 配置
4. **使用了HDFS**：如果配置了 `default_hdfs_dir`，可能保存在HDFS上

### 解决方法：

```bash
# 1. 检查训练配置
grep -r "default_local_dir\|default_hdfs_dir" verl/trainer/config/

# 2. 检查训练脚本
grep -i "checkpoint\|save" run_ppo*.sh

# 3. 查看训练日志中的保存信息
grep -i "save\|checkpoint" ppo.log
```

## 💡 实用脚本

创建一个查找脚本：

```bash
#!/bin/bash
# find_checkpoints.sh

echo "=== 查找Checkpoint位置 ==="
echo ""

# 从当前目录查找
echo "1. 当前目录下的checkpoints:"
find . -type d -name "checkpoints" -maxdepth 3 2>/dev/null

echo ""
echo "2. 所有global_step目录:"
find . -type d -name "global_step_*" -maxdepth 5 2>/dev/null | head -10

echo ""
echo "3. Critic模型位置:"
find . -type d \( -name "Critic" -o -name "critic" \) 2>/dev/null

echo ""
echo "4. 最新的checkpoint:"
LATEST=$(find . -type d -path "*/checkpoints/*/global_step_*" 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST" ]; then
    echo "   $LATEST"
    echo ""
    echo "   内容:"
    ls -la "$LATEST" 2>/dev/null
else
    echo "   未找到checkpoint"
fi
```

保存为 `find_checkpoints.sh`，然后运行：
```bash
chmod +x find_checkpoints.sh
./find_checkpoints.sh
```

## 🎯 总结

1. **默认位置**：`checkpoints/${project_name}/${experiment_name}/global_step_{step}/`
2. **从运行脚本的目录开始查找**
3. **使用 `find` 命令快速定位**
4. **检查训练日志中的保存路径信息**

按照这些方法，应该能找到你的checkpoint！🔍
