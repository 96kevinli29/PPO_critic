# 推送到 PPO_critic 仓库说明

## 重要：当前项目就是 PPO_critic

你的目录 `hyl` 的 git 远程已指向 **https://github.com/96kevinli29/PPO_critic**，因此**不需要**再 `cd` 到别的路径。  
在**本项目根目录**（即 `hyl`）下执行 git 命令即可。

## 为什么不能直接复制之前的命令

之前示例里写的：

```bash
cd /path/to/PPO_critic   # ← 这是占位符，不是真实路径！
```

- **`/path/to/PPO_critic`** 只是“请换成你机器上 PPO_critic 的路径”的占位符。
- 你的情况里，**PPO_critic 的路径就是当前项目**：`/data_storage/lixiao/research_proj_xiao/hyl`。
- 所以应使用：`cd /data_storage/lixiao/research_proj_xiao/hyl`，或直接在当前目录执行，无需再 cd。

## 若 Git 报 “dubious ownership”

在本机执行一次（把当前目录加入安全名单）：

```bash
git config --global --add safe.directory /data_storage/lixiao/research_proj_xiao/hyl
```

## 正确的推送步骤（在项目根目录执行）

在 **`/data_storage/lixiao/research_proj_xiao/hyl`** 下执行：

```bash
# 1. 进入项目根目录（若你已在 hyl 下可省略）
cd /data_storage/lixiao/research_proj_xiao/hyl

# 2. 添加要同步的文件（按需调整列表）
git add run_eval_critic.sh run_qwen3_ppo.sh \
  scripts/eval_critic_prediction.py scripts/convert_critic_to_hf.py \
  README.md CHANGELOG.md docs/

# 3. 提交
git commit -m "sync: run_eval_critic, run_qwen3_ppo, eval/convert scripts, 2026-03 update doc"

# 4. 推送到 GitHub（分支名以你实际为准，一般是 main 或 master）
git push origin main
```

若远程默认分支是 `master`，最后一行改为：

```bash
git push origin master
```

## 若 push 失败可能原因

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `Permission denied (publickey)` | 未配置 SSH 或 GitHub 未加本机公钥 | 配置 SSH key 并添加到 GitHub |
| `Updates were rejected` | 远程有新提交，本地未拉取 | 先 `git pull --rebase origin main` 再 push |
| `dubious ownership` | 当前目录未被 Git 视为安全 | 执行上面 `git config --global --add safe.directory ...` |
