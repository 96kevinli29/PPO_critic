# HYL

基于 [verl](verl/) 的 Qwen3 PPO 与评估实验仓库。本说明侧重**环境与依赖复现**，便于他人克隆后对齐运行环境。

**Git 无法也不应提交完整 Conda/系统环境**（体积大、与驱动/CUDA 绑定）。若需要「别人下载就能用」的**全量环境**，请使用下面的 **Docker 镜像**；你在本机构建并推到 GHCR / Docker Hub 后，他人只需 `docker pull` + `docker run` 即可。

---

## 推荐：Docker 一键环境（全量打包）

镜像内包含：**Miniconda `verl-ppo-new`（Python 3.10）、CUDA 12.4、PyTorch（cu124）、`environment/requirements-hyl.txt` 全部依赖（含 vLLM、flash-attn）、可编辑安装的 `verl`**，并在 `/workspace/hyl/miniconda3` 建立与 `scripts/hyl_env.sh` 一致的符号链接。

### 前置条件（使用方）

- Linux + [NVIDIA 驱动](https://www.nvidia.com/Download/index.aspx)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)（`docker run --gpus all` 可用）

### 用户与权限：请用普通用户（非 root）

日常操作（训练、`docker login` / `docker run`、挂载到本机目录）请**始终用普通 Linux 用户**（勿长期用 root）。文档里 **`hyl`** 仅为本仓库示例用户名；协作者请使用**自己的用户名**。脚本名 `hyl_env.sh` 与用户名无关。

- **Docker**：把 `hyl` 加入 `docker` 组（`sudo usermod -aG docker hyl`），**重新登录**或 `newgrp docker` 后，**不要**长期用 `sudo docker`。若用 `sudo docker` 往挂载卷里写文件，新建文件属主常为 **root**，`hyl` 会无法读写 `hyl_outputs`、`hyl_logs`、`data` 等目录。
- **若目录已被 root 创建**：在宿主机进入项目根后执行  
  `sudo chown -R hyl:hyl .`  
  或只改产物目录：`sudo chown -R hyl:hyl hyl_outputs hyl_logs data`。

### 你方：构建并上传镜像（发布一次）

在**已装 Docker、驱动、nvidia-container-toolkit**的机器上，于仓库根目录执行：

```bash
cd hyl
DOCKER_BUILDKIT=1 docker build -t hyl-env:latest .
# 首次构建可能较久（flash-attn 需编译）
```

#### 上传到 GitHub（GHCR，推荐）

GitHub 的容器仓库地址是 **`ghcr.io`**（GitHub Container Registry），**不**是往 git 仓库里传镜像文件，而是把镜像推到与你账号关联的「包」里。

1. **在 GitHub 创建访问令牌（只需做一次）**  
   - 打开：**Settings → Developer settings → Personal access tokens**。  
   - 建 **Classic** 或 **Fine-grained** 均可；至少需要能向 GHCR 写入：Classic 勾选 **`write:packages`**（若要读私有包再加 `read:packages`）。  
   - 生成后**复制保存**（页面关闭后无法再看到明文）。

2. **在本机登录 GHCR**（把 `YOUR_GITHUB_USERNAME` 换成你的 GitHub 用户名，`TOKEN` 换成上一步的令牌）：

```bash
echo TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

3. **打标签并推送**（**镜像名须全小写**，把 `owner` 换成你的用户名或组织名的小写形式）：

```bash
docker tag hyl-env:latest ghcr.io/owner/hyl:latest
docker push ghcr.io/owner/hyl:latest
```

4. **在网页上查看与公开（可选）**  
   - 打开 GitHub 右上角头像 → **Your packages**，应能看到 `hyl`。  
   - 若希望任何人能 `docker pull`，进入该包 → **Package settings** → **Change visibility** → **Public**。

他人拉取：

```bash
docker pull ghcr.io/owner/hyl:latest
```

若包为 **Private**，拉取方需用有权限的账号先 `docker login ghcr.io`。

也可推到 Docker Hub：`docker tag hyl-env:latest docker.io/yourname/hyl:latest && docker login docker.io && docker push docker.io/yourname/hyl:latest`。

### 使用方：拉镜像直接进环境

```bash
docker pull ghcr.io/owner/hyl:latest   # owner 为发布方 GitHub 用户名/组织（小写）；或 docker.io/yourname/hyl:latest

# 交互 shell（挂载当前目录下的 data / 输出目录，便于放模型与日志）
mkdir -p data hyl_outputs hyl_logs
docker run --gpus all -it --rm \
  -v "$(pwd)/data:/workspace/hyl/data" \
  -v "$(pwd)/hyl_outputs:/workspace/hyl/hyl_outputs" \
  -v "$(pwd)/hyl_logs:/workspace/hyl/hyl_logs" \
  -w /workspace/hyl \
  ghcr.io/owner/hyl:latest

# 容器内：
# source scripts/hyl_env.sh ppo   # 或 eval
# bash run_ppo_stage1.sh
```

使用仓库里的 `docker-compose.yml`（同样会挂载 `data`、`hyl_outputs`、`hyl_logs`）：

```bash
docker compose build
docker compose run --rm hyl
```

说明：

- 镜像**不包含**大模型权重；将 HF 权重放在宿主机 `data/` 或挂载目录，用 `MODEL_ROOT` 等指向（见 `run_ppo.sh`）。
- 若你更新了仓库代码，使用方需**重新构建镜像**或**挂载覆盖** `/workspace/hyl`：  
  `docker run ... -v "$(pwd):/workspace/hyl" ghcr.io/owner/hyl:latest`（用本地克隆覆盖镜像内代码）。

---

## 系统要求（本机 Conda 安装时）

- Linux，NVIDIA GPU + 对应驱动
- Python **≥ 3.10**
- 建议 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 conda

## 1. 获取代码

```bash
git clone <你的仓库 URL> hyl
cd hyl
```

## 2. Conda 环境与路径约定

`scripts/hyl_env.sh` 默认使用：

- 环境名：`verl-ppo-new`
- Python 路径：`$HYLBASE/miniconda3/envs/verl-ppo-new/bin/python`

即：**期望在仓库根目录下存在 `miniconda3/envs/verl-ppo-new`**（与当前机器布局一致）。若你的 conda 装在其他位置，请编辑 `scripts/hyl_env.sh` 中的 `VERL_ENV` / `PYTHON_CMD` 及相关 `LD_LIBRARY_PATH`，或在本机将 miniconda 安装/链接到仓库根下的 `miniconda3`。

创建环境示例：

```bash
# 在仓库根目录，按你的 conda 路径调整前缀
conda create -n verl-ppo-new python=3.10 -y
conda activate verl-ppo-new
```

## 3. 安装顺序（PyTorch → 依赖 → verl）

`environment/requirements-hyl.txt` 已包含 **verl 全部 `requirements.txt`、vLLM、flash-attn**，与 `verl/setup.py` 里可选依赖 `[vllm]`（`vllm>=0.8.5,<=0.12.0`）和 `[gpu]`（含 `flash-attn`）对齐。因 **PyTorch 必须与 CUDA 版本匹配**，不能由单一 txt 代劳，请按下述顺序操作。

### 3.0 本机安装检查清单（按序核对）

**节点要求、安装顺序、镜像与包说明**已全部写在 **`environment/requirements-hyl.txt` 文件开头的注释**里；发给别人时**只发该一个文件**即可（`pip` 会忽略 `#` 注释行，只安装文件末尾的包）。下表为摘要；**版本号以你机器为准**（`pip show torch vllm`）。

| 序号 | 检查项 | 本仓库约定 / 示例 | 状态 |
|:----:|--------|---------------------|:----:|
| 1 | 系统与用户 | Linux；**普通用户**跑训练（勿长期用 root；示例名 `hyl`）；`nvidia-smi` 正常；磁盘空间充足 | ☐ |
| 2 | Conda | 已安装 Miniconda/Anaconda；若不用「项目根下 `miniconda3`」，须改 `hyl_env.sh` 里 `PYTHON_CMD` | ☐ |
| 3 | Conda 环境 | 环境名 **`verl-ppo-new`**，**Python 3.10**（`conda create -n verl-ppo-new python=3.10`） | ☐ |
| 4 | PyTorch + CUDA | **仅 CUDA 版**（示例：**torch 2.6.x + cu124**，与 [PyTorch 官网](https://pytorch.org/get-started/locally/) 一致）；`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` | ☐ |
| 5 | vLLM 与其余 pip 依赖 | 在**仓库根目录**：`pip install -r environment/requirements-hyl.txt`（内含 **vLLM**、**flash-attn** 等；可按该文件注释换国内 PyPI / PyTorch 镜像） | ☐ |
| 6 | 安装 verl 包 | `pip install -e verl` | ☐ |
| 7 | 密钥与 HF | 按需 `cp .env.example .env`；国内拉模型可设 `HF_ENDPOINT=https://hf-mirror.com`（见 `.env.example`） | ☐ |
| 8 | 入口与自检 | `source scripts/hyl_env.sh eval`；再运行下文「快速检查」中的 `python -c ...` | ☐ |

**若你使用其他环境名（如 `verl`）或 Python 3.12**：可以，但必须**同步修改** `scripts/hyl_env.sh` 中的 `VERL_ENV` / `PYTHON_CMD`，并自行承担与上游 verl / 本仓库测试环境不一致的风险。默认以 **`verl-ppo-new` + Python 3.10** 为准。

### 3.1 先安装 PyTorch（CUDA）

按显卡与系统从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装 **CUDA 版** `torch`（不要先 `pip install` 本仓库再装 torch，否则易出现 CPU 版 torch 或版本冲突）。

### 3.2 再安装本仓库聚合依赖（含 vLLM + FlashAttention）

在已激活的 conda 环境中，**在仓库根目录**执行：

```bash
pip install -r environment/requirements-hyl.txt
```

若 `flash-attn` 编译失败，在已装好 CUDA 版 PyTorch 的前提下可再试：

```bash
pip install flash-attn --no-build-isolation
```

若仍失败，可暂时不装 `flash-attn`：`scripts/hyl_env.sh` 会回退到 `VERL_ATTN_IMPLEMENTATION=sdpa`（见脚本内说明）。

### 3.3 以可编辑方式安装 verl

```bash
pip install -e verl
```

可选等价写法（与上面 `requirements-hyl.txt` 重叠，一般不必再用）：

```bash
pip install -e "verl[vllm,gpu]"
```

若遇 `numpy` 与上游约束不一致，`hyl_env.sh` 在 PPO 模式下会提示；可按需执行 `pip install 'numpy<2.0.0'`。

## 4. 环境变量与 `.env`

- 复制 `cp .env.example .env`，填写 `WANDB_API_KEY`、`HF_TOKEN` 等（**勿将 `.env` 提交到 git**）。
- 训练/评估前由入口脚本 `source scripts/hyl_env.sh ppo` 或 `source scripts/hyl_env.sh eval` 设置 `PYTHONPATH`、Ray、W&B 目录、NCCL 等；详见 `scripts/hyl_env.sh` 头部注释。

可覆盖的常用变量包括：`HYLBASE`、`OUTPUTS`、`HYL_LOG_DIR`、`CUDA_VISIBLE_DEVICES`、`HYL_MULTI_NODE`（多机 IB 时）等。

## 5. 模型与数据

大体积权重与数据默认**不在**仓库内（见 `.gitignore`）。请将 Hugging Face 格式的模型放到本地路径，并通过 `MODEL_ROOT`、`ACTOR_MODEL_PATH` 等环境变量指向（见 `run_ppo.sh` 等脚本注释）。

## 6. 快速检查

```bash
source scripts/hyl_env.sh eval
python -c "import verl, torch, vllm; import flash_attn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'vllm', vllm.__version__)"
```

若未安装 `flash-attn`，把 `import flash_attn` 去掉即可。

## 7. 目录说明（简要）

| 路径 | 说明 |
|------|------|
| `verl/` | 上游 verl 子模块式拷贝，可编辑安装 `pip install -e verl` |
| `scripts/hyl_env.sh` | 统一环境：conda、PYTHONPATH、Ray、W&B、OUTPUTS |
| `environment/requirements-hyl.txt` | **对外只发此文件即可**：注释内为节点/verl 配置与安装顺序；末尾为 pip 依赖（`verl/requirements.txt` + vLLM + flash-attn） |
| `Dockerfile` / `docker-compose.yml` | **全量环境镜像**（推荐给他人「拉镜像即用」） |
| `run_ppo.sh` / `run_ppo_stage*.sh` | PPO 训练入口 |
| `run_eval_*.sh` | 评估脚本 |

更细的参数说明以各 `run_*.sh` 内注释为准。

## 8. 环境与用户（对外发布摘要）

复制下面内容到仓库 **About**、组内文档或 Issue，即可说明「本仓库假定什么环境、什么用户」：

| 项目 | 约定 |
|------|------|
| **Linux 登录用户** | **`hyl`**（普通用户，**不要**用 root 跑训练/写挂载目录；Docker 见上文「用户与权限」） |
| **Conda 环境名** | **`verl-ppo-new`**（Python 3.10）；路径约定见 `scripts/hyl_env.sh`（`$HYLBASE/miniconda3/envs/verl-ppo-new`） |
| **环境说明（对外）** | 只发 **`environment/requirements-hyl.txt`**（文件内注释写清节点与安装顺序；**须先装 CUDA 版 PyTorch**） |
| **环境入口脚本** | `source scripts/hyl_env.sh ppo` 或 `source scripts/hyl_env.sh eval` |
| **全量容器环境（可选）** | 根目录 `Dockerfile` 构建；可推至 GHCR，见上文「Docker 一键环境」 |
| **密钥** | 复制 `.env.example` 为 `.env`，**勿提交 git** |

若你指的是 **「退出当前 shell / conda」**：`conda deactivate`（退出 conda）；`exit`（退出当前终端会话）；切换用户用 `su - <用户名>` 或重新 SSH。
