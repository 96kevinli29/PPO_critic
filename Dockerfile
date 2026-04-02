# HYL 全量训练环境：PyTorch + vLLM + flash-attn + verl 依赖，与 scripts/hyl_env.sh 路径约定一致。
# 构建较慢（flash-attn 需编译），建议：DOCKER_BUILDKIT=1 docker build -t hyl-env:latest .
#
# 节点前提（典型单机训练）：NVIDIA 驱动足够新（一般 525+ 可跑 CUDA 12.x 容器）；磁盘建议 ≥30GB 空闲（构建峰值）。
# 宿主机驱动显示的「CUDA Version」可与容器内 12.4 不完全相同，由驱动前向兼容；若 run 时报 CUDA 错再考虑换 cu122 基础镜像。
#
# 基础镜像：CUDA 12.4 + devel（含 nvcc，供 flash-attn 编译）
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 装到固定路径；运行时链到项目根下的 miniconda3，与 hyl_env.sh 一致
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN wget -q ${MINICONDA_URL} -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda3 && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/miniconda3/bin:${PATH}

# 新版 Miniconda 在非交互环境需先接受默认频道 ToS，否则 conda create 失败
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n verl-ppo-new python=3.10 -y && conda clean -afy

# 仅用该环境的 pip/python
ENV PATH=/opt/miniconda3/envs/verl-ppo-new/bin:${PATH}

# PyTorch（CUDA 12.4 wheel）；须先于 requirements-hyl 中的 vllm/flash-attn
# 官方源在国内/共享出口易超时；可换镜像重试构建，例如：
#   docker build --build-arg PYTORCH_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cu124/ -t hyl-env:latest .
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
      --default-timeout=1200 \
      --retries 30 \
      torch torchvision torchaudio \
      --index-url "${PYTORCH_INDEX_URL}"

WORKDIR /workspace/hyl

# 依赖清单（利用缓存层：改代码不触发重装依赖时可优化为分步 COPY，此处一次性 COPY 简单可靠）
COPY environment/requirements-hyl.txt environment/requirements-hyl.txt
COPY verl/requirements.txt verl/requirements.txt

# flash-attn 编译并行数；共享构建节点内存紧可调低：docker build --build-arg MAX_JOBS=2
ARG MAX_JOBS=8
ENV MAX_JOBS=${MAX_JOBS}

RUN pip install -r environment/requirements-hyl.txt

COPY . .

# 与 hyl_env.sh 一致：$HYLBASE/miniconda3/envs/verl-ppo-new/bin/python
RUN ln -sfn /opt/miniconda3 /workspace/hyl/miniconda3

RUN pip install -e verl

# 默认进入项目根；由用户 source scripts/hyl_env.sh 后再跑 run_*.sh
SHELL ["/bin/bash", "-lc"]
CMD ["/bin/bash"]
