# 使用 CUDA 11.3.1 基础镜像
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# 设置非交互模式，避免交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 更新系统并添加支持 Python 3.9 的 PPA 源
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# 安装 Python 3.9、开发头文件、构建工具和基础依赖
RUN apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    build-essential \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 手动安装最新版本的 pip，确保正确安装到 Python 3.9 环境
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# 使用清华源配置 pip
RUN python3.9 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    python3.9 -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 创建 python 的符号链接指向 python3.9
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# 确认 pip 已正确安装
RUN python3.9 -m pip --version

# 设置工作目录
WORKDIR /app

# 复制只包含依赖的文件，利用缓存
COPY requirements.txt /app/

# 升级 pip 并安装项目依赖（不包含 torch, torchvision, torchaudio）
RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install --progress-bar=on -r requirements.txt --no-cache-dir && \
    python3.9 -m pip install cupy-cuda113 imageio[ffmpeg]

# 升级 pip 并安装 PyTorch 及其依赖（针对 CUDA 11.3），显示下载进度
RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install --progress-bar=on torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://mirrors.aliyun.com/pytorch-wheels/cu113/

# 复制剩余的项目文件
COPY . /app

# 打印 Python 和 pip 版本以验证安装
RUN python3.9 --version
RUN python3.9 -m pip --version

# 创建必要的文件夹
RUN mkdir -p /app/ref_video \
    && mkdir -p /app/out_video \
    && mkdir -p /app/Audio2Head/Audio2Head/input_wav \
    && mkdir -p /app/Audio2Head/Audio2Head/input_img

# 确保脚本有执行权限
RUN chmod +x /app/Audio2Head/Audio2Head/run_inference.sh

# 定义启动命令
CMD ["bash", "-c", "/app/Audio2Head/Audio2Head/run_inference.sh && python /app/batch_psnr.py"]
