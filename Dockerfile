# Dockerfile - 基于CNN的常见动物种类识别系统
# 使用Python 3.9官方镜像作为基础镜像

FROM python:3.9-slim

LABEL maintainer="zhifengjin050@gmail.com"
LABEL description="基于CNN的常见动物种类识别系统 - ResNet18迁移学习"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖（使用清华镜像加速）
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制源代码
COPY src/ ./src/

# 创建模型和数据目录
RUN mkdir -p models data logs

# 暴露端口
EXPOSE 5000

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# 启动Web服务
CMD ["python", "src/app.py"]