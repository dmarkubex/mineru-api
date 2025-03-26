#!/bin/bash

# 显示基本环境信息
echo "启动MinerU API服务..."
echo "当前路径: $(pwd)"

# 检查NVIDIA GPU状态
echo "检查NVIDIA GPU状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "GPU可用，将使用GPU模式"
else
    echo "警告：未检测到NVIDIA GPU，服务可能无法正常运行"
fi

# 激活MinerU虚拟环境
echo "激活MinerU虚拟环境..."
source /opt/mineru_venv/bin/activate || {
    echo "错误：无法激活MinerU虚拟环境"
    exit 1
}

# 确认Python环境
echo "使用Python: $(which python)"
echo "Python版本: $(python --version)"

# 设置环境变量，启用GPU支持
export FLAGS_use_gpu=1
export PADDLEPADDLE_INSTALL_GPU=1
export LD_LIBRARY_PATH="/opt/mineru_venv/lib/python3.10/site-packages/paddle/libs:$LD_LIBRARY_PATH"

# 检查依赖是否完整，如果不完整则使用清华源安装
if ! python -c "import uvicorn" &>/dev/null; then
    echo "检测到uvicorn未安装，使用清华源安装..."
    pip install uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# 启动API服务
echo "启动API服务..."
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 