#!/bin/bash

# 设置错误时退出
set -e

echo "====== MinerU API 服务启动脚本 ======"

# 检查 Docker 是否已安装并运行
if ! command -v docker &> /dev/null; then
    echo "错误：Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查本地是否有mineru镜像
if ! docker images | grep -q "mineru"; then
    echo "错误：本地没有找到mineru镜像，请确保mineru:latest镜像已存在"
    exit 1
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p data/uploads data/results
echo "✓ 数据目录已创建"

# 停止并移除现有容器
echo "停止并移除现有容器..."
docker-compose down || docker compose down || true
docker rm -f mineru-api || true

# 强制重新构建API服务
echo "强制重新构建API服务..."
docker-compose build --no-cache || docker compose build --no-cache

# 启动API服务
echo "启动API服务..."
docker-compose up -d || docker compose up -d

echo ""
echo "====== MinerU API 服务启动完成 ======"
echo "访问地址: http://localhost:8000"
echo "服务健康检查: http://localhost:8000/health"
echo "查看日志请使用: docker logs -f mineru-api"
echo ""

# 不再自动显示日志
# docker-compose logs -f || docker compose logs -f 