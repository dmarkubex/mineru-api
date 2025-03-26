# 使用 MinerU 基础镜像
FROM mineru:latest

# 设置工作目录
WORKDIR /app

# 添加LibreOffice安装
RUN apt-get update && \
    apt-get install -y libreoffice --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 复制 API 服务代码和启动脚本
COPY api_server.py /app/
COPY requirements.txt /app/
COPY start_api.sh /app/

# 确保启动脚本可执行
RUN chmod +x /app/start_api.sh

# 设置Paddle相关环境变量，启用GPU支持
ENV LD_LIBRARY_PATH=/opt/mineru_venv/lib/python3.10/site-packages/paddle/libs:$LD_LIBRARY_PATH
ENV FLAGS_call_stack_level=2
ENV FLAGS_use_gpu=1
ENV PADDLEPADDLE_INSTALL_GPU=1

# 在MinerU虚拟环境中使用清华源安装依赖
RUN bash -c "source /opt/mineru_venv/bin/activate && \
    pip install --no-cache-dir -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    echo '已安装的依赖:' && \
    pip list"

# 创建数据目录
RUN mkdir -p /app/data/uploads /app/data/results

# 暴露端口
EXPOSE 8000

# 使用启动脚本
CMD ["/app/start_api.sh"] 