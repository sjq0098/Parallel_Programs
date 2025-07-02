#!/bin/bash

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 检查CUDA是否可用
echo "检查CUDA环境..."
nvcc --version

# 编译
echo "开始编译..."
nvcc -o benchmark benchmark.cu -I. -lcublas -lcudart -O3 -std=c++11 -Xcompiler -fopenmp

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "编译失败，请检查错误信息。"
    exit 1
fi

echo "编译成功！"

# 设置参数
DATA_PATH="anndata/"
MAX_QUERY_COUNT=10000  # 增加最大查询数量
REPEAT_COUNT=5
WARMUP_COUNT=100

# 运行基准测试
echo "开始运行实验..."
./benchmark $DATA_PATH $MAX_QUERY_COUNT $REPEAT_COUNT $WARMUP_COUNT

# 检查是否成功运行
if [ $? -ne 0 ]; then
    echo "实验运行失败，请检查错误信息。"
    exit 1
fi

echo "实验完成！结果已保存到 benchmark_results.csv"

