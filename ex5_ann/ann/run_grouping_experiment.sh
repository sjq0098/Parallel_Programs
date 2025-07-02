#!/bin/bash

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 检查CUDA是否可用
echo "检查CUDA环境..."
nvcc --version

# 编译
echo "开始编译分组策略实验程序..."
nvcc -o grouping_strategy_benchmark grouping_strategy_benchmark.cu -I. -lcublas -lcudart -O3 -std=c++11 -Xcompiler -fopenmp

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "编译失败，请检查错误信息。"
    exit 1
fi

echo "编译成功！"

# 设置参数
DATA_PATH="anndata/"
QUERY_COUNT=2000    # 查询数量
REPEAT_COUNT=5      # 重复次数
WARMUP_COUNT=100    # 暖机查询数

# 检查数据文件是否存在
echo "检查数据文件..."
if [ ! -f "${DATA_PATH}DEEP100K.query.fbin" ]; then
    echo "错误: 查询文件 ${DATA_PATH}DEEP100K.query.fbin 不存在"
    exit 1
fi

if [ ! -f "${DATA_PATH}DEEP100K.base.100k.fbin" ]; then
    echo "错误: 基础数据文件 ${DATA_PATH}DEEP100K.base.100k.fbin 不存在"
    exit 1
fi

if [ ! -f "${DATA_PATH}DEEP100K.gt.query.100k.top100.bin" ]; then
    echo "错误: 真值文件 ${DATA_PATH}DEEP100K.gt.query.100k.top100.bin 不存在"
    exit 1
fi

# 检查IVF文件
IVF_PATH="file/"
if [ -f "${IVF_PATH}ivf_flat_centroids_256.fbin" ] && [ -f "${IVF_PATH}ivf_flat_invlists_256.bin" ]; then
    echo "发现IVF数据文件，将进行分组策略测试"
    echo "使用文件: ${IVF_PATH}ivf_flat_centroids_256.fbin, ${IVF_PATH}ivf_flat_invlists_256.bin"
else
    echo "错误: 未发现IVF数据文件"
    echo "需要的文件:"
    echo "  ${IVF_PATH}ivf_flat_centroids_256.fbin"
    echo "  ${IVF_PATH}ivf_flat_invlists_256.bin"
    exit 1
fi

# 运行分组策略实验
echo "开始运行分组策略实验..."
echo "参数: 查询数=${QUERY_COUNT}, 重复次数=${REPEAT_COUNT}, 暖机次数=${WARMUP_COUNT}"
./grouping_strategy_benchmark $DATA_PATH $QUERY_COUNT $REPEAT_COUNT $WARMUP_COUNT

# 检查是否成功运行
if [ $? -ne 0 ]; then
    echo "实验运行失败，请检查错误信息。"
    exit 1
fi

echo "实验完成！结果已保存到 grouping_strategy_results.csv"

