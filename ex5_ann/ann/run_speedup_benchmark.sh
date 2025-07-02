#!/bin/bash

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 检查CUDA是否可用
echo "检查CUDA环境..."
nvcc --version

# 编译
echo "开始编译加速比测试程序..."
nvcc -o speedup_benchmark speedup_benchmark.cu -I. -lcublas -lcudart -O3 -std=c++11 -Xcompiler -fopenmp

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "编译失败，请检查错误信息。"
    exit 1
fi

echo "编译成功！"

# 设置参数
DATA_PATH="anndata/"
QUERY_COUNT=2000    # 固定查询数量
REPEAT_COUNT=5      # 重复次数
WARMUP_COUNT=100    # 暖机查询数
NPROBE=10          # IVF的nprobe参数

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

# 检查IVF文件（可选）
IVF_PATH="file/"
if [ -f "${IVF_PATH}ivf_flat_centroids_256.fbin" ] && [ -f "${IVF_PATH}ivf_flat_invlists_256.bin" ]; then
    echo "发现IVF数据文件，将包含IVF测试"
    echo "使用文件: ${IVF_PATH}ivf_flat_centroids_256.fbin, ${IVF_PATH}ivf_flat_invlists_256.bin"
    IVF_AVAILABLE=true
else
    echo "未发现IVF数据文件，将跳过IVF测试"
    echo "需要的文件:"
    echo "  ${IVF_PATH}ivf_flat_centroids_256.fbin"
    echo "  ${IVF_PATH}ivf_flat_invlists_256.bin"
    IVF_AVAILABLE=false
fi

# 运行加速比测试
echo "开始运行加速比测试..."
echo "参数: 查询数=${QUERY_COUNT}, 重复次数=${REPEAT_COUNT}, 暖机次数=${WARMUP_COUNT}, nprobe=${NPROBE}"
./speedup_benchmark $DATA_PATH $QUERY_COUNT $REPEAT_COUNT $WARMUP_COUNT $NPROBE

# 检查是否成功运行
if [ $? -ne 0 ]; then
    echo "测试运行失败，请检查错误信息。"
    exit 1
fi

echo "测试完成！结果已保存到 speedup_results.csv"

