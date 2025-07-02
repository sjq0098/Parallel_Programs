#!/bin/bash

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 检查CUDA是否可用
echo "检查CUDA环境..."
nvcc --version

# 编译
echo "开始编译高级分组策略实验程序..."
nvcc -o advanced_grouping_strategies advanced_grouping_strategies.cu -I. -lcublas -lcudart -O3 -std=c++11 -Xcompiler -fopenmp

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "编译失败，请检查错误信息。"
    exit 1
fi

echo "编译成功！"

# 设置参数
DATA_PATH="anndata/"
QUERY_COUNT=2000    # 查询数量
REPEAT_COUNT=3      # 重复次数
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
    echo "发现IVF数据文件，将进行高级分组策略测试"
    echo "使用文件: ${IVF_PATH}ivf_flat_centroids_256.fbin, ${IVF_PATH}ivf_flat_invlists_256.bin"
else
    echo "错误: 未发现IVF数据文件"
    echo "需要的文件:"
    echo "  ${IVF_PATH}ivf_flat_centroids_256.fbin"
    echo "  ${IVF_PATH}ivf_flat_invlists_256.bin"
    exit 1
fi

# 运行高级分组策略实验
echo "开始运行高级分组策略实验..."
echo "参数: 查询数=${QUERY_COUNT}, 重复次数=${REPEAT_COUNT}, 暖机次数=${WARMUP_COUNT}"
./advanced_grouping_strategies $DATA_PATH $QUERY_COUNT $REPEAT_COUNT $WARMUP_COUNT

# 检查是否成功运行
if [ $? -ne 0 ]; then
    echo "实验运行失败，请检查错误信息。"
    exit 1
fi

echo "实验完成！结果已保存到 advanced_grouping_results.csv"


echo ""
echo "=== 高级分组策略实验完成 ==="
echo "主要结果文件:"
echo "  - advanced_grouping_results.csv (详细数据)"
echo "  - advanced_grouping_plots/ (可视化图表目录)"
echo ""
echo "测试的高级分组策略:"
echo "  1. 基准无分组 (对比基准)"
echo "  2. 自适应批大小 (根据查询相似度动态调整批大小)"
echo "  3. 负载均衡 (基于计算复杂度均匀分配查询)"
echo "  4. 局部性感知 (利用查询序列的局部相关性)"
echo "  5. 分层分组 (先按簇分组，再按相似度细分)"
echo "  6. 时间感知自适应 (基于查询方差估计复杂度)"
echo ""
echo "与基础分组策略的主要改进:"
echo "  - 更智能的批大小调整"
echo "  - 考虑计算负载均衡"
echo "  - 利用数据局部性"
echo "  - 分层优化策略" 