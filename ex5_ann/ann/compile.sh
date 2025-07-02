#!/bin/bash

# 简单的编译脚本，编译GPU IVF搜索程序

echo "编译GPU IVF搜索程序..."

# 设置编译参数
NVCC_FLAGS="-std=c++11 -O3 -Xcompiler -fopenmp"
CUDA_LIBS="-lcudart -lcublas"

# 编译命令
nvcc ${NVCC_FLAGS} main_ivf_gpu.cu -o main_ivf_gpu ${CUDA_LIBS}

if [ $? -eq 0 ]; then
    echo "编译成功！生成可执行文件: main_ivf_gpu"
    echo ""
    echo "运行方法:"
    echo "  ./main_ivf_gpu"
    echo ""
    echo "注意: 确保有以下数据文件:"
    echo "  - anndata/DEEP100K.base.100k.fbin"
    echo "  - anndata/DEEP100K.query.fbin"
    echo "  - anndata/DEEP100K.gt.query.100k.top100.bin"
    echo "  - file/ivf_flat_centroids_256.fbin"
    echo "  - file/ivf_flat_invlists_256.bin"
else
    echo "编译失败！请检查CUDA环境和源代码。"
    exit 1
fi 