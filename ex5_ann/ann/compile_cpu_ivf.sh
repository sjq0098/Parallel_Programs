#!/bin/bash

# 编译CPU版本的IVF搜索程序

echo "编译CPU IVF搜索程序..."

# 设置编译参数
CXX_FLAGS="-std=c++14 -O3 -fopenmp -DHAVE_CXX0X -fpic -ftree-vectorize"

# 编译命令
g++ ${CXX_FLAGS} main_ivf.cc -o main_ivf_cpu

if [ $? -eq 0 ]; then
    echo "编译成功！生成可执行文件: main_ivf_cpu"
    echo ""
    echo "运行方法:"
    echo "  ./main_ivf_cpu"
    echo ""
    echo "注意: 确保有以下数据文件:"
    echo "  - anndata/DEEP100K.base.100k.fbin"
    echo "  - anndata/DEEP100K.query.fbin"
    echo "  - anndata/DEEP100K.gt.query.100k.top100.bin"
    echo "  - file/ivf_flat_centroids_256.fbin"
    echo "  - file/ivf_flat_invlists_256.bin"
    echo ""
    echo "对比使用:"
    echo "  CPU版本: ./main_ivf_cpu"
    echo "  GPU版本: ./main_ivf_gpu"
else
    echo "编译失败！请检查源代码和依赖。"
    exit 1
fi 