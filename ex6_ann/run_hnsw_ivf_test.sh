#!/bin/bash

echo "=== 编译 MPI HNSW+IVF 混合算法测试程序 ==="

# 设置编译参数
MPICXX_FLAGS="-O3 -march=native -std=c++17 -fopenmp"
INCLUDE_DIRS="-I. -I./hnswlib"
LIBS="-lm"

# 编译主程序
echo "编译 main_mpi_hnsw_ivf..."
mpicxx $MPICXX_FLAGS $INCLUDE_DIRS main_mpi_hnsw_ivf.cc -o main_mpi_hnsw_ivf $LIBS

if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi

echo "编译成功!"

# 检查数据文件
if [ ! -f "anndata/DEEP100K.query.fbin" ]; then
    echo "错误: 找不到数据文件 anndata/DEEP100K.query.fbin"
    exit 1
fi

echo ""
echo "=== 运行 MPI HNSW+IVF 混合算法测试 ==="
echo "使用4个MPI进程..."

# 设置环境变量
export OMP_NUM_THREADS=4
export OMPI_ALLOW_RUN_AS_ROOT=1

# 运行测试
mpirun -np 4 --oversubscribe ./main_mpi_hnsw_ivf

echo ""
echo "测试完成!"

# 清理
echo "清理编译文件..."
rm -f main_mpi_hnsw_ivf

echo "脚本执行完毕。" 