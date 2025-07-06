#!/bin/bash

echo "=== 快速编译和测试 MPI HNSW+IVF 算法 ==="

# 设置编译参数
MPICXX_FLAGS="-O3 -march=native -std=c++17 -fopenmp"
INCLUDE_DIRS="-I. -I./hnswlib"
LIBS="-lm"

# 编译快速测试程序
echo "编译快速测试程序..."
mpicxx $MPICXX_FLAGS $INCLUDE_DIRS quick_hnsw_ivf_test.cc -o quick_hnsw_ivf_test $LIBS

if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi

echo "编译成功!"

# 检查数据文件
if [ ! -f "anndata/DEEP100K.query.fbin" ]; then
    echo "错误: 找不到数据文件"
    exit 1
fi

echo ""
echo "=== 运行快速测试 ==="

# 设置环境变量
export OMP_NUM_THREADS=4
export OMPI_ALLOW_RUN_AS_ROOT=1

# 运行快速测试
mpirun -np 4 --oversubscribe ./quick_hnsw_ivf_test

echo ""
echo "快速测试完成!"

# 清理
rm -f quick_hnsw_ivf_test

echo "脚本执行完毕。" 