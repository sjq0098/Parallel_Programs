#!/bin/bash

# 设置编译选项
CXX=g++
CXXFLAGS="-O2 -mavx2 -march=core-avx2 -std=c++11 -fopenmp"

# 创建文件夹
mkdir -p files
mkdir -p results

echo "编译原始版本..."
$CXX main.cc -o main $CXXFLAGS

echo "编译优化版本..."
$CXX main_op.cc -o main_op $CXXFLAGS

echo "编译PQ优化测试程序..."
$CXX main_test_pq.cc -o test_pq $CXXFLAGS

echo "构建完成!"
echo "运行 './run_all_tests.sh' 来测试所有算法"
echo "运行 './test_pq' 测试PQ优化效果" 