#!/bin/bash

# 创建结果目录
mkdir -p results 2>/dev/null || mkdir results

# 检测操作系统类型
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    EXEC="build/ivf_benchmark.exe"
else
    EXEC="build/ivf_benchmark"
fi

# 检查可执行文件是否存在
if [ ! -f "$EXEC" ]; then
    echo "错误: 找不到可执行文件 $EXEC"
    echo "请先运行 ./build_ivf.sh 编译程序"
    exit 1
fi

# 运行测试
echo "开始运行测试..."
echo "结果将保存在 results 目录中"

# 测试暴力搜索方法
echo "测试暴力搜索方法..."
$EXEC flat > results/flat_results.csv

# 测试IVF方法
echo "测试IVF相关方法..."
$EXEC ivf > results/ivf_results.csv

# 测试IVFPQ方法
echo "测试IVFPQ相关方法..."
$EXEC ivfpq > results/ivfpq_results.csv


echo "测试完成，结果已保存到results目录" 