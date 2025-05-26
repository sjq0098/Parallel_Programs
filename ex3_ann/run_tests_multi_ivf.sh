#!/bin/bash

# 创建结果目录
mkdir -p results 2>/dev/null || mkdir results

# 检测操作系统类型
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    EXEC="build/multi_ivf_benchmark.exe"
else
    EXEC="build/multi_ivf_benchmark"
fi

# 检查可执行文件是否存在
if [ ! -f "$EXEC" ]; then
    echo "错误: 找不到可执行文件 $EXEC"
    echo "请先运行 ./build_multi_ivf.sh 编译程序"
    exit 1
fi

# 运行测试
echo "开始运行多查询并行测试..."
echo "结果将保存在 results/multi_ivf_results.csv 中"

# 运行测试并将结果输出到CSV文件
$EXEC > results/multi_ivf_results.csv

