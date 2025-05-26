#!/bin/bash

# 确保目录存在
mkdir -p build 2>/dev/null || mkdir build

# 编译程序
echo "编译多查询并行IVF测试程序..."

# Windows平台编译
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    g++ -std=c++17 -O3 -mavx2 -mfma -fopenmp -D_WIN32 -o build/multi_ivf_benchmark.exe main_multi_ivf.cc -Ihnswlib -I.
    
    # 检查编译结果
    if [ $? -eq 0 ]; then
        echo "编译成功，执行文件位于 build/multi_ivf_benchmark.exe"
    else
        echo "编译失败，请检查错误信息"
        exit 1
    fi
# Linux/Unix平台编译
else
    g++ -std=c++17 -O3 -mavx2 -mfma -fopenmp -o build/multi_ivf_benchmark main_multi_ivf.cc -Ihnswlib -I. -lstdc++fs
    
    # 检查编译结果
    if [ $? -eq 0 ]; then
        echo "编译成功，执行文件位于 build/multi_ivf_benchmark"
    else
        echo "编译失败，请检查错误信息"
        exit 1
    fi
fi

echo "使用方法："
echo "  ./build/multi_ivf_benchmark"
echo "结果将显示不同方法、参数和线程数下的批量查询性能"