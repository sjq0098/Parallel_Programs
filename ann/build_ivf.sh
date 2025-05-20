#!/bin/bash

# 确保目录存在
mkdir -p build 2>/dev/null || mkdir build

# 编译程序
echo "编译IVF/Flat测试程序..."

# Windows平台编译
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    g++ -std=c++17 -O3 -mavx2 -mfma -fopenmp -D_WIN32 -o build/ivf_benchmark.exe main_ivf.cc -Ihnswlib -I.
    
    # 检查编译结果
    if [ $? -eq 0 ]; then
        echo "编译成功，执行文件位于 build/ivf_benchmark.exe"
    else
        echo "编译失败，请检查错误信息"
        exit 1
    fi
# Linux/Unix平台编译
else
    g++ -std=c++17 -O3 -mavx2 -mfma -fopenmp -o build/ivf_benchmark main_ivf.cc -Ihnswlib -I. -lstdc++fs
    
    # 检查编译结果
    if [ $? -eq 0 ]; then
        echo "编译成功，执行文件位于 build/ivf_benchmark"
    else
        echo "编译失败，请检查错误信息"
        exit 1
    fi
fi

echo "使用方法："
echo "  ./build/ivf_benchmark flat   # 测试暴力搜索方法"
echo "  ./build/ivf_benchmark ivf    # 测试IVF相关方法"
echo "  ./build/ivf_benchmark ivfpq  # 测试IVFPQ相关方法"
echo "  ./build/ivf_benchmark all    # 测试所有方法" 