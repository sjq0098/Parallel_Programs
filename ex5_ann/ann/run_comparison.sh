#!/bin/bash

# IVF搜索性能对比脚本

echo "=== IVF搜索性能对比测试 ==="
echo ""

# 检查可执行文件是否存在
if [ ! -f "./main_ivf_cpu" ]; then
    echo "CPU版本可执行文件不存在，尝试编译..."
    if [ -f "compile_cpu_ivf.sh" ]; then
        chmod +x compile_cpu_ivf.sh
        ./compile_cpu_ivf.sh
    else
        echo "编译脚本不存在，请手动编译CPU版本"
        exit 1
    fi
fi

if [ ! -f "./main_ivf_gpu" ]; then
    echo "GPU版本可执行文件不存在，尝试编译..."
    if [ -f "compile.sh" ]; then
        chmod +x compile.sh
        ./compile.sh
    else
        echo "编译脚本不存在，请手动编译GPU版本"
        exit 1
    fi
fi

echo "检查数据文件..."
if [ ! -f "anndata/DEEP100K.base.100k.fbin" ] || [ ! -f "file/ivf_flat_centroids_256.fbin" ]; then
    echo "警告: 数据文件可能不完整，请确保以下文件存在:"
    echo "  - anndata/DEEP100K.base.100k.fbin"
    echo "  - anndata/DEEP100K.query.fbin"
    echo "  - anndata/DEEP100K.gt.query.100k.top100.bin"
    echo "  - file/ivf_flat_centroids_256.fbin"
    echo "  - file/ivf_flat_invlists_256.bin"
    echo ""
fi

echo "开始性能对比测试..."
echo ""

# 运行CPU版本
echo "=== 运行CPU IVF版本 ==="
time ./main_ivf_cpu
echo ""

# 运行GPU版本
echo "=== 运行GPU IVF版本 ==="
time ./main_ivf_gpu
echo ""

echo "=== 对比测试完成 ==="
echo ""
echo "分析建议:"
echo "1. 比较平均recall，确保两个版本的精度相当"
echo "2. 比较平均延迟，GPU版本在大批量查询时应有显著优势"
echo "3. 注意GPU版本的batch_size设置，影响性能表现"
echo "4. 如果GPU版本性能不如预期，检查CUDA环境和GPU利用率" 