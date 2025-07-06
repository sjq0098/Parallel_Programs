#!/bin/bash

# ====================================================================
# 轻量级IVF+HNSW 独立测试脚本
# 保持完整的实验设计，只测试轻量级IVF+HNSW算法
# ====================================================================

set -e

# 实验配置
MPI_PROCESSES=4
OMP_THREADS=4
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="ivf_hnsw_results_${TIMESTAMP}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}======================================================================"
echo "          轻量级IVF+HNSW 独立测试实验"
echo "======================================================================${NC}"
echo "MPI进程数: $MPI_PROCESSES"
echo "OpenMP线程数: $OMP_THREADS"
echo "结果目录: $RESULT_DIR"

# 设置环境
export OMP_NUM_THREADS=$OMP_THREADS

# 创建结果目录
mkdir -p "$RESULT_DIR"

# 检查依赖文件
echo -e "\n${BLUE}检查实验环境...${NC}"
required_files=(
    "mpi_ivf_hnsw_lightweight.h"
    "comparison_test_ivf_hnsw.cc"
    "anndata/DEEP100K.query.fbin"
    "anndata/DEEP100K.gt.query.100k.top100.bin"
    "anndata/DEEP100K.base.100k.fbin"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}错误: 必需文件不存在: $file${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ 环境检查通过${NC}"

# 编译轻量级IVF+HNSW
echo -e "\n${BLUE}编译轻量级IVF+HNSW...${NC}"
mpic++ -O3 -std=c++11 -fopenmp -DWITH_OPENMP \
    -I. -Ihnswlib \
    comparison_test_ivf_hnsw.cc \
    -o ivf_hnsw_test \
    2> "$RESULT_DIR/compile.log"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 编译成功${NC}"
else
    echo -e "${RED}✗ 编译失败，查看日志: $RESULT_DIR/compile.log${NC}"
    exit 1
fi

# 系统暖机
echo -e "\n${BLUE}系统暖机...${NC}"
cat > warmup_dummy.cc << 'EOF'
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <chrono>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    #pragma omp parallel
    {
        double sum = 0.0;
        for (int i = 0; i < 5000000; ++i) {
            sum += i * 0.001;
        }
    }
    
    std::vector<float> dummy(500000);
    for (int i = 0; i < 500000; ++i) {
        dummy[i] = i * 0.1f;
    }
    
    MPI_Finalize();
    return 0;
}
EOF

mpic++ -O3 -fopenmp warmup_dummy.cc -o warmup_dummy 2>/dev/null
mpirun -np $MPI_PROCESSES ./warmup_dummy > /dev/null 2>&1
rm -f warmup_dummy.cc warmup_dummy
sleep 2
echo -e "${GREEN}✓ 系统暖机完成${NC}"

# 运行轻量级IVF+HNSW测试
echo -e "\n${BLUE}开始轻量级IVF+HNSW测试...${NC}"
echo "测试配置: 快速配置、平衡配置、高精度配置"
echo "测试参数: 暖机20次、重复3轮、1000条查询"
echo "预计时间: 3-10分钟"

start_time=$(date +%s)

# 运行测试
timeout 1800 mpirun -np $MPI_PROCESSES ./ivf_hnsw_test \
    > "$RESULT_DIR/test_output.log" 2>&1

test_result=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

if [ $test_result -eq 0 ]; then
    echo -e "${GREEN}✓ 轻量级IVF+HNSW测试完成${NC}"
    echo "测试耗时: ${duration}秒"
    
    # 提取CSV结果
    grep "IVF+HNSW," "$RESULT_DIR/test_output.log" > "$RESULT_DIR/results.csv" 2>/dev/null || true
    
    # 检查结果
    if [ -f "$RESULT_DIR/results.csv" ] && [ -s "$RESULT_DIR/results.csv" ]; then
        echo -e "${GREEN}✓ 结果提取成功${NC}"
        
        # 显示结果摘要
        echo -e "\n${YELLOW}=== 轻量级IVF+HNSW 测试结果摘要 ===${NC}"
        echo "配置,召回率,延迟(μs),构建时间(ms)"
        while IFS=',' read -r alg config nlist nprobe M efC efS recall_mean recall_std lat_mean lat_std build_time repeat; do
            echo "$config,$recall_mean,$lat_mean,$build_time"
        done < "$RESULT_DIR/results.csv"
        
        # 提取最佳性能
        echo -e "\n${YELLOW}=== 性能统计 ===${NC}"
        tail -5 "$RESULT_DIR/test_output.log" | grep -E "(最高召回率|最低延迟|索引构建)" || true
        
    else
        echo -e "${RED}⚠️  未找到有效的CSV结果${NC}"
    fi
    
elif [ $test_result -eq 124 ]; then
    echo -e "${RED}✗ 测试超时(30分钟限制)${NC}"
else
    echo -e "${RED}✗ 测试失败，退出码: $test_result${NC}"
fi

# 生成详细报告
echo -e "\n${BLUE}生成实验报告...${NC}"
cat > "$RESULT_DIR/experiment_report.md" << EOF
# 轻量级IVF+HNSW 测试报告

## 实验信息
- **测试时间**: $(date)
- **算法**: 轻量级IVF+HNSW
- **MPI进程数**: $MPI_PROCESSES
- **OpenMP线程数**: $OMP_THREADS
- **测试耗时**: ${duration}秒

## 测试配置
- 快速配置: nlist=64, nprobe=4, M=8, efC=100, efS=50
- 平衡配置: nlist=64, nprobe=8, M=12, efC=100, efS=80
- 高精度配置: nlist=128, nprobe=16, M=16, efC=150, efS=100

## 实验设计
- **数据集**: DEEP100K (100k向量, 96维)
- **暖机**: 20次查询预热
- **测试**: 每配置重复3轮，每轮1000次查询
- **指标**: 召回率、延迟、构建时间

## 算法特点
- **IVF索引**: 先进行聚类粗筛选
- **暴力搜索**: 在选中簇内使用精确距离计算
- **MPI并行**: 多进程分布式计算
- **轻量级**: 避免每簇构建HNSW索引，提高构建速度

## 结果文件
- \`results.csv\`: CSV格式测试数据
- \`test_output.log\`: 完整测试日志
- \`compile.log\`: 编译日志

## 系统环境
- **操作系统**: $(uname -a)
- **编译器**: $(mpic++ --version | head -1)

---
*自动生成于 $(date)*
EOF

# 清理临时文件
rm -f ivf_hnsw_test
rm -f temp_centroids_*.bin

echo -e "\n${GREEN}=== 轻量级IVF+HNSW 独立测试完成 ===${NC}"
echo -e "结果目录: ${BLUE}$RESULT_DIR${NC}"
echo -e "主要文件:"
echo -e "  📊 ${YELLOW}results.csv${NC} - 测试数据"
echo -e "  📋 ${YELLOW}test_output.log${NC} - 完整日志"
echo -e "  📄 ${YELLOW}experiment_report.md${NC} - 实验报告"

echo -e "\n💡 要测试HNSW+IVF，请运行: ${YELLOW}./run_hnsw_ivf_only.sh${NC}" 