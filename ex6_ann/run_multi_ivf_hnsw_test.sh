#!/bin/bash

# ====================================================================
# 原版IVF+HNSW多次运行测试脚本
# 使用预训练码本进行多配置对比实验
# ====================================================================

set -e

# 实验配置
MPI_PROCESSES=4
OMP_THREADS=4
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="multi_ivf_hnsw_results_${TIMESTAMP}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}======================================================================"
echo "              原版IVF+HNSW多次运行测试实验"
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
    "mpi_ivf_hnsw.h"
    "multi_run_ivf_hnsw_test.cc"
    "anndata/DEEP100K.query.fbin"
    "anndata/DEEP100K.gt.query.100k.top100.bin"
    "anndata/DEEP100K.base.100k.fbin"
    "files/pq4_codebook.bin"
    "files/pq8_codebook.bin"
    "files/pq16_codebook.bin"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}错误: 必需文件不存在: $file${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ 环境检查通过${NC}"

# 编译程序
echo -e "\n${BLUE}编译原版IVF+HNSW测试程序...${NC}"
mpic++ -O3 -std=c++11 -fopenmp -DWITH_OPENMP \
    -I. -Ihnswlib \
    multi_run_ivf_hnsw_test.cc \
    -o multi_ivf_hnsw_test \
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
        for (int i = 0; i < 3000000; ++i) {
            sum += i * 0.001;
        }
    }
    
    std::vector<float> dummy(300000);
    for (int i = 0; i < 300000; ++i) {
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

# 运行多次IVF+HNSW测试
echo -e "\n${BLUE}开始原版IVF+HNSW多次运行测试...${NC}"
echo "测试配置: 9种配置组合 (PQ4/PQ8/PQ16 × 快速/平衡/高精度)"
echo "测试参数: 暖机10次、重复5轮、每轮200次查询"
echo "预计时间: 30-60分钟"

start_time=$(date +%s)

# 运行测试
timeout 3600 mpirun -np $MPI_PROCESSES ./multi_ivf_hnsw_test \
    > "$RESULT_DIR/test_output.log" 2>&1

test_result=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

if [ $test_result -eq 0 ]; then
    echo -e "${GREEN}✓ 原版IVF+HNSW测试完成${NC}"
    echo "测试耗时: ${duration}秒"
    
    # 提取CSV结果
    echo "算法,配置,nlist,nprobe,M,efC,efS,召回率,召回率std,延迟μs,延迟std,构建时间ms,重复次数" > "$RESULT_DIR/results.csv"
    grep "原版IVF+HNSW," "$RESULT_DIR/test_output.log" >> "$RESULT_DIR/results.csv" 2>/dev/null || true
    
    # 检查结果
    if [ -f "$RESULT_DIR/results.csv" ] && [ -s "$RESULT_DIR/results.csv" ]; then
        echo -e "${GREEN}✓ 结果提取成功${NC}"
        
        # 显示结果摘要
        echo -e "\n${YELLOW}=== 原版IVF+HNSW 测试结果摘要 ===${NC}"
        
        echo -e "\n${CYAN}PQ4 码本结果:${NC}"
        awk -F',' '$2 ~ /PQ4/ {printf "  %-12s | 召回率: %.4f | 延迟: %6.1fμs | 构建: %6.0fms\n", $2, $8, $10, $12}' "$RESULT_DIR/results.csv" | head -3
        
        echo -e "\n${CYAN}PQ8 码本结果:${NC}"
        awk -F',' '$2 ~ /PQ8/ {printf "  %-12s | 召回率: %.4f | 延迟: %6.1fμs | 构建: %6.0fms\n", $2, $8, $10, $12}' "$RESULT_DIR/results.csv" | head -3
        
        echo -e "\n${CYAN}PQ16 码本结果:${NC}"
        awk -F',' '$2 ~ /PQ16/ {printf "  %-12s | 召回率: %.4f | 延迟: %6.1fμs | 构建: %6.0fms\n", $2, $8, $10, $12}' "$RESULT_DIR/results.csv" | head -3
        
        # 查找最佳性能
        echo -e "\n${YELLOW}=== 性能统计 ===${NC}"
        
        # 最高召回率
        best_recall=$(awk -F',' 'NR>1 {print $8}' "$RESULT_DIR/results.csv" | sort -nr | head -1)
        best_recall_config=$(awk -F',' -v max="$best_recall" '$8 == max {print $2; exit}' "$RESULT_DIR/results.csv")
        echo "最高召回率: $best_recall ($best_recall_config)"
        
        # 最低延迟
        best_latency=$(awk -F',' 'NR>1 {print $10}' "$RESULT_DIR/results.csv" | sort -n | head -1)
        best_latency_config=$(awk -F',' -v min="$best_latency" '$10 == min {print $2; exit}' "$RESULT_DIR/results.csv")
        echo "最低延迟: ${best_latency}μs ($best_latency_config)"
        
        # 最快构建
        best_build=$(awk -F',' 'NR>1 {print $12}' "$RESULT_DIR/results.csv" | sort -n | head -1)
        best_build_config=$(awk -F',' -v min="$best_build" '$12 == min {print $2; exit}' "$RESULT_DIR/results.csv")
        echo "最快构建: ${best_build}ms ($best_build_config)"
        
    else
        echo -e "${RED}⚠️  未找到有效的CSV结果${NC}"
    fi
    
elif [ $test_result -eq 124 ]; then
    echo -e "${RED}✗ 测试超时(60分钟限制)${NC}"
else
    echo -e "${RED}✗ 测试失败，退出码: $test_result${NC}"
fi

# 生成详细报告
echo -e "\n${BLUE}生成实验报告...${NC}"
cat > "$RESULT_DIR/experiment_report.md" << EOF
# 原版IVF+HNSW多次运行测试报告

## 实验信息
- **测试时间**: $(date)
- **算法**: 原版IVF+HNSW
- **MPI进程数**: $MPI_PROCESSES
- **OpenMP线程数**: $OMP_THREADS
- **测试耗时**: ${duration}秒

## 测试配置
使用3种PQ码本 × 3种参数配置 = 9种测试组合：

### PQ4码本配置
- 快速: nlist=256, nprobe=8, M=8, efC=100, efS=50
- 平衡: nlist=256, nprobe=16, M=12, efC=150, efS=80
- 高精度: nlist=256, nprobe=32, M=16, efC=200, efS=100

### PQ8码本配置
- 快速: nlist=256, nprobe=8, M=8, efC=100, efS=50
- 平衡: nlist=256, nprobe=16, M=12, efC=150, efS=80
- 高精度: nlist=256, nprobe=32, M=16, efC=200, efS=100

### PQ16码本配置
- 快速: nlist=256, nprobe=8, M=8, efC=100, efS=50
- 平衡: nlist=256, nprobe=16, M=12, efC=150, efS=80
- 高精度: nlist=256, nprobe=32, M=16, efC=200, efS=100

## 实验设计
- **数据集**: DEEP100K (100k向量, 96维)
- **暖机**: 10次查询预热
- **测试**: 每配置重复5轮，每轮200次查询
- **指标**: 召回率、延迟、构建时间

## 算法特点
- **IVF索引**: 使用预训练PQ码本作为聚类中心
- **HNSW索引**: 每个簇内构建独立的HNSW图索引
- **MPI并行**: 多进程分布式计算
- **预训练码本**: 利用已有的PQ4/PQ8/PQ16码本

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
rm -f multi_ivf_hnsw_test

echo -e "\n${GREEN}=== 原版IVF+HNSW多次运行测试完成 ===${NC}"
echo -e "结果目录: ${BLUE}$RESULT_DIR${NC}"
echo -e "主要文件:"
echo -e "  📊 ${YELLOW}results.csv${NC} - 测试数据"
echo -e "  📋 ${YELLOW}test_output.log${NC} - 完整日志"
echo -e "  📄 ${YELLOW}experiment_report.md${NC} - 实验报告"

echo -e "\n💡 要查看详细结果: ${YELLOW}cat $RESULT_DIR/results.csv | column -t -s','${NC}" 