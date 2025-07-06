#!/bin/bash

# ====================================================================
# 算法对比实验脚本: HNSW+IVF vs IVF+HNSW 性能基准测试
# 包含暖机、多次重复、统计分析和详细报告生成
# ====================================================================

set -e

# 实验配置
MPI_PROCESSES=4
OMP_THREADS=4
OUTPUT_DIR="comparison_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_ID="exp_${TIMESTAMP}"

# 颜色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${PURPLE}=== $1 ===${NC}"
}

# 检查依赖
check_dependencies() {
    log_section "检查实验环境"
    
    # 检查MPI
    if ! command -v mpirun &> /dev/null; then
        log_error "MPI未安装或不在PATH中"
        exit 1
    fi
    
    # 检查编译器
    if ! command -v mpic++ &> /dev/null; then
        log_error "MPI C++编译器未找到"
        exit 1
    fi
    
    # 检查数据文件
    if [ ! -d "anndata" ]; then
        log_error "数据目录 anndata/ 不存在"
        exit 1
    fi
    
    required_files=(
        "anndata/DEEP100K.query.fbin"
        "anndata/DEEP100K.gt.query.100k.top100.bin"
        "anndata/DEEP100K.base.100k.fbin"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "必需的数据文件不存在: $file"
            exit 1
        fi
    done
    
    # 检查头文件
    if [ ! -f "mpi_hnsw_ivf_optimized.h" ]; then
        log_error "HNSW+IVF优化版头文件不存在"
        exit 1
    fi
    
    if [ ! -f "mpi_ivf_hnsw_lightweight.h" ]; then
        log_error "轻量级IVF+HNSW头文件不存在"
        exit 1
    fi
    
    log_success "环境检查通过"
    log_info "MPI进程数: $MPI_PROCESSES"
    log_info "OpenMP线程数: $OMP_THREADS"
}

# 创建输出目录
setup_output_directory() {
    log_section "创建输出目录"
    
    mkdir -p "$OUTPUT_DIR/$EXPERIMENT_ID"
    export OMP_NUM_THREADS=$OMP_THREADS
    
    log_success "输出目录创建: $OUTPUT_DIR/$EXPERIMENT_ID"
}

# 编译测试程序
compile_programs() {
    log_section "编译测试程序"
    
    # 编译HNSW+IVF优化版测试程序
    log_info "编译HNSW+IVF优化版..."
    mpic++ -O3 -std=c++11 -fopenmp -DWITH_OPENMP \
        -I. -Ihnswlib \
        comparison_test_hnsw_ivf.cc \
        -o comparison_test_hnsw_ivf \
        2> "$OUTPUT_DIR/$EXPERIMENT_ID/compile_hnsw_ivf.log"
    
    if [ $? -eq 0 ]; then
        log_success "HNSW+IVF优化版编译成功"
    else
        log_error "HNSW+IVF优化版编译失败，查看日志: $OUTPUT_DIR/$EXPERIMENT_ID/compile_hnsw_ivf.log"
        exit 1
    fi
    
    # 编译轻量级IVF+HNSW测试程序
    log_info "编译轻量级IVF+HNSW..."
    mpic++ -O3 -std=c++11 -fopenmp -DWITH_OPENMP \
        -I. -Ihnswlib \
        comparison_test_ivf_hnsw.cc \
        -o comparison_test_ivf_hnsw \
        2> "$OUTPUT_DIR/$EXPERIMENT_ID/compile_ivf_hnsw.log"
    
    if [ $? -eq 0 ]; then
        log_success "轻量级IVF+HNSW编译成功"
    else
        log_error "轻量级IVF+HNSW编译失败，查看日志: $OUTPUT_DIR/$EXPERIMENT_ID/compile_ivf_hnsw.log"
        exit 1
    fi
}

# 系统暖机
system_warmup() {
    log_section "系统暖机"
    
    log_info "进行系统级暖机（CPU频率调整、缓存预热）..."
    
    # 创建简单的暖机程序
    cat > warmup_dummy.cc << 'EOF'
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    // CPU密集计算暖机
    #pragma omp parallel
    {
        double sum = 0.0;
        for (int i = 0; i < 10000000; ++i) {
            sum += i * 0.001;
        }
    }
    
    // 内存访问暖机
    std::vector<float> dummy(1000000);
    for (int i = 0; i < 1000000; ++i) {
        dummy[i] = i * 0.1f;
    }
    
    MPI_Finalize();
    return 0;
}
EOF
    
    mpic++ -O3 -fopenmp warmup_dummy.cc -o warmup_dummy 2>/dev/null
    mpirun -np $MPI_PROCESSES ./warmup_dummy > /dev/null 2>&1
    rm -f warmup_dummy.cc warmup_dummy
    
    # 短暂休息让系统稳定
    sleep 2
    
    log_success "系统暖机完成"
}

# 运行HNSW+IVF测试
run_hnsw_ivf_test() {
    log_section "运行HNSW+IVF优化版测试"
    
    local output_file="$OUTPUT_DIR/$EXPERIMENT_ID/hnsw_ivf_results.csv"
    local log_file="$OUTPUT_DIR/$EXPERIMENT_ID/hnsw_ivf_test.log"
    
    log_info "开始HNSW+IVF测试..."
    log_info "结果输出: $output_file"
    log_info "日志输出: $log_file"
    
    # 运行测试并捕获输出
    timeout 1800 mpirun -np $MPI_PROCESSES ./comparison_test_hnsw_ivf \
        > >(tee "$log_file") 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "HNSW+IVF测试完成"
        
        # 提取CSV结果
        grep "HNSW+IVF优化版," "$log_file" > "$output_file"
        log_info "结果已保存到: $output_file"
    else
        log_error "HNSW+IVF测试失败或超时"
        return 1
    fi
}

# 运行轻量级IVF+HNSW测试
run_ivf_hnsw_test() {
    log_section "运行轻量级IVF+HNSW测试"
    
    local output_file="$OUTPUT_DIR/$EXPERIMENT_ID/ivf_hnsw_results.csv"
    local log_file="$OUTPUT_DIR/$EXPERIMENT_ID/ivf_hnsw_test.log"
    
    log_info "开始轻量级IVF+HNSW测试..."
    log_info "结果输出: $output_file"
    log_info "日志输出: $log_file"
    
    # 运行测试并捕获输出
    timeout 1800 mpirun -np $MPI_PROCESSES ./comparison_test_ivf_hnsw \
        > >(tee "$log_file") 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "轻量级IVF+HNSW测试完成"
        
        # 提取CSV结果
        grep "IVF+HNSW," "$log_file" > "$output_file"
        log_info "结果已保存到: $output_file"
    else
        log_error "轻量级IVF+HNSW测试失败或超时"
        return 1
    fi
}

# 合并和分析结果
analyze_results() {
    log_section "结果分析"
    
    local hnsw_ivf_file="$OUTPUT_DIR/$EXPERIMENT_ID/hnsw_ivf_results.csv"
    local ivf_hnsw_file="$OUTPUT_DIR/$EXPERIMENT_ID/ivf_hnsw_results.csv"
    local combined_file="$OUTPUT_DIR/$EXPERIMENT_ID/combined_results.csv"
    local analysis_file="$OUTPUT_DIR/$EXPERIMENT_ID/performance_analysis.txt"
    
    # 合并结果
    echo "算法,配置,nlist,nprobe,M,efC,efS,max_candidates,recall_mean,recall_std,latency_us_mean,latency_us_std,build_time_ms,repeat_count" > "$combined_file"
    
    if [ -f "$hnsw_ivf_file" ]; then
        cat "$hnsw_ivf_file" >> "$combined_file"
    fi
    
    if [ -f "$ivf_hnsw_file" ]; then
        # IVF+HNSW文件需要添加max_candidates列（设为N/A）
        sed 's/\(IVF+HNSW,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\)/\1N\/A,/' "$ivf_hnsw_file" >> "$combined_file"
    fi
    
    log_success "结果已合并到: $combined_file"
    
    # 生成分析报告
    python3 << 'EOF' > "$analysis_file"
import csv
import sys
from collections import defaultdict

def analyze_results(filename):
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except FileNotFoundError:
        print("结果文件不存在")
        return
    
    hnsw_ivf_results = [row for row in data if 'HNSW+IVF' in row['算法']]
    ivf_hnsw_results = [row for row in data if row['算法'] == 'IVF+HNSW']
    
    print("=" * 60)
    print("                 算法性能对比分析报告")
    print("=" * 60)
    
    if hnsw_ivf_results:
        print("\n🔹 HNSW+IVF 优化版结果:")
        for result in hnsw_ivf_results:
            print(f"  配置: {result['配置']}")
            print(f"    召回率: {float(result['recall_mean']):.4f} ± {float(result['recall_std']):.4f}")
            print(f"    延迟: {float(result['latency_us_mean']):.1f} ± {float(result['latency_us_std']):.1f} μs")
            print(f"    构建时间: {result['build_time_ms']} ms")
            print()
    
    if ivf_hnsw_results:
        print("🔹 IVF+HNSW 结果:")
        for result in ivf_hnsw_results:
            print(f"  配置: {result['配置']}")
            print(f"    召回率: {float(result['recall_mean']):.4f} ± {float(result['recall_std']):.4f}")
            print(f"    延迟: {float(result['latency_us_mean']):.1f} ± {float(result['latency_us_std']):.1f} μs")
            print(f"    构建时间: {result['build_time_ms']} ms")
            print()
    
    # 最佳性能对比
    if hnsw_ivf_results and ivf_hnsw_results:
        print("=" * 60)
        print("                    性能对比总结")
        print("=" * 60)
        
        # 最高召回率
        best_hnsw_recall = max(hnsw_ivf_results, key=lambda x: float(x['recall_mean']))
        best_ivf_recall = max(ivf_hnsw_results, key=lambda x: float(x['recall_mean']))
        
        print(f"\n📊 最高召回率对比:")
        print(f"  HNSW+IVF: {float(best_hnsw_recall['recall_mean']):.4f} ({best_hnsw_recall['配置']})")
        print(f"  IVF+HNSW: {float(best_ivf_recall['recall_mean']):.4f} ({best_ivf_recall['配置']})")
        
        # 最低延迟
        best_hnsw_latency = min(hnsw_ivf_results, key=lambda x: float(x['latency_us_mean']))
        best_ivf_latency = min(ivf_hnsw_results, key=lambda x: float(x['latency_us_mean']))
        
        print(f"\n⚡ 最低延迟对比:")
        print(f"  HNSW+IVF: {float(best_hnsw_latency['latency_us_mean']):.1f} μs ({best_hnsw_latency['配置']})")
        print(f"  IVF+HNSW: {float(best_ivf_latency['latency_us_mean']):.1f} μs ({best_ivf_latency['配置']})")
        
        # 综合建议
        print(f"\n💡 算法选择建议:")
        if float(best_hnsw_recall['recall_mean']) > float(best_ivf_recall['recall_mean']):
            print("  • 高精度场景推荐: HNSW+IVF (更高召回率)")
        else:
            print("  • 高精度场景推荐: IVF+HNSW (更高召回率)")
            
        if float(best_hnsw_latency['latency_us_mean']) < float(best_ivf_latency['latency_us_mean']):
            print("  • 实时性场景推荐: HNSW+IVF (更低延迟)")
        else:
            print("  • 实时性场景推荐: IVF+HNSW (更低延迟)")

if __name__ == "__main__":
    analyze_results(sys.argv[1])
EOF
    
    python3 - "$combined_file"
    
    log_success "分析报告已生成: $analysis_file"
}

# 生成性能图表（如果有matplotlib）
generate_charts() {
    log_section "生成性能图表"
    
    if ! python3 -c "import matplotlib" 2>/dev/null; then
        log_warning "matplotlib未安装，跳过图表生成"
        return
    fi
    
    local combined_file="$OUTPUT_DIR/$EXPERIMENT_ID/combined_results.csv"
    local chart_output="$OUTPUT_DIR/$EXPERIMENT_ID"
    
    python3 << EOF > /dev/null 2>&1
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('$combined_file')
    
    # 召回率对比图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    hnsw_data = df[df['算法'].str.contains('HNSW+IVF')]
    ivf_data = df[df['算法'] == 'IVF+HNSW']
    
    x_pos = np.arange(len(hnsw_data))
    plt.bar(x_pos - 0.2, hnsw_data['recall_mean'], 0.4, 
            label='HNSW+IVF', alpha=0.8, color='skyblue')
    if len(ivf_data) > 0:
        plt.bar(x_pos + 0.2, ivf_data['recall_mean'], 0.4, 
                label='IVF+HNSW', alpha=0.8, color='lightcoral')
    
    plt.xlabel('配置')
    plt.ylabel('召回率')
    plt.title('召回率对比')
    plt.legend()
    plt.xticks(x_pos, hnsw_data['配置'], rotation=45)
    
    # 延迟对比图
    plt.subplot(2, 2, 2)
    plt.bar(x_pos - 0.2, hnsw_data['latency_us_mean'], 0.4, 
            label='HNSW+IVF', alpha=0.8, color='skyblue')
    if len(ivf_data) > 0:
        plt.bar(x_pos + 0.2, ivf_data['latency_us_mean'], 0.4, 
                label='IVF+HNSW', alpha=0.8, color='lightcoral')
    
    plt.xlabel('配置')
    plt.ylabel('延迟 (μs)')
    plt.title('延迟对比')
    plt.legend()
    plt.xticks(x_pos, hnsw_data['配置'], rotation=45)
    
    # 召回率-延迟权衡图
    plt.subplot(2, 2, 3)
    plt.scatter(hnsw_data['latency_us_mean'], hnsw_data['recall_mean'], 
               s=100, alpha=0.7, c='skyblue', label='HNSW+IVF')
    if len(ivf_data) > 0:
        plt.scatter(ivf_data['latency_us_mean'], ivf_data['recall_mean'], 
                   s=100, alpha=0.7, c='lightcoral', label='IVF+HNSW')
    
    plt.xlabel('延迟 (μs)')
    plt.ylabel('召回率')
    plt.title('召回率-延迟权衡')
    plt.legend()
    
    # 构建时间对比
    plt.subplot(2, 2, 4)
    plt.bar(x_pos - 0.2, hnsw_data['build_time_ms'], 0.4, 
            label='HNSW+IVF', alpha=0.8, color='skyblue')
    if len(ivf_data) > 0:
        plt.bar(x_pos + 0.2, ivf_data['build_time_ms'], 0.4, 
                label='IVF+HNSW', alpha=0.8, color='lightcoral')
    
    plt.xlabel('配置')
    plt.ylabel('构建时间 (ms)')
    plt.title('索引构建时间对比')
    plt.legend()
    plt.xticks(x_pos, hnsw_data['配置'], rotation=45)
    
    plt.tight_layout()
    plt.savefig('$chart_output/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("图表已生成")
    
except Exception as e:
    print(f"图表生成失败: {e}")
EOF
    
    if [ -f "$chart_output/performance_comparison.png" ]; then
        log_success "性能对比图表已生成: $chart_output/performance_comparison.png"
    else
        log_warning "图表生成失败"
    fi
}

# 清理临时文件
cleanup() {
    log_section "清理临时文件"
    
    rm -f comparison_test_hnsw_ivf comparison_test_ivf_hnsw
    rm -f temp_centroids_*.bin
    
    log_success "清理完成"
}

# 生成实验报告
generate_experiment_report() {
    log_section "生成实验报告"
    
    local report_file="$OUTPUT_DIR/$EXPERIMENT_ID/experiment_report.md"
    
    cat > "$report_file" << EOF
# 算法对比实验报告

## 实验信息
- **实验ID**: $EXPERIMENT_ID
- **时间**: $(date)
- **MPI进程数**: $MPI_PROCESSES
- **OpenMP线程数**: $OMP_THREADS
- **数据集**: DEEP100K

## 实验目的
对比 HNSW+IVF 优化版和 IVF+HNSW 两种混合索引算法的性能表现，包括：
- 召回率 (Recall)
- 查询延迟 (Latency)
- 索引构建时间 (Build Time)

## 实验设计
- **暖机阶段**: 100次查询预热
- **正式测试**: 每配置重复5轮，每轮1000次查询
- **统计方法**: 计算均值和标准差，消除偶发抖动
- **测试配置**: 快速配置、平衡配置、高精度配置

## 结果文件
- \`combined_results.csv\`: 完整测试数据
- \`performance_analysis.txt\`: 性能分析报告
- \`performance_comparison.png\`: 可视化对比图表 (如果可用)

## 测试环境
- **操作系统**: $(uname -a)
- **编译器**: $(mpic++ --version | head -1)
- **MPI实现**: $(mpirun --version | head -1)

## 文件说明
- 原始日志文件: \`*_test.log\`
- 编译日志: \`compile_*.log\`
- CSV结果: \`*_results.csv\`

---
*自动生成于 $(date)*
EOF

    log_success "实验报告已生成: $report_file"
}

# 主实验流程
main() {
    echo -e "${CYAN}"
    echo "======================================================================"
    echo "          MPI并行向量搜索算法对比实验 v1.0"
    echo "======================================================================"
    echo -e "${NC}"
    
    # 开始时间记录
    local start_time=$(date +%s)
    
    # 执行实验步骤
    check_dependencies
    setup_output_directory
    compile_programs
    system_warmup
    
    # 运行算法测试
    if ! run_hnsw_ivf_test; then
        log_error "HNSW+IVF测试失败，继续轻量级IVF+HNSW测试"
    fi
    
    sleep 5  # 算法间休息
    
    if ! run_ivf_hnsw_test; then
        log_error "轻量级IVF+HNSW测试失败"
    fi
    
    # 结果分析和报告
    analyze_results
    generate_charts
    generate_experiment_report
    cleanup
    
    # 计算总耗时
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    log_section "实验完成"
    log_success "实验总耗时: ${hours}小时${minutes}分钟${seconds}秒"
    log_success "结果目录: $OUTPUT_DIR/$EXPERIMENT_ID"
    
    echo -e "\n${GREEN}🎉 对比实验成功完成! 🎉${NC}"
    echo -e "查看结果: ${BLUE}cd $OUTPUT_DIR/$EXPERIMENT_ID${NC}"
    echo -e "主要结果文件:"
    echo -e "  📊 ${YELLOW}combined_results.csv${NC} - 完整数据"
    echo -e "  📋 ${YELLOW}performance_analysis.txt${NC} - 分析报告"
    echo -e "  📈 ${YELLOW}performance_comparison.png${NC} - 性能图表"
    echo -e "  📄 ${YELLOW}experiment_report.md${NC} - 实验报告"
}

# 信号处理
trap 'log_error "实验被中断"; cleanup; exit 1' INT TERM

# 执行主函数
main "$@" 