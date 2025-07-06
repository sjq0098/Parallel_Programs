#!/bin/bash

# ====================================================================
# 算法结果对比分析脚本
# 用于比较独立运行的HNSW+IVF和轻量级IVF+HNSW的结果
# ====================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${CYAN}======================================================================"
echo "                    算法性能对比分析工具"
echo "======================================================================${NC}"

# 检查参数
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}用法: $0 [选项]${NC}"
    echo ""
    echo "选项:"
    echo "  auto          - 自动查找最新的结果目录进行对比"
    echo "  <dir1> <dir2> - 指定两个结果目录进行对比"
    echo "  list          - 列出可用的结果目录"
    echo ""
    echo "示例:"
    echo "  $0 auto"
    echo "  $0 hnsw_ivf_results_20241206_143022 ivf_hnsw_results_20241206_143545"
    echo "  $0 list"
    exit 0
fi

# 列出可用目录
if [ "$1" == "list" ]; then
    echo -e "${BLUE}可用的结果目录:${NC}"
    echo ""
    echo -e "${YELLOW}HNSW+IVF 结果:${NC}"
    ls -dt hnsw_ivf_results_* 2>/dev/null | head -5 || echo "  (无)"
    echo ""
    echo -e "${YELLOW}IVF+HNSW 结果:${NC}"
    ls -dt ivf_hnsw_results_* 2>/dev/null | head -5 || echo "  (无)"
    exit 0
fi

# 自动查找最新目录
if [ "$1" == "auto" ]; then
    echo -e "${BLUE}自动查找最新的结果目录...${NC}"
    
    HNSW_IVF_DIR=$(ls -dt hnsw_ivf_results_* 2>/dev/null | head -1)
    IVF_HNSW_DIR=$(ls -dt ivf_hnsw_results_* 2>/dev/null | head -1)
    
    if [ -z "$HNSW_IVF_DIR" ] || [ -z "$IVF_HNSW_DIR" ]; then
        echo -e "${RED}错误: 找不到足够的结果目录${NC}"
        echo "请先运行 ./run_hnsw_ivf_only.sh 和 ./run_ivf_hnsw_only.sh"
        exit 1
    fi
    
    echo -e "HNSW+IVF: ${GREEN}$HNSW_IVF_DIR${NC}"
    echo -e "IVF+HNSW: ${GREEN}$IVF_HNSW_DIR${NC}"
    
elif [ $# -eq 2 ]; then
    HNSW_IVF_DIR="$1"
    IVF_HNSW_DIR="$2"
    
    if [ ! -d "$HNSW_IVF_DIR" ]; then
        echo -e "${RED}错误: 目录不存在: $HNSW_IVF_DIR${NC}"
        exit 1
    fi
    
    if [ ! -d "$IVF_HNSW_DIR" ]; then
        echo -e "${RED}错误: 目录不存在: $IVF_HNSW_DIR${NC}"
        exit 1
    fi
    
else
    echo -e "${RED}错误: 参数数量不正确${NC}"
    echo "使用 $0 help 查看帮助"
    exit 1
fi

# 检查结果文件
HNSW_IVF_CSV="$HNSW_IVF_DIR/results.csv"
IVF_HNSW_CSV="$IVF_HNSW_DIR/results.csv"

if [ ! -f "$HNSW_IVF_CSV" ]; then
    echo -e "${RED}错误: HNSW+IVF结果文件不存在: $HNSW_IVF_CSV${NC}"
    exit 1
fi

if [ ! -f "$IVF_HNSW_CSV" ]; then
    echo -e "${RED}错误: IVF+HNSW结果文件不存在: $IVF_HNSW_CSV${NC}"
    exit 1
fi

# 创建对比结果目录
COMPARISON_DIR="comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$COMPARISON_DIR"

echo -e "\n${BLUE}开始结果对比分析...${NC}"
echo -e "对比结果将保存在: ${GREEN}$COMPARISON_DIR${NC}"

# 合并CSV数据
echo -e "\n${BLUE}合并测试数据...${NC}"
echo "算法,配置,nlist,nprobe,M,efC,efS,max_candidates,recall_mean,recall_std,latency_us_mean,latency_us_std,build_time_ms,repeat_count" > "$COMPARISON_DIR/combined_results.csv"

# 添加HNSW+IVF数据
if [ -s "$HNSW_IVF_CSV" ]; then
    cat "$HNSW_IVF_CSV" >> "$COMPARISON_DIR/combined_results.csv"
    echo -e "${GREEN}✓ HNSW+IVF数据已合并${NC}"
else
    echo -e "${YELLOW}⚠️  HNSW+IVF结果文件为空${NC}"
fi

# 添加IVF+HNSW数据（添加max_candidates列）
if [ -s "$IVF_HNSW_CSV" ]; then
    sed 's/\(IVF+HNSW,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\)/\1N\/A,/' "$IVF_HNSW_CSV" >> "$COMPARISON_DIR/combined_results.csv"
    echo -e "${GREEN}✓ IVF+HNSW数据已合并${NC}"
else
    echo -e "${YELLOW}⚠️  IVF+HNSW结果文件为空${NC}"
fi

# 生成详细分析报告
echo -e "\n${BLUE}生成对比分析报告...${NC}"

python3 << 'EOF' > "$COMPARISON_DIR/performance_comparison.txt"
import csv
import sys
from datetime import datetime

def analyze_comparison(csv_file):
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except FileNotFoundError:
        print("错误: 找不到CSV文件")
        return

    hnsw_ivf_data = [row for row in data if 'HNSW+IVF' in row['算法']]
    ivf_hnsw_data = [row for row in data if row['算法'] == 'IVF+HNSW']

    print("=" * 80)
    print("                      算法性能对比分析报告")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 基本统计
    print("📊 测试数据统计:")
    print(f"  HNSW+IVF测试配置数: {len(hnsw_ivf_data)}")
    print(f"  IVF+HNSW测试配置数: {len(ivf_hnsw_data)}")
    print()

    # HNSW+IVF 详细结果
    if hnsw_ivf_data:
        print("🔹 HNSW+IVF 优化版详细结果:")
        print("  配置          召回率     延迟(μs)   构建时间(ms)")
        print("  " + "-" * 50)
        for row in hnsw_ivf_data:
            recall = float(row['recall_mean'])
            latency = float(row['latency_us_mean'])
            build_time = int(row['build_time_ms'])
            print(f"  {row['配置']:<12} {recall:.4f}     {latency:>7.1f}    {build_time:>8}")
        print()

    # IVF+HNSW 详细结果
    if ivf_hnsw_data:
        print("🔹 轻量级IVF+HNSW 详细结果:")
        print("  配置          召回率     延迟(μs)   构建时间(ms)")
        print("  " + "-" * 50)
        for row in ivf_hnsw_data:
            recall = float(row['recall_mean'])
            latency = float(row['latency_us_mean'])
            build_time = int(row['build_time_ms'])
            print(f"  {row['配置']:<12} {recall:.4f}     {latency:>7.1f}    {build_time:>8}")
        print()

    # 性能对比
    if hnsw_ivf_data and ivf_hnsw_data:
        print("=" * 80)
        print("                        关键性能指标对比")
        print("=" * 80)

        # 最佳召回率对比
        best_hnsw_recall = max(hnsw_ivf_data, key=lambda x: float(x['recall_mean']))
        best_ivf_recall = max(ivf_hnsw_data, key=lambda x: float(x['recall_mean']))

        h_recall = float(best_hnsw_recall['recall_mean'])
        i_recall = float(best_ivf_recall['recall_mean'])

        print("📈 召回率对比:")
        print(f"  HNSW+IVF 最佳: {h_recall:.4f} ({best_hnsw_recall['配置']})")
        print(f"  IVF+HNSW 最佳: {i_recall:.4f} ({best_ivf_recall['配置']})")
        
        recall_diff = ((h_recall - i_recall) / i_recall) * 100
        if recall_diff > 0:
            print(f"  ✅ HNSW+IVF召回率领先 {recall_diff:.2f}%")
        else:
            print(f"  ✅ IVF+HNSW召回率领先 {-recall_diff:.2f}%")
        print()

        # 最佳延迟对比
        best_hnsw_latency = min(hnsw_ivf_data, key=lambda x: float(x['latency_us_mean']))
        best_ivf_latency = min(ivf_hnsw_data, key=lambda x: float(x['latency_us_mean']))

        h_latency = float(best_hnsw_latency['latency_us_mean'])
        i_latency = float(best_ivf_latency['latency_us_mean'])

        print("⚡ 延迟对比:")
        print(f"  HNSW+IVF 最佳: {h_latency:.1f}μs ({best_hnsw_latency['配置']})")
        print(f"  IVF+HNSW 最佳: {i_latency:.1f}μs ({best_ivf_latency['配置']})")
        
        latency_diff = ((i_latency - h_latency) / i_latency) * 100
        if latency_diff > 0:
            print(f"  ✅ HNSW+IVF延迟降低 {latency_diff:.1f}%")
        else:
            print(f"  ✅ IVF+HNSW延迟降低 {-latency_diff:.1f}%")
        print()

        # 构建时间对比
        avg_hnsw_build = sum(int(row['build_time_ms']) for row in hnsw_ivf_data) / len(hnsw_ivf_data)
        avg_ivf_build = sum(int(row['build_time_ms']) for row in ivf_hnsw_data) / len(ivf_hnsw_data)

        print("🏗️  构建时间对比:")
        print(f"  HNSW+IVF 平均: {avg_hnsw_build:.0f}ms")
        print(f"  IVF+HNSW 平均: {avg_ivf_build:.0f}ms")
        
        build_diff = ((avg_hnsw_build - avg_ivf_build) / avg_ivf_build) * 100
        if build_diff > 0:
            print(f"  ✅ IVF+HNSW构建更快 {build_diff:.1f}%")
        else:
            print(f"  ✅ HNSW+IVF构建更快 {-build_diff:.1f}%")
        print()

        # 综合评估
        print("=" * 80)
        print("                          综合评估与建议")
        print("=" * 80)

        print("💡 算法特点总结:")
        print()
        print("🔸 HNSW+IVF 优化版:")
        print("  • 策略: 先全局HNSW粗筛 → IVF聚类精化")
        print("  • 优势: 高精度搜索，适合召回率要求高的场景")
        print("  • 特点: 需要构建全局HNSW索引，内存和构建时间开销较大")
        print()
        print("🔸 轻量级IVF+HNSW:")
        print("  • 策略: 先IVF聚类粗筛 → 簇内暴力搜索")
        print("  • 优势: 构建快速，内存友好")
        print("  • 特点: 避免复杂的每簇HNSW构建，适合快速部署")
        print()

        print("🎯 应用场景建议:")
        
        if h_recall > i_recall * 1.02:  # 召回率显著更高
            print("  • 高精度检索推荐: HNSW+IVF")
            print("    (召回率优势明显，适合精度敏感应用)")
        
        if i_latency < h_latency * 0.8:  # 延迟显著更低
            print("  • 实时检索推荐: IVF+HNSW")
            print("    (延迟优势明显，适合实时性要求高的应用)")
        
        if avg_ivf_build < avg_hnsw_build * 0.5:  # 构建时间显著更短
            print("  • 快速部署推荐: IVF+HNSW")
            print("    (构建时间短，适合需要频繁更新索引的场景)")

        print()
        print("⚖️  权衡分析:")
        accuracy_score = h_recall / max(h_recall, i_recall)
        speed_score = min(h_latency, i_latency) / h_latency
        build_score = min(avg_hnsw_build, avg_ivf_build) / avg_hnsw_build

        print(f"  HNSW+IVF - 精度: {accuracy_score:.2f}, 速度: {speed_score:.2f}, 构建: {build_score:.2f}")
        
        accuracy_score_i = i_recall / max(h_recall, i_recall)
        speed_score_i = min(h_latency, i_latency) / i_latency
        build_score_i = min(avg_hnsw_build, avg_ivf_build) / avg_ivf_build

        print(f"  IVF+HNSW - 精度: {accuracy_score_i:.2f}, 速度: {speed_score_i:.2f}, 构建: {build_score_i:.2f}")

if __name__ == "__main__":
    analyze_comparison(sys.argv[1])
EOF

python3 - "$COMPARISON_DIR/combined_results.csv"

# 生成Markdown报告
echo -e "\n${BLUE}生成Markdown报告...${NC}"
cat > "$COMPARISON_DIR/README.md" << EOF
# 算法性能对比分析

## 对比信息
- **HNSW+IVF结果**: $HNSW_IVF_DIR
- **IVF+HNSW结果**: $IVF_HNSW_DIR
- **对比时间**: $(date)

## 文件说明
- \`combined_results.csv\`: 合并的测试数据
- \`performance_comparison.txt\`: 详细性能分析报告
- \`README.md\`: 本说明文件

## 快速查看
查看详细分析报告:
\`\`\`bash
cat performance_comparison.txt
\`\`\`

查看CSV数据:
\`\`\`bash
column -t -s',' combined_results.csv
\`\`\`

## 原始数据位置
- HNSW+IVF原始数据: \`$HNSW_IVF_DIR/\`
- IVF+HNSW原始数据: \`$IVF_HNSW_DIR/\`
EOF

# 生成简单的可视化（如果有Python matplotlib）
echo -e "\n${BLUE}尝试生成性能图表...${NC}"
python3 << EOF > /dev/null 2>&1 || echo -e "${YELLOW}⚠️  跳过图表生成 (需要matplotlib)${NC}"
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv('$COMPARISON_DIR/combined_results.csv')
    
    # 创建对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    hnsw_data = df[df['算法'].str.contains('HNSW+IVF')]
    ivf_data = df[df['算法'] == 'IVF+HNSW']
    
    # 召回率对比
    ax1.bar(range(len(hnsw_data)), hnsw_data['recall_mean'], 
           alpha=0.7, label='HNSW+IVF', color='skyblue')
    ax1.bar(range(len(ivf_data)), ivf_data['recall_mean'], 
           alpha=0.7, label='IVF+HNSW', color='lightcoral')
    ax1.set_title('召回率对比')
    ax1.set_ylabel('召回率')
    ax1.legend()
    
    # 延迟对比  
    ax2.bar(range(len(hnsw_data)), hnsw_data['latency_us_mean'], 
           alpha=0.7, label='HNSW+IVF', color='skyblue')
    ax2.bar(range(len(ivf_data)), ivf_data['latency_us_mean'], 
           alpha=0.7, label='IVF+HNSW', color='lightcoral')
    ax2.set_title('延迟对比')
    ax2.set_ylabel('延迟 (μs)')
    ax2.legend()
    
    # 构建时间对比
    ax3.bar(range(len(hnsw_data)), hnsw_data['build_time_ms'], 
           alpha=0.7, label='HNSW+IVF', color='skyblue')
    ax3.bar(range(len(ivf_data)), ivf_data['build_time_ms'], 
           alpha=0.7, label='IVF+HNSW', color='lightcoral')
    ax3.set_title('构建时间对比')
    ax3.set_ylabel('构建时间 (ms)')
    ax3.legend()
    
    # 召回率-延迟权衡
    ax4.scatter(hnsw_data['latency_us_mean'], hnsw_data['recall_mean'], 
               s=100, alpha=0.7, label='HNSW+IVF', color='skyblue')
    ax4.scatter(ivf_data['latency_us_mean'], ivf_data['recall_mean'], 
               s=100, alpha=0.7, label='IVF+HNSW', color='lightcoral')
    ax4.set_xlabel('延迟 (μs)')
    ax4.set_ylabel('召回率')
    ax4.set_title('召回率-延迟权衡')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('$COMPARISON_DIR/performance_charts.png', dpi=300, bbox_inches='tight')
    print("✓ 性能图表已生成")
    
except ImportError:
    pass
except Exception as e:
    print(f"图表生成失败: {e}")
EOF

# 显示对比结果摘要
echo -e "\n${PURPLE}=== 对比分析完成 ===${NC}"
echo -e "结果目录: ${GREEN}$COMPARISON_DIR${NC}"
echo ""
echo -e "${YELLOW}主要文件:${NC}"
echo -e "  📊 ${BLUE}combined_results.csv${NC} - 合并的测试数据"
echo -e "  📋 ${BLUE}performance_comparison.txt${NC} - 详细分析报告"
echo -e "  📄 ${BLUE}README.md${NC} - 说明文档"
if [ -f "$COMPARISON_DIR/performance_charts.png" ]; then
    echo -e "  📈 ${BLUE}performance_charts.png${NC} - 性能图表"
fi

echo ""
echo -e "${YELLOW}快速查看报告:${NC}"
echo -e "  ${CYAN}cat $COMPARISON_DIR/performance_comparison.txt${NC}"

echo ""
echo -e "🎉 ${GREEN}对比分析完成！${NC}" 