 #!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
import sys

# 检测当前操作系统
os_system = platform.system()
print(f"当前操作系统: {os_system}")

# 设置支持中文的字体
if os_system == 'Windows':
    # 设置控制台编码为UTF-8
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    
    # 尝试加载Windows中文字体
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',  # 黑体
        'C:/Windows/Fonts/msyh.ttf',    # 微软雅黑
        'C:/Windows/Fonts/simsun.ttc',  # 宋体
    ]
    
    font_found = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                mpl.font_manager.fontManager.addfont(font_path)
                mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
                print(f"已加载字体: {font_path}")
                font_found = True
                break
            except Exception as e:
                print(f"加载字体失败: {e}")
    
    if not font_found:
        print("未找到中文字体，使用默认字体")
        mpl.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
else:
    # 其他系统尝试使用通用字体
    mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'sans-serif']

# 确保负号正确显示
mpl.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = 'results/report_figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Neon结果数据
data = [
    {"algorithm": "暴力串行", "recall": 0.99995, "latency": 16147.30, "speedup": 1.00},
    {"algorithm": "暴力并行（Neon）", "recall": 0.99995, "latency": 1015.31, "speedup": 15.90},
    {"algorithm": "SQ_Neon(openmp)", "recall": 0.97965, "latency": 997.68, "speedup": 16.18},
    {"algorithm": "SQ_Neon", "recall": 0.97965, "latency": 7909.72, "speedup": 2.04},
    {"algorithm": "PQ8", "recall": 0.23865, "latency": 1210.82, "speedup": 13.34},
    {"algorithm": "PQ16+rerank(openmp)", "recall": 0.97565, "latency": 580.73, "speedup": 27.80},
    {"algorithm": "PQ16+rerank", "recall": 0.92711, "latency": 2683.44, "speedup": 6.02},
]

# 使用英文名称版本（防止中文显示问题）
data_en = [
    {"algorithm": "Serial Brute Force", "recall": 0.99995, "latency": 16147.30, "speedup": 1.00},
    {"algorithm": "Parallel Brute Force (Neon)", "recall": 0.99995, "latency": 1015.31, "speedup": 15.90},
    {"algorithm": "SQ_Neon(openmp)", "recall": 0.97965, "latency": 997.68, "speedup": 16.18},
    {"algorithm": "SQ_Neon", "recall": 0.97965, "latency": 7909.72, "speedup": 2.04},
    {"algorithm": "PQ8", "recall": 0.23865, "latency": 1210.82, "speedup": 13.34},
    {"algorithm": "PQ16+rerank(openmp)", "recall": 0.97565, "latency": 580.73, "speedup": 27.80},
    {"algorithm": "PQ16+rerank", "recall": 0.92711, "latency": 2683.44, "speedup": 6.02},
]

# 为可视化提取数据
algorithms = [item["algorithm"] for item in data]
recalls = [item["recall"] for item in data]
latencies = [item["latency"] for item in data]
speedups = [item["speedup"] for item in data]

# 英文版本数据
algorithms_en = [item["algorithm"] for item in data_en]
recalls_en = [item["recall"] for item in data_en]
latencies_en = [item["latency"] for item in data_en]
speedups_en = [item["speedup"] for item in data_en]

# 可视化1：延迟对比（毫秒单位）
def plot_latency_comparison():
    # 中文版
    try:
        plt.figure(figsize=(12, 7))
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f1c40f', '#1abc9c', '#34495e']
        
        # 按从小到大排序
        sorted_indices = np.argsort(latencies)
        sorted_algorithms = [algorithms[i] for i in sorted_indices]
        sorted_latencies = [latencies[i] for i in sorted_indices]
        sorted_colors = [colors[i % len(colors)] for i in sorted_indices]
        
        bars = plt.barh(sorted_algorithms, sorted_latencies, color=sorted_colors, alpha=0.8)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 200, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f} μs', va='center')
        
        plt.xlabel('延迟 (微秒)', fontsize=14)
        plt.title('Neon加速ANN算法延迟对比', fontsize=16)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/neon_latency_comparison.pdf', bbox_inches='tight')
        plt.savefig(f'{output_dir}/neon_latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成延迟对比图 (中文版)")
    except Exception as e:
        print(f"生成中文版延迟对比图失败: {e}")
    
    # 英文版
    try:
        plt.figure(figsize=(12, 7))
        sorted_indices = np.argsort(latencies_en)
        sorted_algorithms = [algorithms_en[i] for i in sorted_indices]
        sorted_latencies = [latencies_en[i] for i in sorted_indices]
        sorted_colors = [colors[i % len(colors)] for i in sorted_indices]
        
        bars = plt.barh(sorted_algorithms, sorted_latencies, color=sorted_colors, alpha=0.8)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 200, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f} μs', va='center')
        
        plt.xlabel('Latency (μs)', fontsize=14)
        plt.title('Neon Accelerated ANN Algorithm Latency Comparison', fontsize=16)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/neon_latency_comparison_en.pdf', bbox_inches='tight')
        plt.savefig(f'{output_dir}/neon_latency_comparison_en.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成延迟对比图 (英文版)")
    except Exception as e:
        print(f"生成英文版延迟对比图失败: {e}")

# 可视化2：加速比对比
def plot_speedup_comparison():
    # 中文版
    try:
        plt.figure(figsize=(12, 7))
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f1c40f', '#1abc9c', '#34495e']
        
        # 按从大到小排序
        sorted_indices = np.argsort(speedups)[::-1]
        sorted_algorithms = [algorithms[i] for i in sorted_indices]
        sorted_speedups = [speedups[i] for i in sorted_indices]
        sorted_colors = [colors[i % len(colors)] for i in sorted_indices]
        
        bars = plt.barh(sorted_algorithms, sorted_speedups, color=sorted_colors, alpha=0.8)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}x', va='center')
        
        plt.xlabel('加速比 (相对于暴力串行)', fontsize=14)
        plt.title('Neon加速ANN算法性能提升对比', fontsize=16)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/neon_speedup_comparison.pdf', bbox_inches='tight')
        plt.savefig(f'{output_dir}/neon_speedup_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成加速比对比图 (中文版)")
    except Exception as e:
        print(f"生成中文版加速比对比图失败: {e}")
    
    # 英文版
    try:
        plt.figure(figsize=(12, 7))
        sorted_indices = np.argsort(speedups_en)[::-1]
        sorted_algorithms = [algorithms_en[i] for i in sorted_indices]
        sorted_speedups = [speedups_en[i] for i in sorted_indices]
        sorted_colors = [colors[i % len(colors)] for i in sorted_indices]
        
        bars = plt.barh(sorted_algorithms, sorted_speedups, color=sorted_colors, alpha=0.8)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}x', va='center')
        
        plt.xlabel('Speedup (relative to Serial Brute Force)', fontsize=14)
        plt.title('Performance Improvement of Neon Accelerated ANN Algorithms', fontsize=16)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/neon_speedup_comparison_en.pdf', bbox_inches='tight')
        plt.savefig(f'{output_dir}/neon_speedup_comparison_en.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成加速比对比图 (英文版)")
    except Exception as e:
        print(f"生成英文版加速比对比图失败: {e}")

# 可视化3：召回率-延迟散点图
def plot_recall_latency_scatter():
    # 中文版
    try:
        plt.figure(figsize=(12, 7))
        
        # 散点大小基于加速比
        sizes = [s * 20 for s in speedups]
        
        scatter = plt.scatter(latencies, recalls, s=sizes, c=speedups, cmap='viridis', 
                             alpha=0.7, edgecolors='black', linewidths=1)
        
        # 添加算法标签
        for i, alg in enumerate(algorithms):
            plt.annotate(alg, (latencies[i], recalls[i]), 
                        textcoords="offset points", 
                        xytext=(5,5), 
                        ha='left')
        
        plt.xscale('log')  # 对数坐标更适合显示不同量级的延迟
        plt.xlabel('延迟 (微秒，对数坐标)', fontsize=14)
        plt.ylabel('召回率', fontsize=14)
        plt.title('Neon加速ANN算法性能权衡：召回率 vs 延迟', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # 添加目标召回率线
        plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='目标召回率 (0.9)')
        
        # 添加色条，表示加速比
        cbar = plt.colorbar(scatter)
        cbar.set_label('加速比', fontsize=12)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/neon_recall_latency_scatter.pdf', bbox_inches='tight')
        plt.savefig(f'{output_dir}/neon_recall_latency_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成召回率-延迟散点图 (中文版)")
    except Exception as e:
        print(f"生成中文版召回率-延迟散点图失败: {e}")
    
    # 英文版
    try:
        plt.figure(figsize=(12, 7))
        
        sizes = [s * 20 for s in speedups_en]
        
        scatter = plt.scatter(latencies_en, recalls_en, s=sizes, c=speedups_en, cmap='viridis', 
                             alpha=0.7, edgecolors='black', linewidths=1)
        
        for i, alg in enumerate(algorithms_en):
            plt.annotate(alg, (latencies_en[i], recalls_en[i]), 
                        textcoords="offset points", 
                        xytext=(5,5), 
                        ha='left')
        
        plt.xscale('log')
        plt.xlabel('Latency (μs, log scale)', fontsize=14)
        plt.ylabel('Recall', fontsize=14)
        plt.title('Neon Accelerated ANN Algorithm Performance Tradeoff: Recall vs Latency', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target Recall (0.9)')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Speedup', fontsize=12)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/neon_recall_latency_scatter_en.pdf', bbox_inches='tight')
        plt.savefig(f'{output_dir}/neon_recall_latency_scatter_en.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成召回率-延迟散点图 (英文版)")
    except Exception as e:
        print(f"生成英文版召回率-延迟散点图失败: {e}")

# 生成总结分析文本
def generate_analysis_latex():
    with open(f'{output_dir}/neon_analysis.tex', 'w', encoding='utf-8') as f:
        f.write("\\section{Neon加速算法分析}\n\n")
        
        f.write("\\subsection{性能概述}\n\n")
        f.write("本节对使用ARM Neon指令集加速的ANN算法进行性能评估。Neon是ARM架构的SIMD（单指令多数据）扩展，能够并行处理多个数据，特别适合于向量距离计算等操作。实验结果显示，Neon指令集能够显著提升ANN算法的性能，特别是当结合OpenMP多线程技术时。\n\n")
        
        f.write("\\subsection{延迟和加速比分析}\n\n")
        
        # 找出最佳算法
        min_latency_idx = np.argmin(latencies)
        max_speedup_idx = np.argmax(speedups)
        max_recall_idx = np.argmax(recalls)
        
        # 找出在保证高召回率(>=0.9)的情况下性能最好的算法
        high_recall_indices = [i for i, r in enumerate(recalls) if r >= 0.9]
        best_high_recall_idx = min(high_recall_indices, key=lambda i: latencies[i])
        
        f.write("实验测试了7种不同配置的算法，包括串行和并行的暴力搜索、标量量化（SQ）以及乘积量化（PQ）方法。从实验结果中可以观察到以下几点：\n\n")
        
        f.write("\\begin{itemize}\n")
        f.write(f"\\item 性能最高的算法是{algorithms[min_latency_idx]}，其查询延迟仅为{latencies[min_latency_idx]:.2f}微秒，相对于基准串行算法的加速比达到{speedups[min_latency_idx]:.2f}倍。\n")
        f.write(f"\\item 在保证高召回率（$\\geq 0.9$）的条件下，{algorithms[best_high_recall_idx]}表现最佳，其延迟为{latencies[best_high_recall_idx]:.2f}微秒，加速比为{speedups[best_high_recall_idx]:.2f}倍。\n")
        f.write("\\item Neon指令集对暴力搜索的加速效果显著，将延迟从16147.30微秒降至1015.31微秒，提速约16倍。\n")
        f.write("\\item OpenMP多线程结合Neon指令集可以进一步提升性能，如SQ\\_Neon(openmp)比SQ\\_Neon快约8倍。\n")
        f.write("\\item PQ16+rerank方法结合OpenMP能够在保持高召回率的同时提供最佳性能，是推荐的最优配置。\n")
        f.write("\\end{itemize}\n\n")
        
        f.write("\\subsection{应用建议}\n\n")
        f.write("基于实验结果，我们提出以下应用建议：\n\n")
        
        f.write("\\begin{itemize}\n")
        f.write("\\item 对于需要100\\%精确结果的应用，推荐使用Neon加速的并行暴力搜索，可获得约16倍加速。\n")
        f.write("\\item 对于大多数实际应用，PQ16+rerank(openmp)提供了最佳的性能-精度平衡点，召回率高达97.6\\%，同时比基准算法快近28倍。\n")
        f.write("\\item 在有OpenMP支持的环境中，应优先使用支持并行的实现，因为它们普遍比相应的串行版本快3-8倍。\n")
        f.write("\\item 简单的SQ量化方法结合Neon和OpenMP也能提供很好的性能，是一个低复杂度但高效的选择。\n")
        f.write("\\end{itemize}\n\n")
        
        f.write("综上所述，ARM Neon指令集为ANN算法提供了显著的性能提升，特别是当结合OpenMP多线程技术时。在实际应用中，应根据精度需求和硬件环境选择合适的算法配置。\n")
    
    print(f"已生成Neon分析报告: {output_dir}/neon_analysis.tex")

# 主函数
def main():
    try:
        print("开始生成Neon加速ANN算法性能分析图表...")
        
        # 生成各种可视化
        plot_latency_comparison()
        plot_speedup_comparison()
        plot_recall_latency_scatter()
        
        # 生成LaTeX分析文本
        generate_analysis_latex()
        
        print(f"图表生成完成！所有图表和分析已保存到 {output_dir} 目录")
        
        # 保存数据供参考
        import pandas as pd
        pd.DataFrame(data).to_csv(f'{output_dir}/neon_data.csv', index=False, encoding='utf-8-sig')
        
    except Exception as e:
        import traceback
        print(f"生成报告图表时出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()