#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IVF算法及其多线程实现的简化可视化分析
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
sns.set_style("whitegrid")

def load_data():
    """直接创建数据"""
    # 根据观察到的数据模式直接构建数据
    implementations = []
    nlists = []
    nprobes = []
    recalls = []
    latencies = []
    
    # 串行实现数据 (前28行)
    serial_data = [
        # nlist=64
        (64, 4, 0.879, 1494), (64, 8, 0.953, 3157), (64, 12, 0.9785, 4576), (64, 16, 0.9885, 6161),
        (64, 20, 0.992, 7170), (64, 24, 0.995, 8978), (64, 32, 0.999, 12172),
        # nlist=128
        (128, 4, 0.8505, 745), (128, 8, 0.936, 1703), (128, 12, 0.9645, 2405), (128, 16, 0.976, 3046),
        (128, 20, 0.983, 3998), (128, 24, 0.9885, 5043), (128, 32, 0.9935, 6788),
        # nlist=256
        (256, 4, 0.817, 406), (256, 8, 0.914, 959), (256, 12, 0.948, 1261), (256, 16, 0.9635, 1792),
        (256, 20, 0.974, 2260), (256, 24, 0.981, 2742), (256, 32, 0.989, 3630),
        # nlist=512
        (512, 4, 0.739, 273), (512, 8, 0.858, 483), (512, 12, 0.9085, 746), (512, 16, 0.933, 1104),
        (512, 20, 0.9495, 1130), (512, 24, 0.9575, 1472), (512, 32, 0.973, 2139),
    ]
    
    # OpenMP实现数据 (接下来28行)
    openmp_data = [
        # nlist=64
        (64, 4, 0.879, 1281), (64, 8, 0.953, 891), (64, 12, 0.9785, 1576), (64, 16, 0.9885, 1479),
        (64, 20, 0.992, 2545), (64, 24, 0.995, 3168), (64, 32, 0.999, 3010),
        # nlist=128
        (128, 4, 0.8505, 450), (128, 8, 0.936, 518), (128, 12, 0.9645, 674), (128, 16, 0.976, 884),
        (128, 20, 0.983, 1221), (128, 24, 0.9885, 1275), (128, 32, 0.9935, 1468),
        # nlist=256
        (256, 4, 0.817, 312), (256, 8, 0.914, 387), (256, 12, 0.948, 466), (256, 16, 0.9635, 575),
        (256, 20, 0.974, 585), (256, 24, 0.981, 594), (256, 32, 0.989, 988),
        # nlist=512
        (512, 4, 0.739, 406), (512, 8, 0.858, 378), (512, 12, 0.9085, 348), (512, 16, 0.933, 599),
        (512, 20, 0.9495, 418), (512, 24, 0.9575, 460), (512, 32, 0.973, 507),
    ]
    
    # Pthread实现数据 (最后28行)
    pthread_data = [
        # nlist=64
        (64, 4, 0.879, 1231), (64, 8, 0.953, 1330), (64, 12, 0.9785, 1484), (64, 16, 0.9885, 1685),
        (64, 20, 0.992, 1966), (64, 24, 0.995, 2367), (64, 32, 0.999, 2658),
        # nlist=128
        (128, 4, 0.8505, 997), (128, 8, 0.936, 1102), (128, 12, 0.9645, 1196), (128, 16, 0.976, 1362),
        (128, 20, 0.983, 1515), (128, 24, 0.9885, 1584), (128, 32, 0.9935, 1752),
        # nlist=256
        (256, 4, 0.817, 903), (256, 8, 0.914, 987), (256, 12, 0.948, 1065), (256, 16, 0.9635, 1100),
        (256, 20, 0.974, 1134), (256, 24, 0.981, 1260), (256, 32, 0.989, 1424),
        # nlist=512
        (512, 4, 0.739, 869), (512, 8, 0.858, 908), (512, 12, 0.9085, 958), (512, 16, 0.933, 985),
        (512, 20, 0.9495, 1054), (512, 24, 0.9575, 1070), (512, 32, 0.973, 1109),
    ]
    
    # 组合数据
    for impl, data in [('Serial', serial_data), ('OpenMP', openmp_data), ('Pthread', pthread_data)]:
        for nlist, nprobe, recall, latency in data:
            implementations.append(impl)
            nlists.append(nlist)
            nprobes.append(nprobe)
            recalls.append(recall)
            latencies.append(latency)
    
    df = pd.DataFrame({
        'Implementation': implementations,
        'nlist': nlists,
        'nprobe': nprobes,
        'Recall': recalls,
        'Latency(us)': latencies
    })
    
    # 计算QPS和加速比
    df['QPS'] = 1000000 / df['Latency(us)']
    
    # 计算加速比
    speedups = []
    for _, row in df.iterrows():
        if row['Implementation'] == 'Serial':
            speedups.append(1.0)
        else:
            # 找到对应的串行版本
            serial_row = df[(df['Implementation'] == 'Serial') & 
                          (df['nlist'] == row['nlist']) & 
                          (df['nprobe'] == row['nprobe'])]
            if len(serial_row) > 0:
                speedup = serial_row.iloc[0]['Latency(us)'] / row['Latency(us)']
                speedups.append(speedup)
            else:
                speedups.append(1.0)
    
    df['Speedup'] = speedups
    
    return df

def plot_comprehensive_analysis(df):
    """绘制综合分析图"""
    # 创建包含6个子图的综合图表
    fig = plt.figure(figsize=(18, 15))
    
    # 图1: 召回率-延迟权衡 (不同nlist值，串行实现)
    ax1 = plt.subplot(2, 3, 1)
    serial_data = df[df['Implementation'] == 'Serial']
    nlist_values = sorted(serial_data['nlist'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(nlist_values)))
    
    for i, nlist in enumerate(nlist_values):
        data = serial_data[serial_data['nlist'] == nlist]
        ax1.plot(data['Recall'] * 100, data['Latency(us)'], 'o-', 
                color=colors[i], label=f'nlist={nlist}', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Recall (%)', fontsize=12)
    ax1.set_ylabel('Latency (μs)', fontsize=12)
    ax1.set_title('Recall-Latency Tradeoff\n(Serial Implementation)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 图2: 不同实现方式的性能比较
    ax2 = plt.subplot(2, 3, 2)
    nlist_256_data = df[df['nlist'] == 256]
    implementations = nlist_256_data['Implementation'].unique()
    impl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, impl in enumerate(implementations):
        data = nlist_256_data[nlist_256_data['Implementation'] == impl]
        ax2.plot(data['Recall'] * 100, data['Latency(us)'], 'o-', 
                color=impl_colors[i], label=impl, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Recall (%)', fontsize=12)
    ax2.set_ylabel('Latency (μs)', fontsize=12)
    ax2.set_title('Implementation Comparison\n(nlist=256)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 图3: OpenMP加速比热力图
    ax3 = plt.subplot(2, 3, 3)
    openmp_data = df[df['Implementation'] == 'OpenMP']
    speedup_matrix = openmp_data.pivot(index='nlist', columns='nprobe', values='Speedup')
    
    im = ax3.imshow(speedup_matrix.values, cmap='RdYlGn', aspect='auto')
    ax3.set_xticks(range(len(speedup_matrix.columns)))
    ax3.set_xticklabels(speedup_matrix.columns)
    ax3.set_yticks(range(len(speedup_matrix.index)))
    ax3.set_yticklabels(speedup_matrix.index)
    ax3.set_xlabel('nprobe', fontsize=12)
    ax3.set_ylabel('nlist', fontsize=12)
    ax3.set_title('OpenMP Speedup Heatmap', fontsize=12, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(speedup_matrix.index)):
        for j in range(len(speedup_matrix.columns)):
            text = ax3.text(j, i, f'{speedup_matrix.iloc[i, j]:.1f}x',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax3, label='Speedup')
    
    # 图4: nprobe对延迟的影响
    ax4 = plt.subplot(2, 3, 4)
    fixed_nlist_data = df[df['nlist'] == 256]
    
    for impl in fixed_nlist_data['Implementation'].unique():
        data = fixed_nlist_data[fixed_nlist_data['Implementation'] == impl]
        ax4.plot(data['nprobe'], data['Latency(us)'], 'o-', label=impl, linewidth=2, markersize=6)
    
    ax4.set_xlabel('nprobe', fontsize=12)
    ax4.set_ylabel('Latency (μs)', fontsize=12)
    ax4.set_title('nprobe Impact on Latency\n(nlist=256)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 图5: 平均加速比对比
    ax5 = plt.subplot(2, 3, 5)
    avg_speedup = df[df['Implementation'] != 'Serial'].groupby('Implementation')['Speedup'].mean()
    bars = ax5.bar(avg_speedup.index, avg_speedup.values, color=['#ff7f0e', '#2ca02c'])
    ax5.set_ylabel('Average Speedup', fontsize=12)
    ax5.set_title('Average Speedup Comparison', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 添加数值标注
    for bar, value in zip(bars, avg_speedup.values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 图6: nlist对召回率的影响
    ax6 = plt.subplot(2, 3, 6)
    fixed_nprobe_data = df[df['nprobe'] == 16]
    
    for impl in fixed_nprobe_data['Implementation'].unique():
        data = fixed_nprobe_data[fixed_nprobe_data['Implementation'] == impl]
        ax6.plot(data['nlist'], data['Recall'] * 100, 'o-', label=impl, linewidth=2, markersize=6)
    
    ax6.set_xlabel('nlist', fontsize=12)
    ax6.set_ylabel('Recall (%)', fontsize=12)
    ax6.set_title('nlist Impact on Recall\n(nprobe=16)', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/ivf_comprehensive_analysis_en.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_performance_summary(df):
    """生成性能总结"""
    print("\n" + "="*80)
    print("IVF Algorithm Multi-threading Implementation Performance Analysis Summary")
    print("="*80)
    
    # 计算总体统计
    summary_stats = df.groupby('Implementation').agg({
        'Speedup': ['mean', 'max', 'min'],
        'Latency(us)': ['mean', 'min'],
        'Recall': ['mean', 'max']
    }).round(3)
    
    print("\n1. Overall Performance Statistics:")
    print(summary_stats)
    
    # 最佳配置分析
    print("\n2. Best Configuration Analysis:")
    for impl in df['Implementation'].unique():
        impl_data = df[df['Implementation'] == impl]
        
        # 最高召回率配置
        best_recall = impl_data.loc[impl_data['Recall'].idxmax()]
        print(f"\n{impl} Implementation:")
        print(f"  Highest Recall: {best_recall['Recall']:.4f} (nlist={best_recall['nlist']}, nprobe={best_recall['nprobe']}, latency={best_recall['Latency(us)']:.0f}μs)")
        
        # 最低延迟配置
        best_latency = impl_data.loc[impl_data['Latency(us)'].idxmin()]
        print(f"  Lowest Latency: {best_latency['Latency(us)']:.0f}μs (nlist={best_latency['nlist']}, nprobe={best_latency['nprobe']}, recall={best_latency['Recall']:.4f})")
        
        if impl != 'Serial':
            best_speedup = impl_data.loc[impl_data['Speedup'].idxmax()]
            print(f"  Highest Speedup: {best_speedup['Speedup']:.2f}x (nlist={best_speedup['nlist']}, nprobe={best_speedup['nprobe']})")

def main():
    """主函数"""
    print("Starting IVF algorithm multi-threading implementation visualization analysis...")
    
    # 加载数据
    df = load_data()
    print(f"Data loading completed, total {len(df)} records")
    
    # 创建图片目录
    import os
    os.makedirs('plots', exist_ok=True)
    
    # 生成综合分析图
    print("\nGenerating comprehensive analysis charts...")
    plot_comprehensive_analysis(df)
    
    # 生成性能总结
    generate_performance_summary(df)
    
    print("\nAnalysis completed! Charts saved to plots/ directory")

if __name__ == "__main__":
    main() 