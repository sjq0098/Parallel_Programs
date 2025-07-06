#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HNSW算法及其多线程实现的综合可视化分析
分析参数影响和多线程实现效率
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

def load_and_process_data(filename):
    """加载和处理HNSW数据"""
    df = pd.read_csv(filename)
    
    # 解析参数信息
    df['M'] = df['M'].str.extract(r'M=(\d+)').astype(int)
    df['efC'] = df['efC'].str.extract(r'efC=(\d+)').astype(int)
    df['efS'] = df['efS'].str.extract(r'efS=(\d+)').astype(int)
    
    # 提取实现方法
    df['Implementation'] = df['Method'].str.extract(r'HNSW\(([^)]+)\)')
    
    # 计算QPS (每秒查询数)
    df['QPS'] = 1000000 / df['Latency(us)']
    
    # 计算相对于串行版本的加速比
    serial_data = df[df['Implementation'] == 'Serial'].copy()
    serial_data = serial_data.set_index(['M', 'efC', 'efS'])
    
    def calculate_speedup(row):
        try:
            serial_latency = serial_data.loc[(row['M'], row['efC'], row['efS']), 'Latency(us)']
            return serial_latency / row['Latency(us)']
        except:
            return 1.0
    
    df['Speedup'] = df.apply(calculate_speedup, axis=1)
    
    return df

def plot_comprehensive_analysis(df):
    """绘制HNSW综合分析图"""
    # 创建包含6个子图的综合图表
    fig = plt.figure(figsize=(18, 15))
    
    # 图1: 不同M值的召回率-延迟权衡（串行实现，efC=100）
    ax1 = plt.subplot(2, 3, 1)
    serial_data = df[(df['Implementation'] == 'Serial') & (df['efC'] == 100)]
    M_values = sorted(serial_data['M'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(M_values)))
    
    for i, M in enumerate(M_values):
        data = serial_data[serial_data['M'] == M]
        ax1.plot(data['Recall'] * 100, data['Latency(us)'], 'o-', 
                color=colors[i], label=f'M={M}', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Recall (%)', fontsize=12)
    ax1.set_ylabel('Latency (μs)', fontsize=12)
    ax1.set_title('Recall-Latency Tradeoff by M Value\n(Serial, efC=100)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 图2: 不同实现方式的性能比较（M=16, efC=100）
    ax2 = plt.subplot(2, 3, 2)
    comparison_data = df[(df['M'] == 16) & (df['efC'] == 100)]
    implementations = comparison_data['Implementation'].unique()
    impl_colors = ['#1f77b4', '#ff7f0e']
    
    for i, impl in enumerate(implementations):
        data = comparison_data[comparison_data['Implementation'] == impl]
        ax2.plot(data['Recall'] * 100, data['Latency(us)'], 'o-', 
                color=impl_colors[i], label=impl, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Recall (%)', fontsize=12)
    ax2.set_ylabel('Latency (μs)', fontsize=12)
    ax2.set_title('Implementation Comparison\n(M=16, efC=100)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 图3: OpenMP加速比热力图（M=16）
    ax3 = plt.subplot(2, 3, 3)
    openmp_data = df[(df['Implementation'] == 'OpenMP') & (df['M'] == 16)]
    
    if len(openmp_data) > 0:
        speedup_matrix = openmp_data.pivot(index='efC', columns='efS', values='Speedup')
        
        im = ax3.imshow(speedup_matrix.values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=2.0)
        ax3.set_xticks(range(len(speedup_matrix.columns)))
        ax3.set_xticklabels(speedup_matrix.columns)
        ax3.set_yticks(range(len(speedup_matrix.index)))
        ax3.set_yticklabels(speedup_matrix.index)
        ax3.set_xlabel('efS', fontsize=12)
        ax3.set_ylabel('efC', fontsize=12)
        ax3.set_title('OpenMP Speedup Heatmap\n(M=16)', fontsize=12, fontweight='bold')
        
        # 添加数值标注
        for i in range(len(speedup_matrix.index)):
            for j in range(len(speedup_matrix.columns)):
                if not pd.isna(speedup_matrix.iloc[i, j]):
                    text = ax3.text(j, i, f'{speedup_matrix.iloc[i, j]:.2f}x',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3, label='Speedup')
    
    # 图4: efS对延迟的影响（M=16, efC=100）
    ax4 = plt.subplot(2, 3, 4)
    efs_data = df[(df['M'] == 16) & (df['efC'] == 100)]
    
    for impl in efs_data['Implementation'].unique():
        data = efs_data[efs_data['Implementation'] == impl]
        ax4.plot(data['efS'], data['Latency(us)'], 'o-', label=impl, linewidth=2, markersize=6)
    
    ax4.set_xlabel('efS Value', fontsize=12)
    ax4.set_ylabel('Latency (μs)', fontsize=12)
    ax4.set_title('efS Impact on Latency\n(M=16, efC=100)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 图5: 平均加速比对比（不同M值）
    ax5 = plt.subplot(2, 3, 5)
    avg_speedup_by_M = df[df['Implementation'] == 'OpenMP'].groupby('M')['Speedup'].mean()
    
    bars = ax5.bar(avg_speedup_by_M.index.astype(str), avg_speedup_by_M.values, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(avg_speedup_by_M))))
    ax5.set_ylabel('Average Speedup', fontsize=12)
    ax5.set_xlabel('M Value', fontsize=12)
    ax5.set_title('Average Speedup by M Value\n(OpenMP vs Serial)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 添加数值标注
    for bar, value in zip(bars, avg_speedup_by_M.values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 图6: efC对召回率的影响（M=16, efS=200）
    ax6 = plt.subplot(2, 3, 6)
    efc_data = df[(df['M'] == 16) & (df['efS'] == 200)]
    
    for impl in efc_data['Implementation'].unique():
        data = efc_data[efc_data['Implementation'] == impl]
        ax6.plot(data['efC'], data['Recall'] * 100, 'o-', label=impl, linewidth=2, markersize=6)
    
    ax6.set_xlabel('efC Value', fontsize=12)
    ax6.set_ylabel('Recall (%)', fontsize=12)
    ax6.set_title('efC Impact on Recall\n(M=16, efS=200)', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/hnsw_comprehensive_analysis_en.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_parameter_sensitivity(df):
    """绘制参数敏感性分析"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1: M值对性能的影响（efC=100, efS=200）
    m_effect_data = df[(df['efC'] == 100) & (df['efS'] == 200)]
    
    for impl in m_effect_data['Implementation'].unique():
        data = m_effect_data[m_effect_data['Implementation'] == impl]
        ax1.plot(data['M'], data['Latency(us)'], 'o-', label=impl, linewidth=2, markersize=6)
    
    ax1.set_xlabel('M Value', fontsize=12)
    ax1.set_ylabel('Latency (μs)', fontsize=12)
    ax1.set_title('M Value Impact on Latency\n(efC=100, efS=200)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 图2: M值对召回率的影响（efC=100, efS=200）
    for impl in m_effect_data['Implementation'].unique():
        data = m_effect_data[m_effect_data['Implementation'] == impl]
        ax2.plot(data['M'], data['Recall'] * 100, 'o-', label=impl, linewidth=2, markersize=6)
    
    ax2.set_xlabel('M Value', fontsize=12)
    ax2.set_ylabel('Recall (%)', fontsize=12)
    ax2.set_title('M Value Impact on Recall\n(efC=100, efS=200)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: efS和efC的联合效应热力图（串行实现，M=16）
    heatmap_data = df[(df['Implementation'] == 'Serial') & (df['M'] == 16)]
    recall_matrix = heatmap_data.pivot(index='efC', columns='efS', values='Recall')
    
    im3 = ax3.imshow(recall_matrix.values, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(range(len(recall_matrix.columns)))
    ax3.set_xticklabels(recall_matrix.columns)
    ax3.set_yticks(range(len(recall_matrix.index)))
    ax3.set_yticklabels(recall_matrix.index)
    ax3.set_xlabel('efS Value', fontsize=12)
    ax3.set_ylabel('efC Value', fontsize=12)
    ax3.set_title('Recall Heatmap\n(Serial, M=16)', fontsize=12, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(recall_matrix.index)):
        for j in range(len(recall_matrix.columns)):
            if not pd.isna(recall_matrix.iloc[i, j]):
                text = ax3.text(j, i, f'{recall_matrix.iloc[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im3, ax=ax3, label='Recall')
    
    # 图4: 延迟热力图（串行实现，M=16）
    latency_matrix = heatmap_data.pivot(index='efC', columns='efS', values='Latency(us)')
    
    im4 = ax4.imshow(latency_matrix.values, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(latency_matrix.columns)))
    ax4.set_xticklabels(latency_matrix.columns)
    ax4.set_yticks(range(len(latency_matrix.index)))
    ax4.set_yticklabels(latency_matrix.index)
    ax4.set_xlabel('efS Value', fontsize=12)
    ax4.set_ylabel('efC Value', fontsize=12)
    ax4.set_title('Latency Heatmap (μs)\n(Serial, M=16)', fontsize=12, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(latency_matrix.index)):
        for j in range(len(latency_matrix.columns)):
            if not pd.isna(latency_matrix.iloc[i, j]):
                text = ax4.text(j, i, f'{int(latency_matrix.iloc[i, j])}',
                               ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im4, ax=ax4, label='Latency (μs)')
    
    plt.tight_layout()
    plt.savefig('plots/hnsw_parameter_sensitivity_en.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_performance_summary(df):
    """生成性能总结"""
    print("\n" + "="*80)
    print("HNSW Algorithm Multi-threading Implementation Performance Analysis Summary")
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
        print(f"  Highest Recall: {best_recall['Recall']:.4f} (M={best_recall['M']}, efC={best_recall['efC']}, efS={best_recall['efS']}, latency={best_recall['Latency(us)']:.0f}μs)")
        
        # 最低延迟配置
        best_latency = impl_data.loc[impl_data['Latency(us)'].idxmin()]
        print(f"  Lowest Latency: {best_latency['Latency(us)']:.0f}μs (M={best_latency['M']}, efC={best_latency['efC']}, efS={best_latency['efS']}, recall={best_latency['Recall']:.4f})")
        
        if impl != 'Serial':
            best_speedup = impl_data.loc[impl_data['Speedup'].idxmax()]
            print(f"  Highest Speedup: {best_speedup['Speedup']:.2f}x (M={best_speedup['M']}, efC={best_speedup['efC']}, efS={best_speedup['efS']})")
    
    # 参数影响分析
    print("\n3. Parameter Impact Analysis:")
    print(f"  M range: {df['M'].min()} - {df['M'].max()}")
    print(f"  efC range: {df['efC'].min()} - {df['efC'].max()}")
    print(f"  efS range: {df['efS'].min()} - {df['efS'].max()}")
    print(f"  Recall range: {df['Recall'].min():.3f} - {df['Recall'].max():.3f}")
    print(f"  Latency range: {df['Latency(us)'].min():.0f} - {df['Latency(us)'].max():.0f} μs")
    
    # 最优配置推荐
    print("\n4. Optimal Configuration Recommendations:")
    
    # 高召回率场景
    high_recall_configs = df[df['Recall'] >= 0.999].nsmallest(3, 'Latency(us)')
    print("\n  High Recall Scenarios (Recall ≥ 99.9%):")
    for i, (_, config) in enumerate(high_recall_configs.iterrows()):
        print(f"    {i+1}. {config['Implementation']}: M={config['M']}, efC={config['efC']}, efS={config['efS']} - "
              f"Recall={config['Recall']:.4f}, Latency={config['Latency(us)']:.0f}μs")
    
    # 低延迟场景
    low_latency_configs = df[df['Recall'] >= 0.95].nsmallest(3, 'Latency(us)')
    print("\n  Low Latency Scenarios (Recall ≥ 95%):")
    for i, (_, config) in enumerate(low_latency_configs.iterrows()):
        print(f"    {i+1}. {config['Implementation']}: M={config['M']}, efC={config['efC']}, efS={config['efS']} - "
              f"Recall={config['Recall']:.4f}, Latency={config['Latency(us)']:.0f}μs")

def main():
    """主函数"""
    print("Starting HNSW algorithm multi-threading implementation visualization analysis...")
    
    # 加载数据
    df = load_and_process_data('hnsw_results.csv')
    print(f"Data loading completed, total {len(df)} records")
    
    # 创建图片目录
    import os
    os.makedirs('plots', exist_ok=True)
    
    # 生成综合分析图
    print("\nGenerating comprehensive analysis charts...")
    plot_comprehensive_analysis(df)
    
    print("Generating parameter sensitivity analysis...")
    plot_parameter_sensitivity(df)
    
    # 生成性能总结
    generate_performance_summary(df)
    
    print("\nAnalysis completed! Charts saved to plots/ directory")

if __name__ == "__main__":
    main() 