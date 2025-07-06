#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IVF算法及其多线程实现的综合可视化分析
分析参数影响和多线程实现效率
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
sns.set_style("whitegrid")

def load_and_process_data(filename):
    """加载和处理数据"""
    df = pd.read_csv(filename)
    
    # 数据格式分析：Method列包含方法名和nlist，Configuration列包含nprobe
    # 需要重新构建正确的数据结构
    
    # 读取原始数据并重新整理
    data_rows = []
    current_method = None
    current_nlist = None
    
    for idx, row in df.iterrows():
        # 检查是否是新的方法和nlist组合
        if 'IVF-Flat' in str(row.iloc[0]):
            # 这是方法行，提取方法和nlist
            method_part = str(row.iloc[0])
            if 'IVF-Flat(Serial)' in method_part:
                current_method = '串行'
            elif 'IVF-Flat(OpenMP)' in method_part:
                current_method = 'OpenMP'
            elif 'IVF-Flat(pthread)' in method_part:
                current_method = 'Pthread'
            else:
                current_method = '串行'  # 默认
                
            # 提取nlist
            if 'nlist=' in method_part:
                nlist_match = pd.Series([method_part]).str.extract(r'nlist=(\d+)')
                if not nlist_match.iloc[0, 0] is None:
                    current_nlist = int(nlist_match.iloc[0, 0])
        else:
            # 这是nlist值行
            if 'nlist=' in str(row.iloc[0]):
                nlist_match = pd.Series([str(row.iloc[0])]).str.extract(r'nlist=(\d+)')
                if not nlist_match.iloc[0, 0] is None:
                    current_nlist = int(nlist_match.iloc[0, 0])
                    current_method = '串行'  # 如果没有明确方法，默认为串行
        
        # 提取nprobe和性能数据
        if 'nprobe=' in str(row.iloc[1]):
            nprobe_match = pd.Series([str(row.iloc[1])]).str.extract(r'nprobe=(\d+)')
            if not nprobe_match.iloc[0, 0] is None:
                nprobe = int(nprobe_match.iloc[0, 0])
                recall = float(row.iloc[2])
                latency = float(row.iloc[3])
                
                data_rows.append({
                    'Implementation': current_method,
                    'nlist': current_nlist,
                    'nprobe': nprobe,
                    'Recall': recall,
                    'Latency(us)': latency
                })
    
    # 创建新的DataFrame
    df = pd.DataFrame(data_rows)
    
    # 如果没有找到数据，尝试另一种解析方法
    if len(df) == 0:
        # 直接从原始数据重新读取
        original_df = pd.read_csv(filename)
        
        # 重新构建数据
        implementations = []
        nlists = []
        nprobes = []
        recalls = []
        latencies = []
        
        for i in range(0, len(original_df), 28):  # 每28行为一组（4个nlist × 7个nprobe）
            if i + 27 < len(original_df):
                # 确定实现方法
                if i == 0:
                    impl = '串行'
                elif i == 28:
                    impl = 'OpenMP'
                elif i == 56:
                    impl = 'Pthread'
                else:
                    impl = '串行'
                
                # 处理每组数据
                for j in range(28):
                    row_idx = i + j
                    if row_idx < len(original_df):
                        row = original_df.iloc[row_idx]
                        
                        # 确定nlist
                        nlist_row = j // 7  # 每7行一个nlist
                        nlist_values = [64, 128, 256, 512]
                        nlist = nlist_values[nlist_row]
                        
                        # 确定nprobe
                        nprobe_idx = j % 7
                        nprobe_values = [4, 8, 12, 16, 20, 24, 32]
                        nprobe = nprobe_values[nprobe_idx]
                        
                        implementations.append(impl)
                        nlists.append(nlist)
                        nprobes.append(nprobe)
                        recalls.append(float(row.iloc[2]))
                        latencies.append(float(row.iloc[3]))
        
        df = pd.DataFrame({
            'Implementation': implementations,
            'nlist': nlists,
            'nprobe': nprobes,
            'Recall': recalls,
            'Latency(us)': latencies
        })
    
    # 计算QPS (每秒查询数)
    df['QPS'] = 1000000 / df['Latency(us)']
    
    # 计算相对于串行版本的加速比
    serial_data = df[df['Implementation'] == '串行'].copy()
    serial_data = serial_data.set_index(['nlist', 'nprobe'])
    
    def calculate_speedup(row):
        try:
            serial_latency = serial_data.loc[(row['nlist'], row['nprobe']), 'Latency(us)']
            return serial_latency / row['Latency(us)']
        except:
            return 1.0
    
    df['Speedup'] = df.apply(calculate_speedup, axis=1)
    
    return df

def plot_recall_latency_analysis(df):
    """绘制召回率-延迟分析图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1: 不同nlist值的召回率-延迟权衡（针对串行实现）
    serial_data = df[df['Implementation'] == '串行']
    nlist_values = sorted(serial_data['nlist'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(nlist_values)))
    
    for i, nlist in enumerate(nlist_values):
        data = serial_data[serial_data['nlist'] == nlist]
        ax1.plot(data['Recall'] * 100, data['Latency(us)'], 'o-', 
                color=colors[i], label=f'nlist={nlist}', linewidth=2, markersize=6)
    
    ax1.set_xlabel('召回率 (%)', fontsize=12)
    ax1.set_ylabel('延迟 (μs)', fontsize=12)
    ax1.set_title('不同nlist值的召回率-延迟权衡\n(串行实现)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 图2: 不同实现方式的性能比较（nlist=256）
    nlist_256_data = df[df['nlist'] == 256]
    implementations = nlist_256_data['Implementation'].unique()
    impl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, impl in enumerate(implementations):
        data = nlist_256_data[nlist_256_data['Implementation'] == impl]
        ax2.plot(data['Recall'] * 100, data['Latency(us)'], 'o-', 
                color=impl_colors[i], label=impl, linewidth=2, markersize=6)
    
    ax2.set_xlabel('召回率 (%)', fontsize=12)
    ax2.set_ylabel('延迟 (μs)', fontsize=12)
    ax2.set_title('不同实现方式的性能比较\n(nlist=256)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 图3: nprobe对延迟的影响
    for i, nlist in enumerate([128, 256, 512]):
        data = serial_data[serial_data['nlist'] == nlist]
        ax3.plot(data['nprobe'], data['Latency(us)'], 'o-', 
                label=f'nlist={nlist}', linewidth=2, markersize=6)
    
    ax3.set_xlabel('nprobe值', fontsize=12)
    ax3.set_ylabel('延迟 (μs)', fontsize=12)
    ax3.set_title('nprobe对延迟的影响\n(串行实现)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 图4: nprobe对召回率的影响
    for i, nlist in enumerate([128, 256, 512]):
        data = serial_data[serial_data['nlist'] == nlist]
        ax4.plot(data['nprobe'], data['Recall'] * 100, 'o-', 
                label=f'nlist={nlist}', linewidth=2, markersize=6)
    
    ax4.set_xlabel('nprobe值', fontsize=12)
    ax4.set_ylabel('召回率 (%)', fontsize=12)
    ax4.set_title('nprobe对召回率的影响\n(串行实现)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/ivf_recall_latency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_speedup_analysis(df):
    """绘制加速比分析图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1: 加速比热力图（OpenMP）
    openmp_data = df[df['Implementation'] == 'OpenMP']
    speedup_matrix_omp = openmp_data.pivot(index='nlist', columns='nprobe', values='Speedup')
    
    im1 = ax1.imshow(speedup_matrix_omp.values, cmap='RdYlGn', aspect='auto')
    ax1.set_xticks(range(len(speedup_matrix_omp.columns)))
    ax1.set_xticklabels(speedup_matrix_omp.columns)
    ax1.set_yticks(range(len(speedup_matrix_omp.index)))
    ax1.set_yticklabels(speedup_matrix_omp.index)
    ax1.set_xlabel('nprobe值', fontsize=12)
    ax1.set_ylabel('nlist值', fontsize=12)
    ax1.set_title('OpenMP加速比热力图', fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(speedup_matrix_omp.index)):
        for j in range(len(speedup_matrix_omp.columns)):
            text = ax1.text(j, i, f'{speedup_matrix_omp.iloc[i, j]:.1f}x',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=ax1, label='加速比')
    
    # 图2: 加速比热力图（Pthread）
    pthread_data = df[df['Implementation'] == 'Pthread']
    speedup_matrix_ptd = pthread_data.pivot(index='nlist', columns='nprobe', values='Speedup')
    
    im2 = ax2.imshow(speedup_matrix_ptd.values, cmap='RdYlGn', aspect='auto')
    ax2.set_xticks(range(len(speedup_matrix_ptd.columns)))
    ax2.set_xticklabels(speedup_matrix_ptd.columns)
    ax2.set_yticks(range(len(speedup_matrix_ptd.index)))
    ax2.set_yticklabels(speedup_matrix_ptd.index)
    ax2.set_xlabel('nprobe值', fontsize=12)
    ax2.set_ylabel('nlist值', fontsize=12)
    ax2.set_title('Pthread加速比热力图', fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(speedup_matrix_ptd.index)):
        for j in range(len(speedup_matrix_ptd.columns)):
            text = ax2.text(j, i, f'{speedup_matrix_ptd.iloc[i, j]:.1f}x',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im2, ax=ax2, label='加速比')
    
    # 图3: 不同nlist值下的加速比比较
    nlist_values = [128, 256, 512]
    x = np.arange(len(df['nprobe'].unique()))
    width = 0.35
    
    for i, nlist in enumerate(nlist_values):
        omp_speedup = openmp_data[openmp_data['nlist'] == nlist]['Speedup'].values
        ptd_speedup = pthread_data[pthread_data['nlist'] == nlist]['Speedup'].values
        
        if i == 0:
            ax3.bar(x - width/2, omp_speedup, width/len(nlist_values), 
                   label=f'OpenMP nlist={nlist}', alpha=0.8)
            ax3.bar(x + width/2, ptd_speedup, width/len(nlist_values), 
                   label=f'Pthread nlist={nlist}', alpha=0.8)
        else:
            ax3.bar(x - width/2 + i*width/len(nlist_values), omp_speedup, width/len(nlist_values), 
                   alpha=0.8)
            ax3.bar(x + width/2 + i*width/len(nlist_values), ptd_speedup, width/len(nlist_values), 
                   alpha=0.8)
    
    ax3.set_xlabel('nprobe值', fontsize=12)
    ax3.set_ylabel('加速比', fontsize=12)
    ax3.set_title('不同nlist值下的加速比比较', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['nprobe'].unique())
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: 平均加速比对比
    avg_speedup = df[df['Implementation'] != '串行'].groupby('Implementation')['Speedup'].mean()
    bars = ax4.bar(avg_speedup.index, avg_speedup.values, color=['#ff7f0e', '#2ca02c'])
    ax4.set_ylabel('平均加速比', fontsize=12)
    ax4.set_title('不同实现方式的平均加速比', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标注
    for bar, value in zip(bars, avg_speedup.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/ivf_speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_parameter_sensitivity(df):
    """绘制参数敏感性分析"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1: nlist对性能的影响（固定nprobe=16）
    fixed_nprobe_data = df[df['nprobe'] == 16]
    
    for impl in fixed_nprobe_data['Implementation'].unique():
        data = fixed_nprobe_data[fixed_nprobe_data['Implementation'] == impl]
        ax1.plot(data['nlist'], data['Latency(us)'], 'o-', label=impl, linewidth=2, markersize=6)
    
    ax1.set_xlabel('nlist值', fontsize=12)
    ax1.set_ylabel('延迟 (μs)', fontsize=12)
    ax1.set_title('nlist对延迟的影响\n(nprobe=16)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 图2: nlist对召回率的影响（固定nprobe=16）
    for impl in fixed_nprobe_data['Implementation'].unique():
        data = fixed_nprobe_data[fixed_nprobe_data['Implementation'] == impl]
        ax2.plot(data['nlist'], data['Recall'] * 100, 'o-', label=impl, linewidth=2, markersize=6)
    
    ax2.set_xlabel('nlist值', fontsize=12)
    ax2.set_ylabel('召回率 (%)', fontsize=12)
    ax2.set_title('nlist对召回率的影响\n(nprobe=16)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: nprobe对性能的影响（固定nlist=256）
    fixed_nlist_data = df[df['nlist'] == 256]
    
    for impl in fixed_nlist_data['Implementation'].unique():
        data = fixed_nlist_data[fixed_nlist_data['Implementation'] == impl]
        ax3.plot(data['nprobe'], data['Latency(us)'], 'o-', label=impl, linewidth=2, markersize=6)
    
    ax3.set_xlabel('nprobe值', fontsize=12)
    ax3.set_ylabel('延迟 (μs)', fontsize=12)
    ax3.set_title('nprobe对延迟的影响\n(nlist=256)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 图4: 效率分析（QPS vs Recall）
    for impl in df['Implementation'].unique():
        data = df[df['Implementation'] == impl]
        scatter = ax4.scatter(data['Recall'] * 100, data['QPS'], 
                            s=50, alpha=0.7, label=impl)
    
    ax4.set_xlabel('召回率 (%)', fontsize=12)
    ax4.set_ylabel('QPS (查询/秒)', fontsize=12)
    ax4.set_title('查询效率分析\n(QPS vs 召回率)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('plots/ivf_parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_performance_summary(df):
    """生成性能总结表"""
    print("\n" + "="*80)
    print("IVF算法多线程实现性能分析总结")
    print("="*80)
    
    # 计算总体统计
    summary_stats = df.groupby('Implementation').agg({
        'Speedup': ['mean', 'max', 'min'],
        'Latency(us)': ['mean', 'min'],
        'Recall': ['mean', 'max']
    }).round(3)
    
    print("\n1. 整体性能统计:")
    print(summary_stats)
    
    # 最佳配置分析
    print("\n2. 最佳配置分析:")
    for impl in df['Implementation'].unique():
        impl_data = df[df['Implementation'] == impl]
        
        # 最高召回率配置
        best_recall = impl_data.loc[impl_data['Recall'].idxmax()]
        print(f"\n{impl}实现:")
        print(f"  最高召回率: {best_recall['Recall']:.4f} (nlist={best_recall['nlist']}, nprobe={best_recall['nprobe']}, 延迟={best_recall['Latency(us)']:.0f}μs)")
        
        # 最低延迟配置
        best_latency = impl_data.loc[impl_data['Latency(us)'].idxmin()]
        print(f"  最低延迟: {best_latency['Latency(us)']:.0f}μs (nlist={best_latency['nlist']}, nprobe={best_latency['nprobe']}, 召回率={best_latency['Recall']:.4f})")
        
        if impl != '串行':
            best_speedup = impl_data.loc[impl_data['Speedup'].idxmax()]
            print(f"  最高加速比: {best_speedup['Speedup']:.2f}x (nlist={best_speedup['nlist']}, nprobe={best_speedup['nprobe']})")
    
    # 参数影响分析
    print("\n3. 参数影响分析:")
    print(f"  nlist范围: {df['nlist'].min()} - {df['nlist'].max()}")
    print(f"  nprobe范围: {df['nprobe'].min()} - {df['nprobe'].max()}")
    print(f"  召回率范围: {df['Recall'].min():.3f} - {df['Recall'].max():.3f}")
    print(f"  延迟范围: {df['Latency(us)'].min():.0f} - {df['Latency(us)'].max():.0f} μs")

def main():
    """主函数"""
    print("开始IVF算法多线程实现可视化分析...")
    
    # 加载数据
    df = load_and_process_data('ivf_results.csv')
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    # 创建图片目录
    import os
    os.makedirs('plots', exist_ok=True)
    
    # 生成各种分析图
    print("\n生成召回率-延迟分析图...")
    plot_recall_latency_analysis(df)
    
    print("生成加速比分析图...")
    plot_speedup_analysis(df)
    
    print("生成参数敏感性分析图...")
    plot_parameter_sensitivity(df)
    
    # 生成性能总结
    generate_performance_summary(df)
    
    print("\n分析完成！所有图表已保存到 plots/ 目录")

if __name__ == "__main__":
    main() 