#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams

# 设置英文字体
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

# 设置图形样式
plt.style.use('default')
sns.set_palette("tab10")

def load_and_preprocess_data(filename='parameter_analysis_results.csv'):
    """Load and preprocess experimental data"""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run parameter analysis first.")
        # Create dummy data for testing
        df = pd.DataFrame({
            'algorithm': ['KDTree Approx', 'KDTree SIMD Parallel', 'LSH Improved', 'LSH SIMD Parallel'],
            'recall_mean': [0.40, 0.42, 0.85, 0.86],
            'latency_mean_us': [800, 650, 1800, 1200],
            'speedup': [5.5, 6.8, 2.8, 3.5],
            'parallel': ['false', 'true', 'false', 'true'],
            'max_search_nodes': [8000, 8000, None, None],
            'num_tables': [None, None, 120, 120],
            'hash_bits': [None, None, 14, 14],
            'min_candidates': [None, None, 2000, 2000]
        })
        print("Using dummy data for demonstration.")
    
    # Add computed columns
    df['recall_percent'] = df['recall_mean'] * 100
    df['algorithm_type'] = df['algorithm'].apply(lambda x: 'KDTree' if 'KDTree' in x else 'LSH')
    df['is_parallel'] = df['parallel'] == 'true'
    
    # Handle missing parameter columns
    parameter_columns = ['max_search_nodes', 'num_tables', 'hash_bits', 'min_candidates', 'search_radius']
    for col in parameter_columns:
        if col not in df.columns:
            df[col] = None
    
    # Convert numeric parameters
    numeric_params = ['max_search_nodes', 'num_tables', 'hash_bits', 'min_candidates', 'search_radius']
    for param in numeric_params:
        if param in df.columns:
            df[param] = pd.to_numeric(df[param], errors='coerce')
    
    # Fill in some default parameter values based on algorithm type
    if 'max_search_nodes' in df.columns:
        df.loc[df['algorithm_type'] == 'KDTree', 'max_search_nodes'] = df.loc[
            df['algorithm_type'] == 'KDTree', 'max_search_nodes'].fillna(8000)
    
    if 'num_tables' in df.columns:
        df.loc[df['algorithm_type'] == 'LSH', 'num_tables'] = df.loc[
            df['algorithm_type'] == 'LSH', 'num_tables'].fillna(120)
    
    if 'hash_bits' in df.columns:
        df.loc[df['algorithm_type'] == 'LSH', 'hash_bits'] = df.loc[
            df['algorithm_type'] == 'LSH', 'hash_bits'].fillna(14)
    
    return df

def plot_comprehensive_analysis(df):
    """创建综合分析图表"""
    fig = plt.figure(figsize=(20, 16))
    
    # 创建6个子图的布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. KDTree Recall vs QPS (连线图)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_kdtree_recall_vs_qps(df, ax1)
    
    # 2. LSH Recall vs QPS (连线图)  
    ax2 = fig.add_subplot(gs[0, 1])
    plot_lsh_recall_vs_qps(df, ax2)
    
    # 3. Speedup Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    plot_speedup_comparison(df, ax3)
    
    # 4. KDTree Parameter Sensitivity - Search Nodes
    ax4 = fig.add_subplot(gs[1, 0])
    plot_kdtree_parameter_sensitivity(df, ax4)
    
    # 5. LSH Parameter Sensitivity - Hash Tables
    ax5 = fig.add_subplot(gs[1, 1])
    plot_lsh_hash_tables_sensitivity(df, ax5)
    
    # 6. LSH Parameter Sensitivity - Hash Bits
    ax6 = fig.add_subplot(gs[1, 2])
    plot_lsh_hash_bits_sensitivity(df, ax6)
    
    # 7. LSH Parameter Sensitivity - Candidates
    ax7 = fig.add_subplot(gs[2, 0])
    plot_lsh_candidates_sensitivity(df, ax7)
    
    # 8. Parallel vs Non-parallel Comparison
    ax8 = fig.add_subplot(gs[2, 1:])
    plot_parallel_comparison_detailed(df, ax8)
    
    plt.suptitle('Comprehensive Parameter Analysis for ANN Algorithms', fontsize=16, fontweight='bold')
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print("✓ Comprehensive analysis chart saved as 'comprehensive_analysis.png'")

def plot_kdtree_recall_vs_qps(df, ax):
    """KDTree算法：召回率 vs QPS 连线图"""
    kdtree_data = df[df['algorithm_type'] == 'KDTree'].copy()
    kdtree_data['qps'] = 1000000 / kdtree_data['latency_mean_us']  # 转换为QPS
    
    # 按搜索节点数分组
    search_nodes_values = sorted(kdtree_data['max_search_nodes'].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(search_nodes_values)))
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
    
    for i, nodes in enumerate(search_nodes_values):
        subset = kdtree_data[kdtree_data['max_search_nodes'] == nodes]
        
        # 分别绘制并行和非并行版本
        non_parallel = subset[~subset['is_parallel']].sort_values('recall_percent')
        parallel = subset[subset['is_parallel']].sort_values('recall_percent')
        
        if len(non_parallel) > 0:
            ax.plot(non_parallel['recall_percent'], non_parallel['qps'], 
                   color=colors[i], marker=markers[i % len(markers)], 
                   linestyle='-', linewidth=2, markersize=8,
                   label=f'Nodes={int(nodes)} (Sequential)')
        
        if len(parallel) > 0:
            ax.plot(parallel['recall_percent'], parallel['qps'], 
                   color=colors[i], marker=markers[i % len(markers)], 
                   linestyle='--', linewidth=2, markersize=8,
                   label=f'Nodes={int(nodes)} (Parallel)')
    
    ax.set_xlabel('Recall (%)')
    ax.set_ylabel('QPS')
    ax.set_title('KDTree: Recall vs QPS')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_yscale('log')

def plot_lsh_recall_vs_qps(df, ax):
    """LSH算法：召回率 vs QPS 连线图"""
    lsh_data = df[df['algorithm_type'] == 'LSH'].copy()
    lsh_data['qps'] = 1000000 / lsh_data['latency_mean_us']  # 转换为QPS
    
    # 按哈希表数量分组
    table_values = sorted(lsh_data['num_tables'].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(table_values)))
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
    
    for i, tables in enumerate(table_values):
        subset = lsh_data[lsh_data['num_tables'] == tables]
        
        # 分别绘制并行和非并行版本
        non_parallel = subset[~subset['is_parallel']].sort_values('recall_percent')
        parallel = subset[subset['is_parallel']].sort_values('recall_percent')
        
        if len(non_parallel) > 0:
            ax.plot(non_parallel['recall_percent'], non_parallel['qps'], 
                   color=colors[i], marker=markers[i % len(markers)], 
                   linestyle='-', linewidth=2, markersize=8,
                   label=f'Tables={int(tables)} (Sequential)')
        
        if len(parallel) > 0:
            ax.plot(parallel['recall_percent'], parallel['qps'], 
                   color=colors[i], marker=markers[i % len(markers)], 
                   linestyle='--', linewidth=2, markersize=8,
                   label=f'Tables={int(tables)} (Parallel)')
    
    ax.set_xlabel('Recall (%)')
    ax.set_ylabel('QPS')
    ax.set_title('LSH: Recall vs QPS')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_yscale('log')

def plot_speedup_comparison(df, ax):
    """绘制加速比对比图"""
    # 准备数据
    non_parallel_speedup = []
    parallel_speedup = []
    
    for algo_base in ['KDTree Approx', 'LSH Improved']:
        if algo_base == 'KDTree Approx':
            parallel_algo = 'KDTree SIMD Parallel'
        else:
            parallel_algo = 'LSH SIMD Parallel'
        
        non_par_data = df[df['algorithm'] == algo_base]
        par_data = df[df['algorithm'] == parallel_algo]
        
        if len(non_par_data) > 0 and len(par_data) > 0:
            non_parallel_speedup.extend(non_par_data['speedup'].tolist())
            parallel_speedup.extend(par_data['speedup'].tolist())
    
    # 加速比对比箱线图
    speedup_data = [non_parallel_speedup, parallel_speedup]
    bp = ax.boxplot(speedup_data, labels=['Sequential', 'Parallel'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Speedup')
    ax.set_title('Parallel vs Sequential Speedup')
    ax.grid(True, alpha=0.3)

def plot_kdtree_parameter_sensitivity(df, ax):
    """KDTree参数敏感性分析：搜索节点数"""
    kdtree_data = df[df['algorithm_type'] == 'KDTree']
    
    if 'max_search_nodes' in kdtree_data.columns:
        kdtree_data = kdtree_data.copy()
        kdtree_data['max_search_nodes'] = pd.to_numeric(kdtree_data['max_search_nodes'], errors='coerce')
        grouped = kdtree_data.groupby(['max_search_nodes', 'is_parallel']).agg({
            'recall_percent': 'mean',
            'speedup': 'mean'
        }).reset_index()
        
        # 分别绘制并行和非并行
        for is_parallel, color, linestyle, label in [(False, 'blue', '-', 'Sequential'), 
                                                    (True, 'red', '--', 'Parallel')]:
            subset = grouped[grouped['is_parallel'] == is_parallel]
            if len(subset) > 0:
                ax.plot(subset['max_search_nodes'], subset['recall_percent'], 
                       color=color, linestyle=linestyle, marker='o', linewidth=2, markersize=6,
                       label=f'{label} (Recall)')
        
        ax.set_xlabel('Max Search Nodes')
        ax.set_ylabel('Recall (%)')
        ax.set_title('KDTree: Search Nodes Impact')
        ax.legend()
        ax.grid(True, alpha=0.3)

def plot_lsh_hash_tables_sensitivity(df, ax):
    """LSH参数敏感性分析：哈希表数量"""
    lsh_data = df[df['algorithm_type'] == 'LSH']
    
    if 'num_tables' in lsh_data.columns:
        lsh_data = lsh_data.copy()
        lsh_data['num_tables'] = pd.to_numeric(lsh_data['num_tables'], errors='coerce')
        grouped = lsh_data.groupby(['num_tables', 'is_parallel']).agg({
            'recall_percent': 'mean',
            'speedup': 'mean'
        }).reset_index()
        
        # 分别绘制并行和非并行
        for is_parallel, color, linestyle, label in [(False, 'green', '-', 'Sequential'), 
                                                    (True, 'orange', '--', 'Parallel')]:
            subset = grouped[grouped['is_parallel'] == is_parallel]
            if len(subset) > 0:
                ax.plot(subset['num_tables'], subset['recall_percent'], 
                       color=color, linestyle=linestyle, marker='s', linewidth=2, markersize=6,
                       label=f'{label} (Recall)')
        
        ax.set_xlabel('Number of Hash Tables')
        ax.set_ylabel('Recall (%)')
        ax.set_title('LSH: Hash Tables Impact')
        ax.legend()
        ax.grid(True, alpha=0.3)

def plot_lsh_hash_bits_sensitivity(df, ax):
    """LSH参数敏感性分析：哈希位数"""
    lsh_data = df[df['algorithm_type'] == 'LSH']
    
    if 'hash_bits' in lsh_data.columns:
        lsh_data = lsh_data.copy()
        lsh_data['hash_bits'] = pd.to_numeric(lsh_data['hash_bits'], errors='coerce')
        grouped = lsh_data.groupby(['hash_bits', 'is_parallel']).agg({
            'recall_percent': 'mean',
            'latency_mean_us': 'mean'
        }).reset_index()
        
        ax2 = ax.twinx()
        
        # 分别绘制并行和非并行
        for is_parallel, color, linestyle, label in [(False, 'cyan', '-', 'Sequential'), 
                                                    (True, 'magenta', '--', 'Parallel')]:
            subset = grouped[grouped['is_parallel'] == is_parallel]
            if len(subset) > 0:
                line1 = ax.plot(subset['hash_bits'], subset['recall_percent'], 
                               color=color, linestyle=linestyle, marker='^', linewidth=2, markersize=6,
                               label=f'{label} (Recall)')
                line2 = ax2.plot(subset['hash_bits'], subset['latency_mean_us'], 
                                color=color, linestyle=linestyle, marker='v', linewidth=2, markersize=6, alpha=0.7,
                                label=f'{label} (Latency)')
        
        ax.set_xlabel('Hash Bits')
        ax.set_ylabel('Recall (%)', color='black')
        ax2.set_ylabel('Latency (μs)', color='gray')
        ax.set_title('LSH: Hash Bits Impact')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

def plot_lsh_candidates_sensitivity(df, ax):
    """LSH参数敏感性分析：候选点数量"""
    lsh_data = df[df['algorithm_type'] == 'LSH']
    
    if 'min_candidates' in lsh_data.columns:
        lsh_data = lsh_data.copy()
        lsh_data['min_candidates'] = pd.to_numeric(lsh_data['min_candidates'], errors='coerce')
        grouped = lsh_data.groupby(['min_candidates', 'is_parallel']).agg({
            'recall_percent': 'mean',
            'speedup': 'mean'
        }).reset_index()
        
        ax2 = ax.twinx()
        
        # 分别绘制并行和非并行
        for is_parallel, color, linestyle, label in [(False, 'purple', '-', 'Sequential'), 
                                                    (True, 'brown', '--', 'Parallel')]:
            subset = grouped[grouped['is_parallel'] == is_parallel]
            if len(subset) > 0:
                line1 = ax.plot(subset['min_candidates'], subset['recall_percent'], 
                               color=color, linestyle=linestyle, marker='D', linewidth=2, markersize=6,
                               label=f'{label} (Recall)')
                line2 = ax2.plot(subset['min_candidates'], subset['speedup'], 
                                color=color, linestyle=linestyle, marker='*', linewidth=2, markersize=8, alpha=0.7,
                                label=f'{label} (Speedup)')
        
        ax.set_xlabel('Min Candidates')
        ax.set_ylabel('Recall (%)', color='black')
        ax2.set_ylabel('Speedup', color='gray')
        ax.set_title('LSH: Candidates Impact')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

def plot_parallel_comparison_detailed(df, ax):
    """详细的并行对比分析"""
    algorithms = df['algorithm'].unique()
    
    recall_means = []
    speedup_means = []
    algorithm_labels = []
    colors = []
    
    for algo in algorithms:
        subset = df[df['algorithm'] == algo]
        recall_means.append(subset['recall_percent'].mean())
        speedup_means.append(subset['speedup'].mean())
        algorithm_labels.append(algo.replace(' ', '\n'))
        
        if 'Parallel' in algo:
            colors.append('lightcoral')
        else:
            colors.append('lightblue')
    
    bars = ax.bar(algorithm_labels, speedup_means, color=colors, alpha=0.7, edgecolor='black')
    
    # 添加召回率标注
    for i, (bar, recall) in enumerate(zip(bars, recall_means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{recall:.1f}%', ha='center', va='bottom', fontsize=8, weight='bold')
    
    ax.set_ylabel('Speedup')
    ax.set_title('Algorithm Performance Comparison\n(Speedup with Recall% annotations)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)



def generate_summary_report(df):
    """Generate comprehensive summary report"""
    print("\n" + "="*80)
    print("Parameter Analysis Experiment Summary Report")
    print("="*80)
    
    # Basic statistics
    print(f"\nTotal experiments: {len(df)}")
    print(f"Algorithm types: {df['algorithm'].nunique()} types")
    print(f"Parameter combinations: {len(df)} combinations")
    
    # Performance statistics
    print(f"\nPerformance Statistics:")
    print(f"Recall range: {df['recall_percent'].min():.2f}% - {df['recall_percent'].max():.2f}%")
    print(f"Latency range: {df['latency_mean_us'].min():.2f} - {df['latency_mean_us'].max():.2f} μs")
    print(f"Speedup range: {df['speedup'].min():.2f}x - {df['speedup'].max():.2f}x")
    
    # Parallel performance analysis
    print(f"\nParallel Performance Analysis:")
    parallel_data = df[df['is_parallel'] == True]
    non_parallel_data = df[df['is_parallel'] == False]
    
    if len(parallel_data) > 0 and len(non_parallel_data) > 0:
        print(f"Parallel version average speedup: {parallel_data['speedup'].mean():.2f}x")
        print(f"Sequential version average speedup: {non_parallel_data['speedup'].mean():.2f}x")
        print(f"Parallel improvement: {(parallel_data['speedup'].mean() / non_parallel_data['speedup'].mean() - 1) * 100:.1f}%")
    
    # Best configuration recommendations
    print(f"\nBest Configuration Recommendations:")
    
    # High recall configuration
    high_recall = df[df['recall_percent'] >= 80]
    if len(high_recall) > 0:
        best_balanced = high_recall.loc[high_recall['speedup'].idxmax()]
        print(f"High recall configuration (>80%): {best_balanced['algorithm']}")
        print(f"  Recall: {best_balanced['recall_percent']:.2f}%")
        print(f"  Speedup: {best_balanced['speedup']:.2f}x")
        print(f"  Latency: {best_balanced['latency_mean_us']:.2f} μs")
    
    # High speed configuration
    best_speed = df.loc[df['speedup'].idxmax()]
    print(f"\nHigh speed configuration: {best_speed['algorithm']}")
    print(f"  Recall: {best_speed['recall_percent']:.2f}%")
    print(f"  Speedup: {best_speed['speedup']:.2f}x")
    print(f"  Latency: {best_speed['latency_mean_us']:.2f} μs")
    
    # Algorithm type comparison
    print(f"\nAlgorithm Type Comparison:")
    for algo_type in df['algorithm_type'].unique():
        subset = df[df['algorithm_type'] == algo_type]
        print(f"{algo_type}:")
        print(f"  Average recall: {subset['recall_percent'].mean():.2f}%")
        print(f"  Average speedup: {subset['speedup'].mean():.2f}x")
        print(f"  Average latency: {subset['latency_mean_us'].mean():.2f} μs")

def main():
    """Main function for comprehensive analysis"""
    print("Loading experimental data...")
    
    try:
        df = load_and_preprocess_data()
        
        print("Generating comprehensive visualization charts...")
        
        # Generate comprehensive analysis chart
        plot_comprehensive_analysis(df)
        
        # Generate summary report
        generate_summary_report(df)
        
        print("\nVisualization analysis completed!")
        print("Generated files:")
        print("- comprehensive_analysis.png: Complete parameter analysis with 8 subplots")
        print("  * KDTree Recall vs QPS")
        print("  * LSH Recall vs QPS") 
        print("  * Speedup Comparison")
        print("  * KDTree Parameter Sensitivity")
        print("  * LSH Hash Tables Sensitivity")
        print("  * LSH Hash Bits Sensitivity")
        print("  * LSH Candidates Sensitivity")
        print("  * Parallel vs Sequential Detailed Comparison")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Try with dummy data as fallback
        print("\nTrying with dummy data...")
        try:
            df_dummy = pd.DataFrame({
                'algorithm': ['KDTree Approx', 'KDTree SIMD Parallel', 'LSH Improved', 'LSH SIMD Parallel'],
                'recall_mean': [0.40, 0.42, 0.85, 0.86],
                'latency_mean_us': [800, 650, 1800, 1200],
                'speedup': [5.5, 6.8, 2.8, 3.5],
                'parallel': ['false', 'true', 'false', 'true'],
                'max_search_nodes': [8000, 8000, None, None],
                'num_tables': [None, None, 120, 120],
                'hash_bits': [None, None, 14, 14],
                'min_candidates': [None, None, 2000, 2000]
            })
            df_dummy['recall_percent'] = df_dummy['recall_mean'] * 100
            df_dummy['algorithm_type'] = df_dummy['algorithm'].apply(lambda x: 'KDTree' if 'KDTree' in x else 'LSH')
            df_dummy['is_parallel'] = df_dummy['parallel'] == 'true'
            
            plot_comprehensive_analysis(df_dummy)
            print("✓ Generated visualization with dummy data")
            
        except Exception as e2:
            print(f"Failed to generate with dummy data: {e2}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 