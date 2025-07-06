#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final comparison visualization for KDTree Hybrid Optimized vs LSH SIMD Parallel
Based on actual experimental results
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_final_comparison():
    """Create final comparison based on experimental results"""
    
    # Based on actual experimental results and main_cpu_only.cpp output
    algorithms_data = {
        'Algorithm': [
            'Flat Search',
            'KDTree CPU', 
            'LSH CPU',
            'KDTree Approx 90%',
            'LSH Improved 90%',
            'KDTree SIMD Parallel', 
            'LSH SIMD Parallel',
            'KDTree Ensemble Parallel',
            'Flat Search SIMD Parallel',
            'KDTree Hybrid Optimized'
        ],
        'Recall (%)': [99.995, 27.6, 53.4, 15.13, 88.31, 42.49, 85.86, 100.0, 99.995, 99.88],
        'Latency (μs)': [3469, 480, 920, 124, 1796, 685, 1027, 4579, 809, 1607],
        'Speedup': [1.0, 7.23, 3.77, 27.98, 1.93, 5.06, 3.38, 0.76, 4.29, 2.16],
        'Type': ['Baseline', 'Basic', 'Basic', 'Optimized', 'Optimized', 'SIMD+Parallel', 'SIMD+Parallel', 'Advanced', 'Advanced', 'Hybrid']
    }
    
    df = pd.DataFrame(algorithms_data)
    
    # Create the comprehensive comparison figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main Performance Chart - Recall vs QPS with connected lines (like your reference)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Calculate QPS
    df['QPS'] = 1000000 / df['Latency (μs)']
    
    # Define algorithm groups and colors
    groups = {
        'Baseline & Basic': ['Flat Search', 'KDTree CPU', 'LSH CPU'],
        'Optimized Variants': ['KDTree Approx 90%', 'LSH Improved 90%'],
        'SIMD+Parallel': ['KDTree SIMD Parallel', 'LSH SIMD Parallel'],
        'Advanced & Hybrid': ['KDTree Ensemble Parallel', 'Flat Search SIMD Parallel', 'KDTree Hybrid Optimized']
    }
    
    colors = ['gray', 'orange', 'blue', 'red']
    markers = ['o', 's', '^', 'D']
    
    for i, (group_name, algos) in enumerate(groups.items()):
        group_data = df[df['Algorithm'].isin(algos)].sort_values('Recall (%)')
        if len(group_data) > 0:
            ax1.plot(group_data['Recall (%)'], group_data['QPS'], 
                    color=colors[i], marker=markers[i], linewidth=3, markersize=10,
                    label=group_name, alpha=0.8)
            
            # Add algorithm labels
            for _, row in group_data.iterrows():
                ax1.annotate(row['Algorithm'].replace(' ', '\n'), 
                           (row['Recall (%)'], row['QPS']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('Recall (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('QPS (Queries Per Second)', fontsize=14, fontweight='bold')
    ax1.set_title('ANN Algorithm Performance Comparison: Recall vs QPS', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_yscale('log')
    ax1.set_ylim(100, 10000)
    
    # 2. Speedup Comparison Bar Chart
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Highlight the two target algorithms
    target_algos = ['KDTree Hybrid Optimized', 'LSH SIMD Parallel']
    colors_bar = ['red' if alg in target_algos else 'lightblue' for alg in df['Algorithm']]
    
    bars = ax2.barh(range(len(df)), df['Speedup'], color=colors_bar, alpha=0.7, edgecolor='black')
    
    # Add recall annotations
    for i, (bar, recall) in enumerate(zip(bars, df['Recall (%)'])):
        width = bar.get_width()
        ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{recall:.1f}%', ha='left', va='center', fontsize=9, weight='bold')
    
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels([alg.replace(' ', '\n') for alg in df['Algorithm']], fontsize=9)
    ax2.set_xlabel('Speedup vs Flat Search', fontsize=12)
    ax2.set_title('Algorithm Speedup Comparison\n(Red = Target Algorithms)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Recall vs Latency Trade-off
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Scatter plot with different markers for different types
    type_colors = {'Baseline': 'gray', 'Basic': 'lightblue', 'Optimized': 'orange', 
                   'SIMD+Parallel': 'blue', 'Advanced': 'green', 'Hybrid': 'red'}
    
    for algo_type in df['Type'].unique():
        subset = df[df['Type'] == algo_type]
        ax3.scatter(subset['Latency (μs)'], subset['Recall (%)'], 
                   c=type_colors.get(algo_type, 'black'), 
                   s=100, alpha=0.7, label=algo_type,
                   marker='D' if algo_type == 'Hybrid' else 'o')
    
    # Highlight target algorithms
    target_data = df[df['Algorithm'].isin(target_algos)]
    for _, row in target_data.iterrows():
        ax3.annotate(row['Algorithm'], 
                    (row['Latency (μs)'], row['Recall (%)']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax3.set_xlabel('Latency (μs)', fontsize=12)
    ax3.set_ylabel('Recall (%)', fontsize=12)
    ax3.set_title('Latency vs Recall Trade-off', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xscale('log')
    
    # 4. Algorithm Evolution Timeline
    ax4 = fig.add_subplot(gs[1, 2])
    
    evolution_order = [
        'Flat Search', 'KDTree CPU', 'LSH CPU',
        'KDTree Approx 90%', 'LSH Improved 90%',
        'KDTree SIMD Parallel', 'LSH SIMD Parallel',
        'Flat Search SIMD Parallel', 'KDTree Ensemble Parallel',
        'KDTree Hybrid Optimized'
    ]
    
    evolution_data = df.set_index('Algorithm').loc[evolution_order].reset_index()
    
    ax4.plot(range(len(evolution_data)), evolution_data['Recall (%)'], 
             'go-', linewidth=3, markersize=8, label='Recall (%)', alpha=0.8)
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(range(len(evolution_data)), evolution_data['Speedup'], 
                  'ro-', linewidth=3, markersize=8, label='Speedup', alpha=0.8)
    
    ax4.set_xticks(range(len(evolution_data)))
    ax4.set_xticklabels([alg.replace(' ', '\n') for alg in evolution_data['Algorithm']], 
                       rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Recall (%)', color='green', fontsize=12)
    ax4_twin.set_ylabel('Speedup', color='red', fontsize=12)
    ax4.set_title('Algorithm Evolution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center left')
    
    # 5. Performance Matrix Heatmap
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Create performance matrix (normalized scores)
    metrics = ['Recall', 'Speed', 'Efficiency']
    algorithms_short = [alg.replace(' ', '\n') for alg in df['Algorithm'][-4:]]  # Last 4 algorithms
    
    # Normalize metrics to 0-1 scale
    recall_norm = df['Recall (%)'].iloc[-4:] / 100
    speed_norm = df['Speedup'].iloc[-4:] / df['Speedup'].max()
    efficiency_norm = (df['Recall (%)'] / df['Latency (μs)']).iloc[-4:] / (df['Recall (%)'] / df['Latency (μs)']).max()
    
    performance_matrix = np.array([recall_norm, speed_norm, efficiency_norm])
    
    im = ax5.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax5.set_xticks(range(len(algorithms_short)))
    ax5.set_xticklabels(algorithms_short, fontsize=9)
    ax5.set_yticks(range(len(metrics)))
    ax5.set_yticklabels(metrics)
    ax5.set_title('Performance Matrix\n(Advanced Algorithms)', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(algorithms_short)):
            text = ax5.text(j, i, f'{performance_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9, weight='bold')
    
    plt.colorbar(im, ax=ax5, shrink=0.6)
    
    # 6. Target Algorithms Detailed Comparison
    ax6 = fig.add_subplot(gs[2, 1:])
    
    target_comparison = df[df['Algorithm'].isin(target_algos)]
    
    metrics_comp = ['Recall (%)', 'Latency (μs)', 'Speedup', 'QPS']
    x = np.arange(len(metrics_comp))
    width = 0.35
    
    kdtree_values = [target_comparison[target_comparison['Algorithm'] == 'KDTree Hybrid Optimized'][metric].iloc[0] 
                     for metric in metrics_comp]
    lsh_values = [target_comparison[target_comparison['Algorithm'] == 'LSH SIMD Parallel'][metric].iloc[0] 
                  for metric in metrics_comp]
    
    # Normalize for comparison (except recall which is already percentage)
    kdtree_normalized = [kdtree_values[0], kdtree_values[1]/1000, kdtree_values[2], kdtree_values[3]/1000]
    lsh_normalized = [lsh_values[0], lsh_values[1]/1000, lsh_values[2], lsh_values[3]/1000]
    
    bars1 = ax6.bar(x - width/2, kdtree_normalized, width, label='KDTree Hybrid Optimized', 
                    color='blue', alpha=0.7)
    bars2 = ax6.bar(x + width/2, lsh_normalized, width, label='LSH SIMD Parallel', 
                    color='red', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar1, bar2, kdtree_val, lsh_val) in enumerate(zip(bars1, bars2, kdtree_values, lsh_values)):
        ax6.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5,
                f'{kdtree_val:.1f}' if i != 1 else f'{kdtree_val:.0f}',
                ha='center', va='bottom', fontsize=10, weight='bold')
        ax6.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                f'{lsh_val:.1f}' if i != 1 else f'{lsh_val:.0f}',
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax6.set_xlabel('Performance Metrics', fontsize=12)
    ax6.set_ylabel('Normalized Values', fontsize=12)
    ax6.set_title('Target Algorithms Detailed Comparison\n(Latency in ms, QPS in k)', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(['Recall (%)', 'Latency (ms)', 'Speedup', 'QPS (k)'])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Comprehensive ANN Algorithm Analysis: KDTree Hybrid Optimized vs LSH SIMD Parallel', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add summary text box
    summary_text = "Key Findings:\n"
    summary_text += "• KDTree Hybrid Optimized: 99.88% recall, 1607μs latency, 2.16x speedup\n"
    summary_text += "• LSH SIMD Parallel: 85.86% recall, 1027μs latency, 3.38x speedup\n"
    summary_text += "• KDTree achieves near-perfect recall with moderate speedup\n"
    summary_text += "• LSH provides good balance of recall and speed\n"
    summary_text += "• Both outperform basic implementations significantly"
    
    fig.text(0.02, 0.02, summary_text, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Save the figure
    output_file = 'final_algorithms_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Final comparison chart saved as '{output_file}'")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    import os
    print("Creating final algorithms comparison...")
    create_final_comparison()
    print("✓ Final comparison completed!") 