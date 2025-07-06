#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified visualization script for ANN algorithm parameter analysis
"""

import sys
import os

# Set non-interactive backend before importing matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for direct saving
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Error importing matplotlib: {e}")
    print("Please install matplotlib: pip install matplotlib")
    sys.exit(1)

try:
    import pandas as pd
except ImportError as e:
    print(f"Error importing pandas: {e}")
    print("Please install pandas: pip install pandas")
    sys.exit(1)

def create_dummy_data():
    """Create dummy data for testing when CSV file is not available"""
    return pd.DataFrame({
        'algorithm': [
            'Flat Search', 'KDTree CPU', 'LSH CPU', 'KDTree Approx', 'LSH Improved',
            'KDTree SIMD Parallel', 'LSH SIMD Parallel', 'KDTree Ensemble', 
            'Flat Search SIMD Parallel', 'KDTree Hybrid Optimized'
        ],
        'recall_mean': [0.99995, 0.276, 0.534, 0.151, 0.883, 0.425, 0.859, 1.0, 0.99995, 1.0],
        'latency_mean_us': [3469, 480, 920, 124, 1796, 685, 1027, 4579, 809, 3491],
        'speedup': [1.0, 7.23, 3.77, 27.98, 1.93, 5.06, 3.38, 0.76, 4.29, 0.99],
        'parallel': ['false', 'false', 'false', 'false', 'false', 'true', 'true', 'true', 'true', 'true'],
        'max_search_nodes': [None, 1, 1, 15000, None, 8000, None, 10000, None, 8000],
        'num_tables': [None, None, 15, None, 120, None, 120, None, None, None],
        'hash_bits': [None, None, 18, None, 14, None, 14, None, None, None],
        'min_candidates': [None, None, 150, None, 2000, None, 2000, None, None, None]
    })

def plot_comprehensive_analysis():
    """Create comprehensive analysis with all plots"""
    print("Creating comprehensive analysis chart...")
    
    # Load or create data
    csv_file = 'parameter_analysis_results.csv'
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded data from {csv_file}")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            df = create_dummy_data()
            print("✓ Using dummy data for demonstration")
    else:
        df = create_dummy_data()
        print(f"✓ File {csv_file} not found, using dummy data for demonstration")
    
    # Preprocess data
    df['recall_percent'] = df['recall_mean'] * 100
    df['algorithm_type'] = df['algorithm'].apply(lambda x: 'KDTree' if 'KDTree' in x else 
                                                           'LSH' if 'LSH' in x else 'Flat')
    df['is_parallel'] = df['parallel'] == 'true'
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    # 1. Recall vs Speedup scatter plot (like your reference image)
    ax1 = fig.add_subplot(gs[0, :2])
    colors = {'KDTree': 'blue', 'LSH': 'red', 'Flat': 'green'}
    markers = {'KDTree': 'o', 'LSH': 's', 'Flat': '^'}
    
    for algo_type in df['algorithm_type'].unique():
        subset = df[df['algorithm_type'] == algo_type]
        
        # Sequential algorithms
        seq_data = subset[~subset['is_parallel']]
        if len(seq_data) > 0:
            seq_sorted = seq_data.sort_values('recall_percent')
            ax1.plot(seq_sorted['recall_percent'], seq_sorted['speedup'], 
                    color=colors[algo_type], marker=markers[algo_type], 
                    linestyle='-', linewidth=2, markersize=8, alpha=0.8,
                    label=f'{algo_type} (Sequential)')
        
        # Parallel algorithms
        par_data = subset[subset['is_parallel']]
        if len(par_data) > 0:
            par_sorted = par_data.sort_values('recall_percent')
            ax1.plot(par_sorted['recall_percent'], par_sorted['speedup'], 
                    color=colors[algo_type], marker=markers[algo_type], 
                    linestyle='--', linewidth=2, markersize=8, alpha=0.8,
                    label=f'{algo_type} (Parallel)')
    
    ax1.set_xlabel('Recall (%)', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Algorithm Performance: Recall vs Speedup', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')
    
    # 2. Algorithm comparison bar chart
    ax2 = fig.add_subplot(gs[0, 2])
    algorithms = df['algorithm'].tolist()
    speedups = df['speedup'].tolist()
    recalls = df['recall_percent'].tolist()
    
    # Color bars based on parallel vs sequential
    bar_colors = ['lightcoral' if parallel else 'lightblue' for parallel in df['is_parallel']]
    
    bars = ax2.bar(range(len(algorithms)), speedups, color=bar_colors, alpha=0.7, edgecolor='black')
    
    # Add recall annotations
    for i, (bar, recall) in enumerate(zip(bars, recalls)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{recall:.1f}%', ha='center', va='bottom', fontsize=8, weight='bold')
    
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('Algorithm Speedup Comparison\n(with Recall% annotations)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels([alg.replace(' ', '\n') for alg in algorithms], rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Latency vs Recall scatter
    ax3 = fig.add_subplot(gs[1, 0])
    for algo_type in df['algorithm_type'].unique():
        subset = df[df['algorithm_type'] == algo_type]
        
        # Sequential
        seq_data = subset[~subset['is_parallel']]
        if len(seq_data) > 0:
            ax3.scatter(seq_data['latency_mean_us'], seq_data['recall_percent'], 
                       color=colors[algo_type], marker=markers[algo_type], s=100, alpha=0.7,
                       label=f'{algo_type} (Sequential)')
        
        # Parallel
        par_data = subset[subset['is_parallel']]
        if len(par_data) > 0:
            ax3.scatter(par_data['latency_mean_us'], par_data['recall_percent'], 
                       color=colors[algo_type], marker=markers[algo_type], s=100, alpha=0.7,
                       facecolors='none', edgecolors=colors[algo_type], linewidth=2,
                       label=f'{algo_type} (Parallel)')
    
    ax3.set_xlabel('Latency (μs)', fontsize=12)
    ax3.set_ylabel('Recall (%)', fontsize=12)
    ax3.set_title('Latency vs Recall Trade-off', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xscale('log')
    
    # 4. Parallel vs Sequential comparison
    ax4 = fig.add_subplot(gs[1, 1])
    parallel_speedup = df[df['is_parallel']]['speedup'].tolist()
    sequential_speedup = df[~df['is_parallel']]['speedup'].tolist()
    
    bp = ax4.boxplot([sequential_speedup, parallel_speedup], 
                     labels=['Sequential', 'Parallel'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax4.set_ylabel('Speedup', fontsize=12)
    ax4.set_title('Parallel vs Sequential\nSpeedup Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Memory efficiency proxy (inverse of latency)
    ax5 = fig.add_subplot(gs[1, 2])
    df['efficiency'] = df['recall_percent'] / df['latency_mean_us'] * 1000  # Efficiency metric
    
    algorithms_short = [alg.replace(' ', '\n') for alg in df['algorithm']]
    efficiency_values = df['efficiency']
    
    bars = ax5.bar(range(len(algorithms_short)), efficiency_values, 
                   color=bar_colors, alpha=0.7, edgecolor='black')
    
    ax5.set_ylabel('Efficiency\n(Recall% / Latency)', fontsize=12)
    ax5.set_title('Algorithm Efficiency\n(Higher is Better)', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(len(algorithms_short)))
    ax5.set_xticklabels(algorithms_short, rotation=45, ha='right', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6-8. Parameter sensitivity analysis
    plot_parameter_analysis(df, fig, gs)
    
    plt.suptitle('Comprehensive Parameter Analysis for ANN Algorithms', fontsize=16, fontweight='bold')
    
    # Save the figure
    output_file = 'comprehensive_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Comprehensive analysis chart saved as '{output_file}'")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

def plot_parameter_analysis(df, fig, gs):
    """Add parameter sensitivity analysis plots"""
    
    # 6. KDTree parameter analysis
    ax6 = fig.add_subplot(gs[2, 0])
    kdtree_data = df[df['algorithm_type'] == 'KDTree']
    if len(kdtree_data) > 0 and 'max_search_nodes' in kdtree_data.columns:
        kdtree_clean = kdtree_data.dropna(subset=['max_search_nodes'])
        if len(kdtree_clean) > 0:
            ax6.scatter(kdtree_clean['max_search_nodes'], kdtree_clean['recall_percent'], 
                       c='blue', s=100, alpha=0.7)
            ax6.set_xlabel('Max Search Nodes', fontsize=11)
            ax6.set_ylabel('Recall (%)', fontsize=11)
            ax6.set_title('KDTree: Search Nodes\nvs Recall', fontsize=11, fontweight='bold')
            ax6.grid(True, alpha=0.3)
    
    # 7. LSH hash tables analysis
    ax7 = fig.add_subplot(gs[2, 1])
    lsh_data = df[df['algorithm_type'] == 'LSH']
    if len(lsh_data) > 0 and 'num_tables' in lsh_data.columns:
        lsh_clean = lsh_data.dropna(subset=['num_tables'])
        if len(lsh_clean) > 0:
            ax7.scatter(lsh_clean['num_tables'], lsh_clean['recall_percent'], 
                       c='red', s=100, alpha=0.7)
            ax7.set_xlabel('Number of Hash Tables', fontsize=11)
            ax7.set_ylabel('Recall (%)', fontsize=11)
            ax7.set_title('LSH: Hash Tables\nvs Recall', fontsize=11, fontweight='bold')
            ax7.grid(True, alpha=0.3)
    
    # 8. Algorithm evolution (speedup vs recall trade-off)
    ax8 = fig.add_subplot(gs[2, 2])
    
    # Create evolution path
    base_algorithms = ['Flat Search', 'KDTree CPU', 'LSH CPU']
    improved_algorithms = ['KDTree Approx', 'LSH Improved']
    parallel_algorithms = ['KDTree SIMD Parallel', 'LSH SIMD Parallel', 'KDTree Hybrid Optimized']
    
    for alg_group, color, label in [
        (base_algorithms, 'gray', 'Baseline'),
        (improved_algorithms, 'orange', 'Improved'),
        (parallel_algorithms, 'purple', 'Parallel')
    ]:
        subset = df[df['algorithm'].isin(alg_group)]
        if len(subset) > 0:
            ax8.scatter(subset['speedup'], subset['recall_percent'], 
                       c=color, s=100, alpha=0.7, label=label)
    
    ax8.set_xlabel('Speedup', fontsize=11)
    ax8.set_ylabel('Recall (%)', fontsize=11)
    ax8.set_title('Algorithm Evolution:\nSpeedup vs Recall', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=10)
    ax8.set_xscale('log')

def main():
    """Main function"""
    print("=" * 60)
    print("ANN Algorithm Parameter Analysis Visualization")
    print("=" * 60)
    
    try:
        plot_comprehensive_analysis()
        print("\n✓ Visualization completed successfully!")
        print("\nGenerated files:")
        print("- comprehensive_analysis.png: Complete 8-subplot analysis")
        print("  * Algorithm Performance (Recall vs Speedup)")
        print("  * Speedup Comparison with Recall annotations")
        print("  * Latency vs Recall Trade-off")
        print("  * Parallel vs Sequential Distribution")
        print("  * Algorithm Efficiency Comparison")
        print("  * KDTree Parameter Sensitivity")
        print("  * LSH Parameter Sensitivity")
        print("  * Algorithm Evolution Analysis")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 