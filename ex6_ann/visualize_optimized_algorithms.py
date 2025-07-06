#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specialized visualization for optimized ANN algorithms comparison
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

def create_dummy_optimized_data():
    """Create dummy data for optimized algorithms testing"""
    kdtree_data = []
    lsh_data = []
    
    # KDTree Hybrid Optimized data
    tree_counts = [5, 8, 10, 12, 15]
    search_nodes = [3000, 5000, 8000, 12000, 15000]
    
    for trees in tree_counts:
        for nodes in search_nodes:
            # Simulate realistic performance data
            recall = min(0.99, 0.75 + (trees/20) + (nodes/20000))
            latency = 1000 + (trees * 200) + (nodes * 0.3)
            speedup = 3500 / latency
            
            kdtree_data.append({
                'algorithm': 'KDTree Hybrid Optimized',
                'parallel': 'true',
                'num_trees': trees,
                'search_nodes': nodes,
                'num_tables': -1,
                'hash_bits': -1,
                'search_radius': -1,
                'min_candidates': -1,
                'recall_mean': recall,
                'recall_std': 0.02,
                'latency_mean_us': latency,
                'latency_std_us': latency * 0.1,
                'speedup': speedup
            })
    
    # LSH SIMD Parallel data
    lsh_configs = [
        (60, 14, 8, 1500),   # Fast config
        (90, 14, 10, 2000),  # Balanced config
        (120, 14, 10, 2000), # Original optimal config
        (150, 14, 12, 2500), # High recall config
        (120, 12, 10, 2000), # Low bits config
        (120, 16, 10, 2000), # High bits config
        (120, 14, 5, 2000),  # Small radius config
        (120, 14, 15, 2000), # Large radius config
        (120, 14, 10, 1000), # Few candidates config
        (120, 14, 10, 4000), # Many candidates config
        (200, 16, 12, 3000), # Ultra high config
    ]
    
    for tables, bits, radius, candidates in lsh_configs:
        # Simulate realistic LSH performance
        recall = min(0.95, 0.50 + (tables/300) + (bits/40) + (radius/50) + (candidates/10000))
        latency = 500 + (tables * 8) + (bits * 50) + (radius * 20) + (candidates * 0.5)
        speedup = 3500 / latency
        
        lsh_data.append({
            'algorithm': 'LSH SIMD Parallel',
            'parallel': 'true',
            'num_trees': -1,
            'search_nodes': -1,
            'num_tables': tables,
            'hash_bits': bits,
            'search_radius': radius,
            'min_candidates': candidates,
            'recall_mean': recall,
            'recall_std': 0.03,
            'latency_mean_us': latency,
            'latency_std_us': latency * 0.15,
            'speedup': speedup
        })
    
    return pd.DataFrame(kdtree_data + lsh_data)

def load_optimized_data(filename='optimized_algorithms_results.csv'):
    """Load optimized algorithms experimental data"""
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            print(f"✓ Loaded data from {filename}")
            return df
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    print(f"✓ File {filename} not found, using dummy data for demonstration")
    return create_dummy_optimized_data()

def plot_optimized_algorithms_analysis():
    """Create comprehensive analysis for optimized algorithms"""
    print("Creating optimized algorithms analysis chart...")
    
    df = load_optimized_data()
    df['recall_percent'] = df['recall_mean'] * 100
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)
    
    # 1. Overall Performance Comparison (Recall vs QPS)
    ax1 = fig.add_subplot(gs[0, :2])
    df['qps'] = 1000000 / df['latency_mean_us']
    
    kdtree_data = df[df['algorithm'] == 'KDTree Hybrid Optimized']
    lsh_data = df[df['algorithm'] == 'LSH SIMD Parallel']
    
    # Plot KDTree performance
    if len(kdtree_data) > 0:
        kdtree_sorted = kdtree_data.sort_values('recall_percent')
        ax1.plot(kdtree_sorted['recall_percent'], kdtree_sorted['qps'], 
                'bo-', linewidth=3, markersize=8, label='KDTree Hybrid Optimized', alpha=0.8)
    
    # Plot LSH performance
    if len(lsh_data) > 0:
        lsh_sorted = lsh_data.sort_values('recall_percent')
        ax1.plot(lsh_sorted['recall_percent'], lsh_sorted['qps'], 
                'rs-', linewidth=3, markersize=8, label='LSH SIMD Parallel', alpha=0.8)
    
    ax1.set_xlabel('Recall (%)', fontsize=14)
    ax1.set_ylabel('QPS (Queries Per Second)', fontsize=14)
    ax1.set_title('Optimized Algorithms: Recall vs QPS Performance', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_yscale('log')
    
    # 2. Speedup Comparison
    ax2 = fig.add_subplot(gs[0, 2])
    algorithms = ['KDTree Hybrid\nOptimized', 'LSH SIMD\nParallel']
    speedups = [kdtree_data['speedup'].mean() if len(kdtree_data) > 0 else 0,
                lsh_data['speedup'].mean() if len(lsh_data) > 0 else 0]
    recalls = [kdtree_data['recall_percent'].mean() if len(kdtree_data) > 0 else 0,
               lsh_data['recall_percent'].mean() if len(lsh_data) > 0 else 0]
    
    bars = ax2.bar(algorithms, speedups, color=['lightblue', 'lightcoral'], alpha=0.7, edgecolor='black')
    
    # Add recall annotations
    for bar, recall in zip(bars, recalls):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{recall:.1f}%', ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax2.set_ylabel('Average Speedup', fontsize=12)
    ax2.set_title('Average Speedup Comparison\n(with Recall% annotations)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Efficiency Analysis (top right)
    ax3 = fig.add_subplot(gs[0, 3])
    df['efficiency'] = df['recall_percent'] / df['latency_mean_us'] * 1000
    
    efficiency_data = df.groupby('algorithm')['efficiency'].agg(['mean', 'std']).reset_index()
    
    x_pos = range(len(efficiency_data))
    ax3.bar(x_pos, efficiency_data['mean'], 
           yerr=efficiency_data['std'], capsize=5,
           color=['lightblue', 'lightcoral'], alpha=0.7, edgecolor='black')
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([alg.replace(' ', '\n') for alg in efficiency_data['algorithm']], fontsize=10)
    ax3.set_ylabel('Efficiency\n(Recall% / Latency)', fontsize=11)
    ax3.set_title('Algorithm Efficiency\n(Higher is Better)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. KDTree Parameter Analysis - Trees vs Performance
    ax4 = fig.add_subplot(gs[1, 0])
    if len(kdtree_data) > 0:
        kdtree_trees = kdtree_data.groupby('num_trees').agg({
            'recall_percent': 'mean',
            'speedup': 'mean'
        }).reset_index()
        
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(kdtree_trees['num_trees'], kdtree_trees['recall_percent'], 
                        'bo-', linewidth=2, markersize=6, label='Recall')
        line2 = ax4_twin.plot(kdtree_trees['num_trees'], kdtree_trees['speedup'], 
                             'ro-', linewidth=2, markersize=6, label='Speedup')
        
        ax4.set_xlabel('Number of Trees', fontsize=11)
        ax4.set_ylabel('Recall (%)', color='blue', fontsize=11)
        ax4_twin.set_ylabel('Speedup', color='red', fontsize=11)
        ax4.set_title('KDTree: Trees Impact', fontsize=12, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='center right')
        ax4.grid(True, alpha=0.3)
    
    # 5. KDTree Parameter Analysis - Search Nodes vs Performance
    ax5 = fig.add_subplot(gs[1, 1])
    if len(kdtree_data) > 0:
        kdtree_nodes = kdtree_data.groupby('search_nodes').agg({
            'recall_percent': 'mean',
            'latency_mean_us': 'mean'
        }).reset_index()
        
        ax5_twin = ax5.twinx()
        
        line1 = ax5.plot(kdtree_nodes['search_nodes'], kdtree_nodes['recall_percent'], 
                        'go-', linewidth=2, markersize=6, label='Recall')
        line2 = ax5_twin.plot(kdtree_nodes['search_nodes'], kdtree_nodes['latency_mean_us'], 
                             'mo-', linewidth=2, markersize=6, label='Latency')
        
        ax5.set_xlabel('Search Nodes', fontsize=11)
        ax5.set_ylabel('Recall (%)', color='green', fontsize=11)
        ax5_twin.set_ylabel('Latency (μs)', color='magenta', fontsize=11)
        ax5.set_title('KDTree: Search Nodes Impact', fontsize=12, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='center right')
        ax5.grid(True, alpha=0.3)
    
    # 6. LSH Parameter Analysis - Hash Tables vs Performance
    ax6 = fig.add_subplot(gs[1, 2])
    if len(lsh_data) > 0:
        lsh_tables = lsh_data.groupby('num_tables').agg({
            'recall_percent': 'mean',
            'speedup': 'mean'
        }).reset_index()
        
        ax6_twin = ax6.twinx()
        
        line1 = ax6.plot(lsh_tables['num_tables'], lsh_tables['recall_percent'], 
                        'co-', linewidth=2, markersize=6, label='Recall')
        line2 = ax6_twin.plot(lsh_tables['num_tables'], lsh_tables['speedup'], 
                             'yo-', linewidth=2, markersize=6, label='Speedup')
        
        ax6.set_xlabel('Hash Tables', fontsize=11)
        ax6.set_ylabel('Recall (%)', color='cyan', fontsize=11)
        ax6_twin.set_ylabel('Speedup', color='orange', fontsize=11)
        ax6.set_title('LSH: Hash Tables Impact', fontsize=12, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='center right')
        ax6.grid(True, alpha=0.3)
    
    # 7. LSH Parameter Analysis - Hash Bits vs Performance
    ax7 = fig.add_subplot(gs[1, 3])
    if len(lsh_data) > 0:
        lsh_bits = lsh_data.groupby('hash_bits').agg({
            'recall_percent': 'mean',
            'latency_mean_us': 'mean'
        }).reset_index()
        
        ax7_twin = ax7.twinx()
        
        line1 = ax7.plot(lsh_bits['hash_bits'], lsh_bits['recall_percent'], 
                        'ko-', linewidth=2, markersize=6, label='Recall')
        line2 = ax7_twin.plot(lsh_bits['hash_bits'], lsh_bits['latency_mean_us'], 
                             'ro-', linewidth=2, markersize=6, label='Latency')
        
        ax7.set_xlabel('Hash Bits', fontsize=11)
        ax7.set_ylabel('Recall (%)', color='black', fontsize=11)
        ax7_twin.set_ylabel('Latency (μs)', color='red', fontsize=11)
        ax7.set_title('LSH: Hash Bits Impact', fontsize=12, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax7.legend(lines, labels, loc='center right')
        ax7.grid(True, alpha=0.3)
    
    # 8. 3D Parameter Analysis for KDTree
    ax8 = fig.add_subplot(gs[2, :2], projection='3d')
    if len(kdtree_data) > 0:
        scatter = ax8.scatter(kdtree_data['num_trees'], kdtree_data['search_nodes'], 
                             kdtree_data['recall_percent'], 
                             c=kdtree_data['speedup'], cmap='viridis', s=60, alpha=0.7)
        ax8.set_xlabel('Number of Trees', fontsize=10)
        ax8.set_ylabel('Search Nodes', fontsize=10)
        ax8.set_zlabel('Recall (%)', fontsize=10)
        ax8.set_title('KDTree: 3D Parameter Space\n(Color = Speedup)', fontsize=11, fontweight='bold')
        plt.colorbar(scatter, ax=ax8, shrink=0.5, aspect=10)
    
    # 9. LSH Multi-parameter Heatmap
    ax9 = fig.add_subplot(gs[2, 2:])
    if len(lsh_data) > 0:
        # Create pivot table for heatmap
        pivot_data = lsh_data.pivot_table(
            values='recall_percent', 
            index='num_tables', 
            columns='hash_bits', 
            aggfunc='mean'
        )
        
        im = ax9.imshow(pivot_data.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax9.set_xticks(range(len(pivot_data.columns)))
        ax9.set_xticklabels(pivot_data.columns)
        ax9.set_yticks(range(len(pivot_data.index)))
        ax9.set_yticklabels(pivot_data.index)
        ax9.set_xlabel('Hash Bits', fontsize=11)
        ax9.set_ylabel('Hash Tables', fontsize=11)
        ax9.set_title('LSH: Tables vs Bits Recall Heatmap', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                if not pd.isna(pivot_data.iloc[i, j]):
                    text = ax9.text(j, i, f'{pivot_data.iloc[i, j]:.1f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax9, shrink=0.8)
    
    # 10. Performance Trade-offs Scatter Plot
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.scatter(kdtree_data['latency_mean_us'], kdtree_data['recall_percent'], 
                c='blue', alpha=0.6, s=80, label='KDTree Hybrid', marker='o')
    ax10.scatter(lsh_data['latency_mean_us'], lsh_data['recall_percent'], 
                c='red', alpha=0.6, s=80, label='LSH SIMD', marker='s')
    
    ax10.set_xlabel('Latency (μs)', fontsize=11)
    ax10.set_ylabel('Recall (%)', fontsize=11)
    ax10.set_title('Latency vs Recall Trade-off', fontsize=12, fontweight='bold')
    ax10.grid(True, alpha=0.3)
    ax10.legend()
    ax10.set_xscale('log')
    
    # 11. Parameter Sensitivity Analysis
    ax11 = fig.add_subplot(gs[3, 1])
    
    # Calculate coefficient of variation for each parameter
    kdtree_cv_trees = kdtree_data.groupby('num_trees')['recall_percent'].std().mean() / kdtree_data['recall_percent'].mean()
    kdtree_cv_nodes = kdtree_data.groupby('search_nodes')['recall_percent'].std().mean() / kdtree_data['recall_percent'].mean()
    
    lsh_cv_tables = lsh_data.groupby('num_tables')['recall_percent'].std().mean() / lsh_data['recall_percent'].mean()
    lsh_cv_bits = lsh_data.groupby('hash_bits')['recall_percent'].std().mean() / lsh_data['recall_percent'].mean()
    
    params = ['KDTree\nTrees', 'KDTree\nNodes', 'LSH\nTables', 'LSH\nBits']
    sensitivities = [kdtree_cv_trees, kdtree_cv_nodes, lsh_cv_tables, lsh_cv_bits]
    
    bars = ax11.bar(params, sensitivities, color=['lightblue', 'blue', 'lightcoral', 'red'], alpha=0.7)
    ax11.set_ylabel('Parameter Sensitivity\n(Coefficient of Variation)', fontsize=10)
    ax11.set_title('Parameter Sensitivity Analysis', fontsize=12, fontweight='bold')
    ax11.grid(True, alpha=0.3, axis='y')
    
    # 12. Best Configurations Summary
    ax12 = fig.add_subplot(gs[3, 2:])
    
    # Find best configurations
    best_kdtree = kdtree_data.loc[kdtree_data['recall_percent'].idxmax()] if len(kdtree_data) > 0 else None
    best_lsh = lsh_data.loc[lsh_data['recall_percent'].idxmax()] if len(lsh_data) > 0 else None
    
    fastest_kdtree = kdtree_data.loc[kdtree_data['speedup'].idxmax()] if len(kdtree_data) > 0 else None
    fastest_lsh = lsh_data.loc[lsh_data['speedup'].idxmax()] if len(lsh_data) > 0 else None
    
    ax12.axis('off')
    
    summary_text = "Best Configuration Summary\n\n"
    summary_text += "Highest Recall Configurations:\n"
    if best_kdtree is not None:
        summary_text += f"• KDTree: {best_kdtree['num_trees']} trees, {best_kdtree['search_nodes']} nodes\n"
        summary_text += f"  Recall: {best_kdtree['recall_percent']:.1f}%, Speedup: {best_kdtree['speedup']:.2f}x\n"
    if best_lsh is not None:
        summary_text += f"• LSH: {best_lsh['num_tables']} tables, {best_lsh['hash_bits']} bits\n"
        summary_text += f"  Recall: {best_lsh['recall_percent']:.1f}%, Speedup: {best_lsh['speedup']:.2f}x\n\n"
    
    summary_text += "Fastest Configurations:\n"
    if fastest_kdtree is not None:
        summary_text += f"• KDTree: {fastest_kdtree['num_trees']} trees, {fastest_kdtree['search_nodes']} nodes\n"
        summary_text += f"  Recall: {fastest_kdtree['recall_percent']:.1f}%, Speedup: {fastest_kdtree['speedup']:.2f}x\n"
    if fastest_lsh is not None:
        summary_text += f"• LSH: {fastest_lsh['num_tables']} tables, {fastest_lsh['hash_bits']} bits\n"
        summary_text += f"  Recall: {fastest_lsh['recall_percent']:.1f}%, Speedup: {fastest_lsh['speedup']:.2f}x\n"
    
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Comprehensive Analysis: KDTree Hybrid Optimized vs LSH SIMD Parallel', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save the figure
    output_file = 'optimized_algorithms_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Optimized algorithms analysis saved as '{output_file}'")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

def main():
    """Main function"""
    print("=" * 70)
    print("Optimized ANN Algorithms Analysis Visualization")
    print("=" * 70)
    
    try:
        plot_optimized_algorithms_analysis()
        print("\n✓ Visualization completed successfully!")
        print("\nGenerated files:")
        print("- optimized_algorithms_analysis.png: Complete 12-subplot analysis")
        print("  * Overall Performance Comparison (Recall vs QPS)")
        print("  * Average Speedup Comparison")
        print("  * Algorithm Efficiency Analysis")
        print("  * KDTree Trees Parameter Impact")
        print("  * KDTree Search Nodes Parameter Impact") 
        print("  * LSH Hash Tables Parameter Impact")
        print("  * LSH Hash Bits Parameter Impact")
        print("  * KDTree 3D Parameter Space Visualization")
        print("  * LSH Multi-parameter Heatmap")
        print("  * Latency vs Recall Trade-off")
        print("  * Parameter Sensitivity Analysis")
        print("  * Best Configurations Summary")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 