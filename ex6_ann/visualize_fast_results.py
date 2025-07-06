import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8')
rcParams['figure.figsize'] = (16, 12)
rcParams['font.size'] = 10

def load_and_process_data():
    """Load and process data"""
    try:
        df = pd.read_csv('optimized_algorithms_fast_results.csv')
        print("Successfully loaded data, shape:", df.shape)
        print("\nData preview:")
        print(df.head())
        return df
    except FileNotFoundError:
        print("CSV file not found, creating sample data...")
        # Create sample data if file doesn't exist
        data = {
            'algorithm': ['KDTree Hybrid Optimized'] * 4 + ['LSH SIMD Parallel'] * 4,
            'parallel': [True] * 8,
            'num_trees': [5, 8, 10, 12, -1, -1, -1, -1],
            'search_nodes': [3000, 5000, 8000, 12000, -1, -1, -1, -1],
            'num_tables': [-1, -1, -1, -1, 60, 90, 120, 150],
            'hash_bits': [-1, -1, -1, -1, 14, 14, 14, 14],
            'search_radius': [-1, -1, -1, -1, 8, 10, 10, 12],
            'min_candidates': [-1, -1, -1, -1, 1500, 2000, 2000, 2500],
            'recall_mean': [0.999, 1.000, 1.000, 1.000, 0.734, 0.789, 0.888, 0.894],
            'latency_mean_us': [1441.66, 2423.06, 3824.51, 6246.45, 1165.10, 1325.89, 2126.67, 2190.91],
            'speedup': [2.08, 1.24, 0.78, 0.48, 2.57, 2.26, 1.41, 1.37]
        }
        df = pd.DataFrame(data)
        return df

def create_comprehensive_analysis():
    """Create comprehensive analysis charts"""
    df = load_and_process_data()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('KDTree Hybrid Optimized vs LSH SIMD Parallel: Parameter Analysis and Performance Comparison', fontsize=16, fontweight='bold')
    
    # Separate data for two algorithms
    kdtree_data = df[df['algorithm'] == 'KDTree Hybrid Optimized'].copy()
    lsh_data = df[df['algorithm'] == 'LSH SIMD Parallel'].copy()
    
    # 1. Recall comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(kdtree_data))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, kdtree_data['recall_mean'], width, 
                   label='KDTree Hybrid', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, lsh_data['recall_mean'], width,
                   label='LSH SIMD', color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Configuration ID')
    ax1.set_ylabel('Recall')
    ax1.set_title('(a) Recall Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Config {i+1}' for i in range(len(kdtree_data))])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Query latency comparison
    ax2 = axes[0, 1]
    bars1 = ax2.bar(x_pos - width/2, kdtree_data['latency_mean_us'], width,
                   label='KDTree Hybrid', color='skyblue', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, lsh_data['latency_mean_us'], width,
                   label='LSH SIMD', color='lightcoral', alpha=0.8)
    
    ax2.set_xlabel('Configuration ID')
    ax2.set_ylabel('Query Latency (μs)')
    ax2.set_title('(b) Query Latency Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Config {i+1}' for i in range(len(kdtree_data))])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Speedup comparison
    ax3 = axes[0, 2]
    bars1 = ax3.bar(x_pos - width/2, kdtree_data['speedup'], width,
                   label='KDTree Hybrid', color='skyblue', alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, lsh_data['speedup'], width,
                   label='LSH SIMD', color='lightcoral', alpha=0.8)
    
    ax3.set_xlabel('Configuration ID')
    ax3.set_ylabel('Speedup')
    ax3.set_title('(c) Speedup Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Config {i+1}' for i in range(len(kdtree_data))])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
    
    # 4. KDTree parameter analysis
    ax4 = axes[1, 0]
    # Create scatter plot of tree count vs search nodes
    scatter = ax4.scatter(kdtree_data['num_trees'], kdtree_data['search_nodes'], 
                         c=kdtree_data['recall_mean'], s=kdtree_data['speedup']*100,
                         cmap='viridis', alpha=0.7)
    ax4.set_xlabel('Number of Trees')
    ax4.set_ylabel('Number of Search Nodes')
    ax4.set_title('(d) KDTree Parameter Analysis\n(Color=Recall, Size=Speedup)')
    
    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Recall')
    
    # Add data labels
    for i, row in kdtree_data.iterrows():
        ax4.annotate(f'({row["num_trees"]},{row["search_nodes"]})', 
                    (row['num_trees'], row['search_nodes']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 5. LSH parameter analysis
    ax5 = axes[1, 1]
    # Create relationship plot between table count and recall
    ax5.plot(lsh_data['num_tables'], lsh_data['recall_mean'], 'o-', 
            color='lightcoral', linewidth=2, markersize=8, label='Recall')
    ax5_twin = ax5.twinx()
    ax5_twin.plot(lsh_data['num_tables'], lsh_data['latency_mean_us'], 's-', 
                 color='steelblue', linewidth=2, markersize=8, label='Latency')
    
    ax5.set_xlabel('Number of Hash Tables')
    ax5.set_ylabel('Recall', color='lightcoral')
    ax5_twin.set_ylabel('Query Latency (μs)', color='steelblue')
    ax5.set_title('(e) LSH Parameter Trade-off Analysis')
    ax5.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # 6. Pareto frontier analysis
    ax6 = axes[1, 2]
    # Plot Pareto frontier of recall vs latency
    ax6.scatter(kdtree_data['recall_mean'], kdtree_data['latency_mean_us'], 
               s=100, color='skyblue', alpha=0.8, label='KDTree Hybrid', marker='o')
    ax6.scatter(lsh_data['recall_mean'], lsh_data['latency_mean_us'], 
               s=100, color='lightcoral', alpha=0.8, label='LSH SIMD', marker='s')
    
    # Connect KDTree points to form frontier
    kdtree_sorted = kdtree_data.sort_values('recall_mean')
    ax6.plot(kdtree_sorted['recall_mean'], kdtree_sorted['latency_mean_us'], 
            '--', color='skyblue', alpha=0.5)
    
    # Connect LSH points to form frontier
    lsh_sorted = lsh_data.sort_values('recall_mean')
    ax6.plot(lsh_sorted['recall_mean'], lsh_sorted['latency_mean_us'], 
            '--', color='lightcoral', alpha=0.5)
    
    ax6.set_xlabel('Recall')
    ax6.set_ylabel('Query Latency (μs)')
    ax6.set_title('(f) Pareto Frontier: Recall vs Latency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add configuration labels
    for i, row in df.iterrows():
        if row['algorithm'] == 'KDTree Hybrid Optimized':
            label = f"T{row['num_trees']}"
        else:
            label = f"{row['num_tables']}"
        ax6.annotate(label, (row['recall_mean'], row['latency_mean_us']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('parameter_analysis_fast.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create separate parameter sensitivity analysis chart
    create_parameter_sensitivity_analysis(df)

def create_parameter_sensitivity_analysis(df):
    """Create parameter sensitivity analysis chart"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Algorithm Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
    
    kdtree_data = df[df['algorithm'] == 'KDTree Hybrid Optimized'].copy()
    lsh_data = df[df['algorithm'] == 'LSH SIMD Parallel'].copy()
    
    # 1. KDTree: number of trees vs performance
    ax1 = axes[0, 0]
    ax1.plot(kdtree_data['num_trees'], kdtree_data['recall_mean'], 'o-', 
            color='blue', label='Recall', linewidth=2, markersize=8)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(kdtree_data['num_trees'], kdtree_data['speedup'], 's-', 
                 color='red', label='Speedup', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Trees')
    ax1.set_ylabel('Recall', color='blue')
    ax1_twin.set_ylabel('Speedup', color='red')
    ax1.set_title('(a) KDTree: Number of Trees vs Performance')
    ax1.grid(True, alpha=0.3)
    
    # 2. KDTree: number of search nodes vs performance
    ax2 = axes[0, 1]
    ax2.plot(kdtree_data['search_nodes'], kdtree_data['latency_mean_us'], 'o-', 
            color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Search Nodes')
    ax2.set_ylabel('Query Latency (μs)')
    ax2.set_title('(b) KDTree: Number of Search Nodes vs Latency')
    ax2.grid(True, alpha=0.3)
    
    # 3. LSH: number of hash tables vs performance
    ax3 = axes[1, 0]
    ax3.plot(lsh_data['num_tables'], lsh_data['recall_mean'], 'o-', 
            color='purple', label='Recall', linewidth=2, markersize=8)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(lsh_data['num_tables'], lsh_data['latency_mean_us'], 's-', 
                 color='orange', label='Latency', linewidth=2, markersize=8)
    
    ax3.set_xlabel('Number of Hash Tables')
    ax3.set_ylabel('Recall', color='purple')
    ax3_twin.set_ylabel('Query Latency (μs)', color='orange')
    ax3.set_title('(c) LSH: Number of Hash Tables vs Performance')
    ax3.grid(True, alpha=0.3)
    
    # 4. Optimal configuration identification
    ax4 = axes[1, 1]
    # Calculate performance score (recall * speedup)
    kdtree_data['performance_score'] = kdtree_data['recall_mean'] * kdtree_data['speedup']
    lsh_data['performance_score'] = lsh_data['recall_mean'] * lsh_data['speedup']
    
    configs = [f'KDT-{row["num_trees"]}T' for _, row in kdtree_data.iterrows()] + \
              [f'LSH-{row["num_tables"]}' for _, row in lsh_data.iterrows()]
    scores = list(kdtree_data['performance_score']) + list(lsh_data['performance_score'])
    colors = ['skyblue'] * len(kdtree_data) + ['lightcoral'] * len(lsh_data)
    
    bars = ax4.bar(range(len(configs)), scores, color=colors, alpha=0.8)
    ax4.set_xlabel('Algorithm Configuration')
    ax4.set_ylabel('Performance Score (Recall × Speedup)')
    ax4.set_title('(d) Comprehensive Performance Score Comparison')
    ax4.set_xticks(range(len(configs)))
    ax4.set_xticklabels(configs, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 标注最高得分
    max_idx = np.argmax(scores)
    max_score = scores[max_idx]
    ax4.annotate(f'best: {max_score:.3f}', 
                xy=(max_idx, max_score), xytext=(max_idx, max_score + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_fast.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_performance_summary():
    """Generate performance summary report"""
    df = load_and_process_data()
    
    print("\n" + "="*80)
    print("Algorithm Performance Summary Report")
    print("="*80)
    
    kdtree_data = df[df['algorithm'] == 'KDTree Hybrid Optimized']
    lsh_data = df[df['algorithm'] == 'LSH SIMD Parallel']
    
    print("\n【KDTree Hybrid Optimized Analysis】")
    print("-" * 40)
    best_kdtree = kdtree_data.loc[kdtree_data['recall_mean'].idxmax()]
    fastest_kdtree = kdtree_data.loc[kdtree_data['speedup'].idxmax()]
    
    print(f"Best recall config: {best_kdtree['num_trees']} trees, {best_kdtree['search_nodes']} search nodes")
    print(f"  - Recall: {best_kdtree['recall_mean']:.4f}")
    print(f"  - Latency: {best_kdtree['latency_mean_us']:.2f} μs")
    print(f"  - Speedup: {best_kdtree['speedup']:.2f}x")
    
    print(f"\nFastest config: {fastest_kdtree['num_trees']} trees, {fastest_kdtree['search_nodes']} search nodes")
    print(f"  - Recall: {fastest_kdtree['recall_mean']:.4f}")
    print(f"  - Latency: {fastest_kdtree['latency_mean_us']:.2f} μs")
    print(f"  - Speedup: {fastest_kdtree['speedup']:.2f}x")
    
    print("\n【LSH SIMD Parallel Analysis】")
    print("-" * 40)
    best_lsh = lsh_data.loc[lsh_data['recall_mean'].idxmax()]
    fastest_lsh = lsh_data.loc[lsh_data['speedup'].idxmax()]
    
    print(f"Best recall config: {best_lsh['num_tables']} tables, {best_lsh['hash_bits']} bits, radius {best_lsh['search_radius']}")
    print(f"  - Recall: {best_lsh['recall_mean']:.4f}")
    print(f"  - Latency: {best_lsh['latency_mean_us']:.2f} μs") 
    print(f"  - Speedup: {best_lsh['speedup']:.2f}x")
    
    print(f"\nFastest config: {fastest_lsh['num_tables']} tables, {fastest_lsh['hash_bits']} bits, radius {fastest_lsh['search_radius']}")
    print(f"  - Recall: {fastest_lsh['recall_mean']:.4f}")
    print(f"  - Latency: {fastest_lsh['latency_mean_us']:.2f} μs")
    print(f"  - Speedup: {fastest_lsh['speedup']:.2f}x")
    
    print("\n【Key Findings】")
    print("-" * 40)
    print("1. KDTree Hybrid Optimized:")
    print("   - Achieves perfect recall (99.9%-100%)")
    print("   - Clear parameter trade-offs: more trees and search nodes → higher recall but slower")
    print("   - Best balance point: 5 trees + 3000 search nodes (99.9% recall, 2.08x speedup)")
    
    print("\n2. LSH SIMD Parallel:")
    print("   - Medium performance on high-dimensional data (73.4%-89.4% recall)")
    print("   - Large parameter tuning space: table count significantly affects performance")
    print("   - Speed advantage: still achieves 2.5x+ speedup with lower parameter configs")
    
    print("\n3. Algorithm Selection Recommendations:")
    print("   - For high recall: Choose KDTree Hybrid Optimized (5-8 trees)")
    print("   - For high speed: Choose LSH SIMD (60-90 table configs)")
    print("   - For balanced applications: KDTree 5-tree config or LSH 90-table config")

if __name__ == "__main__":
    print("Starting optimized algorithm parameter analysis visualization...")
    create_comprehensive_analysis()
    generate_performance_summary()
    print("\nVisualization analysis completed! Generated files:")
    print("- parameter_analysis_fast.png")
    print("- parameter_sensitivity_fast.png") 