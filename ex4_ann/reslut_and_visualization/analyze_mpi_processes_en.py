import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set font and style settings
try:
    # Try different fonts
    import matplotlib
    import platform
    
    if platform.system() == 'Windows':
        # Common fonts for Windows
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Calibri']
    else:
        # Linux/Mac systems
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # Test if font is available
    test_fig, test_ax = plt.subplots(1, 1, figsize=(1, 1))
    test_ax.text(0.5, 0.5, 'Test', fontsize=12)
    plt.close(test_fig)
    
except Exception as e:
    print(f"Font setting failed, using default: {e}")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

# Try to use seaborn style
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # Use default style

# Load data files
def load_data():
    df_1proc = pd.read_csv('results_mpi_ivf_1proc.csv')
    df_2proc = pd.read_csv('results_mpi_ivf_2proc.csv')
    df_4proc = pd.read_csv('results_mpi_ivf_4proc.csv')
    
    # Merge data
    df_all = pd.concat([df_1proc, df_2proc, df_4proc], ignore_index=True)
    return df_all, df_1proc, df_2proc, df_4proc

def analyze_performance_metrics(df_all):
    """Analyze performance metrics"""
    print("=== Impact Analysis of MPI Process Count on IVF Algorithm Performance ===\n")
    
    # Analyze by process count
    for processes in [1, 2, 4]:
        df_proc = df_all[df_all['mpi_processes'] == processes]
        avg_latency = df_proc['latency_us'].mean()
        avg_build_time = df_proc['build_time_ms'].mean()
        avg_recall = df_proc['recall'].mean()
        efficiency = df_proc['recall'].sum() / df_proc['latency_us'].sum() * 1000
        
        print(f"{processes} processes configuration:")
        print(f"  Average query latency: {avg_latency:.1f}μs")
        print(f"  Average build time: {avg_build_time:.1f}ms")
        print(f"  Average recall: {avg_recall:.4f}")
        print(f"  Algorithm efficiency: {efficiency:.4f}")
        print()
    
    # Calculate speedup
    print("Speedup analysis:")
    df_1 = df_all[df_all['mpi_processes'] == 1]
    df_2 = df_all[df_all['mpi_processes'] == 2]
    df_4 = df_all[df_all['mpi_processes'] == 4]
    
    for nlist in [64, 128, 256, 512]:
        latency_1 = df_1[df_1['nlist'] == nlist]['latency_us'].mean()
        latency_2 = df_2[df_2['nlist'] == nlist]['latency_us'].mean()
        latency_4 = df_4[df_4['nlist'] == nlist]['latency_us'].mean()
        
        speedup_2 = latency_1 / latency_2
        speedup_4 = latency_1 / latency_4
        
        build_1 = df_1[df_1['nlist'] == nlist]['build_time_ms'].iloc[0]
        build_2 = df_2[df_2['nlist'] == nlist]['build_time_ms'].iloc[0]
        build_4 = df_4[df_4['nlist'] == nlist]['build_time_ms'].iloc[0]
        
        build_speedup_2 = build_1 / build_2
        build_speedup_4 = build_1 / build_4
        
        print(f"nlist={nlist}:")
        print(f"  Query speedup: 2 processes={speedup_2:.2f}x, 4 processes={speedup_4:.2f}x")
        print(f"  Build speedup: 2 processes={build_speedup_2:.2f}x, 4 processes={build_speedup_4:.2f}x")

def create_comprehensive_analysis(df_all):
    """Create comprehensive analysis charts"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Impact Analysis of MPI Process Count on IVF Algorithm Performance', fontsize=16, fontweight='bold')
    
    # 1. Query latency comparison (grouped by nlist)
    for nlist in [64, 128, 256, 512]:
        for processes in [1, 2, 4]:
            subset = df_all[(df_all['nlist'] == nlist) & (df_all['mpi_processes'] == processes)]
            if len(subset) > 0:
                axes[0, 0].plot(subset['nprobe'], subset['latency_us'], 
                              marker='o', label=f'{processes} processes nlist={nlist}', linewidth=2)
    
    axes[0, 0].set_xlabel('nprobe')
    axes[0, 0].set_ylabel('Query Latency (μs)')
    axes[0, 0].set_title('Query Latency Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Build time comparison
    nlist_values = [64, 128, 256, 512]
    processes_values = [1, 2, 4]
    
    build_times = {}
    for proc in processes_values:
        build_times[proc] = []
        for nlist in nlist_values:
            subset = df_all[(df_all['nlist'] == nlist) & (df_all['mpi_processes'] == proc)]
            if len(subset) > 0:
                build_times[proc].append(subset['build_time_ms'].iloc[0])
            else:
                build_times[proc].append(0)
    
    x = np.arange(len(nlist_values))
    width = 0.25
    
    for i, proc in enumerate(processes_values):
        axes[0, 1].bar(x + i*width, build_times[proc], width, 
                      label=f'{proc} processes', alpha=0.8)
    
    axes[0, 1].set_xlabel('nlist')
    axes[0, 1].set_ylabel('Build Time (ms)')
    axes[0, 1].set_title('Build Time Comparison')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(nlist_values)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Speedup analysis
    speedup_data = []
    for nlist in nlist_values:
        baseline = df_all[(df_all['nlist'] == nlist) & (df_all['mpi_processes'] == 1)]['latency_us'].mean()
        for proc in [2, 4]:
            current = df_all[(df_all['nlist'] == nlist) & (df_all['mpi_processes'] == proc)]['latency_us'].mean()
            speedup = baseline / current
            speedup_data.append({'nlist': nlist, 'processes': proc, 'speedup': speedup})
    
    speedup_df = pd.DataFrame(speedup_data)
    
    for proc in [2, 4]:
        subset = speedup_df[speedup_df['processes'] == proc]
        axes[1, 0].plot(subset['nlist'], subset['speedup'], 
                       marker='o', label=f'{proc} processes', linewidth=2, markersize=8)
    
    # Add ideal speedup lines
    axes[1, 0].plot(nlist_values, [2]*4, '--', alpha=0.7, label='Ideal 2x Speedup')
    axes[1, 0].plot(nlist_values, [4]*4, '--', alpha=0.7, label='Ideal 4x Speedup')
    
    axes[1, 0].set_xlabel('nlist')
    axes[1, 0].set_ylabel('Speedup')
    axes[1, 0].set_title('Query Latency Speedup')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Efficiency analysis
    efficiency_data = []
    for proc in processes_values:
        for nlist in nlist_values:
            subset = df_all[(df_all['nlist'] == nlist) & (df_all['mpi_processes'] == proc)]
            if len(subset) > 0:
                avg_efficiency = subset['recall'].sum() / subset['latency_us'].sum() * 1000
                efficiency_data.append({'processes': proc, 'nlist': nlist, 'efficiency': avg_efficiency})
    
    efficiency_df = pd.DataFrame(efficiency_data)
    
    for proc in processes_values:
        subset = efficiency_df[efficiency_df['processes'] == proc]
        axes[1, 1].plot(subset['nlist'], subset['efficiency'], 
                       marker='o', label=f'{proc} processes', linewidth=2, markersize=8)
    
    axes[1, 1].set_xlabel('nlist')
    axes[1, 1].set_ylabel('Efficiency (Recall/Latency × 1000)')
    axes[1, 1].set_title('Algorithm Efficiency Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Recall-latency tradeoff analysis (line plots)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    # Group by process count and connect lines
    for i, proc in enumerate(processes_values):
        subset = df_all[df_all['mpi_processes'] == proc].sort_values('recall')
        if len(subset) > 0:
            axes[2, 0].plot(subset['recall'] * 100, subset['latency_us'],
                           marker=markers[i], color=colors[i],
                           linewidth=2, markersize=6, label=f'{proc} processes',
                           markerfacecolor=colors[i], markeredgecolor='white', markeredgewidth=0.5)
    
    axes[2, 0].set_xlabel('Recall (%)')
    axes[2, 0].set_ylabel('Query Latency (μs)')
    axes[2, 0].set_title('Recall-Latency Tradeoff Curves')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_yscale('log')
    axes[2, 0].set_xlim(30, 105)
    
    # 6. Parallel efficiency (relative to ideal case)
    parallel_efficiency = []
    for nlist in nlist_values:
        baseline = df_all[(df_all['nlist'] == nlist) & (df_all['mpi_processes'] == 1)]['latency_us'].mean()
        for proc in [2, 4]:
            current = df_all[(df_all['nlist'] == nlist) & (df_all['mpi_processes'] == proc)]['latency_us'].mean()
            speedup = baseline / current
            efficiency = speedup / proc * 100  # Parallel efficiency percentage
            parallel_efficiency.append({'nlist': nlist, 'processes': proc, 'efficiency': efficiency})
    
    parallel_eff_df = pd.DataFrame(parallel_efficiency)
    
    for proc in [2, 4]:
        subset = parallel_eff_df[parallel_eff_df['processes'] == proc]
        axes[2, 1].plot(subset['nlist'], subset['efficiency'], 
                       marker='o', label=f'{proc} processes', linewidth=2, markersize=8)
    
    axes[2, 1].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Ideal Efficiency (100%)')
    axes[2, 1].set_xlabel('nlist')
    axes[2, 1].set_ylabel('Parallel Efficiency (%)')
    axes[2, 1].set_title('Parallel Efficiency')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Leave space for legend
    plt.savefig('mpi_processes_analysis_en.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_detailed_metrics(df_all):
    """Calculate detailed performance metrics"""
    print("\n=== Detailed Performance Metrics Analysis ===")
    
    # Calculate metrics
    metrics = {}
    for proc in [1, 2, 4]:
        subset = df_all[df_all['mpi_processes'] == proc]
        metrics[proc] = {
            'avg_latency': subset['latency_us'].mean(),
            'avg_build_time': subset['build_time_ms'].mean(),
            'avg_recall': subset['recall'].mean(),
            'total_efficiency': subset['recall'].sum() / subset['latency_us'].sum() * 1000
        }
    
    print("Performance Metrics Summary:")
    print("Processes\tAvg Latency(μs)\tAvg Build Time(ms)\tAvg Recall\tAlgorithm Efficiency")
    for proc in [1, 2, 4]:
        m = metrics[proc]
        print(f"{proc}\t\t{m['avg_latency']:.1f}\t\t{m['avg_build_time']:.1f}\t\t\t{m['avg_recall']:.4f}\t\t{m['total_efficiency']:.4f}")
    
    # Calculate improvement
    print("\nPerformance improvement relative to single process:")
    baseline_latency = metrics[1]['avg_latency']
    baseline_build = metrics[1]['avg_build_time']
    
    for proc in [2, 4]:
        latency_improvement = (baseline_latency - metrics[proc]['avg_latency']) / baseline_latency * 100
        build_improvement = (baseline_build - metrics[proc]['avg_build_time']) / baseline_build * 100
        efficiency_improvement = (metrics[proc]['total_efficiency'] - metrics[1]['total_efficiency']) / metrics[1]['total_efficiency'] * 100
        
        print(f"{proc} processes:")
        print(f"  Query latency improvement: {latency_improvement:.2f}%")
        print(f"  Build time improvement: {build_improvement:.2f}%")
        print(f"  Algorithm efficiency improvement: {efficiency_improvement:.2f}%")

if __name__ == "__main__":
    try:
        df_all, df_1proc, df_2proc, df_4proc = load_data()
        analyze_performance_metrics(df_all)
        calculate_detailed_metrics(df_all)
        create_comprehensive_analysis(df_all)
    except Exception as e:
        print(f"Error occurred during analysis: {e}")
        print("Please ensure CSV files exist and are formatted correctly") 