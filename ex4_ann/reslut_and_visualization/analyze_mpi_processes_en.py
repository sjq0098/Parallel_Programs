import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set English font and style to avoid Chinese font issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

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
    
    # Combine data
    df_all = pd.concat([df_1proc, df_2proc, df_4proc], ignore_index=True)
    return df_all, df_1proc, df_2proc, df_4proc

def analyze_performance_metrics(df_all):
    """Analyze performance metrics"""
    print("=== MPI Process Count Impact Analysis on IVF Algorithm ===\n")
    
    # Group analysis by process count
    for processes in [1, 2, 4]:
        df_proc = df_all[df_all['mpi_processes'] == processes]
        avg_latency = df_proc['latency_us'].mean()
        avg_build_time = df_proc['build_time_ms'].mean()
        avg_recall = df_proc['recall'].mean()
        efficiency = df_proc['recall'].sum() / df_proc['latency_us'].sum() * 1000
        
        print(f"{processes} Process Configuration:")
        print(f"  Average Query Latency: {avg_latency:.1f}μs")
        print(f"  Average Build Time: {avg_build_time:.1f}ms")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Algorithm Efficiency: {efficiency:.4f}")
        print()
    
    # Calculate speedup ratios
    print("Speedup Analysis:")
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
        print(f"  Query Speedup: 2proc={speedup_2:.2f}x, 4proc={speedup_4:.2f}x")
        print(f"  Build Speedup: 2proc={build_speedup_2:.2f}x, 4proc={build_speedup_4:.2f}x")

def create_comprehensive_analysis(df_all):
    """Create comprehensive analysis charts"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('MPI Process Count Impact Analysis on IVF Algorithm Performance', fontsize=16, fontweight='bold')
    
    # 1. Query latency comparison (grouped by nlist)
    for nlist in [64, 128, 256, 512]:
        for processes in [1, 2, 4]:
            subset = df_all[(df_all['nlist'] == nlist) & (df_all['mpi_processes'] == processes)]
            if len(subset) > 0:
                axes[0, 0].plot(subset['nprobe'], subset['latency_us'], 
                              marker='o', label=f'{processes}proc nlist={nlist}', linewidth=2)
    
    axes[0, 0].set_xlabel('nprobe')
    axes[0, 0].set_ylabel('Query Latency (microseconds)')
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
                      label=f'{proc} Processes', alpha=0.8)
    
    axes[0, 1].set_xlabel('nlist')
    axes[0, 1].set_ylabel('Build Time (milliseconds)')
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
                       marker='o', label=f'{proc} Processes', linewidth=2, markersize=8)
    
    # Add ideal speedup lines
    axes[1, 0].plot(nlist_values, [2]*4, '--', alpha=0.7, label='Ideal 2x Speedup')
    axes[1, 0].plot(nlist_values, [4]*4, '--', alpha=0.7, label='Ideal 4x Speedup')
    
    axes[1, 0].set_xlabel('nlist')
    axes[1, 0].set_ylabel('Speedup Ratio')
    axes[1, 0].set_title('Query Latency Speedup Ratio')
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
                       marker='o', label=f'{proc} Processes', linewidth=2, markersize=8)
    
    axes[1, 1].set_xlabel('nlist')
    axes[1, 1].set_ylabel('Efficiency (Recall/Latency × 1000)')
    axes[1, 1].set_title('Algorithm Efficiency Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Recall-latency tradeoff (scatter plot)
    colors = ['blue', 'green', 'red']
    for i, proc in enumerate(processes_values):
        subset = df_all[df_all['mpi_processes'] == proc]
        axes[2, 0].scatter(subset['latency_us'], subset['recall'], 
                          alpha=0.7, s=60, label=f'{proc} Processes', c=colors[i])
    
    axes[2, 0].set_xlabel('Query Latency (microseconds)')
    axes[2, 0].set_ylabel('Recall')
    axes[2, 0].set_title('Recall-Latency Tradeoff')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xscale('log')
    
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
                       marker='o', label=f'{proc} Processes', linewidth=2, markersize=8)
    
    axes[2, 1].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Ideal Efficiency (100%)')
    axes[2, 1].set_xlabel('nlist')
    axes[2, 1].set_ylabel('Parallel Efficiency (%)')
    axes[2, 1].set_title('Parallel Efficiency')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mpi_processes_analysis_en.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_detailed_metrics(df_all):
    """Calculate detailed performance metrics"""
    print("\n=== Detailed Performance Metrics Analysis ===")
    
    # Calculate metrics for each configuration
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
        print(f"{proc}\t\t{m['avg_latency']:.1f}\t\t{m['avg_build_time']:.1f}\t\t{m['avg_recall']:.4f}\t\t{m['total_efficiency']:.4f}")
    
    # Calculate improvement rates
    print("\nPerformance Improvement Relative to Single Process:")
    baseline_latency = metrics[1]['avg_latency']
    baseline_build = metrics[1]['avg_build_time']
    
    for proc in [2, 4]:
        latency_improvement = (baseline_latency - metrics[proc]['avg_latency']) / baseline_latency * 100
        build_improvement = (baseline_build - metrics[proc]['avg_build_time']) / baseline_build * 100
        efficiency_improvement = (metrics[proc]['total_efficiency'] - metrics[1]['total_efficiency']) / metrics[1]['total_efficiency'] * 100
        
        print(f"{proc} Processes:")
        print(f"  Query Latency Improvement: {latency_improvement:.2f}%")
        print(f"  Build Time Improvement: {build_improvement:.2f}%")
        print(f"  Algorithm Efficiency Improvement: {efficiency_improvement:.2f}%")

if __name__ == "__main__":
    try:
        df_all, df_1proc, df_2proc, df_4proc = load_data()
        analyze_performance_metrics(df_all)
        calculate_detailed_metrics(df_all)
        create_comprehensive_analysis(df_all)
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please ensure CSV files exist and are properly formatted") 