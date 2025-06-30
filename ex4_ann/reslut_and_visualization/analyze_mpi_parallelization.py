import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass

def load_and_merge_data():
    """加载并合并1进程和2进程数据"""
    df_1proc = pd.read_csv('results_mpi_ivf_1proc.csv')
    df_2proc = pd.read_csv('results_mpi_ivf_2proc.csv')
    
    # 添加标识列
    df_1proc['config'] = '1 Process'
    df_2proc['config'] = '2 Processes'
    
    # 合并数据
    df_combined = pd.concat([df_1proc, df_2proc], ignore_index=True)
    
    return df_combined, df_1proc, df_2proc

def calculate_parallelization_metrics(df_1proc, df_2proc):
    """计算并行化性能指标"""
    print("=== MPI并行化IVF算法性能分析 ===\n")
    
    metrics = {}
    
    # 按nlist分组分析
    for nlist in [64, 128, 256, 512]:
        print(f"nlist = {nlist}:")
        
        subset_1 = df_1proc[df_1proc['nlist'] == nlist]
        subset_2 = df_2proc[df_2proc['nlist'] == nlist]
        
        # 计算平均性能
        avg_latency_1 = subset_1['latency_us'].mean()
        avg_latency_2 = subset_2['latency_us'].mean()
        avg_recall_1 = subset_1['recall'].mean()
        avg_recall_2 = subset_2['recall'].mean()
        
        build_time_1 = subset_1['build_time_ms'].iloc[0]
        build_time_2 = subset_2['build_time_ms'].iloc[0]
        
        # 计算改进率
        latency_improvement = (avg_latency_1 - avg_latency_2) / avg_latency_1 * 100
        build_improvement = (build_time_1 - build_time_2) / build_time_1 * 100
        speedup = avg_latency_1 / avg_latency_2
        parallel_efficiency = speedup / 2 * 100
        
        metrics[nlist] = {
            'latency_1': avg_latency_1,
            'latency_2': avg_latency_2,
            'latency_improvement': latency_improvement,
            'build_1': build_time_1,
            'build_2': build_time_2,
            'build_improvement': build_improvement,
            'speedup': speedup,
            'parallel_efficiency': parallel_efficiency
        }
        
        print(f"  查询延迟: {avg_latency_1:.1f}μs → {avg_latency_2:.1f}μs (改进{latency_improvement:.1f}%)")
        print(f"  构建时间: {build_time_1:.1f}ms → {build_time_2:.1f}ms (改进{build_improvement:.1f}%)")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  并行效率: {parallel_efficiency:.1f}%")
        print()
    
    return metrics

def analyze_parameter_effects(df_1proc, df_2proc):
    """分析参数对并行化效果的影响"""
    print("=== 参数对并行化效果的影响分析 ===\n")
    
    # nprobe对并行化效果的影响
    print("nprobe参数影响分析:")
    for nlist in [64, 128, 256, 512]:
        print(f"\nnlist = {nlist}:")
        subset_1 = df_1proc[df_1proc['nlist'] == nlist]
        subset_2 = df_2proc[df_2proc['nlist'] == nlist]
        
        for nprobe in [1, 2, 4, 8, 16, 32, 64]:
            latency_1 = subset_1[subset_1['nprobe'] == nprobe]['latency_us'].iloc[0]
            latency_2 = subset_2[subset_2['nprobe'] == nprobe]['latency_us'].iloc[0]
            speedup = latency_1 / latency_2
            improvement = (latency_1 - latency_2) / latency_1 * 100
            
            print(f"  nprobe={nprobe:2d}: {latency_1:4.0f}μs → {latency_2:4.0f}μs (加速{speedup:.2f}x, 改进{improvement:5.1f}%)")

def create_parallelization_analysis_plots(df_combined, df_1proc, df_2proc):
    """创建并行化分析图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MPI Parallelization Analysis: 1 Process vs 2 Processes IVF Algorithm', 
                 fontsize=16, fontweight='bold')
    
    # 1. 延迟对比 - 按nlist分组
    nlist_values = [64, 128, 256, 512]
    nprobe_values = [1, 2, 4, 8, 16, 32, 64]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, nlist in enumerate(nlist_values):
        subset_1 = df_1proc[df_1proc['nlist'] == nlist]
        subset_2 = df_2proc[df_2proc['nlist'] == nlist]
        
        axes[0, 0].plot(subset_1['nprobe'], subset_1['latency_us'], 
                       marker='o', linestyle='-', label=f'1 Proc nlist={nlist}', 
                       color=colors[i], alpha=0.7)
        axes[0, 0].plot(subset_2['nprobe'], subset_2['latency_us'], 
                       marker='s', linestyle='--', label=f'2 Proc nlist={nlist}', 
                       color=colors[i])
    
    axes[0, 0].set_xlabel('nprobe')
    axes[0, 0].set_ylabel('Query Latency (μs)')
    axes[0, 0].set_title('Query Latency: 1 vs 2 Processes')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. 加速比分析
    speedup_data = []
    for nlist in nlist_values:
        subset_1 = df_1proc[df_1proc['nlist'] == nlist]
        subset_2 = df_2proc[df_2proc['nlist'] == nlist]
        
        speedups = []
        for nprobe in nprobe_values:
            latency_1 = subset_1[subset_1['nprobe'] == nprobe]['latency_us'].iloc[0]
            latency_2 = subset_2[subset_2['nprobe'] == nprobe]['latency_us'].iloc[0]
            speedup = latency_1 / latency_2
            speedups.append(speedup)
        
        axes[0, 1].plot(nprobe_values, speedups, marker='o', 
                       label=f'nlist={nlist}', linewidth=2, color=colors[nlist_values.index(nlist)])
    
    axes[0, 1].axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Ideal 2x Speedup')
    axes[0, 1].axhline(y=1, color='gray', linestyle='-', alpha=0.5, label='No Speedup')
    axes[0, 1].set_xlabel('nprobe')
    axes[0, 1].set_ylabel('Speedup Ratio')
    axes[0, 1].set_title('Speedup Analysis by Parameters')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 并行效率
    efficiency_data = []
    for nlist in nlist_values:
        subset_1 = df_1proc[df_1proc['nlist'] == nlist]
        subset_2 = df_2proc[df_2proc['nlist'] == nlist]
        
        efficiencies = []
        for nprobe in nprobe_values:
            latency_1 = subset_1[subset_1['nprobe'] == nprobe]['latency_us'].iloc[0]
            latency_2 = subset_2[subset_2['nprobe'] == nprobe]['latency_us'].iloc[0]
            speedup = latency_1 / latency_2
            efficiency = speedup / 2 * 100  # 并行效率百分比
            efficiencies.append(efficiency)
        
        axes[0, 2].plot(nprobe_values, efficiencies, marker='o', 
                       label=f'nlist={nlist}', linewidth=2, color=colors[nlist_values.index(nlist)])
    
    axes[0, 2].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Ideal Efficiency (100%)')
    axes[0, 2].set_xlabel('nprobe')
    axes[0, 2].set_ylabel('Parallel Efficiency (%)')
    axes[0, 2].set_title('Parallel Efficiency by Parameters')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 构建时间对比
    build_comparison = []
    for nlist in nlist_values:
        build_1 = df_1proc[df_1proc['nlist'] == nlist]['build_time_ms'].iloc[0]
        build_2 = df_2proc[df_2proc['nlist'] == nlist]['build_time_ms'].iloc[0]
        build_comparison.append([build_1, build_2])
    
    build_array = np.array(build_comparison)
    x = np.arange(len(nlist_values))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, build_array[:, 0], width, label='1 Process', alpha=0.8)
    axes[1, 0].bar(x + width/2, build_array[:, 1], width, label='2 Processes', alpha=0.8)
    
    axes[1, 0].set_xlabel('nlist')
    axes[1, 0].set_ylabel('Build Time (ms)')
    axes[1, 0].set_title('Build Time Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(nlist_values)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 召回率-延迟权衡
    for nlist in nlist_values:
        subset_1 = df_1proc[df_1proc['nlist'] == nlist]
        subset_2 = df_2proc[df_2proc['nlist'] == nlist]
        
        axes[1, 1].scatter(subset_1['latency_us'], subset_1['recall'], 
                         alpha=0.7, s=60, label=f'1 Proc nlist={nlist}', 
                         marker='o', color=colors[nlist_values.index(nlist)])
        axes[1, 1].scatter(subset_2['latency_us'], subset_2['recall'], 
                         alpha=0.7, s=60, label=f'2 Proc nlist={nlist}', 
                         marker='s', color=colors[nlist_values.index(nlist)])
    
    axes[1, 1].set_xlabel('Query Latency (μs)')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall-Latency Tradeoff')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    # 6. 相对性能改进热力图
    improvement_matrix = np.zeros((len(nlist_values), len(nprobe_values)))
    
    for i, nlist in enumerate(nlist_values):
        subset_1 = df_1proc[df_1proc['nlist'] == nlist]
        subset_2 = df_2proc[df_2proc['nlist'] == nlist]
        
        for j, nprobe in enumerate(nprobe_values):
            latency_1 = subset_1[subset_1['nprobe'] == nprobe]['latency_us'].iloc[0]
            latency_2 = subset_2[subset_2['nprobe'] == nprobe]['latency_us'].iloc[0]
            improvement = (latency_1 - latency_2) / latency_1 * 100
            improvement_matrix[i, j] = improvement
    
    im = axes[1, 2].imshow(improvement_matrix, cmap='RdYlGn', aspect='auto')
    axes[1, 2].set_xlabel('nprobe')
    axes[1, 2].set_ylabel('nlist')
    axes[1, 2].set_title('Performance Improvement Heatmap (%)')
    axes[1, 2].set_xticks(range(len(nprobe_values)))
    axes[1, 2].set_xticklabels(nprobe_values)
    axes[1, 2].set_yticks(range(len(nlist_values)))
    axes[1, 2].set_yticklabels(nlist_values)
    
    # 添加数值标注
    for i in range(len(nlist_values)):
        for j in range(len(nprobe_values)):
            text = axes[1, 2].text(j, i, f'{improvement_matrix[i, j]:.1f}%',
                                 ha="center", va="center", color="black", fontsize=8)
    
    cbar = plt.colorbar(im, ax=axes[1, 2])
    cbar.set_label('Improvement (%)')
    
    plt.tight_layout()
    plt.savefig('mpi_parallelization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_insights(df_1proc, df_2proc):
    """生成分析洞察"""
    print("\n=== 关键洞察分析 ===\n")
    
    # 1. 最佳并行化场景
    best_scenarios = []
    worst_scenarios = []
    
    for nlist in [64, 128, 256, 512]:
        subset_1 = df_1proc[df_1proc['nlist'] == nlist]
        subset_2 = df_2proc[df_2proc['nlist'] == nlist]
        
        for nprobe in [1, 2, 4, 8, 16, 32, 64]:
            latency_1 = subset_1[subset_1['nprobe'] == nprobe]['latency_us'].iloc[0]
            latency_2 = subset_2[subset_2['nprobe'] == nprobe]['latency_us'].iloc[0]
            improvement = (latency_1 - latency_2) / latency_1 * 100
            
            scenario = f"nlist={nlist}, nprobe={nprobe}"
            if improvement > 20:
                best_scenarios.append((scenario, improvement))
            elif improvement < 5:
                worst_scenarios.append((scenario, improvement))
    
    print("最佳并行化效果场景 (改进>20%):")
    for scenario, improvement in sorted(best_scenarios, key=lambda x: x[1], reverse=True):
        print(f"  {scenario}: {improvement:.1f}% 改进")
    
    print("\n并行化效果较差场景 (改进<5%):")
    for scenario, improvement in sorted(worst_scenarios, key=lambda x: x[1]):
        print(f"  {scenario}: {improvement:.1f}% 改进")
    
    # 2. 参数影响分析
    print("\n参数影响总结:")
    
    # nlist影响
    print("nlist参数影响:")
    for nlist in [64, 128, 256, 512]:
        subset_1 = df_1proc[df_1proc['nlist'] == nlist]
        subset_2 = df_2proc[df_2proc['nlist'] == nlist]
        avg_improvement = ((subset_1['latency_us'].mean() - subset_2['latency_us'].mean()) / 
                          subset_1['latency_us'].mean() * 100)
        print(f"  nlist={nlist}: 平均改进 {avg_improvement:.1f}%")
    
    # nprobe影响
    print("\nnprobe参数影响:")
    for nprobe in [1, 2, 4, 8, 16, 32, 64]:
        improvements = []
        for nlist in [64, 128, 256, 512]:
            subset_1 = df_1proc[(df_1proc['nlist'] == nlist) & (df_1proc['nprobe'] == nprobe)]
            subset_2 = df_2proc[(df_2proc['nlist'] == nlist) & (df_2proc['nprobe'] == nprobe)]
            if len(subset_1) > 0 and len(subset_2) > 0:
                improvement = ((subset_1['latency_us'].iloc[0] - subset_2['latency_us'].iloc[0]) / 
                              subset_1['latency_us'].iloc[0] * 100)
                improvements.append(improvement)
        
        if improvements:
            avg_improvement = np.mean(improvements)
            print(f"  nprobe={nprobe}: 平均改进 {avg_improvement:.1f}%")

if __name__ == "__main__":
    try:
        df_combined, df_1proc, df_2proc = load_and_merge_data()
        
        print("数据加载完成，开始分析...\n")
        
        # 计算并行化指标
        metrics = calculate_parallelization_metrics(df_1proc, df_2proc)
        
        # 分析参数效应
        analyze_parameter_effects(df_1proc, df_2proc)
        
        # 生成可视化
        create_parallelization_analysis_plots(df_combined, df_1proc, df_2proc)
        
        # 生成洞察
        generate_insights(df_1proc, df_2proc)
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 