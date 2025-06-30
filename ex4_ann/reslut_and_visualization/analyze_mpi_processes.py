import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体和样式
try:
    # 尝试使用不同的中文字体
    import matplotlib
    import platform
    
    if platform.system() == 'Windows':
        # Windows系统常见中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong', 'KaiTi', 'Arial Unicode MS', 'DejaVu Sans']
    else:
        # Linux/Mac系统
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 测试中文字体是否可用
    test_fig, test_ax = plt.subplots(1, 1, figsize=(1, 1))
    test_ax.text(0.5, 0.5, '测试', fontsize=12)
    plt.close(test_fig)
    
except Exception as e:
    print(f"中文字体设置失败，将使用英文标签: {e}")
    # 如果中文字体不可用，使用英文标签
    USE_ENGLISH = True
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

# 尝试使用seaborn样式，如果失败则使用默认样式
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # 使用默认样式

# 读取数据文件
def load_data():
    df_1proc = pd.read_csv('results_mpi_ivf_1proc.csv')
    df_2proc = pd.read_csv('results_mpi_ivf_2proc.csv')
    df_4proc = pd.read_csv('results_mpi_ivf_4proc.csv')
    
    # 合并数据
    df_all = pd.concat([df_1proc, df_2proc, df_4proc], ignore_index=True)
    return df_all, df_1proc, df_2proc, df_4proc

def analyze_performance_metrics(df_all):
    """分析性能指标"""
    print("=== MPI进程数对IVF算法性能影响分析 ===\n")
    
    # 按进程数分组分析
    for processes in [1, 2, 4]:
        df_proc = df_all[df_all['mpi_processes'] == processes]
        avg_latency = df_proc['latency_us'].mean()
        avg_build_time = df_proc['build_time_ms'].mean()
        avg_recall = df_proc['recall'].mean()
        efficiency = df_proc['recall'].sum() / df_proc['latency_us'].sum() * 1000
        
        print(f"{processes}进程配置:")
        print(f"  平均查询延迟: {avg_latency:.1f}μs")
        print(f"  平均构建时间: {avg_build_time:.1f}ms")
        print(f"  平均召回率: {avg_recall:.4f}")
        print(f"  算法效率: {efficiency:.4f}")
        print()
    
    # 计算加速比
    print("加速比分析:")
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
        print(f"  查询加速比: 2进程={speedup_2:.2f}x, 4进程={speedup_4:.2f}x")
        print(f"  构建加速比: 2进程={build_speedup_2:.2f}x, 4进程={build_speedup_4:.2f}x")

def create_comprehensive_analysis(df_all):
    """创建综合分析图表"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('MPI进程数对IVF算法性能影响分析', fontsize=16, fontweight='bold')
    
    # 1. 查询延迟对比 (按nlist分组)
    for nlist in [64, 128, 256, 512]:
        for processes in [1, 2, 4]:
            subset = df_all[(df_all['nlist'] == nlist) & (df_all['mpi_processes'] == processes)]
            if len(subset) > 0:
                axes[0, 0].plot(subset['nprobe'], subset['latency_us'], 
                              marker='o', label=f'{processes}进程 nlist={nlist}', linewidth=2)
    
    axes[0, 0].set_xlabel('nprobe')
    axes[0, 0].set_ylabel('查询延迟 (微秒)')
    axes[0, 0].set_title('查询延迟对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. 构建时间对比
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
                      label=f'{proc}进程', alpha=0.8)
    
    axes[0, 1].set_xlabel('nlist')
    axes[0, 1].set_ylabel('构建时间 (毫秒)')
    axes[0, 1].set_title('构建时间对比')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(nlist_values)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 加速比分析
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
                       marker='o', label=f'{proc}进程', linewidth=2, markersize=8)
    
    # 添加理想加速比线
    axes[1, 0].plot(nlist_values, [2]*4, '--', alpha=0.7, label='理想2倍加速')
    axes[1, 0].plot(nlist_values, [4]*4, '--', alpha=0.7, label='理想4倍加速')
    
    axes[1, 0].set_xlabel('nlist')
    axes[1, 0].set_ylabel('加速比')
    axes[1, 0].set_title('查询延迟加速比')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 效率分析
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
                       marker='o', label=f'{proc}进程', linewidth=2, markersize=8)
    
    axes[1, 1].set_xlabel('nlist')
    axes[1, 1].set_ylabel('效率 (召回率/延迟 × 1000)')
    axes[1, 1].set_title('算法效率对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. 召回率-延迟权衡 (散点图)
    colors = ['blue', 'green', 'red']
    for i, proc in enumerate(processes_values):
        subset = df_all[df_all['mpi_processes'] == proc]
        axes[2, 0].scatter(subset['latency_us'], subset['recall'], 
                          alpha=0.7, s=60, label=f'{proc}进程', c=colors[i])
    
    axes[2, 0].set_xlabel('查询延迟 (微秒)')
    axes[2, 0].set_ylabel('召回率')
    axes[2, 0].set_title('召回率-延迟权衡')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xscale('log')
    
    # 6. 并行效率 (相对于理想情况)
    parallel_efficiency = []
    for nlist in nlist_values:
        baseline = df_all[(df_all['nlist'] == nlist) & (df_all['mpi_processes'] == 1)]['latency_us'].mean()
        for proc in [2, 4]:
            current = df_all[(df_all['nlist'] == nlist) & (df_all['mpi_processes'] == proc)]['latency_us'].mean()
            speedup = baseline / current
            efficiency = speedup / proc * 100  # 并行效率百分比
            parallel_efficiency.append({'nlist': nlist, 'processes': proc, 'efficiency': efficiency})
    
    parallel_eff_df = pd.DataFrame(parallel_efficiency)
    
    for proc in [2, 4]:
        subset = parallel_eff_df[parallel_eff_df['processes'] == proc]
        axes[2, 1].plot(subset['nlist'], subset['efficiency'], 
                       marker='o', label=f'{proc}进程', linewidth=2, markersize=8)
    
    axes[2, 1].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='理想效率(100%)')
    axes[2, 1].set_xlabel('nlist')
    axes[2, 1].set_ylabel('并行效率 (%)')
    axes[2, 1].set_title('并行效率')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mpi_processes_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_detailed_metrics(df_all):
    """计算详细的性能指标"""
    print("\n=== 详细性能指标分析 ===")
    
    # 计算各项指标
    metrics = {}
    for proc in [1, 2, 4]:
        subset = df_all[df_all['mpi_processes'] == proc]
        metrics[proc] = {
            'avg_latency': subset['latency_us'].mean(),
            'avg_build_time': subset['build_time_ms'].mean(),
            'avg_recall': subset['recall'].mean(),
            'total_efficiency': subset['recall'].sum() / subset['latency_us'].sum() * 1000
        }
    
    print("性能指标汇总:")
    print("进程数\t平均延迟(μs)\t平均构建时间(ms)\t平均召回率\t算法效率")
    for proc in [1, 2, 4]:
        m = metrics[proc]
        print(f"{proc}\t{m['avg_latency']:.1f}\t\t{m['avg_build_time']:.1f}\t\t{m['avg_recall']:.4f}\t\t{m['total_efficiency']:.4f}")
    
    # 计算改进幅度
    print("\n相对于单进程的性能改进:")
    baseline_latency = metrics[1]['avg_latency']
    baseline_build = metrics[1]['avg_build_time']
    
    for proc in [2, 4]:
        latency_improvement = (baseline_latency - metrics[proc]['avg_latency']) / baseline_latency * 100
        build_improvement = (baseline_build - metrics[proc]['avg_build_time']) / baseline_build * 100
        efficiency_improvement = (metrics[proc]['total_efficiency'] - metrics[1]['total_efficiency']) / metrics[1]['total_efficiency'] * 100
        
        print(f"{proc}进程:")
        print(f"  查询延迟改进: {latency_improvement:.2f}%")
        print(f"  构建时间改进: {build_improvement:.2f}%")
        print(f"  算法效率改进: {efficiency_improvement:.2f}%")

if __name__ == "__main__":
    try:
        df_all, df_1proc, df_2proc, df_4proc = load_data()
        analyze_performance_metrics(df_all)
        calculate_detailed_metrics(df_all)
        create_comprehensive_analysis(df_all)
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请确保CSV文件存在且格式正确") 