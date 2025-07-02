import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the speedup data
df = pd.read_csv('ann/speedup_results.csv')

# Rename columns to English for consistency
df = df.rename(columns={
    '方法': 'Method',
    '批大小': 'Batch_Size',
    '基准时间(us)': 'Baseline_Time_us',
    '方法时间(us)': 'Method_Time_us',
    '加速比': 'Speedup_Ratio',
    '召回率': 'Recall_Rate'
})

print("数据概览：")
print(df.head(10))
print(f"\n数据形状: {df.shape}")
print(f"方法类型: {df['Method'].unique()}")
print(f"批大小范围: {sorted(df['Batch_Size'].unique())}")

# Set up the plotting style
plt.style.use('default')
fig = plt.figure(figsize=(20, 15))

# 1. 综合加速比对比图
plt.subplot(2, 3, 1)
methods_with_batch = ['GPU+CPU', 'GPU+GPU', 'GPU_IVF']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

for i, method in enumerate(methods_with_batch):
    method_data = df[df['Method'] == method]
    plt.plot(method_data['Batch_Size'], method_data['Speedup_Ratio'], 
             marker=markers[i], color=colors[i], linewidth=3, markersize=8,
             label=method, alpha=0.8)

# Add CPU_IVF as horizontal line
cpu_ivf_speedup = df[df['Method'] == 'CPU_IVF']['Speedup_Ratio'].iloc[0]
plt.axhline(y=cpu_ivf_speedup, color='red', linestyle='--', linewidth=2, 
            label=f'CPU_IVF (Baseline): {cpu_ivf_speedup:.2f}x', alpha=0.7)

plt.title('Speedup Ratio Comparison Across Methods\n(Query Count = 2000)', fontsize=14, fontweight='bold')
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Speedup Ratio', fontsize=12)
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# 2. GPU并行对暴力搜索的加速效应
plt.subplot(2, 3, 2)
brute_force_methods = ['GPU+CPU', 'GPU+GPU']
for i, method in enumerate(brute_force_methods):
    method_data = df[df['Method'] == method]
    plt.plot(method_data['Batch_Size'], method_data['Speedup_Ratio'], 
             marker=markers[i], color=colors[i], linewidth=3, markersize=8,
             label=method, alpha=0.8)

plt.title('GPU Parallel Acceleration for Brute Force Search\n(vs CPU Baseline)', fontsize=14, fontweight='bold')
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Speedup Ratio', fontsize=12)
plt.xscale('log', base=2)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# 3. GPU对IVF算法的加速比
plt.subplot(2, 3, 3)
# Compare CPU_IVF vs GPU_IVF
gpu_ivf_data = df[df['Method'] == 'GPU_IVF']
plt.plot(gpu_ivf_data['Batch_Size'], gpu_ivf_data['Speedup_Ratio'], 
         marker='^', color='#2ca02c', linewidth=3, markersize=8,
         label='GPU_IVF', alpha=0.8)

plt.axhline(y=cpu_ivf_speedup, color='red', linestyle='--', linewidth=2, 
            label=f'CPU_IVF: {cpu_ivf_speedup:.2f}x', alpha=0.7)

plt.title('GPU Acceleration for IVF Algorithm\n(GPU_IVF vs CPU_IVF)', fontsize=14, fontweight='bold')
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Speedup Ratio', fontsize=12)
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# 4. 加速比效率分析
plt.subplot(2, 3, 4)
# Calculate efficiency (speedup per batch size)
efficiency_data = []
for method in methods_with_batch:
    method_data = df[df['Method'] == method]
    for _, row in method_data.iterrows():
        efficiency = row['Speedup_Ratio'] / row['Batch_Size']
        efficiency_data.append({
            'Method': method,
            'Batch_Size': row['Batch_Size'],
            'Efficiency': efficiency
        })

eff_df = pd.DataFrame(efficiency_data)
eff_pivot = eff_df.pivot_table(index='Batch_Size', columns='Method', values='Efficiency')

for i, method in enumerate(methods_with_batch):
    if method in eff_pivot.columns:
        plt.plot(eff_pivot.index, eff_pivot[method], 
                marker=markers[i], color=colors[i], linewidth=3, markersize=8,
                label=method, alpha=0.8)

plt.title('Speedup Efficiency\n(Speedup Ratio / Batch Size)', fontsize=14, fontweight='bold')
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Efficiency', fontsize=12)
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# 5. 方法时间对比
plt.subplot(2, 3, 5)
for i, method in enumerate(methods_with_batch):
    method_data = df[df['Method'] == method]
    plt.plot(method_data['Batch_Size'], method_data['Method_Time_us'], 
             marker=markers[i], color=colors[i], linewidth=3, markersize=8,
             label=method, alpha=0.8)

# Add baseline and CPU_IVF
baseline_time = df['Baseline_Time_us'].iloc[0]
cpu_ivf_time = df[df['Method'] == 'CPU_IVF']['Method_Time_us'].iloc[0]

plt.axhline(y=baseline_time, color='black', linestyle='-', linewidth=2, 
            label=f'CPU Baseline: {baseline_time:.1f}μs', alpha=0.7)
plt.axhline(y=cpu_ivf_time, color='red', linestyle='--', linewidth=2, 
            label=f'CPU_IVF: {cpu_ivf_time:.1f}μs', alpha=0.7)

plt.title('Execution Time Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Execution Time (μs)', fontsize=12)
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# 6. 吞吐量分析 (queries per second)
plt.subplot(2, 3, 6)
query_count = 2000  # Fixed query count for this experiment

for i, method in enumerate(methods_with_batch):
    method_data = df[df['Method'] == method]
    # Calculate throughput: queries per second
    throughput = query_count / (method_data['Method_Time_us'] / 1e6)  # Convert μs to seconds
    plt.plot(method_data['Batch_Size'], throughput, 
             marker=markers[i], color=colors[i], linewidth=3, markersize=8,
             label=method, alpha=0.8)

# Add baselines
baseline_throughput = query_count / (baseline_time / 1e6)
cpu_ivf_throughput = query_count / (cpu_ivf_time / 1e6)

plt.axhline(y=baseline_throughput, color='black', linestyle='-', linewidth=2, 
            label=f'CPU Baseline: {baseline_throughput:.0f} QPS', alpha=0.7)
plt.axhline(y=cpu_ivf_throughput, color='red', linestyle='--', linewidth=2, 
            label=f'CPU_IVF: {cpu_ivf_throughput:.0f} QPS', alpha=0.7)

plt.title('Throughput Analysis\n(Queries Per Second, Higher is Better)', fontsize=14, fontweight='bold')
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Throughput (QPS)', fontsize=12)
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('speedup_analysis.png', dpi=300, bbox_inches='tight')

# Create detailed analysis table
print("\n=== 详细加速比分析表 ===")
print("方法对比（Query Count = 2000）:")
print("-" * 80)

# Create comparison table
comparison_data = []
for batch_size in sorted(df['Batch_Size'].unique()):
    if batch_size == 1:  # Only CPU_IVF has batch_size=1
        continue
    
    row_data = {'Batch_Size': batch_size}
    
    for method in ['GPU+CPU', 'GPU+GPU', 'GPU_IVF']:
        method_row = df[(df['Method'] == method) & (df['Batch_Size'] == batch_size)]
        if not method_row.empty:
            speedup = method_row['Speedup_Ratio'].iloc[0]
            time = method_row['Method_Time_us'].iloc[0]
            row_data[f'{method}_Speedup'] = speedup
            row_data[f'{method}_Time'] = time
    
    comparison_data.append(row_data)

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False, float_format='%.2f'))

print(f"\nCPU_IVF基准: {cpu_ivf_speedup:.2f}x 加速比")
print(f"CPU基准时间: {baseline_time:.1f}μs")

print("\n=== 关键发现 ===")
print("1. GPU_IVF在大批量处理时获得最高加速比（最高90.41x）")
print("2. GPU+GPU方案随批大小增加表现出显著的性能提升")
print("3. GPU+CPU方案在中等批大小时表现稳定")
print("4. 批大小是影响GPU性能的关键因素")

print("\n可视化分析完成！图表已保存为 'speedup_analysis.png'") 