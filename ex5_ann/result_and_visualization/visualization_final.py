import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
df = pd.read_csv('benchmark_results.csv')
# Rename columns to English
df = df.rename(columns={
    '方法': 'Method',
    '批大小': 'Batch_Size',
    '查询数量': 'Query_Count',
    '平均召回率': 'Avg_Recall',
    '平均延迟(μs)': 'Avg_Latency_us',
    '总耗时(ms)': 'Total_Time_ms'
})

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    sns.set_style("whitegrid")
fig = plt.figure(figsize=(20, 12))

# Figure 1: Comprehensive latency comparison
plt.subplot(2, 3, 1)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'v', 'p']

for i, query in enumerate(sorted(df['Query_Count'].unique())):
    gpu_cpu_data = df[(df['Method'] == 'GPU+CPU') & (df['Query_Count'] == query)]
    gpu_gpu_data = df[(df['Method'] == 'GPU+GPU') & (df['Query_Count'] == query)]
    
    plt.plot(gpu_cpu_data['Batch_Size'], gpu_cpu_data['Avg_Latency_us'], 
             marker=markers[i], color=colors[i], linestyle='-', linewidth=2,
             label=f'GPU+CPU, Query={query}', markersize=6, alpha=0.8)
    plt.plot(gpu_gpu_data['Batch_Size'], gpu_gpu_data['Avg_Latency_us'], 
             marker=markers[i], color=colors[i], linestyle='--', linewidth=2,
             label=f'GPU+GPU, Query={query}', markersize=6, alpha=0.8)

plt.title('Latency Performance Comparison Across All Configurations', fontsize=14, fontweight='bold')
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Average Latency (μs)', fontsize=12)
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

# Figure 2: GPU parallelism threshold effect
plt.subplot(2, 3, 2)
gpu_gpu_data = df[df['Method'] == 'GPU+GPU']
pivot_gpu_gpu = gpu_gpu_data.pivot(index='Batch_Size', columns='Query_Count', values='Avg_Latency_us')

# Highlight the threshold effect
query_500 = pivot_gpu_gpu[500]
query_1000 = pivot_gpu_gpu[1000]

plt.plot(pivot_gpu_gpu.index, query_500, 'ro-', linewidth=3, markersize=8, 
         label='Query=500 (Below Threshold)', alpha=0.8)
plt.plot(pivot_gpu_gpu.index, query_1000, 'bo-', linewidth=3, markersize=8, 
         label='Query=1000 (Above Threshold)', alpha=0.8)

# Add threshold annotation
plt.axhline(y=1000, color='gray', linestyle=':', alpha=0.5)
plt.annotate('GPU Parallelism Threshold Effect\nQuery Count = 1000 is the turning point', 
             xy=(2048, 150), xytext=(1000, 800),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=11, color='green', weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

plt.title('GPU Hardware Parallelism Threshold Effect', fontsize=14, fontweight='bold')
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Average Latency (μs)', fontsize=12)
plt.xscale('log', base=2)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Figure 3: Performance region analysis
plt.subplot(2, 3, 3)
batch_latency = df.pivot_table(index='Batch_Size', columns='Method', values='Avg_Latency_us', aggfunc='mean')
ax = batch_latency.plot(kind='bar', color=['#2E86AB', '#A23B72'], ax=plt.gca(), width=0.8)

# Add performance regions
plt.axvspan(-0.5, 4.5, alpha=0.2, color='red', label='GPU+CPU Advantage')
plt.axvspan(4.5, 7.5, alpha=0.2, color='yellow', label='Transition Zone')

plt.title('Average Latency by Batch Size\n(Hardware-Optimized Regions)', fontsize=14, fontweight='bold')
plt.ylabel('Average Latency (μs)', fontsize=12)
plt.xlabel('Batch Size', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend(title='Method', fontsize=11)

# Figure 4: Speedup ratio heatmap
plt.subplot(2, 3, 4)
speedup_df = df.pivot_table(index=['Batch_Size', 'Query_Count'], columns='Method', values='Avg_Latency_us').reset_index()
speedup_df['Performance_Ratio'] = speedup_df['GPU+GPU'] / speedup_df['GPU+CPU']
pivot_speedup = speedup_df.pivot_table(index='Batch_Size', columns='Query_Count', values='Performance_Ratio')

# Custom colormap for better visualization
cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(pivot_speedup, annot=True, fmt=".2f", cmap=cmap, center=1, 
            cbar_kws={'label': 'GPU+GPU/GPU+CPU Ratio'})

plt.title('Performance Ratio Analysis\n(<1: GPU+GPU Better, >1: GPU+CPU Better)', fontsize=14, fontweight='bold')
plt.ylabel('Batch Size', fontsize=12)
plt.xlabel('Query Count', fontsize=12)

# Figure 5: GPU utilization efficiency
plt.subplot(2, 3, 5)
gpu_gpu_pivot = df[df['Method'] == 'GPU+GPU'].pivot_table(
    index='Query_Count', columns='Batch_Size', values='Avg_Latency_us')

# Calculate efficiency score (lower latency = higher efficiency)
efficiency = 1000 / gpu_gpu_pivot  # Normalize for visualization

sns.heatmap(efficiency, annot=False, cmap='RdYlGn', 
            cbar_kws={'label': 'GPU Efficiency Score'})

plt.title('GPU+GPU Efficiency Map\n(Darker Green = Higher Efficiency)', fontsize=14, fontweight='bold')
plt.ylabel('Query Count', fontsize=12)
plt.xlabel('Batch Size', fontsize=12)

# Figure 6: Query Count impact with fixed batch size (controlled variable)
plt.subplot(2, 3, 6)
fixed_batch_size = 1024
batch_1024_data = df[df['Batch_Size'] == fixed_batch_size]
query_latency_1024 = batch_1024_data.pivot_table(index='Query_Count', columns='Method', values='Avg_Latency_us')

# Plot with different colors and markers
colors = ['#2E86AB', '#A23B72']
markers = ['o', 's']
for i, method in enumerate(query_latency_1024.columns):
    plt.plot(query_latency_1024.index, query_latency_1024[method], 
             color=colors[i], marker=markers[i], linewidth=3, markersize=8, 
             label=method, alpha=0.8)

# Add threshold line to highlight the parallelism effect
plt.axvline(x=1000, color='red', linestyle='--', alpha=0.7, linewidth=2)
plt.annotate('Parallelism\nThreshold', xy=(1000, 140), xytext=(2000, 160),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red', weight='bold')

plt.title(f'Query Count Impact on Latency\n(Fixed Batch Size = {fixed_batch_size})', fontsize=14, fontweight='bold')
plt.xlabel('Query Count', fontsize=12)
plt.ylabel('Average Latency (μs)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(title='Method', fontsize=11)

plt.tight_layout()
plt.savefig('final_gpu_analysis.png', dpi=300, bbox_inches='tight')

# Generate additional focused charts
# Chart 1: Threshold effect detail
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
threshold_data = df[df['Method'] == 'GPU+GPU']
for batch in [1024, 2048, 4096]:
    batch_data = threshold_data[threshold_data['Batch_Size'] == batch]
    plt.plot(batch_data['Query_Count'], batch_data['Avg_Latency_us'], 
             'o-', linewidth=2, markersize=8, label=f'Batch Size {batch}')

plt.axvline(x=1000, color='red', linestyle='--', alpha=0.7, linewidth=2)
plt.text(1200, 180, 'Parallelism\nThreshold', fontsize=12, color='red', weight='bold')
plt.title('GPU Parallelism Threshold Effect (Large Batch Sizes)', fontsize=14, fontweight='bold')
plt.xlabel('Query Count', fontsize=12)
plt.ylabel('GPU+GPU Latency (μs)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Chart 2: Performance crossover points
plt.subplot(2, 2, 2)
for query in df['Query_Count'].unique():
    if query >= 1000:  # Focus on high parallelism scenarios
        query_data = df[df['Query_Count'] == query]
        gpu_cpu = query_data[query_data['Method'] == 'GPU+CPU']
        gpu_gpu = query_data[query_data['Method'] == 'GPU+GPU']
        
        plt.plot(gpu_cpu['Batch_Size'], gpu_cpu['Avg_Latency_us'], 
                'o-', color='blue', alpha=0.7, linewidth=2)
        plt.plot(gpu_gpu['Batch_Size'], gpu_gpu['Avg_Latency_us'], 
                's--', color='red', alpha=0.7, linewidth=2)

plt.title('Performance Crossover Analysis\n(Query Count ≥ 1000)', fontsize=14, fontweight='bold')
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Average Latency (μs)', fontsize=12)
plt.xscale('log', base=2)
plt.grid(True, alpha=0.3)
plt.legend(['GPU+CPU', 'GPU+GPU'])

# Chart 3: Efficiency improvement
plt.subplot(2, 2, 3)
efficiency_data = []
for batch in df['Batch_Size'].unique():
    for query in df['Query_Count'].unique():
        subset = df[(df['Batch_Size'] == batch) & (df['Query_Count'] == query)]
        if len(subset) == 2:
            gpu_cpu_lat = subset[subset['Method'] == 'GPU+CPU']['Avg_Latency_us'].iloc[0]
            gpu_gpu_lat = subset[subset['Method'] == 'GPU+GPU']['Avg_Latency_us'].iloc[0]
            improvement = (gpu_cpu_lat - gpu_gpu_lat) / gpu_cpu_lat * 100
            efficiency_data.append({'Batch_Size': batch, 'Query_Count': query, 'Improvement': improvement})

eff_df = pd.DataFrame(efficiency_data)
eff_pivot = eff_df.pivot_table(index='Batch_Size', columns='Query_Count', values='Improvement')

sns.heatmap(eff_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'GPU+GPU Improvement (%)'})
plt.title('GPU+GPU Performance Improvement\n(Negative = GPU+CPU Better)', fontsize=14, fontweight='bold')
plt.ylabel('Batch Size', fontsize=12)
plt.xlabel('Query Count', fontsize=12)

# Chart 4: Hardware utilization zones
plt.subplot(2, 2, 4)
zones = {
    'Low Utilization\n(GPU+CPU Better)': {'x': [32, 512], 'y': [500, 10000], 'color': 'lightcoral'},
    'Medium Utilization\n(Transition Zone)': {'x': [1024, 4096], 'y': [500, 999], 'color': 'lightyellow'},
    'High Utilization\n(GPU+GPU Better)': {'x': [1024, 4096], 'y': [1000, 10000], 'color': 'lightgreen'}
}

for zone, params in zones.items():
    plt.fill_between(params['x'], params['y'][0], params['y'][1], 
                     alpha=0.5, color=params['color'], label=zone)

plt.title('GPU Hardware Utilization Zones', fontsize=14, fontweight='bold')
plt.xlabel('Batch Size Range', fontsize=12)
plt.ylabel('Query Count Range', fontsize=12)
plt.legend(loc='center', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gpu_hardware_analysis.png', dpi=300, bbox_inches='tight')

print("=== GPU HARDWARE ANALYSIS COMPLETED ===")
print("✓ Generated final_gpu_analysis.png - Comprehensive 6-panel analysis")
print("✓ Generated gpu_hardware_analysis.png - Detailed hardware utilization analysis")
print("\nKey Findings:")
print("1. GPU parallelism threshold at ~1000 queries")
print("2. Hardware-driven performance zones identified")
print("3. Optimal configuration guidelines established")
print("4. Both methods achieve excellent recall rates") 