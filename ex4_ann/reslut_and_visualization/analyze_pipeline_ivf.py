import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 流水线并行算法数据
pipeline_data = {
    'nlist': [32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512],
    'nprobe': [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64],
    'recall_mean': [0.552300, 0.728501, 0.867652, 0.953253, 0.991701, 0.999950, 0.480800, 0.663651, 0.814553, 0.922154, 0.974252, 0.996000, 0.999950, 0.430099, 0.600900, 0.763402, 0.881603, 0.952903, 0.986702, 0.998450, 0.391200, 0.556250, 0.709951, 0.833303, 0.919053, 0.969852, 0.992651, 0.349800, 0.497000, 0.658152, 0.790402, 0.890003, 0.949103, 0.982152],
    'latency_us_mean': [299, 379, 449, 877, 1239, 2334, 225, 276, 339, 513, 886, 1784, 3259, 212, 261, 280, 425, 853, 1401, 2429, 191, 209, 239, 380, 650, 1260, 1949, 203, 219, 258, 341, 764, 972, 1867],
    'build_time_ms': [118, 118, 118, 118, 118, 118, 74, 74, 74, 74, 74, 74, 74, 98, 98, 98, 98, 98, 98, 98, 122, 122, 122, 122, 122, 122, 122, 202, 202, 202, 202, 202, 202, 202]
}

# 普通MPI并行算法数据
mpi_data = {
    'nlist': [64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512],
    'nprobe': [1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64],
    'recall': [0.489100, 0.675701, 0.817701, 0.918801, 0.970401, 0.995300, 0.999900, 0.437000, 0.611500, 0.769500, 0.883400, 0.950501, 0.985700, 0.998500, 0.388600, 0.555399, 0.709000, 0.832501, 0.914601, 0.968001, 0.991200, 0.353501, 0.495200, 0.656700, 0.790100, 0.890100, 0.946800, 0.981301],
    'latency_us': [240, 278, 311, 454, 651, 1140, 1914, 195, 188, 213, 287, 462, 683, 1127, 168, 186, 187, 216, 288, 412, 643, 134, 139, 153, 200, 218, 287, 415],
    'build_time_ms': [160, 160, 160, 160, 160, 160, 160, 164, 164, 164, 164, 164, 164, 164, 288, 288, 288, 288, 288, 288, 288, 466, 466, 466, 466, 466, 466, 466]
}

# 转换为DataFrame
df_pipeline = pd.DataFrame(pipeline_data)
df_mpi = pd.DataFrame(mpi_data)

# 创建综合分析图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('流水线IVF算法与普通MPI算法性能对比分析', fontsize=16, fontweight='bold')

# 1. 延迟对比 (按nlist分组)
for nlist in [64, 128, 256, 512]:
    pipeline_subset = df_pipeline[df_pipeline['nlist'] == nlist]
    mpi_subset = df_mpi[df_mpi['nlist'] == nlist]
    
    if len(pipeline_subset) > 0 and len(mpi_subset) > 0:
        axes[0, 0].plot(pipeline_subset['nprobe'], pipeline_subset['latency_us_mean'], 
                       marker='o', label=f'流水线 nlist={nlist}', linewidth=2)
        axes[0, 0].plot(mpi_subset['nprobe'], mpi_subset['latency_us'], 
                       marker='s', linestyle='--', label=f'普通MPI nlist={nlist}', linewidth=2)

axes[0, 0].set_xlabel('nprobe')
axes[0, 0].set_ylabel('延迟 (微秒)')
axes[0, 0].set_title('延迟对比')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 2. 召回率对比
for nlist in [64, 128, 256, 512]:
    pipeline_subset = df_pipeline[df_pipeline['nlist'] == nlist]
    mpi_subset = df_mpi[df_mpi['nlist'] == nlist]
    
    if len(pipeline_subset) > 0 and len(mpi_subset) > 0:
        axes[0, 1].plot(pipeline_subset['nprobe'], pipeline_subset['recall_mean'], 
                       marker='o', label=f'流水线 nlist={nlist}', linewidth=2)
        axes[0, 1].plot(mpi_subset['nprobe'], mpi_subset['recall'], 
                       marker='s', linestyle='--', label=f'普通MPI nlist={nlist}', linewidth=2)

axes[0, 1].set_xlabel('nprobe')
axes[0, 1].set_ylabel('召回率')
axes[0, 1].set_title('召回率对比')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 构建时间对比
nlist_values = [64, 128, 256, 512]
pipeline_build_times = []
mpi_build_times = []

for nlist in nlist_values:
    pipeline_build = df_pipeline[df_pipeline['nlist'] == nlist]['build_time_ms'].iloc[0] if len(df_pipeline[df_pipeline['nlist'] == nlist]) > 0 else 0
    mpi_build = df_mpi[df_mpi['nlist'] == nlist]['build_time_ms'].iloc[0] if len(df_mpi[df_mpi['nlist'] == nlist]) > 0 else 0
    pipeline_build_times.append(pipeline_build)
    mpi_build_times.append(mpi_build)

x = np.arange(len(nlist_values))
width = 0.35

axes[0, 2].bar(x - width/2, pipeline_build_times, width, label='流水线算法', alpha=0.8)
axes[0, 2].bar(x + width/2, mpi_build_times, width, label='普通MPI算法', alpha=0.8)
axes[0, 2].set_xlabel('nlist')
axes[0, 2].set_ylabel('构建时间 (毫秒)')
axes[0, 2].set_title('构建时间对比')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(nlist_values)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. 性能效率分析：延迟改善百分比
improvement_data = []
for nlist in [64, 128, 256, 512]:
    pipeline_subset = df_pipeline[df_pipeline['nlist'] == nlist]
    mpi_subset = df_mpi[df_mpi['nlist'] == nlist]
    
    if len(pipeline_subset) > 0 and len(mpi_subset) > 0:
        # 找到共同的nprobe值
        common_nprobes = set(pipeline_subset['nprobe']).intersection(set(mpi_subset['nprobe']))
        for nprobe in sorted(common_nprobes):
            pipeline_latency = pipeline_subset[pipeline_subset['nprobe'] == nprobe]['latency_us_mean'].iloc[0]
            mpi_latency = mpi_subset[mpi_subset['nprobe'] == nprobe]['latency_us'].iloc[0]
            improvement = (mpi_latency - pipeline_latency) / mpi_latency * 100
            improvement_data.append({'nlist': nlist, 'nprobe': nprobe, 'improvement': improvement})

improvement_df = pd.DataFrame(improvement_data)

# 创建热力图
pivot_table = improvement_df.pivot(index='nlist', columns='nprobe', values='improvement')
im = axes[1, 0].imshow(pivot_table.values, cmap='RdYlGn', aspect='auto')
axes[1, 0].set_xticks(range(len(pivot_table.columns)))
axes[1, 0].set_xticklabels(pivot_table.columns)
axes[1, 0].set_yticks(range(len(pivot_table.index)))
axes[1, 0].set_yticklabels(pivot_table.index)
axes[1, 0].set_xlabel('nprobe')
axes[1, 0].set_ylabel('nlist')
axes[1, 0].set_title('延迟改善百分比 (%)')

# 添加数值标注
for i in range(len(pivot_table.index)):
    for j in range(len(pivot_table.columns)):
        if not np.isnan(pivot_table.iloc[i, j]):
            text = axes[1, 0].text(j, i, f'{pivot_table.iloc[i, j]:.1f}%',
                                  ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=axes[1, 0])

# 5. 召回率-延迟权衡分析 (描点连线)
colors_pipeline = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
colors_mpi = ['#17becf', '#bcbd22', '#e377c2', '#8c564b', '#7f7f7f']
markers_pipeline = ['o', 's', '^', 'D', 'v']
markers_mpi = ['o', 's', '^', 'D', 'v']

# 流水线算法 - 按nlist分组连线
nlist_values_pipeline = sorted(df_pipeline['nlist'].unique())
for i, nlist in enumerate(nlist_values_pipeline):
    pipeline_subset = df_pipeline[df_pipeline['nlist'] == nlist].sort_values('recall_mean')
    if len(pipeline_subset) > 0:
        color_idx = i % len(colors_pipeline)
        marker_idx = i % len(markers_pipeline)
        axes[1, 1].plot(pipeline_subset['recall_mean'] * 100, pipeline_subset['latency_us_mean'],
                       marker=markers_pipeline[marker_idx], color=colors_pipeline[color_idx],
                       linewidth=2, markersize=6, label=f'流水线 nlist={nlist}',
                       markerfacecolor=colors_pipeline[color_idx], markeredgecolor='white', markeredgewidth=0.5)

# 普通MPI算法 - 按nlist分组连线
nlist_values_mpi = sorted(df_mpi['nlist'].unique())
for i, nlist in enumerate(nlist_values_mpi):
    mpi_subset = df_mpi[df_mpi['nlist'] == nlist].sort_values('recall')
    if len(mpi_subset) > 0:
        color_idx = i % len(colors_mpi)
        marker_idx = i % len(markers_mpi)
        axes[1, 1].plot(mpi_subset['recall'] * 100, mpi_subset['latency_us'],
                       marker=markers_mpi[marker_idx], color=colors_mpi[color_idx],
                       linewidth=2, markersize=6, linestyle='--', label=f'普通MPI nlist={nlist}',
                       markerfacecolor=colors_mpi[color_idx], markeredgecolor='white', markeredgewidth=0.5)

axes[1, 1].set_xlabel('召回率 (%)')
axes[1, 1].set_ylabel('延迟 (微秒)')
axes[1, 1].set_title('召回率-延迟权衡曲线')
axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_yscale('log')
axes[1, 1].set_xlim(30, 105)

# 6. 效率指标：召回率/延迟比率
df_pipeline['efficiency'] = df_pipeline['recall_mean'] / df_pipeline['latency_us_mean'] * 1000
df_mpi['efficiency'] = df_mpi['recall'] / df_mpi['latency_us'] * 1000

# 按nlist分组显示效率
for nlist in [64, 128, 256, 512]:
    pipeline_subset = df_pipeline[df_pipeline['nlist'] == nlist]
    mpi_subset = df_mpi[df_mpi['nlist'] == nlist]
    
    if len(pipeline_subset) > 0 and len(mpi_subset) > 0:
        axes[1, 2].plot(pipeline_subset['nprobe'], pipeline_subset['efficiency'], 
                       marker='o', label=f'流水线 nlist={nlist}', linewidth=2)
        axes[1, 2].plot(mpi_subset['nprobe'], mpi_subset['efficiency'], 
                       marker='s', linestyle='--', label=f'普通MPI nlist={nlist}', linewidth=2)

axes[1, 2].set_xlabel('nprobe')
axes[1, 2].set_ylabel('效率 (召回率/延迟 × 1000)')
axes[1, 2].set_title('算法效率对比')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(right=0.85)  # 为图例留出空间
plt.savefig('pipeline_ivf_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 计算统计数据
print("=== 流水线算法性能分析报告 ===")
print("\n1. 延迟改善统计:")
print(f"平均延迟改善: {improvement_df['improvement'].mean():.2f}%")
print(f"最大延迟改善: {improvement_df['improvement'].max():.2f}%")
print(f"最小延迟改善: {improvement_df['improvement'].min():.2f}%")

print("\n2. 构建时间对比:")
avg_pipeline_build = np.mean(pipeline_build_times)
avg_mpi_build = np.mean(mpi_build_times)
build_improvement = (avg_mpi_build - avg_pipeline_build) / avg_mpi_build * 100
print(f"流水线算法平均构建时间: {avg_pipeline_build:.1f}ms")
print(f"普通MPI算法平均构建时间: {avg_mpi_build:.1f}ms")
print(f"构建时间改善: {build_improvement:.2f}%")

print("\n3. 效率提升:")
avg_pipeline_efficiency = df_pipeline['efficiency'].mean()
avg_mpi_efficiency = df_mpi['efficiency'].mean()
efficiency_improvement = (avg_pipeline_efficiency - avg_mpi_efficiency) / avg_mpi_efficiency * 100
print(f"流水线算法平均效率: {avg_pipeline_efficiency:.4f}")
print(f"普通MPI算法平均效率: {avg_mpi_efficiency:.4f}")
print(f"效率提升: {efficiency_improvement:.2f}%") 