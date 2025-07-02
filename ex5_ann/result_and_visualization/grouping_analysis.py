import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取两个数据文件
df1 = pd.read_csv('ann/advanced_grouping_results.csv')
df2 = pd.read_csv('ann/grouping_strategy_results.csv')

# 重命名列为英文
df1.columns = ['Strategy', 'Batch_Size', 'Avg_Time_us', 'Avg_Recall', 'Cluster_Overlap_Ratio', 'Matrix_Ops', 'Batch_Count']
df2.columns = ['Strategy', 'Batch_Size', 'Avg_Time_us', 'Avg_Recall', 'Cluster_Overlap_Ratio', 'Matrix_Ops']

# 添加 Batch_Count 列到 df2（如果没有的话）
if 'Batch_Count' not in df2.columns:
    df2['Batch_Count'] = 2000 / df2['Batch_Size']  # 计算batch数量

# 合并数据
df_combined = pd.concat([df1, df2], ignore_index=True)

# 翻译策略名称为英文
strategy_translation = {
    '基准无分组': 'Baseline (No Grouping)',
    '无分组': 'No Grouping',
    '自适应批大小': 'Adaptive Batch Size',
    '负载均衡': 'Load Balancing',
    '局部性感知': 'Locality Aware',
    '分层分组': 'Hierarchical Grouping',
    '时间感知自适应': 'Time-Aware Adaptive',
    '簇相似性分组': 'Cluster Similarity',
    '查询相似性分组': 'Query Similarity',
    '混合策略': 'Hybrid Strategy'
}

df_combined['Strategy_EN'] = df_combined['Strategy'].map(strategy_translation)

# 统一基准策略名称
df_combined.loc[df_combined['Strategy_EN'].isin(['Baseline (No Grouping)', 'No Grouping']), 'Strategy_EN'] = 'Baseline (No Grouping)'

print("=== 数据概览 ===")
print(f"总共数据点: {len(df_combined)}")
print(f"策略数量: {df_combined['Strategy_EN'].nunique()}")
print(f"批大小范围: {sorted(df_combined['Batch_Size'].unique())}")
print("\n各策略数据点数量:")
print(df_combined['Strategy_EN'].value_counts())

# 设置绘图风格
plt.style.use('default')
fig = plt.figure(figsize=(20, 15))

# 1. 簇重叠率 vs 延迟的散点图
plt.subplot(2, 3, 1)
strategies = df_combined['Strategy_EN'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
strategy_colors = dict(zip(strategies, colors))

for strategy in strategies:
    strategy_data = df_combined[df_combined['Strategy_EN'] == strategy]
    plt.scatter(strategy_data['Cluster_Overlap_Ratio'], strategy_data['Avg_Time_us'],
               label=strategy, alpha=0.7, s=100, c=[strategy_colors[strategy]])

plt.xlabel('Cluster Overlap Ratio', fontsize=12)
plt.ylabel('Average Latency (μs)', fontsize=12)
plt.title('Cluster Overlap Ratio vs Latency', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)

# 2. 不同批大小下的簇重叠率对比
plt.subplot(2, 3, 2)
batch_sizes = sorted(df_combined['Batch_Size'].unique())
baseline_data = df_combined[df_combined['Strategy_EN'] == 'Baseline (No Grouping)']

# 绘制基准线
if not baseline_data.empty:
    plt.plot(baseline_data['Batch_Size'], baseline_data['Cluster_Overlap_Ratio'], 
             'k--', linewidth=3, label='Baseline (No Grouping)', alpha=0.8)

# 绘制其他策略
for strategy in strategies:
    if strategy == 'Baseline (No Grouping)':
        continue
    strategy_data = df_combined[df_combined['Strategy_EN'] == strategy]
    if not strategy_data.empty:
        plt.plot(strategy_data['Batch_Size'], strategy_data['Cluster_Overlap_Ratio'],
                marker='o', linewidth=2, markersize=6, label=strategy, alpha=0.8)

plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Cluster Overlap Ratio', fontsize=12)
plt.title('Cluster Overlap Ratio vs Batch Size', fontsize=14, fontweight='bold')
plt.xscale('log', base=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)

# 3. 不同批大小下的延迟对比（标出优于基准的方法）
plt.subplot(2, 3, 3)
# 绘制基准线
if not baseline_data.empty:
    plt.plot(baseline_data['Batch_Size'], baseline_data['Avg_Time_us'], 
             'k--', linewidth=3, label='Baseline (No Grouping)', alpha=0.8)

# 绘制其他策略并标出优于基准的点
for strategy in strategies:
    if strategy == 'Baseline (No Grouping)':
        continue
    strategy_data = df_combined[df_combined['Strategy_EN'] == strategy]
    if not strategy_data.empty:
        plt.plot(strategy_data['Batch_Size'], strategy_data['Avg_Time_us'],
                marker='o', linewidth=2, markersize=6, label=strategy, alpha=0.8)
        
        # 标出优于基准的点
        for _, row in strategy_data.iterrows():
            baseline_time = baseline_data[baseline_data['Batch_Size'] == row['Batch_Size']]['Avg_Time_us']
            if not baseline_time.empty and row['Avg_Time_us'] < baseline_time.iloc[0]:
                plt.scatter(row['Batch_Size'], row['Avg_Time_us'], 
                           s=200, marker='*', color='red', alpha=0.8, zorder=5)

plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Average Latency (μs)', fontsize=12)
plt.title('Latency Comparison (★ = Better than Baseline)', fontsize=14, fontweight='bold')
plt.xscale('log', base=2)
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)

# 4. 策略性能热力图
plt.subplot(2, 3, 4)
# 创建性能提升矩阵
strategies_for_heatmap = [s for s in strategies if s != 'Baseline (No Grouping)']
batch_sizes_for_heatmap = sorted(df_combined['Batch_Size'].unique())

improvement_matrix = np.zeros((len(strategies_for_heatmap), len(batch_sizes_for_heatmap)))

for i, strategy in enumerate(strategies_for_heatmap):
    for j, batch_size in enumerate(batch_sizes_for_heatmap):
        strategy_data = df_combined[(df_combined['Strategy_EN'] == strategy) & 
                                   (df_combined['Batch_Size'] == batch_size)]
        baseline_time = baseline_data[baseline_data['Batch_Size'] == batch_size]['Avg_Time_us']
        
        if not strategy_data.empty and not baseline_time.empty:
            improvement = (baseline_time.iloc[0] - strategy_data['Avg_Time_us'].iloc[0]) / baseline_time.iloc[0] * 100
            improvement_matrix[i, j] = improvement

# 创建热力图
im = plt.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-100, vmax=50)
plt.colorbar(im, label='Performance Improvement (%)')
plt.xticks(range(len(batch_sizes_for_heatmap)), batch_sizes_for_heatmap)
plt.yticks(range(len(strategies_for_heatmap)), strategies_for_heatmap)
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Grouping Strategy', fontsize=12)
plt.title('Performance Improvement Heatmap\n(vs Baseline, % improvement)', fontsize=14, fontweight='bold')

# 添加数值标注
for i in range(len(strategies_for_heatmap)):
    for j in range(len(batch_sizes_for_heatmap)):
        text = plt.text(j, i, f'{improvement_matrix[i, j]:.1f}%',
                       ha="center", va="center", color="black", fontsize=8)

# 5. 簇重叠率效率分析
plt.subplot(2, 3, 5)
# 计算效率指标：簇重叠率 / 延迟
df_combined['Efficiency'] = df_combined['Cluster_Overlap_Ratio'] / (df_combined['Avg_Time_us'] / 1000)  # 标准化到ms

for strategy in strategies:
    strategy_data = df_combined[df_combined['Strategy_EN'] == strategy]
    if not strategy_data.empty:
        plt.plot(strategy_data['Batch_Size'], strategy_data['Efficiency'],
                marker='o', linewidth=2, markersize=6, label=strategy, alpha=0.8)

plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Efficiency (Overlap Ratio / Latency)', fontsize=12)
plt.title('Grouping Efficiency Analysis', fontsize=14, fontweight='bold')
plt.xscale('log', base=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)

# 6. 策略性能综合排名
plt.subplot(2, 3, 6)
# 计算每个策略的综合得分（综合考虑延迟和重叠率）
strategy_scores = []

for strategy in strategies:
    strategy_data = df_combined[df_combined['Strategy_EN'] == strategy]
    if not strategy_data.empty:
        avg_time = strategy_data['Avg_Time_us'].mean()
        avg_overlap = strategy_data['Cluster_Overlap_Ratio'].mean()
        # 综合得分：重叠率越高越好，延迟越低越好
        score = avg_overlap / (avg_time / 1000)  # 标准化
        strategy_scores.append((strategy, score, avg_time, avg_overlap))

strategy_scores.sort(key=lambda x: x[1], reverse=True)

strategies_ranked = [s[0] for s in strategy_scores]
scores = [s[1] for s in strategy_scores]
colors_ranked = [strategy_colors[s] for s in strategies_ranked]

bars = plt.barh(range(len(strategies_ranked)), scores, color=colors_ranked, alpha=0.7)
plt.yticks(range(len(strategies_ranked)), strategies_ranked)
plt.xlabel('Composite Score (Overlap/Latency)', fontsize=12)
plt.title('Overall Strategy Performance Ranking', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# 添加数值标注
for i, (bar, score) in enumerate(zip(bars, scores)):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{score:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('grouping_strategy_analysis.png', dpi=300, bbox_inches='tight')

# 生成详细统计表
print("\n=== 详细性能分析 ===")
print("策略性能排名（综合得分 = 簇重叠率 / 标准化延迟）:")
for i, (strategy, score, avg_time, avg_overlap) in enumerate(strategy_scores):
    print(f"{i+1:2d}. {strategy:25s} | 得分: {score:.4f} | 平均延迟: {avg_time:8.1f}μs | 平均重叠率: {avg_overlap:.4f}")

print("\n=== 批大小影响分析 ===")
for batch_size in sorted(df_combined['Batch_Size'].unique()):
    print(f"\n批大小 {batch_size}:")
    batch_data = df_combined[df_combined['Batch_Size'] == batch_size].sort_values('Avg_Time_us')
    baseline_time = baseline_data[baseline_data['Batch_Size'] == batch_size]['Avg_Time_us']
    
    if not baseline_time.empty:
        baseline_val = baseline_time.iloc[0]
        print(f"  基准延迟: {baseline_val:.1f}μs")
        
        better_strategies = []
        for _, row in batch_data.iterrows():
            if row['Strategy_EN'] != 'Baseline (No Grouping)' and row['Avg_Time_us'] < baseline_val:
                improvement = (baseline_val - row['Avg_Time_us']) / baseline_val * 100
                better_strategies.append((row['Strategy_EN'], improvement, row['Cluster_Overlap_Ratio']))
        
        if better_strategies:
            print("  优于基准的策略:")
            for strategy, improvement, overlap in better_strategies:
                print(f"    - {strategy}: {improvement:.1f}% 改进, 重叠率: {overlap:.4f}")
        else:
            print("  无策略优于基准")

print("\n可视化分析完成！图表已保存为 'grouping_strategy_analysis.png'") 