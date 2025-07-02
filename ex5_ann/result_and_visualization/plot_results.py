import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取CSV结果
df = pd.read_csv('benchmark_results.csv')

# 设置图表风格
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# 创建图表目录
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# 按批大小绘制延迟对比图
plt.figure(figsize=(12, 8))
for method in df['方法'].unique():
    for qcount in df['查询数量'].unique():
        data = df[(df['方法'] == method) & (df['查询数量'] == qcount)]
        plt.plot(data['批大小'], data['平均延迟(μs)'], marker='o', label=f'{method}, 查询数={qcount}')

plt.xlabel('批大小')
plt.ylabel('平均延迟 (μs)')
plt.title('不同方法和查询数量下批大小对延迟的影响')
plt.legend()
plt.grid(True)
plt.savefig('plots/latency_by_batch_size.png', dpi=300, bbox_inches='tight')

# 按查询数量绘制延迟对比图
plt.figure(figsize=(12, 8))
for method in df['方法'].unique():
    for batch in sorted(df['批大小'].unique()):
        data = df[(df['方法'] == method) & (df['批大小'] == batch)]
        plt.plot(data['查询数量'], data['平均延迟(μs)'], marker='o', label=f'{method}, 批大小={batch}')

plt.xlabel('查询数量')
plt.ylabel('平均延迟 (μs)')
plt.title('不同方法和批大小下查询数量对延迟的影响')
plt.legend()
plt.grid(True)
plt.savefig('plots/latency_by_query_count.png', dpi=300, bbox_inches='tight')

# 按批大小绘制召回率对比图
plt.figure(figsize=(12, 8))
for method in df['方法'].unique():
    for qcount in df['查询数量'].unique():
        data = df[(df['方法'] == method) & (df['查询数量'] == qcount)]
        plt.plot(data['批大小'], data['平均召回率'], marker='o', label=f'{method}, 查询数={qcount}')

plt.xlabel('批大小')
plt.ylabel('平均召回率')
plt.title('不同方法和查询数量下批大小对召回率的影响')
plt.legend()
plt.grid(True)
plt.savefig('plots/recall_by_batch_size.png', dpi=300, bbox_inches='tight')

# 创建热力图比较两种方法在不同批大小和查询数下的性能差异
plt.figure(figsize=(14, 10))
pivot_latency = df.pivot_table(
    index='批大小', 
    columns=['方法', '查询数量'], 
    values='平均延迟(μs)'
)
sns.heatmap(pivot_latency, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('不同批大小和查询数量下各方法的平均延迟(μs)')
plt.tight_layout()
plt.savefig('plots/latency_heatmap.png', dpi=300, bbox_inches='tight')

# 绘制条形图比较两种方法
plt.figure(figsize=(14, 10))
grouped = df.groupby(['批大小', '方法'])['平均延迟(μs)'].mean().reset_index()
sns.barplot(x='批大小', y='平均延迟(μs)', hue='方法', data=grouped)
plt.title('不同批大小下各方法的平均延迟(μs)')
plt.tight_layout()
plt.savefig('plots/method_comparison.png', dpi=300, bbox_inches='tight')

print('图表已生成在plots目录下')
