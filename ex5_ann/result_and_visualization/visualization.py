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
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))

# Figure 1: Comparison of average latency with different batch sizes
plt.subplot(2, 2, 1)
batch_latency = df.pivot_table(index='Batch_Size', columns='Method', values='Avg_Latency_us')
batch_latency.plot(kind='bar', ax=plt.gca())
plt.title('Average Latency with Different Batch Sizes')
plt.ylabel('Average Latency (μs)')
plt.xlabel('Batch Size')
plt.xticks(rotation=0)
plt.grid(True)
plt.legend(title='Method')

# Figure 2: Comparison of average latency with different query counts
plt.subplot(2, 2, 2)
query_latency = df.pivot_table(index='Query_Count', columns='Method', values='Avg_Latency_us')
query_latency.plot(kind='bar', ax=plt.gca())
plt.title('Average Latency with Different Query Counts')
plt.ylabel('Average Latency (μs)')
plt.xlabel('Query Count')
plt.xticks(rotation=0)
plt.grid(True)
plt.legend(title='Method')

# Figure 3: Impact of batch size on latency
plt.subplot(2, 2, 3)
for method in df['Method'].unique():
    method_df = df[df['Method'] == method]
    for query in method_df['Query_Count'].unique():
        query_df = method_df[method_df['Query_Count'] == query]
        plt.plot(query_df['Batch_Size'], query_df['Avg_Latency_us'], marker='o', label=f'{method}, Query={query}')
plt.title('Impact of Batch Size on Average Latency')
plt.xlabel('Batch Size')
plt.ylabel('Average Latency (μs)')
plt.xscale('log', base=2)
plt.grid(True)
plt.legend(fontsize='small')

# Figure 4: Speedup analysis
plt.subplot(2, 2, 4)
speedup_df = df.pivot_table(index=['Batch_Size', 'Query_Count'], columns='Method', values='Avg_Latency_us').reset_index()
speedup_df['Speedup_Ratio'] = speedup_df['GPU+GPU'] / speedup_df['GPU+CPU']
pivot_speedup = speedup_df.pivot_table(index='Batch_Size', columns='Query_Count', values='Speedup_Ratio')
sns.heatmap(pivot_speedup, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('GPU+GPU to GPU+CPU Latency Ratio (>1 means GPU+CPU is faster)')
plt.ylabel('Batch Size')
plt.xlabel('Query Count')

plt.tight_layout()
plt.savefig('benchmark_visualization.png', dpi=300)

# Create a line chart showing the relationship between latency and batch size
plt.figure(figsize=(12, 6))
for method in df['Method'].unique():
    avg_latency = df[df['Method'] == method].groupby('Batch_Size')['Avg_Latency_us'].mean()
    plt.plot(avg_latency.index, avg_latency, marker='o', linewidth=2, label=method)

plt.title('Impact of Batch Size on Average Latency (Mean of All Query Counts)')
plt.xlabel('Batch Size')
plt.ylabel('Average Latency (μs)')
plt.xscale('log', base=2)
plt.grid(True)
plt.legend()
plt.savefig('batch_size_impact.png', dpi=300)

# Create a line chart showing the relationship between latency and query count
plt.figure(figsize=(12, 6))
for method in df['Method'].unique():
    avg_latency = df[df['Method'] == method].groupby('Query_Count')['Avg_Latency_us'].mean()
    plt.plot(avg_latency.index, avg_latency, marker='o', linewidth=2, label=method)

plt.title('Impact of Query Count on Average Latency (Mean of All Batch Sizes)')
plt.xlabel('Query Count')
plt.ylabel('Average Latency (μs)')
plt.grid(True)
plt.legend()
plt.savefig('query_count_impact.png', dpi=300)

print("Visualization analysis completed! Charts have been saved as PNG files.") 