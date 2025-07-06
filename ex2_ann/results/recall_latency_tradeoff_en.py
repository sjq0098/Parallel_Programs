import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# 设置中文字体和图表样式
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

# 读取数据
df = pd.read_csv('report_figures/data_used.csv')

# 定义算法分组和颜色
algorithm_groups = {
    'Brute Force Methods': {
        'algorithms': ['Scalar Brute Force', 'SSE Optimized Brute Force', 'AVX Optimized Brute Force'],
        'color': '#e74c3c',
        'marker': 'o',
        'linestyle': '-'
    },
    'Quantization Methods': {
        'algorithms': ['Scalar Quantization (SQ)'],
        'color': '#3498db', 
        'marker': 's',
        'linestyle': '-'
    },
    'Product Quantization (PQ)': {
        'algorithms': ['Product Quantization (PQ) M=4', 'Product Quantization (PQ) M=8', 
                      'Product Quantization (PQ) M=16', 'Product Quantization (PQ) M=32'],
        'color': '#2ecc71',
        'marker': '^',
        'linestyle': '-'
    },
    'Optimized PQ (OPQ)': {
        'algorithms': ['Optimized PQ (OPQ) M=4', 'Optimized PQ (OPQ) M=8',
                      'Optimized PQ (OPQ) M=16', 'Optimized PQ (OPQ) M=32'],
        'color': '#f39c12',
        'marker': 'v',
        'linestyle': '-'
    },
    'Hybrid Methods': {
        'algorithms': ['Hybrid Search (PQ16+Rerank)', 'Hybrid Search (PQ32+Rerank)',
                      'Hybrid Search (OPQ16+Rerank)', 'Hybrid Search (OPQ32+Rerank)'],
        'color': '#9b59b6',
        'marker': 'D',
        'linestyle': '-'
    }
}

# 创建图表
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# 为每个算法组绘制连线
for group_name, group_info in algorithm_groups.items():
    group_data = df[df['algorithm'].isin(group_info['algorithms'])].copy()
    
    if len(group_data) > 0:
        # 按recall排序以确保连线合理
        group_data = group_data.sort_values('recall')
        
        # 绘制连线和点
        ax.plot(group_data['recall'], group_data['latency_ms'], 
               color=group_info['color'], 
               marker=group_info['marker'], 
               linestyle=group_info['linestyle'],
               linewidth=2.5,
               markersize=8,
               label=group_name,
               alpha=0.8)

# 设置图表属性
ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
ax.set_ylabel('Latency (ms)', fontsize=14, fontweight='bold')
ax.set_title('Recall-Latency Trade-off Analysis\nDEEP100K Dataset, K=10', 
             fontsize=16, fontweight='bold', pad=20)

# 设置网格
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# 设置坐标轴范围
ax.set_xlim(0, 1.05)
ax.set_ylim(0, max(df['latency_ms']) * 1.1)

# 添加图例
legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11)
legend.get_frame().set_alpha(0.9)

# 设置刻度标签
ax.tick_params(axis='both', which='major', labelsize=12)

# 添加一些关键点的标注
key_points = [
    ('AVX Optimized Brute Force', 'Best Accuracy\n(Highest Recall)'),
    ('Optimized PQ (OPQ) M=4', 'Fastest Method\n(Lowest Latency)'),
    ('Hybrid Search (PQ32+Rerank)', 'Best Balance\n(High Recall + Reasonable Speed)')
]

for alg_name, annotation in key_points:
    point_data = df[df['algorithm'] == alg_name]
    if len(point_data) > 0:
        x, y = point_data.iloc[0]['recall'], point_data.iloc[0]['latency_ms']
        ax.annotate(annotation, 
                   xy=(x, y), 
                   xytext=(10, 10), 
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                   fontsize=9)

# 调整布局
plt.tight_layout()

# 保存图表
output_file = 'report_figures/recall_latency_tradeoff_en.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图表已保存到: {output_file}")

# 显示图表
plt.show()

# 打印性能总结
print("\n=== Performance Summary ===")
print("Top 3 methods by recall:")
top_recall = df.nlargest(3, 'recall')[['algorithm', 'recall', 'latency_ms']]
for idx, row in top_recall.iterrows():
    print(f"  {row['algorithm']}: Recall={row['recall']:.3f}, Latency={row['latency_ms']:.2f}ms")

print("\nTop 3 fastest methods:")
fastest = df.nsmallest(3, 'latency_ms')[['algorithm', 'recall', 'latency_ms']]
for idx, row in fastest.iterrows():
    print(f"  {row['algorithm']}: Latency={row['latency_ms']:.2f}ms, Recall={row['recall']:.3f}")

print("\nBest trade-off methods (Recall > 0.9, Latency < 3ms):")
balanced = df[(df['recall'] > 0.9) & (df['latency_ms'] < 3.0)][['algorithm', 'recall', 'latency_ms']]
for idx, row in balanced.iterrows():
    print(f"  {row['algorithm']}: Recall={row['recall']:.3f}, Latency={row['latency_ms']:.2f}ms") 