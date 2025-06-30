import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from scipy.stats import ttest_rel

def visualize_and_analyze(csv_path):
    # 1. 读取数据
    df = pd.read_csv(csv_path)
    
    # 2. 汇总统计
    summary = df.groupby('sharding_strategy').agg({
        'latency_us':    ['mean','std'],
        'recall':        ['mean','std'],
        'build_time_ms': ['mean','std']
    })
    print("=== Summary Statistics by Strategy ===")
    print(summary)
    
    # 3. 箱线/小提琴图：Latency 和 Recall
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    sns.boxplot(data=df, x='sharding_strategy', y='latency_us')
    plt.title('Latency Distribution')
    plt.subplot(1,2,2)
    sns.boxplot(data=df, x='sharding_strategy', y='recall')
    plt.title('Recall Distribution')
    plt.tight_layout()
    plt.show()
    
    # 4. 构建时间 vs 平均延迟散点
    mean_stats = df.groupby('sharding_strategy').agg({
        'latency_us':'mean', 'build_time_ms':'mean'
    }).reset_index()
    plt.figure()
    sns.scatterplot(data=mean_stats, x='build_time_ms', y='latency_us', hue='sharding_strategy', s=100)
    plt.xlabel('Build Time (ms)')
    plt.ylabel('Avg Latency (µs)')
    plt.title('Build Time vs Avg Latency')
    plt.grid(True)
    plt.show()
    
    # 5. Recall vs Latency with convex hull & centroids
    strategies = df['sharding_strategy'].unique()
    markers = ['o', 's', '^', 'D']
    colors  = sns.color_palette('tab10', len(strategies))
    plt.figure(figsize=(8, 6))
    for strat, marker, color in zip(strategies, markers, colors):
        sub = df[df['sharding_strategy'] == strat]
        plt.scatter(sub['latency_us'], sub['recall'],
                    marker=marker, c=[color], s=50, edgecolor='k', linewidth=1, alpha=0.7, label=strat)
        pts = sub[['latency_us','recall']].values
        if len(pts)>=3:
            hull = ConvexHull(pts)
            poly = pts[hull.vertices]
            plt.fill(poly[:,0], poly[:,1], color=color, alpha=0.1)
        mx,my = sub['latency_us'].mean(), sub['recall'].mean()
        plt.scatter(mx, my, marker='*', c=[color], s=200, edgecolor='k', linewidth=1.5)
    plt.xscale('log')
    plt.xlabel('Latency (µs)')
    plt.ylabel('Recall@10')
    plt.title('Recall vs Latency by Strategy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 6. Statistical significance: compare RANDOM vs KMEANS for latency
    a = df[df['sharding_strategy']=='RANDOM']['latency_us']
    b = df[df['sharding_strategy']=='KMEANS_BASED']['latency_us']
    stat, pval = ttest_rel(a.sort_index(), b.sort_index())
    print(f"T-test between RANDOM and KMEANS latency: p-value = {pval:.3f}")
    
if __name__ == '__main__':
    visualize_and_analyze('results_mpi_sharded_hnsw.csv')

