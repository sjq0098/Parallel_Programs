#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题

# 读取多查询并行结果
try:
    df = pd.read_csv('results/multi_ivf_results.csv')
    df['Method'] = df['Method'].astype(str)
    
    # 1. 绘制不同方法和线程数下的QPS对比
    plt.figure(figsize=(14, 8))
    
    # 筛选有意义的配置参数组合
    df_filtered = df[(df['nlist'] == 256) & (df['nprobe'] == 16)]
    
    # 按方法分组
    methods = df_filtered['Method'].unique()
    
    for method in methods:
        method_data = df_filtered[df_filtered['Method'] == method]
        if len(method_data) > 1:  # 如果有多个线程数据点
            plt.plot(method_data['threads'], method_data['throughput(QPS)'], 
                    marker='o', label=method, linewidth=2, markersize=8)
        else:  # 如果只有一个数据点（串行方法）
            plt.scatter(method_data['threads'], method_data['throughput(QPS)'], 
                        label=method, marker='*', s=100)
    
    plt.xlabel('线程数', fontsize=14)
    plt.ylabel('吞吐量 (QPS)', fontsize=14)
    plt.title('不同方法和线程数下的查询吞吐量对比 (nlist=256, nprobe=16)', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results/multi_query_throughput.png', dpi=300)
    
    # 2. 绘制加速比
    plt.figure(figsize=(14, 8))
    
    # 获取串行基准值
    serial_data = df[df['Method'].str.contains('Serial')]
    
    # 合并数据，计算加速比
    for nlist in df['nlist'].unique():
        for nprobe in df['nprobe'].unique():
            # 获取当前参数下的串行性能
            serial_perf = serial_data[(serial_data['nlist'] == nlist) & 
                                    (serial_data['nprobe'] == nprobe)]['throughput(QPS)'].values
            
            if len(serial_perf) == 0:
                continue
                
            serial_throughput = serial_perf[0]
            
            # 筛选并行方法数据
            for method in [m for m in methods if 'Serial' not in m]:
                parallel_data = df[(df['Method'] == method) & 
                                (df['nlist'] == nlist) & 
                                (df['nprobe'] == nprobe)]
                
                if parallel_data.empty:
                    continue
                
                # 计算加速比
                parallel_data = parallel_data.copy()
                parallel_data['speedup'] = parallel_data['throughput(QPS)'] / serial_throughput
                
                # 绘制加速比曲线
                plt.plot(parallel_data['threads'], parallel_data['speedup'], 
                        marker='o', label=f'{method} (nlist={nlist}, nprobe={nprobe})',
                        linewidth=2, markersize=8)
    
    # 绘制理想加速比参考线
    max_threads = df['threads'].max()
    plt.plot([1, max_threads], [1, max_threads], 'k--', label='理想线性加速比', alpha=0.5)
    
    plt.xlabel('线程数', fontsize=14)
    plt.ylabel('加速比 (相对于串行方法)', fontsize=14)
    plt.title('多查询并行方法的加速比', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('results/multi_query_speedup.png', dpi=300)
    
    # 3. 综合指标图 - 同时展示召回率和吞吐量
    plt.figure(figsize=(14, 8))
    
    # 筛选有代表性的数据点
    representative_data = df[(df['threads'] == df['threads'].max()) | (df['Method'].str.contains('Serial'))]
    
    # 绘制散点图
    scatter = plt.scatter(representative_data['throughput(QPS)'], 
                          representative_data['avg_recall'],
                          c=representative_data['Method'].astype('category').cat.codes,
                          s=100, alpha=0.7, cmap='viridis')
    
    # 为每个点添加标签
    for i, row in representative_data.iterrows():
        label = f"{row['Method']}\nnlist={row['nlist']}, nprobe={row['nprobe']}"
        if 'Serial' not in row['Method']:
            label += f", threads={row['threads']}"
        plt.annotate(label, (row['throughput(QPS)'], row['avg_recall']),
                    fontsize=8, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('吞吐量 (QPS)', fontsize=14)
    plt.ylabel('平均召回率', fontsize=14)
    plt.title('多查询方法的吞吐量-召回率权衡', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(handles=scatter.legend_elements()[0], labels=representative_data['Method'].unique())
    plt.tight_layout()
    plt.savefig('results/multi_query_performance_tradeoff.png', dpi=300)
    
    print("可视化完成，所有图表已保存到results目录")
    
except Exception as e:
    print(f"生成图表时出错: {e}")
