#!/bin/bash

# 创建结果目录
mkdir -p results 2>/dev/null || mkdir results

# 检测操作系统类型
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    EXEC="build/ivf_benchmark.exe"
else
    EXEC="build/ivf_benchmark"
fi

# 检查可执行文件是否存在
if [ ! -f "$EXEC" ]; then
    echo "错误: 找不到可执行文件 $EXEC"
    echo "请先运行 ./build_ivf.sh 编译程序"
    exit 1
fi

# 运行测试
echo "开始运行测试..."
echo "结果将保存在 results 目录中"

# 测试暴力搜索方法
echo "测试暴力搜索方法..."
$EXEC flat > results/flat_results.csv

# 测试IVF方法
echo "测试IVF相关方法..."
$EXEC ivf > results/ivf_results.csv

# 测试IVFPQ方法
echo "测试IVFPQ相关方法..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    $EXEC ivfpq 2>NUL > results/ivfpq_results.csv
else
    $EXEC ivfpq 2>/dev/null > results/ivfpq_results.csv
fi

# 生成性能对比图
if command -v python3 &>/dev/null; then
    echo "检测到Python3，正在生成性能对比图..."
    
    # 创建Python脚本用于绘制结果
    cat > plot_ivf_results.py << 'EOF'
#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

# 读取Flat搜索结果
try:
    flat_df = pd.read_csv('results/flat_results.csv')
    flat_df['Method'] = flat_df['Method'].astype(str)
    has_flat_results = True
except Exception as e:
    print(f"Warning: 无法加载暴力搜索结果: {e}")
    has_flat_results = False

# 读取IVF结果
try:
    ivf_df = pd.read_csv('results/ivf_results.csv')
    ivf_df['Method'] = ivf_df['Method'].astype(str)
    ivf_df['Configuration'] = ivf_df['Configuration'].astype(str)
    has_ivf_results = True
except Exception as e:
    print(f"Warning: 无法加载IVF结果: {e}")
    has_ivf_results = False

# 读取IVFPQ结果
try:
    ivfpq_df = pd.read_csv('results/ivfpq_results.csv')
    ivfpq_df['Method'] = ivfpq_df['Method'].astype(str)
    ivfpq_df['Configuration'] = ivfpq_df['Configuration'].astype(str)
    has_ivfpq_results = True
except Exception as e:
    print(f"Warning: 无法加载IVFPQ结果: {e}")
    has_ivfpq_results = False

# 提取参数
def extract_params(config):
    params = {}
    for param in config.split(','):
        key, value = param.split('=')
        params[key] = int(value)
    return params

if has_ivf_results:
    # 处理IVF数据
    ivf_df['nlist'] = ivf_df['Configuration'].apply(lambda x: extract_params(x)['nlist'])
    ivf_df['nprobe'] = ivf_df['Configuration'].apply(lambda x: extract_params(x)['nprobe'])
    ivf_df['Implementation'] = ivf_df['Method'].apply(lambda x: re.search(r'\((.*?)\)', x).group(1))
    ivf_df['Algorithm'] = 'IVF-Flat'

if has_ivfpq_results:
    # 处理IVFPQ数据
    ivfpq_df['nlist'] = ivfpq_df['Configuration'].apply(lambda x: extract_params(x)['nlist'])
    ivfpq_df['nprobe'] = ivfpq_df['Configuration'].apply(lambda x: extract_params(x)['nprobe'])
    ivfpq_df['m'] = ivfpq_df['Configuration'].apply(lambda x: extract_params(x)['m'])
    ivfpq_df['Implementation'] = ivfpq_df['Method'].apply(lambda x: re.search(r'\((.*?)\)', x).group(1))
    ivfpq_df['Algorithm'] = 'IVF-PQ'

# 合并数据
all_df = pd.DataFrame()
if has_ivf_results:
    all_df = pd.concat([all_df, ivf_df])
if has_ivfpq_results:
    all_df = pd.concat([all_df, ivfpq_df])

# 图0: 暴力搜索性能比较
if has_flat_results:
    plt.figure(figsize=(10, 6))
    # 提取算法名称和实现类型
    flat_df['Algorithm'] = flat_df['Method'].apply(lambda x: x.split('(')[0].strip())
    
    # 绘制柱状图
    algorithms = flat_df['Algorithm'].unique()
    x = np.arange(len(algorithms))
    width = 0.35
    
    # 召回率柱状图
    ax1 = plt.subplot(1, 2, 1)
    recall_data = [flat_df[flat_df['Algorithm'] == alg]['Recall'].values[0] for alg in algorithms]
    bars1 = ax1.bar(x, recall_data, width, label='Recall')
    ax1.set_ylabel('Recall')
    ax1.set_title('召回率比较')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45)
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # 延迟柱状图
    ax2 = plt.subplot(1, 2, 2)
    latency_data = [flat_df[flat_df['Algorithm'] == alg]['Latency(us)'].values[0] for alg in algorithms]
    bars2 = ax2.bar(x, latency_data, width, label='Latency', color='orange')
    ax2.set_ylabel('Latency (us)')
    ax2.set_title('延迟比较')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=45)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/flat_search_comparison.png', dpi=300)
    print("暴力搜索比较图已保存为 'results/flat_search_comparison.png'")

# 图1: 所有方法性能对比
plt.figure(figsize=(12, 8))

# 绘制暴力搜索结果
if has_flat_results:
    for i, row in flat_df.iterrows():
        plt.scatter(row['Latency(us)'], row['Recall'], 
                    label=row['Method'], marker='*', s=200)

# 绘制IVF和IVFPQ结果
if len(all_df) > 0:
    implementations = all_df['Implementation'].unique()
    algorithms = all_df['Algorithm'].unique()
    
    markers = {'Serial': 'o', 'OpenMP': 's', 'pthread': '^'}
    colors = {'IVF-Flat': 'blue', 'IVF-PQ': 'green'}
    
    for algo in algorithms:
        for impl in implementations:
            df_subset = all_df[(all_df['Algorithm'] == algo) & (all_df['Implementation'] == impl)]
            if not df_subset.empty:
                plt.scatter(df_subset['Latency(us)'], df_subset['Recall'],
                            label=f'{algo} ({impl})',
                            marker=markers.get(impl, 'o'),
                            color=colors.get(algo, 'red'),
                            alpha=0.7, s=50)

plt.xscale('log')
plt.xlabel('延迟 (微秒)', fontsize=12)
plt.ylabel('召回率', fontsize=12)
plt.title('所有搜索方法性能对比', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('results/all_methods_comparison.png', dpi=300)
print("所有方法性能对比图已保存为 'results/all_methods_comparison.png'")

if has_ivf_results:
    # 图2: 不同实现方式的性能比较 (IVF-Flat)
    plt.figure(figsize=(12, 8))
    for impl in ivf_df['Implementation'].unique():
        df_subset = ivf_df[ivf_df['Implementation'] == impl]
        plt.scatter(df_subset['Latency(us)'], df_subset['Recall'], 
                    label=f'IVF-Flat ({impl})', alpha=0.7, s=50)
    
    plt.xscale('log')
    plt.xlabel('延迟 (微秒)', fontsize=12)
    plt.ylabel('召回率', fontsize=12)
    plt.title('IVF-Flat: 不同实现方式的性能比较', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('results/ivf_implementations.png', dpi=300)
    print("IVF实现性能比较图已保存为 'results/ivf_implementations.png'")

if has_ivfpq_results:
    # 图3: 不同实现方式的性能比较 (IVF-PQ)
    plt.figure(figsize=(12, 8))
    for impl in ivfpq_df['Implementation'].unique():
        df_subset = ivfpq_df[ivfpq_df['Implementation'] == impl]
        plt.scatter(df_subset['Latency(us)'], df_subset['Recall'], 
                    label=f'IVF-PQ ({impl})', alpha=0.7, s=50)
    
    plt.xscale('log')
    plt.xlabel('延迟 (微秒)', fontsize=12)
    plt.ylabel('召回率', fontsize=12)
    plt.title('IVF-PQ: 不同实现方式的性能比较', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('results/ivfpq_implementations.png', dpi=300)
    print("IVFPQ实现性能比较图已保存为 'results/ivfpq_implementations.png'")

# 更多图表生成...
if has_ivf_results and has_ivfpq_results:
    # IVF-Flat vs IVF-PQ
    plt.figure(figsize=(12, 8))
    for algo in ['IVF-Flat', 'IVF-PQ']:
        df_subset = all_df[(all_df['Algorithm'] == algo) & (all_df['Implementation'] == 'Serial')]
        plt.scatter(df_subset['Latency(us)'], df_subset['Recall'], 
                    label=algo, alpha=0.7, s=50)
    
    plt.xscale('log')
    plt.xlabel('延迟 (微秒)', fontsize=12)
    plt.ylabel('召回率', fontsize=12)
    plt.title('IVF-Flat vs IVF-PQ (串行实现)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('results/ivf_vs_ivfpq.png', dpi=300)
    print("IVF vs IVFPQ对比图已保存为 'results/ivf_vs_ivfpq.png'")

if has_ivf_results:
    # nlist对性能的影响 (IVF-Flat)
    plt.figure(figsize=(12, 8))
    for nlist in sorted(ivf_df['nlist'].unique()):
        df_subset = ivf_df[(ivf_df['nlist'] == nlist) & (ivf_df['Implementation'] == 'Serial')]
        plt.scatter(df_subset['Latency(us)'], df_subset['Recall'], 
                    label=f'nlist={nlist}', alpha=0.7, s=50)
    
    plt.xscale('log')
    plt.xlabel('延迟 (微秒)', fontsize=12)
    plt.ylabel('召回率', fontsize=12)
    plt.title('IVF-Flat: nlist对性能的影响', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('results/ivf_nlist_impact.png', dpi=300)
    print("nlist对IVF性能的影响图已保存为 'results/ivf_nlist_impact.png'")

    # 加速比分析 (IVF-Flat)
    plt.figure(figsize=(12, 8))
    
    # 计算加速比
    speedup_data = []
    for nlist in sorted(ivf_df['nlist'].unique()):
        for nprobe in sorted(ivf_df['nprobe'].unique()):
            serial_data = ivf_df[(ivf_df['nlist'] == nlist) & 
                              (ivf_df['nprobe'] == nprobe) & 
                              (ivf_df['Implementation'] == 'Serial')]
            
            if len(serial_data) == 0:
                continue
            
            serial_latency = serial_data['Latency(us)'].iloc[0]
            
            for impl in ['OpenMP', 'pthread']:
                parallel_data = ivf_df[(ivf_df['nlist'] == nlist) & 
                                   (ivf_df['nprobe'] == nprobe) & 
                                   (ivf_df['Implementation'] == impl)]
                
                if len(parallel_data) == 0:
                    continue
                
                parallel_latency = parallel_data['Latency(us)'].iloc[0]
                speedup = serial_latency / parallel_latency
                
                speedup_data.append({
                    'nlist': nlist,
                    'nprobe': nprobe,
                    'Implementation': impl,
                    'Speedup': speedup
                })
    
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        
        for impl in speedup_df['Implementation'].unique():
            df_subset = speedup_df[speedup_df['Implementation'] == impl]
            plt.scatter(df_subset['nprobe'], df_subset['Speedup'], 
                        label=impl, alpha=0.7, s=50)
        
        plt.axhline(y=1.0, color='gray', linestyle='--')
        plt.xlabel('nprobe', fontsize=12)
        plt.ylabel('加速比 (Serial/Parallel)', fontsize=12)
        plt.title('IVF-Flat: 并行实现的加速比', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig('results/ivf_speedup.png', dpi=300)
        print("IVF加速比分析图已保存为 'results/ivf_speedup.png'")

print("所有图表已保存到results目录")
EOF

    # 运行Python脚本
    python3 plot_ivf_results.py
else
    echo "未检测到Python3，跳过图表生成"
    echo "如需生成图表，请安装Python3及必要的库(matplotlib, pandas, numpy)"
fi

echo "测试完成，结果已保存到results目录" 