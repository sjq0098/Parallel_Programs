#!/bin/bash

# 创建结果目录
mkdir -p results 2>/dev/null || mkdir results

# 测试所有方法并保存结果
echo "开始运行所有测试..."
echo "结果将保存在 results 目录中"

# 运行所有测试并保存结果到CSV文件
./build/ann_benchmark.exe all > results/all_methods.csv

# 单独运行每个方法并保存详细结果
METHODS=("flat" "sq" "pq4" "pq8" "pq16" "pq32" "pq4rerank" "pq8rerank" "pq16rerank" "pq32rerank" \
         "ivf_n64_p4" "ivf_n64_p8" "ivf_n64_p12" \
         "ivf_n128_p8" "ivf_n128_p12" "ivf_n128_p16" \
         "ivf_n256_p12" "ivf_n256_p16" "ivf_n256_p24" \
         "ivf_n512_p16" "ivf_n512_p20" "ivf_n512_p32" \
         "ivf_omp_n64_p4" "ivf_omp_n64_p8" "ivf_omp_n64_p12" \
         "ivf_omp_n128_p8" "ivf_omp_n128_p12" "ivf_omp_n128_p16" \
         "ivf_omp_n256_p12" "ivf_omp_n256_p16" "ivf_omp_n256_p24" \
         "ivf_omp_n512_p16" "ivf_omp_n512_p20" "ivf_omp_n512_p32" \
         "ivf_ptd_n64_p4" "ivf_ptd_n64_p8" "ivf_ptd_n64_p12" \
         "ivf_ptd_n128_p8" "ivf_ptd_n128_p12" "ivf_ptd_n128_p16" \
         "ivf_ptd_n256_p12" "ivf_ptd_n256_p16" "ivf_ptd_n256_p24" \
         "ivf_ptd_n512_p16" "ivf_ptd_n512_p20" "ivf_ptd_n512_p32")
METHOD_NAMES=("FLAT_SEARCH" "FLAT_SEARCH_SQ" "PQ4" "PQ8" "PQ16" "PQ32" \
              "PQ4_RERANK" "PQ8_RERANK" "PQ16_RERANK" "PQ32_RERANK" \
              "IVF_N64_P4" "IVF_N64_P8" "IVF_N64_P12" \
              "IVF_N128_P8" "IVF_N128_P12" "IVF_N128_P16" \
              "IVF_N256_P12" "IVF_N256_P16" "IVF_N256_P24" \
              "IVF_N512_P16" "IVF_N512_P20" "IVF_N512_P32" \
              "IVF_OMP_N64_P4" "IVF_OMP_N64_P8" "IVF_OMP_N64_P12" \
              "IVF_OMP_N128_P8" "IVF_OMP_N128_P12" "IVF_OMP_N128_P16" \
              "IVF_OMP_N256_P12" "IVF_OMP_N256_P16" "IVF_OMP_N256_P24" \
              "IVF_OMP_N512_P16" "IVF_OMP_N512_P20" "IVF_OMP_N512_P32" \
              "IVF_PTD_N64_P4" "IVF_PTD_N64_P8" "IVF_PTD_N64_P12" \
              "IVF_PTD_N128_P8" "IVF_PTD_N128_P12" "IVF_PTD_N128_P16" \
              "IVF_PTD_N256_P12" "IVF_PTD_N256_P16" "IVF_PTD_N256_P24" \
              "IVF_PTD_N512_P16" "IVF_PTD_N512_P20" "IVF_PTD_N512_P32")

for i in "${!METHODS[@]}"; do
    method=${METHODS[$i]}
    name=${METHOD_NAMES[$i]}
    echo "运行 $name 测试..."
    ./build/ann_benchmark.exe "$method" > "results/${method}_results.txt"
done

# 生成汇总报告
echo "生成汇总报告..."
echo "方法,召回率,延迟(微秒)" > results/summary.csv
for i in "${!METHODS[@]}"; do
    method=${METHODS[$i]}
    name=${METHOD_NAMES[$i]}
    
    # 从文件中提取召回率和延迟
    recall=$(grep "Average recall" "results/${method}_results.txt" | awk '{print $3}')
    latency=$(grep "Average latency" "results/${method}_results.txt" | awk '{print $4}')
    
    echo "$name,$recall,$latency" >> results/summary.csv
done

# 检查Python是否安装
if command -v python &>/dev/null; then
    echo "检测到Python，正在生成性能对比图..."
    
    # 创建简单的Python脚本用于绘制结果
    cat > plot_results.py << 'EOF'
#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if len(sys.argv) < 2:
    print("使用方法: python plot_results.py <results_csv>")
    sys.exit(1)

# 读取CSV数据
df = pd.read_csv(sys.argv[1])

# 设置图表
fig, ax1 = plt.subplots(figsize=(12, 8))

# 绘制召回率条形图
x = np.arange(len(df['方法']))
width = 0.4
bars = ax1.bar(x, df['召回率'], width, color='b', alpha=0.7, label='Recall')
ax1.set_ylim(0, 1.1)
ax1.set_ylabel('Recall')
ax1.set_xticks(x)
ax1.set_xticklabels(df['方法'], rotation=45, ha='right')

# 为条形图添加数值标签
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

# 创建第二个Y轴用于延迟
ax2 = ax1.twinx()
line = ax2.plot(x, df['延迟(微秒)'], 'r-o', label='Latency (μs)')
ax2.set_ylabel('Latency (μs)')
ax2.set_yscale('log')  # 使用对数刻度以便更好地显示不同量级的延迟

# 为线条添加数值标签
for i, value in enumerate(df['延迟(微秒)']):
    ax2.annotate(f'{value:.0f}',
                xy=(i, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', color='r', fontsize=8)

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

# 添加PQ方法与其重排序版本的比较图
plt.figure(figsize=(12, 8))
plt.title('Performance Comparison: PQ vs PQ+Rerank')

# 提取PQ和PQ+重排序方法
pq_methods = [name for name in df['方法'] if name.startswith('PQ') and not name.endswith('RERANK')]
pq_rerank_methods = [name for name in df['方法'] if name.endswith('RERANK')]

# 准备数据
pq_recalls = [df[df['方法'] == method]['召回率'].values[0] for method in pq_methods]
pq_rerank_recalls = [df[df['方法'] == method]['召回率'].values[0] for method in pq_rerank_methods]
pq_latencies = [df[df['方法'] == method]['延迟(微秒)'].values[0] for method in pq_methods]
pq_rerank_latencies = [df[df['方法'] == method]['延迟(微秒)'].values[0] for method in pq_rerank_methods]

# 绘制召回率对比
x = np.arange(len(pq_methods))
width = 0.35
plt.subplot(1, 2, 1)
plt.bar(x - width/2, pq_recalls, width, label='PQ', color='blue', alpha=0.7)
plt.bar(x + width/2, pq_rerank_recalls, width, label='PQ+Rerank', color='green', alpha=0.7)
plt.xticks(x, [m.replace('PQ', '') for m in pq_methods])
plt.ylabel('Recall')
plt.title('Recall Comparison')
plt.legend()
plt.grid(alpha=0.3)

# 绘制延迟对比（对数尺度）
plt.subplot(1, 2, 2)
plt.bar(x - width/2, pq_latencies, width, label='PQ', color='blue', alpha=0.7)
plt.bar(x + width/2, pq_rerank_latencies, width, label='PQ+Rerank', color='green', alpha=0.7)
plt.yscale('log')
plt.xticks(x, [m.replace('PQ', '') for m in pq_methods])
plt.ylabel('Latency (μs)')
plt.title('Latency Comparison (Log Scale)')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/pq_comparison.png', dpi=300)
print("PQ comparison chart saved as 'results/pq_comparison.png'")

# 设置标题和网格
plt.figure(1)
plt.title('Recall and Latency Comparison of Different Search Methods')
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# 保存结果
plt.savefig('results/performance_comparison.png', dpi=300)
print("Performance comparison chart saved as 'results/performance_comparison.png'")
plt.show()
EOF

    # 运行Python脚本绘制图表
    python plot_results.py results/summary.csv

    # 新增IVF方法性能对比图
    cat >> plot_results.py << 'EOF'
import re
ivf_rows = df[df['方法'].str.startswith('IVF_N')].copy()
if not ivf_rows.empty:
    # 解析 nlist 和 nprobe
    def parse_ivf_params(name):
        m = re.match(r'IVF_N(\d+)_P(\d+)', name)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None, None
    ivf_rows['nlist'] = ivf_rows['方法'].apply(lambda x: parse_ivf_params(x)[0])
    ivf_rows['nprobe'] = ivf_rows['方法'].apply(lambda x: parse_ivf_params(x)[1])
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    plt.figure(figsize=(10,7))
    colors = cm.get_cmap('tab10', ivf_rows['nlist'].nunique())
    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']
    for idx, (nlist, group) in enumerate(ivf_rows.groupby('nlist')):
        plt.scatter(group['延迟(微秒)'], group['召回率'],
                    label=f'nlist={nlist}',
                    color=colors(idx),
                    marker=markers[idx%len(markers)],
                    s=80)
        for _, row in group.iterrows():
            plt.annotate(f"P{int(row['nprobe'])}", (row['延迟(微秒)'], row['召回率']),
                         textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
    plt.xscale('log')
    plt.xlabel('Latency (μs)')
    plt.ylabel('Recall')
    plt.title('IVF-Flat Performance: Recall vs Latency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/ivf_performance.png', dpi=300)
    print("IVF方法性能对比图已保存为 'results/ivf_performance.png'")
EOF

    # OpenMP加速比对比图
    cat >> plot_results.py << 'EOF'
# 创建OpenMP加速比对比图
import re
ivf_serial_rows = df[df['方法'].str.startswith('IVF_N')].copy()
ivf_omp_rows = df[df['方法'].str.startswith('IVF_OMP_N')].copy()

if not ivf_serial_rows.empty and not ivf_omp_rows.empty:
    # 为了比较，我们需要提取相同的nlist和nprobe组合
    def parse_params(name):
        m = re.match(r'IVF_(?:OMP_)?N(\d+)_P(\d+)', name)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None, None
    
    # 准备比较数据
    comparison_data = []
    for _, omp_row in ivf_omp_rows.iterrows():
        omp_nlist, omp_nprobe = parse_params(omp_row['方法'])
        # 查找对应的串行版本
        serial_name = f"IVF_N{omp_nlist}_P{omp_nprobe}"
        serial_row = ivf_serial_rows[ivf_serial_rows['方法'] == serial_name]
        if not serial_row.empty:
            speedup = serial_row.iloc[0]['延迟(微秒)'] / omp_row['延迟(微秒)']
            comparison_data.append({
                'nlist': omp_nlist,
                'nprobe': omp_nprobe,
                'serial_latency': serial_row.iloc[0]['延迟(微秒)'],
                'omp_latency': omp_row['延迟(微秒)'],
                'speedup': speedup,
                'recall': omp_row['召回率']  # 假设召回率接近相同
            })
    
    if comparison_data:
        import pandas as pd
        comp_df = pd.DataFrame(comparison_data)
        
        # 按nlist分组绘制加速比
        plt.figure(figsize=(12, 8))
        markers = ['o', 's', '^', 'D']
        colors = plt.cm.tab10(range(4))
        
        for i, (nlist, group) in enumerate(comp_df.groupby('nlist')):
            plt.plot(group['nprobe'], group['speedup'], 
                     marker=markers[i % len(markers)], 
                     color=colors[i % len(colors)],
                     label=f'nlist={nlist}', 
                     linewidth=2)
            
        plt.axhline(y=1.0, color='gray', linestyle='--')
        plt.grid(alpha=0.3)
        plt.xlabel('nprobe')
        plt.ylabel('Speedup (Serial Time / OpenMP Time)')
        plt.title('OpenMP Speedup for Different IVF-Flat Configurations')
        plt.legend()
        
        # 添加加速比标签
        for _, row in comp_df.iterrows():
            plt.annotate(f"{row['speedup']:.2f}x", 
                         (row['nprobe'], row['speedup']),
                         textcoords="offset points", 
                         xytext=(0, 7), 
                         ha='center')
        
        plt.tight_layout()
        plt.savefig('results/ivf_omp_speedup.png', dpi=300)
        print("OpenMP加速比对比图已保存为 'results/ivf_omp_speedup.png'")
        
        # 创建延迟-召回率散点图，比较串行和OpenMP版本
        plt.figure(figsize=(12, 8))
        
        # 扩展数据帧以便比较
        comp_df['serial_config'] = comp_df.apply(lambda x: f'N={x["nlist"]}, P={x["nprobe"]} (Serial)', axis=1)
        comp_df['omp_config'] = comp_df.apply(lambda x: f'N={x["nlist"]}, P={x["nprobe"]} (OMP)', axis=1)
        
        # 在同一图上绘制串行和OpenMP版本的延迟-召回率点
        for i, (nlist, group) in enumerate(comp_df.groupby('nlist')):
            # 串行版
            plt.scatter(group['serial_latency'], group['recall'], 
                      marker='o', s=80, alpha=0.7,
                      color=colors[i % len(colors)], 
                      label=f'Serial (N={nlist})')
            
            # OpenMP版
            plt.scatter(group['omp_latency'], group['recall'], 
                      marker='s', s=80, alpha=0.7,
                      color=colors[i % len(colors)],
                      label=f'OpenMP (N={nlist})')
            
            # 连接相同配置的串行和OpenMP版本
            for _, row in group.iterrows():
                plt.plot([row['serial_latency'], row['omp_latency']], 
                         [row['recall'], row['recall']], 
                         color=colors[i % len(colors)], 
                         linestyle='--', alpha=0.5)
                
                # 在连线中标注配置
                plt.annotate(f"P={row['nprobe']}", 
                             ((row['serial_latency'] + row['omp_latency'])/2, row['recall']),
                             textcoords="offset points", 
                             xytext=(0, 7), 
                             ha='center', 
                             fontsize=8)
        
        plt.xscale('log')
        plt.grid(alpha=0.3)
        plt.xlabel('Latency (μs)')
        plt.ylabel('Recall')
        plt.title('Serial vs OpenMP IVF-Flat: Recall-Latency Trade-off')
        
        # 创建自定义图例
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        plt.savefig('results/ivf_serial_vs_omp.png', dpi=300)
        print("串行vs并行IVF对比图已保存为 'results/ivf_serial_vs_omp.png'")
EOF

    # pthread加速比图与比较
    cat >> plot_results.py << 'EOF'
# pthread加速比图与比较
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 提取各版本IVF数据
ivf_serial_rows = df[df['方法'].str.startswith('IVF_N')].copy()
ivf_omp_rows = df[df['方法'].str.startswith('IVF_OMP_N')].copy()
ivf_ptd_rows = df[df['方法'].str.startswith('IVF_PTD_N')].copy()

if not ivf_ptd_rows.empty:
    # 为pthread创建与OpenMP类似的加速比图
    def parse_params(name):
        m = re.match(r'IVF_(?:PTD_|OMP_)?N(\d+)_P(\d+)', name)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None, None
    
    # 准备比较数据
    ptd_comparison_data = []
    for _, ptd_row in ivf_ptd_rows.iterrows():
        ptd_nlist, ptd_nprobe = parse_params(ptd_row['方法'])
        serial_name = f"IVF_N{ptd_nlist}_P{ptd_nprobe}"
        serial_row = ivf_serial_rows[ivf_serial_rows['方法'] == serial_name]
        if not serial_row.empty:
            speedup = serial_row.iloc[0]['延迟(微秒)'] / ptd_row['延迟(微秒)']
            ptd_comparison_data.append({
                'nlist': ptd_nlist,
                'nprobe': ptd_nprobe,
                'serial_latency': serial_row.iloc[0]['延迟(微秒)'],
                'ptd_latency': ptd_row['延迟(微秒)'],
                'speedup': speedup,
                'recall': ptd_row['召回率']
            })
    
    if ptd_comparison_data:
        ptd_comp_df = pd.DataFrame(ptd_comparison_data)
        
        # 绘制pthread加速比图
        plt.figure(figsize=(12, 8))
        markers = ['o', 's', '^', 'D']
        colors = plt.cm.tab10(range(4))
        
        for i, (nlist, group) in enumerate(ptd_comp_df.groupby('nlist')):
            plt.plot(group['nprobe'], group['speedup'], 
                     marker=markers[i % len(markers)], 
                     color=colors[i % len(colors)],
                     label=f'nlist={nlist}', 
                     linewidth=2)
            
        plt.axhline(y=1.0, color='gray', linestyle='--')
        plt.grid(alpha=0.3)
        plt.xlabel('nprobe')
        plt.ylabel('Speedup (Serial Time / pthread Time)')
        plt.title('pthread Speedup for Different IVF-Flat Configurations')
        plt.legend()
        
        for _, row in ptd_comp_df.iterrows():
            plt.annotate(f"{row['speedup']:.2f}x", 
                         (row['nprobe'], row['speedup']),
                         textcoords="offset points", 
                         xytext=(0, 7), 
                         ha='center')
        
        plt.tight_layout()
        plt.savefig('results/ivf_ptd_speedup.png', dpi=300)
        print("pthread加速比对比图已保存为 'results/ivf_ptd_speedup.png'")

        # 创建pthread与OpenMP的性能比较图
        if not ivf_omp_rows.empty:
            # 准备比较数据
            omp_vs_ptd_data = []
            for _, ptd_row in ivf_ptd_rows.iterrows():
                ptd_nlist, ptd_nprobe = parse_params(ptd_row['方法'])
                omp_name = f"IVF_OMP_N{ptd_nlist}_P{ptd_nprobe}"
                omp_row = ivf_omp_rows[ivf_omp_rows['方法'] == omp_name]
                
                if not omp_row.empty:
                    ratio = omp_row.iloc[0]['延迟(微秒)'] / ptd_row['延迟(微秒)']
                    omp_vs_ptd_data.append({
                        'nlist': ptd_nlist,
                        'nprobe': ptd_nprobe,
                        'omp_latency': omp_row.iloc[0]['延迟(微秒)'],
                        'ptd_latency': ptd_row['延迟(微秒)'],
                        'omp/ptd_ratio': ratio, # OpenMP/pthread 比率，>1表示pthread更快
                        'recall': ptd_row['召回率'] # 用pthread的召回率
                    })
            
            if omp_vs_ptd_data:
                omp_ptd_df = pd.DataFrame(omp_vs_ptd_data)
                
                # 创建图表比较OpenMP与pthread的延迟比率
                plt.figure(figsize=(12, 8))
                markers = ['o', 's', '^', 'D']
                colors = plt.cm.tab10(range(4))
                
                for i, (nlist, group) in enumerate(omp_ptd_df.groupby('nlist')):
                    plt.plot(group['nprobe'], group['omp/ptd_ratio'], 
                             marker=markers[i % len(markers)], 
                             color=colors[i % len(colors)],
                             label=f'nlist={nlist}', 
                             linewidth=2)
                
                plt.axhline(y=1.0, color='gray', linestyle='--', label='Equal Performance')
                plt.grid(alpha=0.3)
                plt.xlabel('nprobe')
                plt.ylabel('Performance Ratio (OpenMP Time / pthread Time)')
                plt.title('OpenMP vs pthread Performance Comparison')
                plt.legend()
                
                for _, row in omp_ptd_df.iterrows():
                    label = "pthread faster" if row['omp/ptd_ratio'] > 1.0 else "OpenMP faster"
                    plt.annotate(f"{row['omp/ptd_ratio']:.2f}", 
                                 (row['nprobe'], row['omp/ptd_ratio']),
                                 textcoords="offset points", 
                                 xytext=(0, 7), 
                                 ha='center')
                
                plt.tight_layout()
                plt.savefig('results/omp_vs_ptd.png', dpi=300)
                print("OpenMP vs pthread性能比较图已保存为 'results/omp_vs_ptd.png'")
                
                # 三种方法的综合性能对比散点图
                plt.figure(figsize=(14, 9))
                
                # 创建模式数据
                patterns = []
                for _, ptd_row in ivf_ptd_rows.iterrows():
                    nlist, nprobe = parse_params(ptd_row['方法'])
                    
                    pattern = {
                        'nlist': nlist,
                        'nprobe': nprobe,
                        'config': f'N={nlist},P={nprobe}',
                        'ptd_latency': ptd_row['延迟(微秒)'],
                        'ptd_recall': ptd_row['召回率']
                    }
                    
                    # 查找对应的OpenMP和串行版本
                    omp_name = f"IVF_OMP_N{nlist}_P{nprobe}"
                    serial_name = f"IVF_N{nlist}_P{nprobe}"
                    
                    omp_row = ivf_omp_rows[ivf_omp_rows['方法'] == omp_name]
                    serial_row = ivf_serial_rows[ivf_serial_rows['方法'] == serial_name]
                    
                    if not omp_row.empty:
                        pattern['omp_latency'] = omp_row.iloc[0]['延迟(微秒)']
                        pattern['omp_recall'] = omp_row.iloc[0]['召回率']
                    
                    if not serial_row.empty:
                        pattern['serial_latency'] = serial_row.iloc[0]['延迟(微秒)']
                        pattern['serial_recall'] = serial_row.iloc[0]['召回率']
                    
                    if ('omp_latency' in pattern) and ('serial_latency' in pattern):
                        patterns.append(pattern)
                
                if patterns:
                    patterns_df = pd.DataFrame(patterns)
                    
                    markers = ['o', 's', '^', 'D']
                    colors = {'Serial': 'blue', 'OpenMP': 'green', 'pthread': 'red'}
                    
                    # 按nlist分组
                    for i, (nlist, group) in enumerate(patterns_df.groupby('nlist')):
                        marker = markers[i % len(markers)]
                        
                        # 绘制串行版本
                        plt.scatter(group['serial_latency'], group['serial_recall'], 
                                  marker=marker, s=100, edgecolor='black', linewidth=1,
                                  color=colors['Serial'], 
                                  label=f'Serial (N={nlist})' if i == 0 else "")
                        
                        # 绘制OpenMP版本
                        plt.scatter(group['omp_latency'], group['omp_recall'], 
                                  marker=marker, s=100, edgecolor='black', linewidth=1,
                                  color=colors['OpenMP'], 
                                  label=f'OpenMP (N={nlist})' if i == 0 else "")
                        
                        # 绘制pthread版本
                        plt.scatter(group['ptd_latency'], group['ptd_recall'], 
                                  marker=marker, s=100, edgecolor='black', linewidth=1,
                                  color=colors['pthread'], 
                                  label=f'pthread (N={nlist})' if i == 0 else "")
                        
                        # 连接同一配置的三个版本
                        for _, row in group.iterrows():
                            xs = [row['serial_latency'], row['omp_latency'], row['ptd_latency']]
                            ys = [row['serial_recall'], row['omp_recall'], row['ptd_recall']]
                            plt.plot(xs, ys, '-', color='gray', alpha=0.5, zorder=0)
                            
                            # 在连线中点标注配置
                            plt.annotate(f"P={row['nprobe']}", 
                                         ((xs[0] + xs[1] + xs[2])/3, (ys[0] + ys[1] + ys[2])/3),
                                         textcoords="offset points", 
                                         xytext=(0, 5), 
                                         ha='center', 
                                         fontsize=8)
                    
                    # 添加图例
                    plt.xscale('log')
                    
                    # 创建颜色图例
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Serial'], markersize=10, label='Serial'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['OpenMP'], markersize=10, label='OpenMP'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['pthread'], markersize=10, label='pthread')
                    ]
                    
                    # 添加nlist图例
                    for i, nlist in enumerate(sorted(patterns_df['nlist'].unique())):
                        legend_elements.append(
                            Line2D([0], [0], marker=markers[i % len(markers)], color='w', 
                                   markerfacecolor='gray', markersize=10, label=f'nlist={nlist}')
                        )
                    
                    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
                    plt.grid(alpha=0.3)
                    plt.xlabel('Latency (μs)', fontsize=12)
                    plt.ylabel('Recall', fontsize=12)
                    plt.title('Three Implementations of IVF-Flat: Recall vs Latency Comparison', fontsize=14)
                    plt.tight_layout()
                    plt.savefig('results/ivf_triple_comparison.png', dpi=300)
                    print("三种IVF实现对比图已保存为 'results/ivf_triple_comparison.png'")
EOF
else
    echo "未检测到Python，跳过图表生成。如需生成图表，请安装Python及必要的包（matplotlib, pandas, numpy）"
    echo "安装后可以运行：python plot_results.py results/summary.csv"
fi

echo "所有测试完成，汇总结果保存在 results/summary.csv" 