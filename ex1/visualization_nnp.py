import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data="""
Size: 1024 elements Average copytime:0.26
Navie result: 50408, Average Time: 3.28 us
Uroll result: 50408, Average Time: 0.92 us
Pair  result: 50408, Average Time: 2.66 us
----------------------------------------
Size: 1035 elements Average copytime:0.24
Navie result: 50633, Average Time: 2.36 us
Uroll result: 50633, Average Time: 0.88 us
Pair  result: 50633, Average Time: 2.68 us
----------------------------------------
Size: 2048 elements Average copytime:0.88
Navie result: 100627, Average Time: 36.52 us
Uroll result: 100627, Average Time: 3.96 us
Pair  result: 100627, Average Time: 13.5 us
----------------------------------------
Size: 4096 elements Average copytime:1.46
Navie result: 201761, Average Time: 18.72 us
Uroll result: 201761, Average Time: 8.04 us
Pair  result: 201761, Average Time: 23.22 us
----------------------------------------
Size: 8192 elements Average copytime:2.62
Navie result: 403178, Average Time: 36.04 us
Uroll result: 403178, Average Time: 16.16 us
Pair  result: 403178, Average Time: 46.32 us
----------------------------------------
Size: 16384 elements Average copytime:5.22
Navie result: 812693, Average Time: 68.62 us
Uroll result: 812693, Average Time: 32.42 us
Pair  result: 812693, Average Time: 93.2 us
----------------------------------------
Size: 32768 elements Average copytime:11.32
Navie result: 1.62794e+06, Average Time: 132.74 us
Uroll result: 1.62794e+06, Average Time: 64.7 us
Pair  result: 1.62794e+06, Average Time: 175.08 us
----------------------------------------
Size: 65536 elements Average copytime:35.16
Navie result: 3.23798e+06, Average Time: 198.94 us
Uroll result: 3.23798e+06, Average Time: 115.18 us
Pair  result: 3.23798e+06, Average Time: 246.74 us
----------------------------------------
Size: 131072 elements Average copytime:196.3
Navie result: 6.47973e+06, Average Time: 436.82 us
Uroll result: 6.47973e+06, Average Time: 186.28 us
Pair  result: 6.47973e+06, Average Time: 465.08 us
----------------------------------------
Size: 231311 elements Average copytime:358.04
Navie result: 1.14354e+07, Average Time: 526.82 us
Uroll result: 1.14354e+07, Average Time: 334.44 us
Pair  result: 1.14354e+07, Average Time: 736.18 us
----------------------------------------
Size: 262144 elements Average copytime:387.22
Navie result: 1.29807e+07, Average Time: 495.04 us
Uroll result: 1.29807e+07, Average Time: 377.52 us
Pair  result: 1.29807e+07, Average Time: 773.1 us
----------------------------------------
Size: 524288 elements Average copytime:798.12
Navie result: 2.59541e+07, Average Time: 1080.8 us
Uroll result: 2.59541e+07, Average Time: 910.6 us
Pair  result: 2.59541e+07, Average Time: 1584.76 us
----------------------------------------
Size: 2313119 elements Average copytime:4108.68
Navie result: 1.1451e+08, Average Time: 5095.86 us
Uroll result: 1.1451e+08, Average Time: 2824.44 us
Pair  result: 1.1451e+08, Average Time: 7422.22 us
----------------------------------------
"""

# 修改后的正则表达式模式
pattern = r"Size: (\d+) elements Average copytime:([\d.]+)\nNavie result: \d+, Average Time: ([\d.-]+) us.*?Uroll result: \d+, Average Time: ([\d.-]+) us.*?Pair  result: \d+, Average Time: ([\d.-]+) us"

blocks = re.findall(pattern, data, re.DOTALL)

# 数据解析
sizes, copytimes, naive, uroll, pair = [], [], [], [], []
for block in blocks:
    size = int(block[0])
    copytime = float(block[1])
    naive_time = float(block[2])
    uroll_time = float(block[3])
    pair_time = float(block[4])
    
    # 过滤负值数据点
    if naive_time > 0 and uroll_time > 0 and pair_time > 0:
        sizes.append(size)
        copytimes.append(copytime)
        naive.append(naive_time)
        uroll.append(uroll_time)
        pair.append(pair_time)

# 创建DataFrame
df = pd.DataFrame({
    'Size': sizes,
    'CopyTime': copytimes,
    'Naive': naive,
    'Uroll': uroll,
    'Pair': pair
}).sort_values('Size')

# 图表1: 综合时间分析（含拷贝时间）
plt.figure(figsize=(12, 7))
plt.loglog(df['Size'], df['CopyTime'], 'D-', markersize=8, label='Copy Time', linewidth=2)
plt.loglog(df['Size'], df['Naive'], 'o-', label='Naive')
plt.loglog(df['Size'], df['Uroll'], 's--', label='Uroll')
plt.loglog(df['Size'], df['Pair'], '^-.', label='Pair')
plt.xlabel('Input Size (log scale)', fontsize=12)
plt.ylabel('Time (µs, log scale)', fontsize=12)
plt.title('Comprehensive Time Analysis with Copy Time', fontsize=14)
plt.grid(True, which='both', linestyle=':', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('time_analysis_with_copy.png', dpi=300)
plt.close()

# 图表2: 拷贝时间占比分析
df['Total_Time'] = df['CopyTime'] + df['Uroll']
df['Copy_Ratio'] = df['CopyTime'] / df['Total_Time'] * 100

plt.figure(figsize=(10, 6))
plt.semilogx(df['Size'], df['Copy_Ratio'], 'mo-', markersize=8)
plt.xlabel('Input Size (log scale)', fontsize=12)
plt.ylabel('Copy Time Ratio (%)', fontsize=12)
plt.title('Memory Copy Time Proportion in Optimal Algorithm', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('copy_ratio_analysis.png', dpi=300)
plt.close()

# 图表3: 内存带宽分析
def compute_bandwidth(size, time):
    bytes_transferred = size * 8  # 假设double类型
    return bytes_transferred / (time * 1e-6) / (1024**3)  # GB/s

df['Copy_BW'] = df.apply(lambda x: compute_bandwidth(x['Size'], x['CopyTime']), axis=1)
df['Uroll_BW'] = df.apply(lambda x: compute_bandwidth(x['Size'], x['Uroll']), axis=1)

plt.figure(figsize=(12, 6))
plt.plot(df['Size'], df['Copy_BW'], 'D-', label='Copy Bandwidth')
plt.plot(df['Size'], df['Uroll_BW'], 's--', label='Uroll Bandwidth')
plt.xscale('log')
plt.xlabel('Input Size (log scale)', fontsize=12)
plt.ylabel('Effective Bandwidth (GB/s)', fontsize=12)
plt.title('Memory Subsystem Performance Analysis', fontsize=14)
plt.grid(True, which='both', linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('bandwidth_analysis.png', dpi=300)
plt.close()