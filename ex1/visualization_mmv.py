import matplotlib.pyplot as plt
import numpy as np

# 实验数据：矩阵尺寸以及对应的运行时间（单位：微秒）
sizes = np.array([128, 150, 256, 512, 1024, 2048])

naive_times = np.array([
    8028.2,      # 128x128
    12179.1,     # 150x150
    48499.3,     # 256x256
    279459,      # 512x512
    1.45637e+06, # 1024x1024
    1.36049e+07  # 2048x2048
])

naive_unroll_times = np.array([
    7569.4,      # 128x128
    10953.2,     # 150x150
    38396.2,     # 256x256
    268174,      # 512x512
    1.44914e+06, # 1024x1024
    1.39215e+07  # 2048x2048
])

cache_opt_times = np.array([
    7429.3,      # 128x128
    11000.5,     # 150x150
    27369.6,     # 256x256
    112442,      # 512x512
    453760,      # 1024x1024
    3.87694e+06  # 2048x2048
])

# 计算加速比
speedup_naive = naive_times / cache_opt_times
speedup_unroll = naive_unroll_times / cache_opt_times

#########################
# 图1：原始运行时间（对数坐标）
#########################
plt.figure(figsize=(8, 6))
plt.plot(sizes, naive_times, marker='o', linestyle='-', label='Naive Time')
plt.plot(sizes, naive_unroll_times, marker='s', linestyle='-', label='Naive Unroll Time')
plt.plot(sizes, cache_opt_times, marker='^', linestyle='-', label='Cache Optimized Time')
plt.xlabel('Matrix Size (n x n) [n]')
plt.ylabel('Time (μs)')
plt.title('Execution Time Comparison')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()

#########################
# 图2：加速比图（Naive / CacheOpt 和 Naive Unroll / CacheOpt）
#########################
plt.figure(figsize=(8, 6))
plt.plot(sizes, speedup_naive, marker='o', linestyle='-', label='Speedup (Naive / CacheOpt)')
plt.plot(sizes, speedup_unroll, marker='s', linestyle='-', label='Speedup (Naive Unroll / CacheOpt)')
plt.xlabel('Matrix Size (n x n) [n]')
plt.ylabel('Speedup Factor')
plt.title('Speedup of Cache Optimized Method')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.show()

#########################
# 图3：柱状图对比（针对每个规模展示三种算法的运行时间）
#########################
# 为了绘制柱状图，将每个规模作为一个分组，每组中有三根柱子
x = np.arange(len(sizes))  # 分组位置
width = 0.25  # 柱子的宽度

plt.figure(figsize=(10, 6))
plt.bar(x - width, naive_times, width=width, label='Naive Time')
plt.bar(x, naive_unroll_times, width=width, label='Naive Unroll Time')
plt.bar(x + width, cache_opt_times, width=width, label='Cache Optimized Time')

plt.xlabel('Matrix Size (n x n)')
plt.ylabel('Time (μs)')
plt.title('Execution Time Comparison per Matrix Size')
plt.xticks(x, sizes)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

