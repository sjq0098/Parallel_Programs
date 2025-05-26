import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据并处理
def extract_value(s):
    if isinstance(s, str) and '=' in s:
        return int(s.split('=')[1])
    return s

# 读取数据，使用更宽松的参数
df = pd.read_csv('results/ivfpq_results.csv', on_bad_lines='skip', engine='python')

# 提取nlist, m, nprobe的值
df['nlist'] = df['nlist'].apply(extract_value)
df['m'] = df['m'].apply(extract_value)
df['nprobe'] = df['nprobe'].apply(extract_value)

# 创建多个子图
plt.figure(figsize=(20, 15))

# 1. Recall vs nprobe for different m values (Serial)
plt.subplot(2, 2, 1)
serial_data = df[df['Method'] == 'IVF-PQ(Serial)']
nlist_value = 128  # 选择一个固定的nlist值
data = serial_data[serial_data['nlist'] == nlist_value]
for m in [8, 16, 32, 48]:
    m_data = data[data['m'] == m]
    plt.plot(m_data['nprobe'], m_data['Recall'], marker='o', label=f'm={m}')
plt.xlabel('nprobe值')
plt.ylabel('召回率 (Recall)')
plt.title(f'不同m值下nprobe对召回率的影响\n(串行版本, nlist={nlist_value})')
plt.legend()
plt.grid(True)

# 2. Latency vs nprobe for different m values (Serial)
plt.subplot(2, 2, 2)
for m in [8, 16, 32, 48]:
    m_data = data[data['m'] == m]
    plt.plot(m_data['nprobe'], m_data['Latency(us)'], marker='o', label=f'm={m}')
plt.xlabel('nprobe值')
plt.ylabel('延迟 (微秒)')
plt.title(f'不同m值下nprobe对延迟的影响\n(串行版本, nlist={nlist_value})')
plt.legend()
plt.grid(True)

# 3. Serial vs OpenMP comparison
plt.subplot(2, 2, 3)
serial_data = df[df['Method'] == 'IVF-PQ(Serial)']
openmp_data = df[df['Method'] == 'IVF-PQ(OpenMP)']

plt.scatter(serial_data['Latency(us)'], serial_data['Recall'], 
           alpha=0.5, label='串行版本')
plt.scatter(openmp_data['Latency(us)'], openmp_data['Recall'], 
           alpha=0.5, label='OpenMP版本')
plt.xlabel('延迟 (微秒)')
plt.ylabel('召回率 (Recall)')
plt.title('串行vs并行性能对比')
plt.legend()
plt.grid(True)

# 4. Heatmap of average Recall for different nlist and m combinations
plt.subplot(2, 2, 4)
pivot_data = df[df['Method'] == 'IVF-PQ(Serial)'].pivot_table(
    values='Recall', 
    index='nlist',
    columns='m',
    aggfunc='mean'
)
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('不同nlist和m组合的平均召回率\n(串行版本)')
plt.xlabel('m值')
plt.ylabel('nlist值')

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('results/performance_analysis.png', dpi=300, bbox_inches='tight')
print("图表已保存为 'results/performance_analysis.png'") 