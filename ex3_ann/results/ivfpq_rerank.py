import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import griddata
from pandas.plotting import parallel_coordinates
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ------------------------
# 全局美化配置
# ------------------------
# 1. 统一字体与字号
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Source Sans Pro', 'Arial', 'Liberation Sans']
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['legend.fontsize'] = 11
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10

# 2. 统一调色板（三色循环）
palette = {
    'IVF-PQ(Serial)+Rerank':    '#2E5AAC',  # 深蓝
    'IVF-PQ(OMP)+Rerank':       '#E07A5F',  # 橙红
    'IVF-PQ(MultiThread)+Rerank': '#3D84A8', # 青蓝
}
rcParams['axes.prop_cycle'] = plt.cycler('color', list(palette.values()))

# 3. 背景与网格
rcParams['axes.facecolor'] = '#F7F7F7'
rcParams['figure.facecolor'] = '#FFFFFF'
rcParams['grid.color'] = '#DDDDDD'
rcParams['grid.linestyle'] = '--'
rcParams['grid.linewidth'] = 0.5

# 4. 边框与刻度
rcParams['axes.edgecolor'] = '#333333'
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

# ------------------------
# 数据载入 & 清洗
# ------------------------
df = pd.read_csv('results/ivfpq_rerank.csv')
for col in ['nlist','nprobe','m','L']:
    df[col] = df[col].str.extract(r'(\d+)').astype(int)
df.rename(columns={'Latency(us)':'latency','Recall':'recall','Method':'method'}, inplace=True)
low, high = df['latency'].quantile([0.01,0.99])
df = df[(df['latency']>=low)&(df['latency']<=high)]

methods = df['method'].unique()
fixed_m, fixed_L = 16, 200  # 可按需修改

# ------------------------
# 1. 并排热图（改进版）
# ------------------------
fig, axes = plt.subplots(1, len(methods), figsize=(4*len(methods), 4), sharey=True)
for ax, mtd in zip(axes, methods):
    sub = df[(df['method']==mtd)&(df['m']==fixed_m)&(df['L']==fixed_L)]
    pivot = sub.pivot_table(index='nlist', columns='nprobe', values='recall', aggfunc='mean')
    im = ax.imshow(pivot, origin='lower', aspect='auto', cmap='Reds')
    ax.set_title(mtd, pad=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)
    if ax is axes[0]:
        ax.set_ylabel('nlist')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

    # colorbar 放右侧，不遮挡
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im, cax=cax, label='Mean Recall')

fig.suptitle(f'Recall Heatmaps by Method (m={fixed_m}, L={fixed_L})', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()
# ------------------------
# 2. 性能包络线
# ------------------------
plt.figure(figsize=(10,6))
for mtd in methods:
    sub = df[df['method']==mtd].sort_values('latency')
    sub['best_recall'] = sub['recall'].cummax()
    plt.plot(sub['latency'], sub['best_recall'], label=mtd, linewidth=2)
    # 标注最高点
    idx = sub['best_recall'].idxmax()
    plt.scatter(sub.loc[idx,'latency'], sub.loc[idx,'best_recall'], s=50)
    plt.annotate(f"{sub.loc[idx,'best_recall']:.2f}",
                 (sub.loc[idx,'latency'], sub.loc[idx,'best_recall']),
                 textcoords="offset points", xytext=(0,8), ha='center')
plt.xscale('log')
plt.xlabel('Latency (μs)')
plt.ylabel('Best Achievable Recall')
plt.title('Performance Envelope: Best Recall vs Latency')
plt.legend(frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------
# 3. 采样散点气泡图
# ------------------------
plt.figure(figsize=(10,6))
for mtd, marker in zip(methods, ['o','s','^']):
    sub = df[df['method']==mtd].sample(frac=0.3, random_state=1)
    sizes = (sub['recall'] - sub['recall'].min()) / (sub['recall'].max() - sub['recall'].min()) * 200 + 20
    plt.scatter(sub['latency'], sub['recall'],
                s=sizes, alpha=0.6, marker=marker, label=mtd)
plt.xscale('log')
plt.xlabel('Latency (μs)')
plt.ylabel('Recall')
plt.title('Bubble Chart: latency vs recall (size~recall)')
plt.legend(frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------
# 4. Recall→Latency 曲线
# ------------------------
targets = np.linspace(0.6, 0.92, 8)
plt.figure(figsize=(10,6))
for mtd in methods:
    mins = []
    sub = df[df['method']==mtd]
    for t in targets:
        sel = sub[sub['recall']>=t]
        mins.append(sel['latency'].min() if not sel.empty else np.nan)
    plt.plot(targets, mins, marker='o', linewidth=2, label=mtd)
plt.yscale('log')
plt.xlabel('Recall Target')
plt.ylabel('Min Latency')
plt.title('Recall→Latency Curve by Method')
plt.legend(frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------
# 5. 延迟分布盒须图
# ------------------------
plt.figure(figsize=(8,6))
data = [df[df['method']==mtd]['latency'] for mtd in methods]
plt.boxplot(data, labels=methods, showfliers=False, patch_artist=True,
            boxprops=dict(facecolor='#EDEDED', edgecolor='#555'),
            medianprops=dict(color='#333'))
plt.yscale('log')
plt.ylabel('Latency (μs)')
plt.title('Latency Distribution by Method')
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------
# 6. 等高线对比图（改进版）
# ------------------------
fig, axes = plt.subplots(1, len(methods), figsize=(4*len(methods), 4))
for ax, mtd in zip(axes, methods):
    sub = df[(df['method']==mtd)&(df['m']==fixed_m)&(df['L']==fixed_L)]
    xi = np.linspace(sub['nlist'].min(), sub['nlist'].max(), 100)
    yi = np.linspace(sub['nprobe'].min(), sub['nprobe'].max(), 100)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata(sub[['nlist','nprobe']], sub['recall'], (XI, YI), method='cubic')

    cs = ax.contourf(XI, YI, ZI, levels=12, cmap='Blues')
    ax.set_title(mtd, pad=6)
    ax.set_xlabel('nlist')
    if ax is axes[0]:
        ax.set_ylabel('nprobe')

    # colorbar 放右侧
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(cs, cax=cax, label='Recall')

fig.suptitle(f'Contour Plots (m={fixed_m}, L={fixed_L})', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()