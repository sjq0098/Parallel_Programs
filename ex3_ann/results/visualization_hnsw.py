#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# —— 1. 加载并预处理 —— #
def load_and_preprocess(path):
    df = pd.read_csv(path)
    # 把 HNSW 列里的 "M=16" 等格式提取数字，其它直接转 int
    def extract(col):
        return (df[col]
                .astype(str)
                .str.replace(r'.*=', '', regex=True)
                .astype(int))
    df['M']   = extract('M')
    df['efC'] = extract('efC')
    df['efS'] = extract('efS')
    return df

# —— 2. 绘制大热力图 —— #
def plot_big_heatmap(df, metric, outfn):
    """
    每个 Method 一个子图，用 pivot_table + FacetGrid
    """
    fmt  = '.3f' if metric=='Recall' else '.0f'
    cmap = 'viridis' if metric=='Recall' else 'rocket'

    g = sns.FacetGrid(df, col='Method', col_wrap=3,
                      sharex=True, sharey=True, despine=False)
    def heatmap(data, **kwargs):
        pt = data.pivot_table(index='efC', 
                              columns='efS', 
                              values=metric,
                              aggfunc='mean')
        ax = plt.gca()
        sns.heatmap(pt, annot=True, fmt=fmt, cmap=cmap,
                    cbar=False, ax=ax, **kwargs)
        ax.set_xlabel('efS')
        ax.set_ylabel('efC')

    g.map_dataframe(heatmap)
    g.set_titles("{col_name}")
    plt.subplots_adjust(top=0.88, left=0.05, right=0.9, bottom=0.05)
    g.fig.suptitle(f"{metric} Heatmaps by Method", fontsize=16)

    # 公共 colorbar
    norm = plt.Normalize(df[metric].min(), df[metric].max())
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = g.fig.add_axes([0.92, 0.30, 0.02, 0.4])
    g.fig.colorbar(sm, cax=cbar_ax, label=metric)

    g.savefig(outfn, dpi=200, bbox_inches='tight')
    plt.close()

# —— 3. 绘制折线对比图 —— #
def plot_line_vs_efS(df, metric, outfn):
    """
    横轴 efS，纵轴 metric，
    不同 Method 用不同颜色，按 M 分面
    """
    g = sns.FacetGrid(df, col='M', col_wrap=4,
                      sharex=True, sharey=False, height=4)
    g.map_dataframe(sns.lineplot, x='efS', y=metric,
                    hue='Method', marker='o')
    g.add_legend(title='Method')
    g.set_axis_labels("efS", metric)
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle(f"{metric} vs efS (grouped by M)", fontsize=16)
    g.savefig(outfn, dpi=200, bbox_inches='tight')
    plt.close()

# —— 主流程 —— #
if __name__ == '__main__':
    sns.set_style("whitegrid")
    # 确保输出目录存在
    os.makedirs('plots/hnsw', exist_ok=True)

    df = load_and_preprocess('results/hnsw_results.csv')  # 或 'results/hnsw_results.csv'
    df = df[df['M'] > 0].copy()

    # 生成热力图
    plot_big_heatmap(df, 'Recall',      'plots/hnsw/recall_heatmap.png')
    plot_big_heatmap(df, 'Latency(us)', 'plots/hnsw/latency_heatmap.png')

    # 生成折线图
    plot_line_vs_efS(df, 'Recall',      'plots/hnsw/recall_vs_efS.png')
    plot_line_vs_efS(df, 'Latency(us)', 'plots/hnsw/latency_vs_efS.png')

    print("已生成四张大图：")
    print(" - plots/hnsw/recall_heatmap.png")
    print(" - plots/hnsw/latency_heatmap.png")
    print(" - plots/hnsw/recall_vs_efS.png")
    print(" - plots/hnsw/latency_vs_efS.png")
