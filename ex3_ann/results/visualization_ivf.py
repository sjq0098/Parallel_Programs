#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def ensure_plot_dir():
    """确保plots目录存在"""
    if not os.path.exists('plots'):
        os.makedirs('plots')

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    # 提取nlist和nprobe的数值
    df['nlist'] = df['nlist'].str.split('=', expand=True)[1].astype(int)
    df['nprobe'] = df['nprobe'].str.split('=', expand=True)[1].astype(int)
    return df

def plot_heatmap_comparison(df, metric, outfn):
    """绘制不同方法在不同参数下的热力图对比"""
    plt.figure(figsize=(15, 10))
    
    # 为每个方法创建子图
    methods = df['Method'].unique()
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    
    for idx, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        pivot = method_data.pivot(index='nlist', columns='nprobe', values=metric)
        
        sns.heatmap(pivot, annot=True, fmt='.3f' if metric=='Recall' else '.0f',
                    cmap='viridis' if metric=='Recall' else 'rocket',
                    ax=axes[idx], cbar=True)
        axes[idx].set_title(f'{method}\n{metric}')
    
    plt.tight_layout()
    plt.savefig(os.path.join('plots', outfn), dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_impact(df, outfn):
    """分析nlist和nprobe对性能的影响"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # nlist对latency的影响
    sns.boxplot(data=df, x='nlist', y='Latency(us)', hue='Method', ax=ax1)
    ax1.set_title('Impact of nlist on Latency')
    ax1.set_xlabel('Number of Lists')
    ax1.set_ylabel('Latency (μs)')
    
    # nprobe对recall的影响
    sns.boxplot(data=df, x='nprobe', y='Recall', hue='Method', ax=ax2)
    ax2.set_title('Impact of nprobe on Recall')
    ax2.set_xlabel('Number of Probes')
    ax2.set_ylabel('Recall')
    
    plt.tight_layout()
    plt.savefig(os.path.join('plots', outfn), dpi=300, bbox_inches='tight')
    plt.close()

def plot_tradeoff_analysis(df, outfn):
    """分析recall和latency的权衡关系，只画连线不加标注"""
    plt.figure(figsize=(10, 8))
    
    # 为每个方法绘制连线图
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        
        # 按 latency 排序，以确保线条方向一致
        method_data = method_data.sort_values(by='Latency(us)')
        
        # 绘制连线和点
        plt.plot(method_data['Latency(us)'], method_data['Recall'],
                 label=method, marker='o', linewidth=2, alpha=0.8)
    
    plt.xlabel('Latency (μs)')
    plt.ylabel('Recall')
    plt.title('Recall-Latency Tradeoff (Line Only, per Method)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('plots', outfn), dpi=300, bbox_inches='tight')
    plt.close()

def plot_method_comparison(df, outfn):
    """比较不同方法在相同参数下的性能"""
    # 选择具有代表性的参数组合
    representative_params = df[
        (df['nlist'].isin([64, 256])) & 
        (df['nprobe'].isin([4, 16, 32]))
    ]
    
    plt.figure(figsize=(12, 6))
    
    # 创建分组柱状图
    sns.barplot(data=representative_params,
                x='Method',
                y='Latency(us)',
                hue='nlist',
                palette='viridis')
    
    plt.title('Method Comparison with Different Parameters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', outfn), dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_sensitivity(df, outfn):
    """分析参数敏感性"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # nlist对latency的敏感性
    sns.lineplot(data=df, x='nlist', y='Latency(us)',
                hue='Method', marker='o', ax=axes[0,0])
    axes[0,0].set_title('Latency Sensitivity to nlist')
    
    # nprobe对latency的敏感性
    sns.lineplot(data=df, x='nprobe', y='Latency(us)',
                hue='Method', marker='o', ax=axes[0,1])
    axes[0,1].set_title('Latency Sensitivity to nprobe')
    
    # nlist对recall的敏感性
    sns.lineplot(data=df, x='nlist', y='Recall',
                hue='Method', marker='o', ax=axes[1,0])
    axes[1,0].set_title('Recall Sensitivity to nlist')
    
    # nprobe对recall的敏感性
    sns.lineplot(data=df, x='nprobe', y='Recall',
                hue='Method', marker='o', ax=axes[1,1])
    axes[1,1].set_title('Recall Sensitivity to nprobe')
    
    plt.tight_layout()
    plt.savefig(os.path.join('plots', outfn), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # 确保plots目录存在
    ensure_plot_dir()
    
    # 设置绘图风格
    sns.set_palette("husl")
    
    # 加载数据
    df = load_and_preprocess_data('results/ivf_results.csv')
    
    # 生成各种可视化
    plot_heatmap_comparison(df, 'Recall', 'recall_heatmap.png')
    plot_heatmap_comparison(df, 'Latency(us)', 'latency_heatmap.png')
    plot_parameter_impact(df, 'parameter_impact.png')
    plot_tradeoff_analysis(df, 'recall_latency_tradeoff.png')
    plot_method_comparison(df, 'method_comparison.png')
    plot_parameter_sensitivity(df, 'parameter_sensitivity.png')
    
    print("Generated visualization files in 'plots' directory:")
    print("1. recall_heatmap.png - 不同方法的Recall热力图")
    print("2. latency_heatmap.png - 不同方法的Latency热力图")
    print("3. parameter_impact.png - 参数对性能的影响")
    print("4. recall_latency_tradeoff.png - Recall和Latency的权衡分析")
    print("5. method_comparison.png - 不同方法的性能对比")
    print("6. parameter_sensitivity.png - 参数敏感性分析")
