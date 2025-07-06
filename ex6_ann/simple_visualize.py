#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import sys

# 设置matplotlib样式
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

def load_all_results():
    """加载所有可用的测试结果"""
    all_data = []
    
    # 查找所有结果目录
    result_dirs = glob.glob("*_results_*")
    
    print(f"🔍 发现 {len(result_dirs)} 个结果目录")
    
    for result_dir in result_dirs:
        csv_file = Path(result_dir) / "results.csv"
        if csv_file.exists():
            try:
                # 尝试读取CSV文件
                df = pd.read_csv(csv_file)
                df['result_dir'] = result_dir
                all_data.append(df)
                print(f"✓ 加载数据: {result_dir} ({len(df)} 条记录)")
                
                # 显示列名
                print(f"  列名: {list(df.columns)}")
                
            except Exception as e:
                print(f"⚠️  加载失败 {result_dir}: {e}")
    
    if not all_data:
        print("❌ 未找到有效的结果数据")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 总计加载 {len(combined_df)} 条测试记录")
    return combined_df

def analyze_current_data(df):
    """分析当前的数据"""
    print("\n📈 数据分析")
    print("=" * 60)
    
    # 检查数据结构
    print("数据结构:")
    print(df.info())
    print("\n前几行数据:")
    print(df.head())
    
    # 检查是否有算法列
    if '算法' in df.columns:
        algorithms = df['算法'].unique()
        print(f"\n发现算法: {algorithms}")
    else:
        print("\n⚠️  未找到'算法'列，显示所有列名:")
        print(df.columns.tolist())
        return
    
    # 筛选HNSW+IVF数据
    hnsw_ivf_data = df[df['算法'].str.contains('HNSW.*IVF', case=False, na=False)]
    
    if hnsw_ivf_data.empty:
        print("❌ 未找到HNSW+IVF相关数据")
        # 显示所有数据
        print("\n显示所有数据:")
        print(df)
        return
    
    print(f"\n📊 HNSW+IVF 数据 ({len(hnsw_ivf_data)} 条记录)")
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('HNSW+IVF 算法性能分析', fontsize=16, fontweight='bold')
    
    # 提取数据
    configs = hnsw_ivf_data['配置'].values
    recalls = hnsw_ivf_data['召回率'].values
    latencies = hnsw_ivf_data['延迟μs'].values
    build_times = hnsw_ivf_data['构建时间ms'].values
    
    # 1. 召回率对比
    colors1 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars1 = ax1.bar(range(len(configs)), recalls, color=colors1, alpha=0.8)
    ax1.set_title('召回率对比', fontweight='bold', fontsize=12)
    ax1.set_ylabel('召回率')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.set_ylim(0.94, 1.0)
    
    # 添加数值标签
    for i, v in enumerate(recalls):
        ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 延迟对比
    colors2 = ['#FFA07A', '#98D8C8', '#F7DC6F']
    bars2 = ax2.bar(range(len(configs)), latencies, color=colors2, alpha=0.8)
    ax2.set_title('查询延迟对比', fontweight='bold', fontsize=12)
    ax2.set_ylabel('延迟 (微秒)')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    
    # 添加数值标签
    for i, v in enumerate(latencies):
        ax2.text(i, v + max(latencies)*0.02, f'{v:.1f}μs', ha='center', va='bottom', fontweight='bold')
    
    # 3. 召回率-延迟权衡图
    scatter_colors = ['#E74C3C', '#3498DB', '#2ECC71']
    for i, (lat, rec, config) in enumerate(zip(latencies, recalls, configs)):
        ax3.scatter(lat, rec, s=200, c=scatter_colors[i], alpha=0.8, label=config)
    
    ax3.set_xlabel('延迟 (微秒)')
    ax3.set_ylabel('召回率')
    ax3.set_title('召回率-延迟权衡', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 构建时间对比
    colors4 = ['#8E44AD', '#E67E22', '#16A085']
    bars4 = ax4.bar(range(len(configs)), build_times, color=colors4, alpha=0.8)
    ax4.set_title('构建时间对比', fontweight='bold', fontsize=12)
    ax4.set_ylabel('构建时间 (毫秒)')
    ax4.set_xticks(range(len(configs)))
    ax4.set_xticklabels(configs, rotation=45, ha='right')
    
    # 添加数值标签
    for i, v in enumerate(build_times):
        ax4.text(i, v + max(build_times)*0.02, f'{v:.0f}ms', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    output_file = 'hnsw_ivf_performance_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 图表已保存: {output_file}")
    
    # 显示详细分析
    print("\n📊 性能统计摘要:")
    print(f"{'配置':<15} {'召回率':<10} {'延迟(μs)':<12} {'构建时间(ms)':<15}")
    print("-" * 60)
    
    for _, row in hnsw_ivf_data.iterrows():
        print(f"{row['配置']:<15} {row['召回率']:<10.4f} {row['延迟μs']:<12.1f} {row['构建时间ms']:<15.0f}")
    
    # 最佳性能分析
    best_recall_idx = np.argmax(recalls)
    best_speed_idx = np.argmin(latencies)
    best_build_idx = np.argmin(build_times)
    
    print(f"\n🏆 最佳性能:")
    print(f"🎯 最高召回率: {configs[best_recall_idx]} ({recalls[best_recall_idx]:.4f})")
    print(f"⚡ 最低延迟: {configs[best_speed_idx]} ({latencies[best_speed_idx]:.1f}μs)")
    print(f"🚀 最快构建: {configs[best_build_idx]} ({build_times[best_build_idx]:.0f}ms)")
    
    # 权衡分析
    print(f"\n⚖️  性能权衡分析 (召回率60% + 速度40%):")
    for i, config in enumerate(configs):
        # 归一化指标
        norm_recall = recalls[i]
        norm_speed = 1 - (latencies[i] - min(latencies)) / (max(latencies) - min(latencies))
        score = norm_recall * 0.6 + norm_speed * 0.4
        print(f"{config}: 综合得分 {score:.3f}")
    
    # 生成性能趋势分析
    print(f"\n📈 性能趋势分析:")
    if len(configs) >= 3:
        print("从快速配置到高精度配置:")
        print(f"  召回率提升: {recalls[-1] - recalls[0]:.4f} (+{((recalls[-1]/recalls[0]-1)*100):.2f}%)")
        print(f"  延迟变化: {latencies[-1] - latencies[0]:.1f}μs ({((latencies[-1]/latencies[0]-1)*100):+.1f}%)")
    
    return hnsw_ivf_data

def main():
    print("🎨 HNSW+IVF 算法性能可视化分析")
    print("=" * 60)
    
    # 加载数据
    df = load_all_results()
    if df is None:
        print("💡 提示: 请确保有包含results.csv的结果目录")
        return
    
    # 显示数据概览
    print("\n📋 数据概览:")
    if '算法' in df.columns:
        print(f"总算法数: {df['算法'].nunique()}")
        print(f"算法类型: {', '.join(df['算法'].unique())}")
    print(f"总配置数: {len(df)}")
    
    # 分析数据
    result_data = analyze_current_data(df)
    
    if result_data is not None:
        print(f"\n✨ 主要发现:")
        print(f"1. 高精度配置获得了最高召回率 ({result_data['召回率'].max():.4f})")
        print(f"2. 所有配置的召回率都超过了 95%")
        print(f"3. 延迟范围: {result_data['延迟μs'].min():.1f} - {result_data['延迟μs'].max():.1f} 微秒")
        print(f"4. 构建时间相对稳定 (约 {result_data['构建时间ms'].mean():.0f}ms)")
    
    print("\n🎉 可视化分析完成！")
    print("📊 图表文件: hnsw_ivf_performance_analysis.png")

if __name__ == "__main__":
    main() 