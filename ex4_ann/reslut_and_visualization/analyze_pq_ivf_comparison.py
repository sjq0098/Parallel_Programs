import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass

def clean_and_load_data():
    """清理和加载数据"""
    # 读取clean_results.csv，这个文件包含了两种算法的清理后数据
    df = pd.read_csv('clean_results.csv')
    
    print("=== 原始数据概览 ===")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    print(f"算法类型: {df['algorithm'].value_counts()}")
    
    # 分离两种算法
    df_pq_ivf = df[df['algorithm'] == 'PQ-IVF'].copy()
    df_ivf_pq = df[df['algorithm'] == 'IVF-PQ'].copy()
    
    print(f"\nPQ-IVF 数据点: {len(df_pq_ivf)}")
    print(f"IVF-PQ 数据点: {len(df_ivf_pq)}")
    
    return df, df_pq_ivf, df_ivf_pq

def calculate_performance_metrics(df_pq_ivf, df_ivf_pq):
    """计算性能指标"""
    print("\n=== PQ-IVF vs IVF-PQ 性能对比分析 ===\n")
    
    # 按参数配置进行对比
    configs = []
    for _, row in df_pq_ivf.iterrows():
        nlist, nprobe, m, ksub = row['nlist'], row['nprobe'], row['m'], row['ksub']
        
        # 查找对应的IVF-PQ配置
        ivf_pq_match = df_ivf_pq[
            (df_ivf_pq['nlist'] == nlist) & 
            (df_ivf_pq['nprobe'] == nprobe) & 
            (df_ivf_pq['m'] == m) & 
            (df_ivf_pq['ksub'] == ksub)
        ]
        
        if len(ivf_pq_match) > 0:
            pq_ivf = row
            ivf_pq = ivf_pq_match.iloc[0]
            
            config = {
                'nlist': nlist,
                'nprobe': nprobe,
                'm': m,
                'ksub': ksub,
                'pq_ivf_recall': pq_ivf['recall'],
                'ivf_pq_recall': ivf_pq['recall'],
                'pq_ivf_latency': pq_ivf['latency_us'],
                'ivf_pq_latency': ivf_pq['latency_us'],
                'pq_ivf_build_time': pq_ivf['build_time_ms'],
                'ivf_pq_build_time': ivf_pq['build_time_ms'],
            }
            
            # 计算改进指标
            config['recall_improvement'] = (ivf_pq['recall'] - pq_ivf['recall']) / pq_ivf['recall'] * 100
            config['latency_change'] = (ivf_pq['latency_us'] - pq_ivf['latency_us']) / pq_ivf['latency_us'] * 100
            config['build_time_change'] = (ivf_pq['build_time_ms'] - pq_ivf['build_time_ms']) / pq_ivf['build_time_ms'] * 100
            
            # 效率指标
            config['pq_ivf_efficiency'] = pq_ivf['recall'] / pq_ivf['latency_us'] * 1000
            config['ivf_pq_efficiency'] = ivf_pq['recall'] / ivf_pq['latency_us'] * 1000
            config['efficiency_improvement'] = (config['ivf_pq_efficiency'] - config['pq_ivf_efficiency']) / config['pq_ivf_efficiency'] * 100
            
            configs.append(config)
            
            print(f"配置 nlist={nlist}, nprobe={nprobe}, m={m}, ksub={ksub}:")
            print(f"  召回率: PQ-IVF={pq_ivf['recall']:.4f} vs IVF-PQ={ivf_pq['recall']:.4f} (改进{config['recall_improvement']:+.1f}%)")
            print(f"  延迟: PQ-IVF={pq_ivf['latency_us']:.0f}μs vs IVF-PQ={ivf_pq['latency_us']:.0f}μs (变化{config['latency_change']:+.1f}%)")
            print(f"  构建时间: PQ-IVF={pq_ivf['build_time_ms']:.0f}ms vs IVF-PQ={ivf_pq['build_time_ms']:.0f}ms (变化{config['build_time_change']:+.1f}%)")
            print(f"  效率: PQ-IVF={config['pq_ivf_efficiency']:.3f} vs IVF-PQ={config['ivf_pq_efficiency']:.3f} (改进{config['efficiency_improvement']:+.1f}%)")
            print()
    
    return pd.DataFrame(configs)

def create_comprehensive_comparison_plots(df, df_pq_ivf, df_ivf_pq, comparison_df):
    """创建综合对比图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PQ-IVF vs IVF-PQ Comprehensive Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 配置标签
    config_labels = []
    for _, row in comparison_df.iterrows():
        label = f"nlist={int(row['nlist'])}\nnprobe={int(row['nprobe'])}\nm={int(row['m'])}"
        config_labels.append(label)
    
    # 1. 召回率对比
    x_pos = np.arange(len(comparison_df))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, comparison_df['pq_ivf_recall'], width, 
                   label='PQ-IVF', alpha=0.8, color='#1f77b4')
    axes[0, 0].bar(x_pos + width/2, comparison_df['ivf_pq_recall'], width, 
                   label='IVF-PQ', alpha=0.8, color='#ff7f0e')
    
    axes[0, 0].set_xlabel('Configuration')
    axes[0, 0].set_ylabel('Recall')
    axes[0, 0].set_title('Recall Comparison')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(config_labels, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 延迟对比
    axes[0, 1].bar(x_pos - width/2, comparison_df['pq_ivf_latency'], width, 
                   label='PQ-IVF', alpha=0.8, color='#1f77b4')
    axes[0, 1].bar(x_pos + width/2, comparison_df['ivf_pq_latency'], width, 
                   label='IVF-PQ', alpha=0.8, color='#ff7f0e')
    
    axes[0, 1].set_xlabel('Configuration')
    axes[0, 1].set_ylabel('Latency (μs)')
    axes[0, 1].set_title('Query Latency Comparison')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(config_labels, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 构建时间对比
    axes[0, 2].bar(x_pos - width/2, comparison_df['pq_ivf_build_time'], width, 
                   label='PQ-IVF', alpha=0.8, color='#1f77b4')
    axes[0, 2].bar(x_pos + width/2, comparison_df['ivf_pq_build_time'], width, 
                   label='IVF-PQ', alpha=0.8, color='#ff7f0e')
    
    axes[0, 2].set_xlabel('Configuration')
    axes[0, 2].set_ylabel('Build Time (ms)')
    axes[0, 2].set_title('Index Build Time Comparison')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(config_labels, rotation=45, ha='right')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 召回率改进
    colors = ['red' if x < 0 else 'green' for x in comparison_df['recall_improvement']]
    bars = axes[1, 0].bar(x_pos, comparison_df['recall_improvement'], 
                         color=colors, alpha=0.7)
    
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_xlabel('Configuration')
    axes[1, 0].set_ylabel('Recall Improvement (%)')
    axes[1, 0].set_title('IVF-PQ vs PQ-IVF: Recall Improvement')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(config_labels, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                       f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # 5. 延迟变化
    colors = ['green' if x < 0 else 'red' for x in comparison_df['latency_change']]
    bars = axes[1, 1].bar(x_pos, comparison_df['latency_change'], 
                         color=colors, alpha=0.7)
    
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].set_xlabel('Configuration')
    axes[1, 1].set_ylabel('Latency Change (%)')
    axes[1, 1].set_title('IVF-PQ vs PQ-IVF: Latency Change')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(config_labels, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                       f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # 6. 效率对比散点图
    axes[1, 2].scatter(comparison_df['pq_ivf_efficiency'], comparison_df['ivf_pq_efficiency'], 
                      s=100, alpha=0.7, color='blue')
    
    # 添加对角线
    min_eff = min(comparison_df['pq_ivf_efficiency'].min(), comparison_df['ivf_pq_efficiency'].min())
    max_eff = max(comparison_df['pq_ivf_efficiency'].max(), comparison_df['ivf_pq_efficiency'].max())
    axes[1, 2].plot([min_eff, max_eff], [min_eff, max_eff], 'r--', alpha=0.7, label='Equal Performance')
    
    axes[1, 2].set_xlabel('PQ-IVF Efficiency (Recall/Latency × 1000)')
    axes[1, 2].set_ylabel('IVF-PQ Efficiency (Recall/Latency × 1000)')
    axes[1, 2].set_title('Algorithm Efficiency Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 为每个点添加配置标签
    for i, (x, y) in enumerate(zip(comparison_df['pq_ivf_efficiency'], comparison_df['ivf_pq_efficiency'])):
        axes[1, 2].annotate(f'C{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('pq_ivf_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_trade_offs(comparison_df):
    """分析权衡关系"""
    print("\n=== 权衡关系深度分析 ===\n")
    
    # 1. 召回率 vs 延迟权衡
    print("1. 召回率 vs 延迟权衡:")
    for i, row in comparison_df.iterrows():
        recall_gain = row['recall_improvement']
        latency_cost = row['latency_change']
        
        if recall_gain > 0 and latency_cost > 0:
            ratio = recall_gain / latency_cost
            print(f"  配置{i+1}: 召回率增加{recall_gain:.1f}%，延迟增加{latency_cost:.1f}%，收益比{ratio:.3f}")
        elif recall_gain > 0 and latency_cost < 0:
            print(f"  配置{i+1}: 召回率增加{recall_gain:.1f}%，延迟减少{abs(latency_cost):.1f}% (双重优势)")
        else:
            print(f"  配置{i+1}: 召回率变化{recall_gain:.1f}%，延迟变化{latency_cost:.1f}%")
    
    # 2. 构建成本分析
    print("\n2. 构建成本分析:")
    avg_build_increase = comparison_df['build_time_change'].mean()
    print(f"  平均构建时间增加: {avg_build_increase:.1f}%")
    
    for i, row in comparison_df.iterrows():
        build_cost = row['build_time_change']
        recall_gain = row['recall_improvement']
        print(f"  配置{i+1}: 构建时间增加{build_cost:.1f}%，换取召回率提升{recall_gain:.1f}%")
    
    # 3. 整体效率分析
    print("\n3. 整体效率分析:")
    avg_efficiency_improvement = comparison_df['efficiency_improvement'].mean()
    print(f"  平均效率改进: {avg_efficiency_improvement:.1f}%")
    
    better_configs = comparison_df[comparison_df['efficiency_improvement'] > 0]
    worse_configs = comparison_df[comparison_df['efficiency_improvement'] <= 0]
    
    print(f"  IVF-PQ表现更好的配置: {len(better_configs)}/{len(comparison_df)}")
    print(f"  PQ-IVF表现更好的配置: {len(worse_configs)}/{len(comparison_df)}")

def generate_parameter_analysis(comparison_df):
    """生成参数影响分析"""
    print("\n=== 参数影响分析 ===\n")
    
    # 按nlist分组分析
    print("nlist参数影响:")
    nlist_groups = comparison_df.groupby('nlist').agg({
        'recall_improvement': 'mean',
        'latency_change': 'mean',
        'efficiency_improvement': 'mean'
    })
    
    for nlist, group in nlist_groups.iterrows():
        print(f"  nlist={int(nlist)}: 召回率改进{group['recall_improvement']:.1f}%, "
              f"延迟变化{group['latency_change']:+.1f}%, 效率改进{group['efficiency_improvement']:.1f}%")
    
    # 按nprobe分组分析
    print("\nnprobe参数影响:")
    nprobe_groups = comparison_df.groupby('nprobe').agg({
        'recall_improvement': 'mean',
        'latency_change': 'mean',
        'efficiency_improvement': 'mean'
    })
    
    for nprobe, group in nprobe_groups.iterrows():
        print(f"  nprobe={int(nprobe)}: 召回率改进{group['recall_improvement']:.1f}%, "
              f"延迟变化{group['latency_change']:+.1f}%, 效率改进{group['efficiency_improvement']:.1f}%")
    
    # 按m分组分析
    print("\nPQ子向量数量(m)影响:")
    m_groups = comparison_df.groupby('m').agg({
        'recall_improvement': 'mean',
        'latency_change': 'mean',
        'efficiency_improvement': 'mean'
    })
    
    for m, group in m_groups.iterrows():
        print(f"  m={int(m)}: 召回率改进{group['recall_improvement']:.1f}%, "
              f"延迟变化{group['latency_change']:+.1f}%, 效率改进{group['efficiency_improvement']:.1f}%")

def create_summary_table(comparison_df):
    """创建总结表格"""
    print("\n=== 性能对比总结表 ===\n")
    
    print("配置\t\t召回率(PQ-IVF)\t召回率(IVF-PQ)\t召回率改进\t延迟(PQ-IVF)\t延迟(IVF-PQ)\t延迟变化\t效率改进")
    print("-" * 120)
    
    for i, row in comparison_df.iterrows():
        config = f"nlist={int(row['nlist'])},nprobe={int(row['nprobe'])},m={int(row['m'])}"
        print(f"{config:<15}\t{row['pq_ivf_recall']:.4f}\t\t{row['ivf_pq_recall']:.4f}\t\t"
              f"{row['recall_improvement']:+.1f}%\t\t{row['pq_ivf_latency']:.0f}μs\t\t"
              f"{row['ivf_pq_latency']:.0f}μs\t\t{row['latency_change']:+.1f}%\t\t{row['efficiency_improvement']:+.1f}%")

if __name__ == "__main__":
    try:
        print("开始分析PQ-IVF vs IVF-PQ性能对比...")
        
        # 加载和清理数据
        df, df_pq_ivf, df_ivf_pq = clean_and_load_data()
        
        # 计算性能指标
        comparison_df = calculate_performance_metrics(df_pq_ivf, df_ivf_pq)
        
        if len(comparison_df) > 0:
            # 生成可视化
            create_comprehensive_comparison_plots(df, df_pq_ivf, df_ivf_pq, comparison_df)
            
            # 分析权衡关系
            analyze_trade_offs(comparison_df)
            
            # 参数影响分析
            generate_parameter_analysis(comparison_df)
            
            # 创建总结表格
            create_summary_table(comparison_df)
            
            print("\n分析完成！生成的文件:")
            print("- pq_ivf_comparison_analysis.png: 综合对比图表")
        else:
            print("警告：没有找到匹配的配置进行对比分析")
            
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 