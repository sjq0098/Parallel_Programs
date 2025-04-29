#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import sys
import platform

# Detect operating system
os_system = platform.system()
print(f"Current OS: {os_system}")

# Use simple ASCII characters for output
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']

# Set plotting style
sns.set_style("whitegrid")

# Create output directory
output_dir = 'results/visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Hardcoded data as fallback
def get_hardcoded_data():
    # English algorithm names
    data = [
        {"algorithm": "Scalar Brute Force", "recall": 0.99995, "latency": 9.30317},
        {"algorithm": "SSE Optimized Brute Force", "recall": 0.99995, "latency": 1.99643},
        {"algorithm": "AVX Optimized Brute Force", "recall": 0.99995, "latency": 1.91721},
        {"algorithm": "Scalar Quantization (SQ)", "recall": 0.987752, "latency": 1.70464},
        {"algorithm": "Product Quantization (PQ) M=4", "recall": 0.10945, "latency": 0.515718},
        {"algorithm": "Product Quantization (PQ) M=8", "recall": 0.238451, "latency": 0.60371},
        {"algorithm": "Product Quantization (PQ) M=16", "recall": 0.434, "latency": 2.6585},
        {"algorithm": "Product Quantization (PQ) M=32", "recall": 0.685149, "latency": 4.4887},
        {"algorithm": "Optimized PQ (OPQ) M=4", "recall": 0.112401, "latency": 0.4069},
        {"algorithm": "Optimized PQ (OPQ) M=8", "recall": 0.237151, "latency": 0.3850},
        {"algorithm": "Optimized PQ (OPQ) M=16", "recall": 0.4303, "latency": 2.6046},
        {"algorithm": "Optimized PQ (OPQ) M=32", "recall": 0.692049, "latency": 5.9351},
        {"algorithm": "Hybrid Search (PQ16+Rerank)", "recall": 0.926455, "latency": 1.023},
        {"algorithm": "Hybrid Search (PQ32+Rerank)", "recall": 0.99825, "latency": 1.814},
        {"algorithm": "Hybrid Search (OPQ16+Rerank)", "recall": 0.926906, "latency": 1.055},
        {"algorithm": "Hybrid Search (OPQ32+Rerank)", "recall": 0.99825, "latency": 1.90}
    ]
    return pd.DataFrame(data)

# Read data from result files
def read_results_from_files():
    results = []
    
    # Dictionary to map Chinese algorithm names to English
    algo_name_map = {
        "暴力搜索": "Brute Force",
        "标量暴力搜索": "Scalar Brute Force",
        "标量Brute Force": "Scalar Brute Force",
        "SSE优化暴力搜索": "SSE Optimized Brute Force",
        "SSE优化Brute Force": "SSE Optimized Brute Force",
        "AVX优化暴力搜索": "AVX Optimized Brute Force",
        "AVX优化Brute Force": "AVX Optimized Brute Force",
        "标量量化(SQ)": "Scalar Quantization (SQ)",
        "乘积量化(PQ)": "Product Quantization (PQ)",
        "优化乘积量化(OPQ)": "Optimized PQ (OPQ)",
        "混合搜索": "Hybrid Search",
        "精确重排序": "Rerank",
        "混合搜索(PQ16+精确重排序)": "Hybrid Search (PQ16+Rerank)",
        "混合搜索(PQ32+精确重排序)": "Hybrid Search (PQ32+Rerank)",
        "混合搜索(OPQ16+精确重排序)": "Hybrid Search (OPQ16+Rerank)",
        "混合搜索(OPQ32+精确重排序)": "Hybrid Search (OPQ32+Rerank)"
    }
    
    # First try to read directly from data_used.csv (preferred method)
    try:
        print("Trying to read from data_used.csv...")
        df = pd.read_csv(f'{output_dir}/data_used.csv', encoding='utf-8-sig')
        
        if not df.empty:
            print(f"Successfully read {len(df)} records from data_used.csv")
            
            # Translate any remaining Chinese algorithm names to English
            for i, row in df.iterrows():
                algo_name = row['algorithm']
                if any(ch for ch in algo_name if ord(ch) > 127):  # Check if contains non-ASCII (likely Chinese)
                    for ch_name, eng_name in algo_name_map.items():
                        if ch_name in algo_name:
                            df.at[i, 'algorithm'] = algo_name.replace(ch_name, eng_name)
                            break
            
            return df
    except Exception as e:
        print(f"Error reading from data_used.csv: {str(e)}, falling back to other methods")
    
    # Rest of the existing code for reading from other files
    try:
        with open('results/pq_test.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Read PQ section
            pq_section = False
            opq_section = False
            rerank_section = False
            
            for line in lines:
                line = line.strip()
                if "PQ性能测试" in line:
                    pq_section = True
                    continue
                elif "OPQ性能测试" in line:
                    pq_section = False
                    opq_section = True
                    continue
                elif "PQ+重排序性能测试" in line:
                    opq_section = False
                    rerank_section = True
                    continue
                
                # Skip headers and separators
                if not line or '---' in line or '平均召回率' in line or '子空间数' in line or '配置' in line or '基础向量' in line:
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        if pq_section:
                            if parts[0] == '暴力搜索':
                                results.append({
                                    'algorithm': "Brute Force",
                                    'recall': float(parts[1]),
                                    'latency': float(parts[2]),
                                    'speedup': float(parts[3].replace('x', ''))
                                })
                            else:
                                results.append({
                                    'algorithm': f"PQ (M={parts[0]})",
                                    'recall': float(parts[1]),
                                    'latency': float(parts[2]),
                                    'speedup': float(parts[3].replace('x', ''))
                                })
                        elif opq_section:
                            results.append({
                                'algorithm': f"OPQ (M={parts[0].replace('(OPQ)', '')})",
                                'recall': float(parts[1]),
                                'latency': float(parts[2]),
                                'speedup': float(parts[3].replace('x', ''))
                            })
                        elif rerank_section:
                            results.append({
                                'algorithm': parts[0].replace('PQ16+R', 'Hybrid PQ16+R'),
                                'recall': float(parts[1]),
                                'latency': float(parts[2]),
                                'speedup': float(parts[3].replace('x', ''))
                            })
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line}, error: {str(e)}")
    except FileNotFoundError:
        print("Warning: pq_test.txt file not found, skipping this data source")
    except Exception as e:
        print(f"Error reading pq_test.txt: {str(e)}")
    
    # Read from main_op files
    for i in range(16):
        try:
            with open(f'results/main_op_{i}.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    algo_name = lines[0].replace('测试方法:', '').strip()
                    
                    # Translate algorithm name to English
                    eng_algo_name = algo_name
                    for ch_name, eng_name in algo_name_map.items():
                        if ch_name in algo_name:
                            eng_algo_name = algo_name.replace(ch_name, eng_name)
                            break
                    
                    recall_line = None
                    latency_line = None
                    
                    # Find recall and latency lines
                    for line in lines:
                        if '平均召回率:' in line:
                            recall_line = line
                        elif '平均延迟(微秒):' in line:
                            latency_line = line
                    
                    if recall_line and latency_line:
                        try:
                            recall = float(recall_line.replace('平均召回率:', '').strip())
                            latency_match = re.search(r'平均延迟\(微秒\): ([\d\.]+)', latency_line)
                            latency = float(latency_match.group(1)) if latency_match else None
                            
                            # Check if algorithm already exists
                            exists = False
                            for r in results:
                                if r['algorithm'] == eng_algo_name:
                                    exists = True
                                    break
                            
                            if not exists and latency is not None:
                                results.append({
                                    'algorithm': eng_algo_name,
                                    'recall': recall,
                                    'latency': latency / 1000,  # microseconds to milliseconds
                                    'speedup': None
                                })
                        except (ValueError, IndexError, AttributeError) as e:
                            print(f"Error parsing main_op_{i}.txt: {str(e)}")
        except FileNotFoundError:
            # Silently skip non-existent files
            pass
        except Exception as e:
            print(f"Error reading main_op_{i}.txt: {str(e)}")
    
    # If no data was read, use example data
    if not results:
        print("Warning: No data could be read from files, using example data instead")
        return get_hardcoded_data()
    
    return pd.DataFrame(results)

# Calculate speedup relative to brute force
def calc_speedup(df):
    # If no brute force, use slowest algorithm as baseline
    if 'Brute Force' not in df['algorithm'].values and 'Scalar Brute Force' not in df['algorithm'].values:
        base_latency = df['latency'].max()
    else:
        # Use brute force as baseline
        base_algo = 'Brute Force' if 'Brute Force' in df['algorithm'].values else 'Scalar Brute Force'
        base_latency = df.loc[df['algorithm'] == base_algo, 'latency'].values[0]
    
    # Calculate speedup
    df['speedup'] = base_latency / df['latency']
    return df

# Plot recall-latency tradeoff
def plot_recall_latency_tradeoff(df):
    plt.figure(figsize=(12, 8))
    
    # Markers for different algorithm types
    markers = {'Brute Force': 'o', 'Scalar Brute Force': 'o', 'SSE Optimized Brute Force': 'o', 'AVX Optimized Brute Force': 'o',
              'Scalar Quantization (SQ)': 's', 
              'PQ': '^', 'OPQ': 'D', 
              'Hybrid Search': '*', 'Hybrid PQ16': 'P'}
    
    # Group algorithms for better visualization
    groups = {
        'Brute Force': ['Brute Force', 'Scalar Brute Force', 'SSE Optimized Brute Force', 'AVX Optimized Brute Force'],
        'SQ': ['Scalar Quantization (SQ)'],
        'PQ': [col for col in df['algorithm'] if 'PQ (M=' in col or 'Product Quantization (PQ) M=' in col],
        'OPQ': [col for col in df['algorithm'] if 'OPQ (M=' in col or 'Optimized PQ (OPQ) M=' in col],
        'Hybrid Methods': [col for col in df['algorithm'] if 'Hybrid Search' in col or 'Hybrid PQ16' in col]
    }
    
    colors = {'Brute Force': 'blue', 'SQ': 'green', 'PQ': 'red', 'OPQ': 'purple', 'Hybrid Methods': 'orange'}
    
    for group_name, group_algos in groups.items():
        group_df = df[df['algorithm'].isin(group_algos)]
        if not group_df.empty:
            for _, row in group_df.iterrows():
                marker = next((markers[m] for m in markers if m in row['algorithm']), 'o')
                plt.scatter(row['latency'], row['recall'], 
                          label=row['algorithm'], 
                          marker=marker, s=100, 
                          color=colors[group_name],
                          alpha=0.7)
    
    # Add target recall line
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target Recall (0.9)')
    
    # Add labels and title
    plt.xlabel('Latency (ms)', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.title('ANN Algorithm Recall-Latency Tradeoff Analysis', fontsize=16)
    
    # Adjust legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    
    plt.xlim(left=0)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/recall_latency_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot speedup comparison
def plot_speedup_comparison(df):
    plt.figure(figsize=(12, 8))
    
    # Filter out rows with no speedup
    df_speedup = df[df['speedup'].notna()]
    if df_speedup.empty:
        return
    
    # Sort by speedup
    df_speedup = df_speedup.sort_values('speedup', ascending=False)
    
    # Set different colors for different categories
    colors = []
    for algo in df_speedup['algorithm']:
        if any(x in algo for x in ['Brute Force', 'Scalar Brute Force', 'SSE Optimized Brute Force', 'AVX Optimized Brute Force']):
            colors.append('blue')
        elif 'SQ' in algo or 'Scalar Quantization' in algo:
            colors.append('green')
        elif 'PQ' in algo and 'OPQ' not in algo and 'Hybrid' not in algo and 'R' not in algo:
            colors.append('red')
        elif 'OPQ' in algo and 'Hybrid' not in algo:
            colors.append('purple')
        else:
            colors.append('orange')
    
    # Plot bar chart
    bars = plt.bar(df_speedup['algorithm'], df_speedup['speedup'], color=colors, alpha=0.7)
    
    # Show speedup values above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}x', ha='center', va='bottom', rotation=0)
    
    # Add labels and title
    plt.ylabel('Speedup Relative to Brute Force', fontsize=14)
    plt.title('Speedup Comparison of ANN Algorithms', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot recall vs speedup bubble chart
def plot_recall_speedup_bubble(df):
    plt.figure(figsize=(12, 8))
    
    # Filter out rows with no speedup
    df_filtered = df[df['speedup'].notna()]
    if df_filtered.empty:
        return
    
    # Define algorithm categories
    categories = {
        'Brute Force': ['Brute Force', 'Scalar Brute Force', 'SSE Optimized Brute Force', 'AVX Optimized Brute Force'],
        'Scalar Quantization': ['Scalar Quantization (SQ)'],
        'Product Quantization': [algo for algo in df_filtered['algorithm'] if ('PQ (M=' in algo or 'Product Quantization (PQ) M=' in algo) and 'OPQ' not in algo and 'Hybrid' not in algo and '+R' not in algo],
        'Optimized PQ': [algo for algo in df_filtered['algorithm'] if 'OPQ' in algo and 'Hybrid' not in algo],
        'Hybrid Methods': [algo for algo in df_filtered['algorithm'] if 'Hybrid Search' in algo or '+R' in algo]
    }
    
    colors = {'Brute Force': 'blue', 'Scalar Quantization': 'green', 'Product Quantization': 'red', 'Optimized PQ': 'purple', 'Hybrid Methods': 'orange'}
    
    # Add target recall line
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target Recall (0.9)')
    
    # Plot scatter points for each category
    for category, algos in categories.items():
        cat_df = df_filtered[df_filtered['algorithm'].isin(algos)]
        if not cat_df.empty:
            # Use inverse of latency for bubble size
            sizes = 2000 / (cat_df['latency'] + 0.1)  # Add 0.1 to avoid division by zero
            plt.scatter(cat_df['speedup'], cat_df['recall'], 
                       s=sizes, alpha=0.7, label=category, color=colors[category])
            
            # Add algorithm name labels
            for _, row in cat_df.iterrows():
                plt.annotate(row['algorithm'], 
                           (row['speedup'], row['recall']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
    
    # Add labels and title
    plt.xlabel('Speedup', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.title('ANN Algorithm Performance: Recall vs Speedup', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/recall_speedup_bubble.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot impact of subspace count on PQ and OPQ
def plot_subspace_impact(df):
    # Extract PQ and OPQ results
    pq_df = df[df['algorithm'].str.contains('PQ \(M=') | df['algorithm'].str.contains('Product Quantization \(PQ\) M=')]
    opq_df = df[df['algorithm'].str.contains('OPQ \(M=') | df['algorithm'].str.contains('Optimized PQ \(OPQ\) M=')]
    
    if pq_df.empty and opq_df.empty:
        return
    
    # Extract subspace count
    def extract_m(algo):
        match = re.search(r'M=(\d+)', algo)
        if match:
            return int(match.group(1))
        return None
    
    pq_df['M'] = pq_df['algorithm'].apply(extract_m)
    opq_df['M'] = opq_df['algorithm'].apply(extract_m)
    
    # Sort by M
    pq_df = pq_df.sort_values('M')
    opq_df = opq_df.sort_values('M')
    
    # Create chart
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot recall curves
    ax1.set_xlabel('Number of Subspaces (M)', fontsize=14)
    ax1.set_ylabel('Recall', fontsize=14, color='blue')
    if not pq_df.empty:
        ax1.plot(pq_df['M'], pq_df['recall'], 'o-', label='PQ Recall', color='blue')
    if not opq_df.empty:
        ax1.plot(opq_df['M'], opq_df['recall'], 's-', label='OPQ Recall', color='lightblue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axhline(y=0.9, color='blue', linestyle='--', alpha=0.5, label='Target Recall (0.9)')
    
    # Create second y-axis for latency
    ax2 = ax1.twinx()
    ax2.set_ylabel('Latency (ms)', fontsize=14, color='red')
    if not pq_df.empty:
        ax2.plot(pq_df['M'], pq_df['latency'], 'o--', label='PQ Latency', color='red')
    if not opq_df.empty:
        ax2.plot(opq_df['M'], opq_df['latency'], 's--', label='OPQ Latency', color='darkred')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Impact of Subspace Count on PQ and OPQ Performance', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/subspace_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot hybrid method performance
def plot_hybrid_performance(df):
    # Extract hybrid methods
    hybrid_df = df[df['algorithm'].str.contains('Hybrid') | df['algorithm'].str.contains('R')]
    
    if hybrid_df.empty:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Sort by recall
    hybrid_df = hybrid_df.sort_values('recall')
    
    # Create bar chart
    x = np.arange(len(hybrid_df))
    width = 0.35
    
    # Plot recall bars
    ax1 = plt.subplot(111)
    bars1 = ax1.bar(x - width/2, hybrid_df['recall'], width, label='Recall', color='blue', alpha=0.7)
    ax1.set_ylabel('Recall', fontsize=14)
    ax1.set_ylim(0, 1.05)
    
    # Plot latency bars on second y-axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, hybrid_df['latency'], width, label='Latency (ms)', color='red', alpha=0.7)
    ax2.set_ylabel('Latency (ms)', fontsize=14)
    
    # Add algorithm names and legend
    ax1.set_xticks(x)
    ax1.set_xticklabels(hybrid_df['algorithm'], rotation=45, ha='right')
    
    # Add target recall line
    ax1.axhline(y=0.9, color='blue', linestyle='--', alpha=0.5, label='Target Recall (0.9)')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Hybrid Search Method Performance Analysis', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hybrid_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main function
def main():
    print("Starting ANN algorithm performance visualization...")
    
    # Try to read data from files
    try:
        df = read_results_from_files()
        if len(df) == 0:
            print("Data read from files is empty, using hardcoded data")
            df = get_hardcoded_data()
    except Exception as e:
        print(f"Failed to read file data: {str(e)}, using hardcoded data")
        df = get_hardcoded_data()
    
    # Calculate speedup if needed
    df = calc_speedup(df)
    
    # Save data for debugging
    print("Data used:")
    print(df)
    df.to_csv(f'{output_dir}/data_used.csv', index=False, encoding='utf-8-sig')
    
    print("Starting to generate charts...")
    # Generate visualization charts
    plot_recall_latency_tradeoff(df)
    plot_speedup_comparison(df)
    plot_recall_speedup_bubble(df)
    plot_subspace_impact(df)
    plot_hybrid_performance(df)
    
    print(f"Visualization complete! Charts saved to {output_dir} directory")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error during execution: {str(e)}")
        traceback.print_exc() 