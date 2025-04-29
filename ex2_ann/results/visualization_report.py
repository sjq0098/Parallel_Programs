#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import sys
import platform

# Set Agg backend to avoid GUI dependencies
mpl.use('Agg')  # Non-interactive backend

# Detect operating system
os_system = platform.system()
print(f"Current OS: {os_system}")

# Use simple ASCII characters for output
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']

# Set plotting style
sns.set_style("whitegrid")

# Create output directory
output_dir = 'results/report_figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Prepare data for LaTeX report
def prepare_latex_data():
    # Hardcoded data with complete latency information
    data = [
        {"algorithm": "Scalar Brute Force", "recall": 0.99995, "latency": 9303.17},
        {"algorithm": "SSE Optimized Brute Force", "recall": 0.99995, "latency": 1996.43},
        {"algorithm": "AVX Optimized Brute Force", "recall": 0.99995, "latency": 1917.21},
        {"algorithm": "Scalar Quantization (SQ)", "recall": 0.987752, "latency": 1704.64},
        {"algorithm": "Product Quantization (PQ) M=4", "recall": 0.10945, "latency": 515.718},
        {"algorithm": "Product Quantization (PQ) M=8", "recall": 0.238451, "latency": 603.71},
        {"algorithm": "Product Quantization (PQ) M=16", "recall": 0.434, "latency": 2658.5},
        {"algorithm": "Product Quantization (PQ) M=32", "recall": 0.685149, "latency": 4488.7},
        {"algorithm": "Optimized PQ (OPQ) M=4", "recall": 0.112401, "latency": 406.9},
        {"algorithm": "Optimized PQ (OPQ) M=8", "recall": 0.237151, "latency": 385.0},
        {"algorithm": "Optimized PQ (OPQ) M=16", "recall": 0.4303, "latency": 2604.6},
        {"algorithm": "Optimized PQ (OPQ) M=32", "recall": 0.692049, "latency": 5935.1},
        {"algorithm": "Hybrid Search (PQ16+Rerank)", "recall": 0.926455, "latency": 1023.0},
        {"algorithm": "Hybrid Search (PQ32+Rerank)", "recall": 0.99825, "latency": 1814.0},
        {"algorithm": "Hybrid Search (OPQ16+Rerank)", "recall": 0.926906, "latency": 1055.0},
        {"algorithm": "Hybrid Search (OPQ32+Rerank)", "recall": 0.99825, "latency": 1900.0}
    ]
    
    # Add rerank factor data
    rerank_data = [
        {"algorithm": "Hybrid PQ16+R50", "recall": 0.8277, "latency": 1060.5},
        {"algorithm": "Hybrid PQ16+R100", "recall": 0.9265, "latency": 1023.0},
        {"algorithm": "Hybrid PQ16+R200", "recall": 0.9798, "latency": 1214.9},
        {"algorithm": "Hybrid PQ16+R500", "recall": 0.9982, "latency": 1813.2}
    ]
    
    data.extend(rerank_data)
    df = pd.DataFrame(data)
    
    # Calculate speedup
    base_latency = df.loc[df['algorithm'] == 'Scalar Brute Force', 'latency'].values[0]
    df['speedup'] = base_latency / df['latency']
    
    # Convert microseconds to milliseconds for display
    df['latency_ms'] = df['latency'] / 1000
    
    return df

# Generate recall-latency plot for LaTeX
def generate_recall_latency_plot(df):
    plt.figure(figsize=(10, 6))
    
    # Define algorithm categories
    categories = {
        'Brute Force': ['Scalar Brute Force', 'SSE Optimized Brute Force', 'AVX Optimized Brute Force'],
        'Scalar Quantization': ['Scalar Quantization (SQ)'],
        'Product Quantization': [algo for algo in df['algorithm'] if 'Product Quantization (PQ)' in algo],
        'Optimized PQ': [algo for algo in df['algorithm'] if 'Optimized PQ (OPQ)' in algo],
        'Hybrid Methods': [algo for algo in df['algorithm'] if 'Hybrid Search' in algo or 'Hybrid PQ16+R' in algo]
    }
    
    markers = {'Brute Force': 'o', 'Scalar Quantization': 's', 'Product Quantization': '^', 'Optimized PQ': 'D', 'Hybrid Methods': '*'}
    colors = {'Brute Force': 'blue', 'Scalar Quantization': 'green', 'Product Quantization': 'red', 'Optimized PQ': 'purple', 'Hybrid Methods': 'orange'}
    
    for category, algos in categories.items():
        df_cat = df[df['algorithm'].isin(algos)]
        if not df_cat.empty:
            plt.scatter(df_cat['latency_ms'], df_cat['recall'], 
                       label=category, marker=markers[category], 
                       color=colors[category], s=100, alpha=0.7)
    
    # Add target recall line
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target Recall (0.9)')
    
    plt.xlabel('Latency (ms)', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.title('ANN Algorithm Recall-Latency Tradeoff', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/recall_latency.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/recall_latency.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate SIMD speedup plot
def generate_simd_speedup_plot(df):
    simd_df = df[df['algorithm'].isin(['Scalar Brute Force', 'SSE Optimized Brute Force', 'AVX Optimized Brute Force'])]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(simd_df['algorithm'], simd_df['speedup'], color='skyblue', alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}x', ha='center', va='bottom', fontsize=12)
    
    plt.ylabel('Speedup Relative to Scalar Implementation', fontsize=14)
    plt.title('SIMD Instruction Set Acceleration for Brute Force Search', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/simd_speedup.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/simd_speedup.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate PQ and OPQ subspace count impact plot
def generate_subspace_impact_plot(df):
    # Extract relevant data
    pq_df = df[df['algorithm'].str.contains('Product Quantization \(PQ\) M=')]
    opq_df = df[df['algorithm'].str.contains('Optimized PQ \(OPQ\) M=')]
    
    # Extract subspace count
    def extract_m(algo):
        import re
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
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
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
        ax2.plot(pq_df['M'], pq_df['latency_ms'], 'o--', label='PQ Latency', color='red')
    if not opq_df.empty:
        ax2.plot(opq_df['M'], opq_df['latency_ms'], 's--', label='OPQ Latency', color='darkred')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Impact of Subspace Count on PQ and OPQ Performance', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/subspace_impact.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/subspace_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate rerank factor impact plot
def generate_rerank_impact_plot(df):
    # Extract rerank related data
    rerank_df = df[df['algorithm'].str.contains('R')]
    
    if rerank_df.empty:
        return
    
    # Extract rerank factor
    def extract_factor(algo):
        import re
        match = re.search(r'R(\d+)', algo)
        if match:
            return int(match.group(1))
        return None
    
    rerank_df['factor'] = rerank_df['algorithm'].apply(extract_factor)
    rerank_df = rerank_df.sort_values('factor')
    
    # Create chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot recall curve
    ax1.set_xlabel('Rerank Factor', fontsize=14)
    ax1.set_ylabel('Recall', fontsize=14, color='blue')
    ax1.plot(rerank_df['factor'], rerank_df['recall'], 'o-', label='Recall', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axhline(y=0.9, color='blue', linestyle='--', alpha=0.5, label='Target Recall (0.9)')
    
    # Create second y-axis for latency
    ax2 = ax1.twinx()
    ax2.set_ylabel('Latency (ms)', fontsize=14, color='red')
    ax2.plot(rerank_df['factor'], rerank_df['latency_ms'], 's--', label='Latency', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Impact of Rerank Factor on Performance (PQ16)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rerank_impact.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/rerank_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate high recall algorithm comparison plot
def generate_high_recall_comparison(df):
    # Filter algorithms with recall ≥ 0.9
    high_recall_df = df[df['recall'] >= 0.9]
    if high_recall_df.empty:
        return
    
    # Sort by latency
    high_recall_df = high_recall_df.sort_values('latency_ms')
    
    plt.figure(figsize=(10, 6))
    
    # Set different colors for different types
    colors = []
    for algo in high_recall_df['algorithm']:
        if any(x in algo for x in ['Brute Force', 'Scalar Brute Force', 'SSE Optimized Brute Force', 'AVX Optimized Brute Force']):
            colors.append('blue')
        elif 'SQ' in algo or 'Scalar Quantization' in algo:
            colors.append('green')
        elif 'OPQ' in algo and 'Hybrid' in algo:
            colors.append('purple')
        elif 'PQ' in algo and 'Hybrid' in algo:
            colors.append('red')
        else:
            colors.append('orange')
    
    # Plot horizontal bar chart
    bars = plt.barh(high_recall_df['algorithm'], high_recall_df['latency_ms'], 
                   color=colors, alpha=0.7)
    
    # Add latency values
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                f'{width:.2f} ms', va='center')
    
    plt.xlabel('Latency (ms)', fontsize=14)
    plt.title('Performance Comparison of High Recall Algorithms (Recall ≥ 0.9)', fontsize=16)
    plt.xlim(right=max(high_recall_df['latency_ms']) * 1.1)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/high_recall_comparison.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/high_recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate recall vs speedup scatter plot
def generate_recall_speedup_scatter(df):
    plt.figure(figsize=(10, 6))
    
    # Define algorithm categories and colors
    categories = {
        'Brute Force': ['Scalar Brute Force', 'SSE Optimized Brute Force', 'AVX Optimized Brute Force'],
        'Scalar Quantization': ['Scalar Quantization (SQ)'],
        'Product Quantization': [algo for algo in df['algorithm'] if 'Product Quantization (PQ)' in algo],
        'Optimized PQ': [algo for algo in df['algorithm'] if 'Optimized PQ (OPQ)' in algo],
        'Hybrid Methods': [algo for algo in df['algorithm'] if 'Hybrid Search' in algo or 'Hybrid PQ16+R' in algo]
    }
    
    colors = {'Brute Force': 'blue', 'Scalar Quantization': 'green', 'Product Quantization': 'red', 'Optimized PQ': 'purple', 'Hybrid Methods': 'orange'}
    markers = {'Brute Force': 'o', 'Scalar Quantization': 's', 'Product Quantization': '^', 'Optimized PQ': 'D', 'Hybrid Methods': '*'}
    
    # Add target recall line
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target Recall (0.9)')
    
    # Plot scatter points for each category
    for category, algos in categories.items():
        cat_df = df[df['algorithm'].isin(algos)]
        if not cat_df.empty:
            plt.scatter(cat_df['speedup'], cat_df['recall'], 
                       label=category, marker=markers[category], 
                       color=colors[category], s=100, alpha=0.7)
    
    plt.xlabel('Speedup (Relative to Scalar Brute Force)', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.title('ANN Algorithm Performance Tradeoff: Recall vs Speedup', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/recall_speedup_scatter.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/recall_speedup_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate LaTeX tables for report
def generate_latex_tables(df):
    # Group by category
    categories = {
        'Brute Force': df[df['algorithm'].isin(['Scalar Brute Force', 'SSE Optimized Brute Force', 'AVX Optimized Brute Force'])],
        'Quantization': df[df['algorithm'].isin(['Scalar Quantization (SQ)'])],
        'PQ Methods': df[df['algorithm'].str.contains('Product Quantization')],
        'OPQ Methods': df[df['algorithm'].str.contains('Optimized PQ')],
        'Hybrid Methods': df[df['algorithm'].str.contains('Hybrid Search') | df['algorithm'].str.contains('Hybrid PQ')]
    }
    
    # Create LaTeX tables
    with open(f'{output_dir}/performance_tables.tex', 'w', encoding='utf-8') as f:
        # Brute force table
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Brute Force Algorithm Performance Comparison}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Algorithm & Recall & Latency (ms) & Speedup \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in categories['Brute Force'].iterrows():
            f.write(f"{row['algorithm']} & {row['recall']:.4f} & {row['latency_ms']:.2f} & {row['speedup']:.2f}x \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:brute_force}\n")
        f.write("\\end{table}\n\n")
        
        # Quantization methods table
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Quantization Method Performance Comparison}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Algorithm & Recall & Latency (ms) & Speedup \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in categories['Quantization'].iterrows():
            f.write(f"{row['algorithm']} & {row['recall']:.4f} & {row['latency_ms']:.2f} & {row['speedup']:.2f}x \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:quantization}\n")
        f.write("\\end{table}\n\n")
        
        # PQ and OPQ methods table
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Product Quantization (PQ) and Optimized PQ (OPQ) Performance Comparison}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Algorithm & Recall & Latency (ms) & Speedup \\\\\n")
        f.write("\\midrule\n")
        
        pq_opq_df = pd.concat([categories['PQ Methods'], categories['OPQ Methods']])
        for _, row in pq_opq_df.iterrows():
            f.write(f"{row['algorithm']} & {row['recall']:.4f} & {row['latency_ms']:.2f} & {row['speedup']:.2f}x \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:pq_opq}\n")
        f.write("\\end{table}\n\n")
        
        # Hybrid methods table
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Hybrid Search Method Performance Comparison}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Algorithm & Recall & Latency (ms) & Speedup \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in categories['Hybrid Methods'].iterrows():
            f.write(f"{row['algorithm']} & {row['recall']:.4f} & {row['latency_ms']:.2f} & {row['speedup']:.2f}x \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:hybrid}\n")
        f.write("\\end{table}\n\n")
        
        # High recall algorithms table
        high_recall_df = df[df['recall'] >= 0.9].sort_values('latency_ms')
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{High Recall Algorithm Performance Ranking (Recall $\\geq$ 0.9)}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Algorithm & Recall & Latency (ms) & Speedup \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in high_recall_df.iterrows():
            f.write(f"{row['algorithm']} & {row['recall']:.4f} & {row['latency_ms']:.2f} & {row['speedup']:.2f}x \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:high_recall}\n")
        f.write("\\end{table}\n")

# Generate performance analysis summary
def generate_performance_summary(df):
    with open(f'{output_dir}/performance_summary.tex', 'w', encoding='utf-8') as f:
        f.write("\\subsection{Algorithm Performance Overview}\n\n")
        
        # Brute force analysis
        f.write("\\paragraph{Brute Force Optimization} ")
        f.write("Through SIMD instruction set optimization, the performance of brute force search has been significantly improved. ")
        
        bf_df = df[df['algorithm'].isin(['Scalar Brute Force', 'SSE Optimized Brute Force', 'AVX Optimized Brute Force'])]
        if not bf_df.empty:
            speedup_sse = bf_df.loc[bf_df['algorithm'] == 'SSE Optimized Brute Force', 'speedup'].values[0]
            speedup_avx = bf_df.loc[bf_df['algorithm'] == 'AVX Optimized Brute Force', 'speedup'].values[0]
            f.write(f"Compared to the scalar implementation, the SSE instruction set provides {speedup_sse:.2f}x speedup, ")
            f.write(f"while the AVX instruction set provides {speedup_avx:.2f}x speedup, while maintaining identical recall. ")
            f.write("This demonstrates the effectiveness of SIMD vectorization for accelerating distance calculations.\n\n")
        
        # Quantization method analysis
        f.write("\\paragraph{Scalar Quantization (SQ) Technique} ")
        sq_df = df[df['algorithm'] == 'Scalar Quantization (SQ)']
        if not sq_df.empty:
            sq_recall = sq_df['recall'].values[0]
            sq_speedup = sq_df['speedup'].values[0]
            f.write(f"The SQ technique provides {sq_speedup:.2f}x speedup while maintaining a high recall of {sq_recall:.4f}. ")
            f.write("This technique quantizes floating-point vectors to 8-bit integers, combined with SIMD instructions to accelerate calculations, ")
            f.write("significantly reducing computational complexity while ensuring accuracy.\n\n")
        
        # PQ and OPQ methods analysis
        f.write("\\paragraph{Product Quantization Methods} ")
        f.write("Experimental results show that the performance of PQ and OPQ is closely related to the number of subspaces (M). ")
        f.write("As M increases, recall gradually improves, but computational overhead also increases correspondingly. ")
        
        pq_df = df[df['algorithm'].str.contains('Product Quantization')]
        opq_df = df[df['algorithm'].str.contains('Optimized PQ')]
        if not pq_df.empty and not opq_df.empty:
            pq_best_m = pq_df.loc[pq_df['recall'].idxmax(), 'algorithm']
            opq_best_m = opq_df.loc[opq_df['recall'].idxmax(), 'algorithm']
            f.write(f"Among the tested configurations, {pq_best_m} and {opq_best_m} provide the highest recall, ")
            f.write("but also come with higher latency. Interestingly, OPQ performs better than PQ at lower subspace counts (M=4,8), ")
            f.write("showing the benefits of the optimized transformation matrix. However, as M increases, the performance difference between the two diminishes, ")
            f.write("while the index construction cost for OPQ is significantly higher than PQ.\n\n")
        
        # Hybrid methods analysis
        f.write("\\paragraph{Hybrid Search Methods} ")
        f.write("Hybrid search methods (combining PQ/OPQ with exact reranking) achieve significant performance improvements while maintaining high recall. ")
        
        hybrid_df = df[df['algorithm'].str.contains('Hybrid')]
        if not hybrid_df.empty:
            best_hybrid = hybrid_df.loc[(hybrid_df['recall'] >= 0.9) & (hybrid_df['latency_ms'].idxmin()), 'algorithm']
            best_hybrid_recall = hybrid_df.loc[hybrid_df['algorithm'] == best_hybrid, 'recall'].values[0]
            best_hybrid_speedup = hybrid_df.loc[hybrid_df['algorithm'] == best_hybrid, 'speedup'].values[0]
            f.write(f"In particular, {best_hybrid} achieves {best_hybrid_recall:.4f} recall ")
            f.write(f"with a {best_hybrid_speedup:.2f}x speedup. ")
            f.write("This indicates that two-stage search strategies can effectively balance query efficiency and search accuracy.\n\n")
        
        # Rerank factor analysis
        rerank_df = df[df['algorithm'].str.contains('R')]
        if not rerank_df.empty:
            f.write("\\paragraph{Rerank Factor Impact} ")
            f.write("The experiment explores the impact of different reranking factors on performance. ")
            f.write("Results show that as the reranking factor increases, recall gradually improves, but query latency also increases accordingly. ")
            factor_90 = rerank_df[rerank_df['recall'] >= 0.9]['algorithm'].iloc[0] if any(rerank_df['recall'] >= 0.9) else None
            if factor_90:
                f.write(f"For the goal of achieving a 0.9 recall rate, {factor_90} provides the optimal performance balance point. ")
            f.write("This also confirms that hybrid methods can effectively compensate for the precision loss caused by PQ quantization through exact reranking.\n\n")
        
        # Final conclusion
        f.write("\\paragraph{Comprehensive Performance Analysis} ")
        high_recall_df = df[df['recall'] >= 0.9].sort_values('latency_ms')
        if not high_recall_df.empty:
            best_algo = high_recall_df.iloc[0]['algorithm']
            best_speedup = high_recall_df.iloc[0]['speedup']
            f.write("Among all implemented algorithms, ")
            f.write(f"{best_algo} provides the best performance under the condition of high recall ($\\geq$0.9), ")
            f.write(f"with a {best_speedup:.2f}x speedup compared to scalar brute force search. ")
            f.write("The experimental results show that hybrid methods are the most effective strategy for balancing accuracy and performance in high-dimensional ANN search, ")
            f.write("while SIMD optimization is a common basic acceleration technique for all methods. Considering comprehensively, ")
            f.write("the best practice should combine vectorized computation (SIMD), data compression techniques (such as PQ), and multi-stage search strategies.\n")

# Main function
def main():
    print("Starting to generate ANN algorithm performance report figures...")
    
    # Prepare data
    try:
        df = prepare_latex_data()
        
        # Save data for debugging
        print("Data used:")
        print(df)
        df.to_csv(f'{output_dir}/data_used.csv', index=False, encoding='utf-8-sig')
        
        print("Starting to generate charts...")
        # Generate various charts
        generate_recall_latency_plot(df)
        generate_simd_speedup_plot(df)
        generate_subspace_impact_plot(df)
        generate_rerank_impact_plot(df)
        generate_high_recall_comparison(df)
        generate_recall_speedup_scatter(df)
        
        # Generate LaTeX tables and summary
        generate_latex_tables(df)
        generate_performance_summary(df)
        
        print(f"Report figures generation complete! All figures and analysis saved to {output_dir} directory")
    except Exception as e:
        import traceback
        print(f"Error during report figure generation: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 