import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import numpy as np

# ---------- Data Preprocessing ----------
def preprocess_csv(file_path):
    corrected_lines = []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if line.strip().startswith('//'):
                continue
            parts = line.strip().split(',')
            if parts[0] == 'Method':
                corrected_lines.append(','.join(parts))
            else:
                if len(parts) == 7:
                    new_parts = parts[:5]
                    latency = parts[5] + parts[6]
                    new_parts.append(latency)
                    corrected_lines.append(','.join(new_parts))
                else:
                    corrected_lines.append(line.strip())
    df = pd.read_csv(StringIO("\n".join(corrected_lines)))
    df['M'] = df['M'].str.replace('M=', '').astype(int)
    df['efC'] = df['efC'].str.replace('efC=', '').astype(int)
    df['efS'] = df['efS'].str.replace('efS=', '').astype(int)
    df['Latency(us)'] = df['Latency(us)'].astype(str).str.replace(',', '').astype(int)
    df['Recall'] = df['Recall'].astype(float)
    return df

DATA_PATH = 'results/hnsw_results.csv'
data = preprocess_csv(DATA_PATH)


# ---------- Global Aesthetic Settings ----------
sns.set_theme(
    style="whitegrid",
    palette="tab10",
    font_scale=1.1,
    rc={
        "axes.linewidth": 1.5,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "legend.frameon": True,
        "legend.framealpha": 0.85
    }
)


# ---------- 1. Latency Comparison: Serial vs OpenMP ----------
def plot_latency_comparison():
    plt.figure(figsize=(8,5))
    methods = ['HNSW(Serial)', 'HNSW(OpenMP)']
    for method in methods:
        df = data[(data['Method']==method) & (data['M']==128) & (data['efC']==100)]
        df = df.sort_values('efS')
        sns.lineplot(x='efS', y='Latency(us)', data=df, marker='o', label=method, linewidth=2)
    plt.title('Latency vs efS (M=128, efC=100)')
    plt.xlabel('efS')
    plt.ylabel('Latency (µs)')
    plt.legend(title='Method')
    plt.tight_layout()
    plt.savefig('1_latency_comparison.png', dpi=300)
    plt.close()


# ---------- 2. Recall Comparison: Serial vs OpenMP ----------
def plot_recall_comparison():
    plt.figure(figsize=(8,5))
    methods = ['HNSW(Serial)', 'HNSW(OpenMP)']
    for method in methods:
        df = data[(data['Method']==method) & (data['M']==128) & (data['efC']==100)]
        df = df.sort_values('efS')
        sns.lineplot(x='efS', y='Recall', data=df, marker='s', label=method, linewidth=2)
    plt.ylim(0.95,1.005)
    plt.title('Recall vs efS (M=128, efC=100)')
    plt.xlabel('efS')
    plt.ylabel('Recall')
    plt.legend(title='Method')
    plt.tight_layout()
    plt.savefig('2_recall_comparison.png', dpi=300)
    plt.close()


# ---------- 3. Violin Plot: Latency Distribution by efC ----------
def plot_latency_violin():
    plt.figure(figsize=(8,5))
    subset = data[(data['Method']=='HNSW(Serial)') & (data['efS']==200)]
    sns.violinplot(x='efC', y='Latency(us)', data=subset, inner='quartile')
    plt.title('Latency Distribution at efS=200 (Serial)')
    plt.xlabel('efC')
    plt.ylabel('Latency (µs)')
    plt.tight_layout()
    plt.savefig('3_latency_violin.png', dpi=300)
    plt.close()


# ---------- 4. Heatmap: Recall (M vs efC) at efS=200 ----------
def plot_recall_heatmap():
    subset = data[(data['Method']=='HNSW(Serial)') & (data['efS']==200)]
    pivot = subset.pivot(index='M', columns='efC', values='Recall')
    plt.figure(figsize=(7,6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap='YlGnBu', cbar_kws={'label':'Recall'})
    plt.title('Recall Heatmap (efS=200, Serial)')
    plt.xlabel('efC')
    plt.ylabel('M')
    plt.tight_layout()
    plt.savefig('4_recall_heatmap.png', dpi=300)
    plt.close()


# ---------- 5. Parallel Coordinates: Performance Across Parameters ----------
from pandas.plotting import parallel_coordinates
def plot_parallel_coords():
    df = data[(data['Method']=='HNSW(Serial)') & (data['efC']==100) & (data['efS'].isin([50,200,500]))]
    df_pc = df[['M','efS','Recall','Latency(us)']].copy()
    df_pc['Latency(us)'] = (df_pc['Latency(us)'] - df_pc['Latency(us)'].min()) / (df_pc['Latency(us)'].max() - df_pc['Latency(us)'].min())
    plt.figure(figsize=(8,5))
    parallel_coordinates(df_pc, class_column='efS', cols=['M','Recall','Latency(us)'], marker='o')
    plt.title('Parallel Coordinates (Serial, efC=100)')
    plt.xlabel('Parameter / Metric')
    plt.ylabel('Normalized Value')
    plt.legend(title='efS', bbox_to_anchor=(1.05,1))
    plt.tight_layout()
    plt.savefig('5_parallel_coords.png', dpi=300)
    plt.close()


# ---------- 6. Pairplot: Overview of All Numeric Relationships (fixed sampling) ----------
def plot_pairplot():
    subset_all = data[data['Method']=='HNSW(OpenMP)']
    n_samples = min(200, len(subset_all))
    subset = subset_all.sample(n=n_samples, random_state=42)
    sns.pairplot(
        subset,
        vars=['M','efC','efS','Recall','Latency(us)'],
        kind='scatter',
        diag_kind='kde',
        corner=True,
        plot_kws={'alpha':0.6, 's':30}
    )
    plt.suptitle('Pairplot (OpenMP sample)', y=1.02)
    plt.savefig('6_pairplot.png', dpi=300)
    plt.close()


# 1) Speedup Curve: Speedup vs efS for fixed M=128, efC=100
def plot_speedup_curve():
    df_s = data[(data['Method']=='HNSW(Serial)')   & (data['M']==128) & (data['efC']==100)].set_index('efS')
    df_o = data[(data['Method']=='HNSW(OpenMP)')   & (data['M']==128) & (data['efC']==100)].set_index('efS')
    common = df_s.index.intersection(df_o.index)
    speedup = df_s.loc[common, 'Latency(us)'] / df_o.loc[common, 'Latency(us)']
    plt.figure(figsize=(8,5))
    sns.lineplot(x=common, y=speedup, marker='o')
    plt.xlabel('efS')
    plt.ylabel('Speedup (Serial / OpenMP)')
    plt.title('Speedup vs efS (M=128, efC=100)')
    plt.tight_layout()
    plt.savefig('speedup_curve.png', dpi=300)
    plt.close()

# 2) Boxplot of Latency by Method
def plot_latency_boxplot():
    plt.figure(figsize=(7,5))
    sns.boxplot(x='Method', y='Latency(us)', data=data, showfliers=False)
    sns.swarmplot(x='Method', y='Latency(us)', data=data, color='0.3', size=3, alpha=0.6)
    plt.yscale('log')
    plt.title('Latency Distribution by Method (all configs)')
    plt.ylabel('Latency (μs, log scale)')
    plt.tight_layout()
    plt.savefig('latency_boxplot.png', dpi=300)
    plt.close()

# 3) Performance Profile
def plot_performance_profile():
    # 针对每个配置计算 ratio = Lat_OP / Lat_SER（越大越好）
    df_ser = data[data['Method']=='HNSW(Serial)'].sort_values(['M','efC','efS'])
    df_o   = data[data['Method']=='HNSW(OpenMP)'] .sort_values(['M','efC','efS'])
    df = pd.merge(
        df_ser[['M','efC','efS','Latency(us)']],
        df_o[['M','efC','efS','Latency(us)']],
        on=['M','efC','efS'],
        suffixes=('_ser','_omp')
    )
    df['ratio'] = df['Latency(us)_ser'] / df['Latency(us)_omp']
    taus = np.linspace(1, df['ratio'].max(), 100)
    prof = [ (df['ratio'] <= tau).mean() for tau in taus ]
    plt.figure(figsize=(7,5))
    plt.step(taus, prof, where='post')
    plt.xlabel(r'Performance Ratio $\tau$')
    plt.ylabel('Proportion of Instances\n(ratio ≤ τ)')
    plt.title('Performance Profile: OpenMP vs Serial')
    plt.tight_layout()
    plt.savefig('performance_profile.png', dpi=300)
    plt.close()

# 4) Ratio Heatmap: Speedup (Serial/OpenMP) for efS vs M at efC=100
def plot_speedup_heatmap():
    # 构造矩阵
    df_ser = data[(data['Method']=='HNSW(Serial)')   & (data['efC']==100)]
    df_o   = data[(data['Method']=='HNSW(OpenMP)')   & (data['efC']==100)]
    merged = pd.merge(
        df_ser[['M','efS','Latency(us)']],
        df_o[['M','efS','Latency(us)']],
        on=['M','efS'],
        suffixes=('_ser','_omp')
    )
    merged['speedup'] = merged['Latency(us)_ser'] / merged['Latency(us)_omp']
    pivot = merged.pivot(index='efS', columns='M', values='speedup')
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label':'Speedup'})
    plt.title('Speedup Heatmap (efC=100)')
    plt.xlabel('M')
    plt.ylabel('efS')
    plt.tight_layout()
    plt.savefig('speedup_heatmap.png', dpi=300)
    plt.close()



# ---------- Main ----------
if __name__ == '__main__':
    plot_latency_comparison()
    plot_recall_comparison()
    plot_latency_violin()
    plot_recall_heatmap()
    plot_parallel_coords()
    plot_pairplot()
    plot_speedup_curve()
    plot_latency_boxplot()
    plot_performance_profile()
    plot_speedup_heatmap()
