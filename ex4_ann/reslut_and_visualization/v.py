import pandas as pd
import matplotlib.pyplot as plt

def plot_vs_x(df, x, y, fixed, out_file, x_label, y_label, title):
    # 过滤掉所有固定参数
    sub = df.copy()
    for k, v in fixed.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        raise ValueError(f"No data for fixed={fixed}")
    
    plt.figure(figsize=(5,4))
    for M in sorted(sub['M'].unique()):
        s2 = sub[sub['M'] == M].sort_values(x)
        plt.plot(s2[x], s2[y],
                 marker='o', linewidth=2, markersize=6,
                 label=f'M={M}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title='M')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def main():
    df = pd.read_csv('results_mpi_ivf_hnsw.csv')
    
    # ==== 控制所有次要变量 ====
    # 固定 nlist=64, efConstruction=100, efSearch=32，扫描 nprobe
    fixed1 = {'nlist': 64, 'efConstruction': 100, 'efSearch': 32}
    plot_vs_x(df, x='nprobe', y='recall', fixed=fixed1,
              out_file='recall_vs_nprobe.png',
              x_label='$n_{probe}$', y_label='Recall@10',
              title='Recall vs $n_{probe}$\n(nlist=64, efC=100, efS=32)')
    plot_vs_x(df, x='nprobe', y='latency_us', fixed=fixed1,
              out_file='latency_vs_nprobe.png',
              x_label='$n_{probe}$', y_label='Latency (µs)',
              title='Latency vs $n_{probe}$\n(nlist=64, efC=100, efS=32)')
    
    # 固定 nlist=64, efConstruction=100, nprobe=8，扫描 efSearch
    fixed2 = {'nlist': 64, 'efConstruction': 100, 'nprobe': 8}
    plot_vs_x(df, x='efSearch', y='recall', fixed=fixed2,
              out_file='recall_vs_efSearch.png',
              x_label='$efSearch$', y_label='Recall@10',
              title='Recall vs $efSearch$\n(nlist=64, efC=100, nprobe=8)')
    plot_vs_x(df, x='efSearch', y='latency_us', fixed=fixed2,
              out_file='latency_vs_efSearch.png',
              x_label='$efSearch$', y_label='Latency (µs)',
              title='Latency vs $efSearch$\n(nlist=64, efC=100, nprobe=8)')
    
    print("Saved 4 plots with controlled parameters.")

if __name__ == '__main__':
    main()
