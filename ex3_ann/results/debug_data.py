#!/usr/bin/env python3
import pandas as pd

def debug_data():
    """调试数据解析"""
    df = pd.read_csv('ivf_results.csv')
    print("原始数据:")
    print(f"总行数: {len(df)}")
    print("\n前5行:")
    print(df.head())
    print("\nConfiguration列的唯一值:")
    print(df['Configuration'].unique())
    
    # 测试正则表达式
    nlist_extracted = df['Configuration'].str.extract(r'nlist=(\d+)')
    nprobe_extracted = df['Configuration'].str.extract(r'nprobe=(\d+)')
    
    print(f"\nnlist提取结果 (前5行):")
    print(nlist_extracted.head())
    print(f"\nnprobe提取结果 (前5行):")
    print(nprobe_extracted.head())
    
    print(f"\n有效行数统计:")
    valid_rows = ~(nlist_extracted.isna() | nprobe_extracted.isna())
    print(f"总行数: {len(df)}")
    print(f"有效行数: {valid_rows.iloc[:, 0].sum()}")
    
    # 检查Method列
    print(f"\nMethod列的唯一值:")
    print(df['Method'].unique())

if __name__ == "__main__":
    debug_data() 