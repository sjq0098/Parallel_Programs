#!/usr/bin/env python3 
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import struct
import time

def load_fbin(filename):
    with open(filename, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        d = struct.unpack('i', f.read(4))[0]
        data = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            data[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return data

def save_codebook(filename, codebook, m, ksub, dim):
    with open(filename, 'wb') as f:
        f.write(struct.pack('Q', m))  # 子空间数量
        f.write(struct.pack('Q', ksub))  # 每个子空间的聚类数量
        f.write(struct.pack('Q', dim))  # 向量维度
        f.write(codebook.astype(np.float32).tobytes())
    print(f"保存码本到 {filename}")

def save_codes(filename, codes, n):
    with open(filename, 'wb') as f:
        f.write(struct.pack('Q', n))  # 向量数量
        f.write(codes.astype(np.uint8).tobytes())
    print(f"保存编码到 {filename}")

def save_rotation_matrix(filename, R, dim):
    with open(filename, 'wb') as f:
        f.write(struct.pack('Q', dim))  # 向量维度
        f.write(R.astype(np.float32).tobytes())
    print(f"保存旋转矩阵到 {filename}")

def compute_pq(data, m, ksub=256, opq=False, iteration=20):
    n, d = data.shape
    assert d % m == 0, f"向量维度 {d} 必须能被子空间数量 {m} 整除"
    
    dsub = d // m
    codebook = np.zeros((m, ksub, dsub), dtype=np.float32)
    
    # 旋转矩阵初始化（仅OPQ使用）
    R = np.eye(d, dtype=np.float32)
    
    if opq:
        print(f"计算OPQ, 子空间数量={m}, 子空间维度={dsub}, 聚类数量={ksub}")
        from scipy.linalg import svd
        
        # 中心化数据
        data_centered = data - np.mean(data, axis=0)
        
        # PCA分解获取旋转矩阵
        cov = np.dot(data_centered.T, data_centered) / n
        U, S, Vt = svd(cov, full_matrices=False)
        R = U.T  # 使用转置确保正确旋转
        
        # 应用旋转
        data_rotated = np.dot(data_centered, R)
    else:
        print(f"计算普通PQ, 子空间数量={m}, 子空间维度={dsub}, 聚类数量={ksub}")
        data_rotated = data
    
    # 计算每个子空间的码本
    codes = np.zeros((n, m), dtype=np.uint8)
    
    for i in range(m):
        print(f"处理子空间 {i+1}/{m}")
        sub_data = data_rotated[:, i*dsub:(i+1)*dsub]
        
        # 使用KMeans计算码本（增加n_init提高稳定性）
        kmeans = KMeans(n_clusters=ksub, n_init=5, max_iter=iteration, verbose=0)
        kmeans.fit(sub_data)
        
        codebook[i] = kmeans.cluster_centers_
        codes[:, i] = kmeans.predict(sub_data)
    
    return codebook.reshape(-1), codes, R

def main():
    # 修复路径分隔符
    data_path = os.path.join("anndata", "DEEP100K.base.100k.fbin")  # 使用os.path.join确保路径分隔符正确
    output_dir = "files"    # 输出目录
    m_list = [4, 8, 16, 32]         # 子空间数量列表
    opq_flags = [False, True]       # 同时生成PQ和OPQ
    
    # 加载并预处理数据
    print(f"加载数据 {data_path}")
    data = load_fbin(data_path)
    n, d = data.shape
    print(f"原始数据: {n} 向量, 维度 {d}")
    
    # L2归一化数据
    data = normalize(data, axis=1, norm='l2')
    print("数据L2归一化完成")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有配置组合
    for use_opq in opq_flags:
        for m in m_list:
            if d % m != 0:
                print(f"跳过 m={m}, 维度{d}无法整除")
                continue
            
            prefix = f"opq{m}" if use_opq else f"pq{m}"
            codebook_file = os.path.join(output_dir, f"{prefix}_codebook.bin")
            codes_file = os.path.join(output_dir, f"{prefix}_codes.bin")
            rotation_file = os.path.join(output_dir, f"{prefix}_rotation.bin")
            
            print(f"\n正在处理 {'OPQ' if use_opq else 'PQ'} m={m}")
            start = time.time()
            codebook, codes, R = compute_pq(data, m, opq=use_opq)
            print(f"计算耗时: {time.time()-start:.2f}s")
            
            save_codebook(codebook_file, codebook, m, 256, d)
            save_codes(codes_file, codes, n)
            if use_opq:
                save_rotation_matrix(rotation_file, R, d)

if __name__ == "__main__":
    main()