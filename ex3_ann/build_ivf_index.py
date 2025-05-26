import numpy as np
import struct
from pathlib import Path
import sys
from sklearn.cluster import KMeans

# --- 配置参数 ---
BASE_DATA_DIR = Path("anndata")
OUTPUT_DIR = Path("files")
VECTOR_FILE = BASE_DATA_DIR / "DEEP100K.base.100k.fbin"
NUM_CLUSTERS_LIST = [64, 128, 256, 512]
PQ_M_LIST = [48]
PQ_NBITS = 8  # 每子量化器位数 => 每个子空间聚类中心数 = 2**PQ_NBITS

def read_fbin(filepath: Path):
    with open(filepath, 'rb') as f:
        n, d = struct.unpack('II', f.read(8))
        data = np.fromfile(f, dtype=np.float32, count=n*d)
    if data.size != n*d:
        raise ValueError(f"文件读取异常: 预计 {n*d} 个 float, 实际 {data.size}")
    return data.reshape(n, d), n, d

def save_centroids(centroids: np.ndarray, filepath: Path):
    nlist, dim = centroids.shape
    with open(filepath, 'wb') as f:
        f.write(struct.pack('i', nlist))
        f.write(struct.pack('i', dim))
        centroids.tofile(f)

def save_invlists(labels: np.ndarray, filepath: Path):
    nlist = labels.max() + 1
    invlists = [[] for _ in range(nlist)]
    for idx, lab in enumerate(labels):
        invlists[lab].append(idx)
    with open(filepath, 'wb') as f:
        f.write(struct.pack('i', nlist))
        for lst in invlists:
            f.write(struct.pack('i', len(lst)))
            if lst:
                f.write(struct.pack(f'{len(lst)}I', *lst))

def build_ivf_flat(vectors: np.ndarray, nlist: int):
    km = KMeans(n_clusters=nlist, random_state=42, n_init='auto', verbose=0)
    km.fit(vectors)
    return km.cluster_centers_.astype(np.float32), km.labels_

def train_pq(
    vectors: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    nlist: int,
    m: int,
    nbits: int,
    out_bin: Path
):
    """
    自己实现 IVF-PQ:
     1) 用 centroids+labels 构造倒排列表
     2) 对每条向量 residual = x - centroid[label]
     3) 将所有 residual 拆成 m 段，每段独立做 KMeans(2**nbits)
     4) PQ 码本 = 每段的质心；codes = 每条 residual 在各段上最靠近的质心索引
     5) 按自定义格式写入 .bin
    """
    n, d = vectors.shape
    ksub = 1 << nbits
    dsub = d // m
    
    # 确保维度可以均匀划分
    if d % m != 0:
        raise ValueError(f"维度 {d} 不能被 m={m} 整除")
    
    print(f"训练 IVF-PQ: nlist={nlist}, m={m}, ksub={ksub}, d={d}, dsub={dsub}")

    # 构造倒排列表
    invlists = [[] for _ in range(nlist)]
    for i, lab in enumerate(labels):
        invlists[lab].append(i)

    # 计算所有残差
    residuals = vectors - centroids[labels]

    # 拆分子空间并训练 PQ 码本
    codebooks = np.zeros((m, ksub, dsub), dtype=np.float32)
    codes = np.zeros((n, m), dtype=np.uint8)
    for pi in range(m):
        sub = residuals[:, pi*dsub:(pi+1)*dsub]
        km = KMeans(n_clusters=ksub, random_state=42, n_init='auto', verbose=0)
        km.fit(sub)
        codebooks[pi] = km.cluster_centers_.astype(np.float32)
        codes[:, pi] = km.predict(sub).astype(np.uint8)

    # 写二进制，使用int32_t格式与C++代码一致
    with open(out_bin, 'wb') as f:
        # 1. 写入基本参数 (4个int32)
        f.write(struct.pack('i', nlist))   # 聚类中心数量
        f.write(struct.pack('i', d))       # 原始向量维度
        f.write(struct.pack('i', m))       # 子向量组数
        f.write(struct.pack('i', ksub))    # 每组的聚类中心数
        
        # 2. 写入IVF聚类中心 (nlist*d个float32)
        f.write(centroids.astype(np.float32).tobytes())
        
        # 3. 写入PQ码本 (m*ksub*dsub个float32)
        # 将(m,ksub,dsub)形状重新排列为一维数组
        pq_codebooks_flat = codebooks.reshape(-1).astype(np.float32)
        f.write(pq_codebooks_flat.tobytes())
        
        # 4. 写入每个聚类的倒排列表和PQ编码
        for list_id in range(nlist):
            # 获取该聚类中的ID列表
            id_list = invlists[list_id]
            list_size = len(id_list)
            
            # 写入列表大小
            f.write(struct.pack('i', list_size))
            
            if list_size > 0:
                # 写入ID列表 (list_size个uint32)
                f.write(np.array(id_list, dtype=np.uint32).tobytes())
                
                # 为每个ID收集对应的PQ编码
                list_codes = np.zeros((list_size, m), dtype=np.uint8)
                for i, idx in enumerate(id_list):
                    list_codes[i] = codes[idx]
                
                # 写入PQ编码 (list_size*m个uint8)
                f.write(list_codes.tobytes())

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    vectors, n, d = read_fbin(VECTOR_FILE)
    print(f"加载向量: 数量={n}, 维度={d}")

    for nlist in NUM_CLUSTERS_LIST:
        c, lab = build_ivf_flat(vectors, nlist)
        cfile = OUTPUT_DIR / f"ivf_flat_centroids_{nlist}.fbin"
        ilist = OUTPUT_DIR / f"ivf_flat_invlists_{nlist}.bin"
        save_centroids(c, cfile)
        save_invlists(lab, ilist)
        print(f"已保存 IVF-Flat: {cfile.name}, {ilist.name}")

        for m in PQ_M_LIST:
            out_bin = OUTPUT_DIR / f"ivf_pq_nlist{nlist}_m{m}_b{PQ_NBITS}.bin"
            print(f"生成 IVF-PQ bin: nlist={nlist}, m={m} -> {out_bin.name}")
            train_pq(vectors, c, lab, nlist, m, PQ_NBITS, out_bin)

    print("全部完成。")

if __name__ == "__main__":
    main()
