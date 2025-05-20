import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

# 生成示例数据
N, d = 20000, 64
X = np.random.random((N, d)).astype('float32')
queries = np.random.random((5, d)).astype('float32')

# 1. 聚类训练
nlist, nprobe, k = 64, 5, 10
kmeans = KMeans(n_clusters=nlist, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# 2. 构建倒排列表
inverted_lists = defaultdict(list)
for idx, label in enumerate(labels):
    inverted_lists[label].append((idx, X[idx]))

# 3. 定义搜索函数
def probe_clusters(q, centroids, nprobe):
    dists = np.linalg.norm(centroids - q, axis=1)
    return np.argpartition(dists, nprobe)[:nprobe]

def search_ivf(q, centroids, inverted_lists, nprobe, k):
    probes = probe_clusters(q, centroids, nprobe)
    candidates = []
    for p in probes:
        for idx, vec in inverted_lists[p]:
            dist = np.linalg.norm(vec - q)
            candidates.append((dist, idx))
    candidates.sort(key=lambda x: x[0])
    return candidates[:k]

# 4. 执行检索
for qi, q in enumerate(queries):
    results = search_ivf(q, centroids, inverted_lists, nprobe, k)
    dists, idxs = zip(*results)
    print(f"Query {qi}: indices={idxs}, distances={dists}")

