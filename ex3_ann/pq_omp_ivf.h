// OpenMP 并行版 IVFPQ 搜索 + Re-ranking（ADC + 全局候选 + 动态调度）
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <omp.h>
#include "pq_ivf.h"  // 引入 IVFPQIndex, compute_pq_distance_table


// OpenMP 并行版 IVFPQ 搜索（ADC）
std::priority_queue<std::pair<float, uint32_t>> ivfpq_search_omp(
    const IVFPQIndex* index,
    float* query,
    size_t k,
    size_t nprobe,
    int num_threads = omp_get_max_threads())
{
    // 1. 串行找最近 nprobe 个簇中心（L2 距离平方）
    struct CentDist { float dist2; uint32_t idx; };
    std::vector<CentDist> cd(index->nlist);
    
    for (size_t i = 0; i < index->nlist; ++i) {
        const float* cptr = index->centroids.data() + i * index->d;
        float dist2 = 0;
        for (size_t d = 0; d < index->d; ++d) {
            float diff = query[d] - cptr[d];
            dist2 += diff * diff;
        }
        cd[i].dist2 = dist2;
        cd[i].idx   = uint32_t(i);
    }
    if (nprobe < index->nlist) {
        std::nth_element(cd.begin(), cd.begin() + nprobe, cd.end(),
            [](auto &a, auto &b){ return a.dist2 < b.dist2; });
    }
    std::vector<uint32_t> probe_list;
    probe_list.reserve(nprobe);
    for (size_t i = 0; i < nprobe && i < index->nlist; ++i)
        probe_list.push_back(cd[i].idx);
    
    // 2. 串行构建 L2 距离表
    std::vector<float> pq_dist_table = compute_pq_distance_table(index, query);

    // 3. 并行扫描倒排列表
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_q(num_threads);
    
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        auto &q = local_q[tid];
        
        #pragma omp for schedule(static)
        for (int pi = 0; pi < (int)probe_list.size(); ++pi) {
            const auto& ids   = index->invlists[probe_list[pi]];
            const auto& codes = index->codes[probe_list[pi]];
            for (size_t i = 0; i < ids.size(); ++i) {
                float dist2 = 0;
                for (size_t m = 0; m < index->m; ++m) {
                    uint8_t code = codes[i*index->m + m];
                    dist2 += pq_dist_table[m*index->ksub + code];
                }
                if (q.size() < k) {
                    q.emplace(dist2, ids[i]);
                } else if (dist2 < q.top().first) {
                    q.pop();
                    q.emplace(dist2, ids[i]);
                }
            }
        }
    }
    
    // 4. 归并
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (auto &q : local_q) {
        while (!q.empty()) {
            auto p = q.top(); q.pop();
            if (result.size() < k) {
                result.push(p);
            } else if (p.first < result.top().first) {
                result.pop();
                result.push(p);
            }
        }
    }
    return result;
}

// OpenMP版本的重排序函数
std::vector<std::pair<float, uint32_t>> rerank_with_omp(
    const float* base,
    const float* query,
    const std::vector<std::pair<float, uint32_t>>& candidates,
    size_t d,
    size_t k,
    int num_threads)
{
    std::vector<std::pair<float, uint32_t>> result(candidates);
    
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < (int)candidates.size(); ++i) {
        uint32_t id = candidates[i].second;
        const float* vec = base + id * d;
        float dist2 = 0;
        for (size_t j = 0; j < d; ++j) {
            float diff = query[j] - vec[j];
            dist2 += diff * diff;
        }
        result[i].first = dist2;
    }
    
    // 部分排序，只保留前k个结果
    if (k < result.size()) {
        std::partial_sort(result.begin(), result.begin() + k, result.end(),
            [](auto &a, auto &b) { return a.first < b.first; });
        result.resize(k);
    } else {
        std::sort(result.begin(), result.end(),
            [](auto &a, auto &b) { return a.first < b.first; });
    }
    
    return result;
}

// OpenMP 并行版 IVFPQ 搜索并重排序
std::priority_queue<std::pair<float, uint32_t>> ivfpq_search_omp_rerank(
    const IVFPQIndex* index,
    const float* base,
    float* query,
    size_t k,               // 最终 top-k
    size_t nprobe,          // 搜索簇数
    size_t L,               // Top-L 候选池大小
    int num_threads = omp_get_max_threads())
{
    // 1. 串行：选最近 nprobe 个聚类中心 (L2 距离平方)
    struct CentDist { float dist2; uint32_t idx; };
    std::vector<CentDist> cd(index->nlist);
    for (size_t i = 0; i < index->nlist; ++i) {
        const float* cptr = index->centroids.data() + i * index->d;
        float sum2 = 0;
        for (size_t d0 = 0; d0 < index->d; ++d0) {
            float diff = query[d0] - cptr[d0];
            sum2 += diff * diff;
        }
        cd[i] = { sum2, uint32_t(i) };
    }
    if (nprobe < index->nlist) {
        std::nth_element(
            cd.begin(), cd.begin() + nprobe, cd.end(),
            [](auto &a, auto &b){ return a.dist2 < b.dist2; }
        );
    }
    std::vector<uint32_t> probe_list;
    probe_list.reserve(nprobe);
    for (size_t i = 0; i < nprobe && i < index->nlist; ++i) {
        probe_list.push_back(cd[i].idx);
    }

    // 2. 串行：构建 PQ L2 距离表
    std::vector<float> pq_dist = compute_pq_distance_table(index, query);

    // 3. 并行生成 Top-L 候选 (Approximate)
    std::vector<std::vector<std::pair<float,uint32_t>>> local_cands(num_threads);
    // 预估每线程候选量并 reserve
    size_t avg_size = 0;
    // 估算所有倒排列表元素总和
    size_t total_inv = 0;
    for (auto lid : probe_list) total_inv += index->invlists[lid].size();
    avg_size = (total_inv + num_threads - 1) / num_threads;
    for (auto &v : local_cands) v.reserve(avg_size);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        auto &cand = local_cands[tid];
        #pragma omp for schedule(dynamic, 4)
        for (int pi = 0; pi < (int)probe_list.size(); ++pi) {
            uint32_t lid = probe_list[pi];
            const auto &ids   = index->invlists[lid];
            const auto &codes = index->codes[lid];
            for (size_t i = 0; i < ids.size(); ++i) {
                float dist2 = 0;
                for (size_t m = 0; m < index->m; ++m) {
                    uint8_t code = codes[i * index->m + m];
                    dist2 += pq_dist[m * index->ksub + code];
                }
                cand.emplace_back(dist2, ids[i]);
            }
        }
    }

    // 4. 合并所有线程候选到一个大 vector
    std::vector<std::pair<float,uint32_t>> all;
    all.reserve(total_inv);
    for (auto &v : local_cands) {
        all.insert(all.end(), v.begin(), v.end());
    }

    // 5. 部分排序选取 Top-L
    size_t topL = std::min(L, all.size());
    if (all.size() > topL) {
        std::nth_element(
            all.begin(), all.begin() + topL, all.end(),
            [](auto &a, auto &b){ return a.first < b.first; }
        );
        all.resize(topL);
    }
    // 完全排序保证顺序
    std::sort(
        all.begin(), all.end(),
        [](auto &a, auto &b){ return a.first < b.first; }
    );

    // 6. 并行重排序: 对 Top-L 做精确 L2
    std::vector<std::pair<float,uint32_t>> reranked;
    reranked.reserve(std::min(k, all.size()));
    // 并行计算精确距离
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < (int)all.size(); ++i) {
        float dist2 = 0;
        uint32_t id = all[i].second;
        const float* vec = base + id * index->d;
        for (size_t d0 = 0; d0 < index->d; ++d0) {
            float diff = query[d0] - vec[d0];
            dist2 += diff * diff;
        }
        // 单线程写入结果
        #pragma omp critical
        reranked.emplace_back(dist2, id);
    }
    // 7. 部分排序取最终 top-k
    if (reranked.size() > k) {
        std::nth_element(
            reranked.begin(), reranked.begin() + k, reranked.end(),
            [](auto &a, auto &b){ return a.first < b.first; }
        );
        reranked.resize(k);
    }
    std::sort(
        reranked.begin(), reranked.end(),
        [](auto &a, auto &b){ return a.first < b.first; }
    );

    // 8. 转为优先队列输出
    std::priority_queue<std::pair<float,uint32_t>> result;
    for (auto &p : reranked) result.push(p);
    return result;
}
