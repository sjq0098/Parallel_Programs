// 使用 SIMD 优化后的 OpenMP IVFPQ 搜索和重排序实现

#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <omp.h>
#include "pq_ivf.h"  // 包含 IVFPQIndex 和 compute_pq_distance_table
#include "simd.h"    // 包含 SIMD 加速函数

// OpenMP 并行版 IVFPQ 搜索（SIMD 优化 ADC）
std::priority_queue<std::pair<float, uint32_t>> ivfpq_search_omp(
    const IVFPQIndex* index,
    float* query,
    size_t k,
    size_t nprobe,
    int num_threads = omp_get_max_threads())
{
    struct CentDist { float dist2; uint32_t idx; };
    std::vector<CentDist> cd(index->nlist);
    for (size_t i = 0; i < index->nlist; ++i) {
        const float* cptr = index->centroids.data() + i * index->d;
        float dist2 = l2_dist_avx2(query, cptr, index->d);
        cd[i] = { dist2, static_cast<uint32_t>(i) };
    }
    if (nprobe < index->nlist) {
        std::nth_element(cd.begin(), cd.begin() + nprobe, cd.end(),
                         [](auto &a, auto &b){ return a.dist2 < b.dist2; });
    }
    std::vector<uint32_t> probe_list;
    probe_list.reserve(nprobe);
    for (size_t i = 0; i < nprobe && i < index->nlist; ++i)
        probe_list.push_back(cd[i].idx);

    std::vector<float> pq_dist_table = compute_pq_distance_table(index, query);

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
                float dist2 = pq_dist_simd(pq_dist_table.data(), &codes[i * index->m], index->m, index->ksub);
                if (q.size() < k) q.emplace(dist2, ids[i]);
                else if (dist2 < q.top().first) { q.pop(); q.emplace(dist2, ids[i]); }
            }
        }
    }
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (auto &q : local_q) {
        while (!q.empty()) {
            auto p = q.top(); q.pop();
            if (result.size() < k) result.push(p);
            else if (p.first < result.top().first) { result.pop(); result.push(p); }
        }
    }
    return result;
}

// OpenMP 并行版精排函数（使用 SIMD 加速 L2）
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
        result[i].first = l2_dist_avx2(query, vec, d);
    }
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
        float sum2 = l2_dist_avx2(query, cptr, index->d);
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
    size_t avg_size = 0;
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
                float dist2 = pq_dist_simd(
                    pq_dist.data(),
                    codes.data() + i * index->m,
                    index->m,
                    index->ksub
                );
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
    std::sort(
        all.begin(), all.end(),
        [](auto &a, auto &b){ return a.first < b.first; }
    );

    // 6. 并行重排序: 对 Top-L 做精确 L2
    std::vector<std::pair<float,uint32_t>> reranked;
    reranked.reserve(std::min(k, all.size()));

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < (int)all.size(); ++i) {
        uint32_t id = all[i].second;
        const float* vec = base + id * index->d;
        float dist2 = l2_dist_avx2(query, vec, index->d);
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

