// omp_ivf.h
#ifndef OMP_IVF_H
#define OMP_IVF_H

#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <omp.h>


// openmp并行版 IVF-Flat 搜索
//   base:        所有原始向量起始地址（row-major）
//   query:       单条查询向量
//   base_number: 向量总数（未在函数体内使用，但可用于扩展或检查）
//   vecdim:      向量维度
//   k:           top-k 大小
//   centroids:   簇中心数组，长度 nlist*vecdim
//   nlist:       簇中心个数
//   invlists:    倒排列表，长度 nlist，每个列表是该簇内向量的索引
//   nprobe:      要扫描的倒排簇数量
//   num_threads: 并行线程数（默认使用所有可用核心）
// 返回：一个 priority_queue，包含 k 个最小距离 (dist, id) 对
inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_omp(float* base,
                    float* query,
                    size_t /*base_number*/,
                    size_t vecdim,
                    size_t k,
                    float* centroids,
                    size_t nlist,
                    const std::vector<std::vector<uint32_t>>& invlists,
                    size_t nprobe,
                    int num_threads = omp_get_max_threads())
{
    struct CentDist { float dist; uint32_t idx; };

    // —— 1. 串行粗筛 ——  
    std::vector<CentDist> cd(nlist);
    for (size_t i = 0; i < nlist; ++i) {
        float dot = 0.f;
        float* cptr = centroids + i * vecdim;
        for (size_t d = 0; d < vecdim; ++d) {
            dot += cptr[d] * query[d];
        }
        cd[i].dist = 1.f - dot;
        cd[i].idx  = uint32_t(i);
    }
    if (nprobe < nlist) {
        std::nth_element(
            cd.begin(), cd.begin() + nprobe, cd.end(),
            [](const CentDist &a, const CentDist &b){
                return a.dist < b.dist;
            }
        );
    }
    std::vector<uint32_t> probe_list(nprobe);
    for (size_t i = 0; i < nprobe; ++i) {
        probe_list[i] = cd[i].idx;
    }

    // —— 2. 并行精排 ——  
    // 为每个线程分配一个 local top-k 队列
    std::vector<std::priority_queue<std::pair<float,uint32_t>>> local_q(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        auto &q = local_q[tid];

        // 将 probe_list 均匀分配给各线程
        #pragma omp for schedule(static)
        for (int pi = 0; pi < (int)nprobe; ++pi) {
            uint32_t list_id = probe_list[pi];
            const auto &vec_ids = invlists[list_id];
            for (uint32_t vid : vec_ids) {
                // 计算 IP 距离
                float dot = 0.f;
                float* vptr = base + vid * vecdim;
                for (size_t d = 0; d < vecdim; ++d) {
                    dot += vptr[d] * query[d];
                }
                float dist = 1.f - dot;

                if (q.size() < k) {
                    q.emplace(dist, vid);
                } else if (dist < q.top().first) {
                    q.emplace(dist, vid);
                    q.pop();
                }
            }
        }
    }

    // —— 3. 合并 local top-k ——  
    std::priority_queue<std::pair<float,uint32_t>> global_q;
    for (int t = 0; t < num_threads; ++t) {
        auto &lq = local_q[t];
        while (!lq.empty()) {
            auto pr = lq.top(); lq.pop();
            if (global_q.size() < k) {
                global_q.push(pr);
            } else if (pr.first < global_q.top().first) {
                global_q.push(pr);
                global_q.pop();
            }
        }
    }

    return global_q;
}

#endif // OMP_IVF_H
