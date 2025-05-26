// omp_multi_ivf.h
#ifndef OMP_MULTI_IVF_H
#define OMP_MULTI_IVF_H

#include <vector>
#include <queue>
#include <cstdint>
#include <omp.h>
#include "flat_ivf.h" // 引入串行IVF实现

// 多查询并行 - 使用OpenMP
// 每个线程分配一个或多个查询，线程内部使用串行方法处理
std::vector<std::priority_queue<std::pair<float, uint32_t>>>
multi_ivf_search_omp(
    float* base,
    float* queries,          // 多个查询向量
    size_t query_num,        // 查询向量的数量
    size_t base_number,
    size_t vecdim,
    size_t k,
    float* centroids,
    size_t nlist,
    const std::vector<std::vector<uint32_t>>& invlists,
    size_t nprobe,
    int num_threads = omp_get_max_threads())
{
    // 结果数组
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_num);
    
    // 使用OpenMP并行处理多个查询
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int q = 0; q < (int)query_num; ++q) {
        float* query = queries + q * vecdim; // 当前查询向量
        
        // 调用串行IVF搜索处理单个查询
        results[q] = flat_ivf_search(
            base, query, base_number, vecdim, k, 
            centroids, nlist, invlists, nprobe
        );
    }
    
    return results;
}

#endif // OMP_MULTI_IVF_H