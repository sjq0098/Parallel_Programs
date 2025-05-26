// ptd_multi_ivf.h
#ifndef PTD_MULTI_IVF_H
#define PTD_MULTI_IVF_H

#include <vector>
#include <queue>
#include <cstdint>
#include <thread>
#include <atomic>
#include "flat_ivf.h" // 引入串行IVF实现

// 线程参数结构体
struct MultiQueryThreadArgs {
    float* base;
    float* queries;              // 所有查询向量
    size_t base_number;
    size_t vecdim;
    size_t k;
    float* centroids;
    size_t nlist;
    const std::vector<std::vector<uint32_t>>* invlists;
    size_t nprobe;
    size_t query_num;           // 查询总数
    std::vector<std::priority_queue<std::pair<float, uint32_t>>>* results; // 结果数组
    std::atomic<size_t>* next_query;  // 原子变量，用于动态分配任务
};

// 线程函数
void* multi_query_thread_fn(void* _args) {
    auto* args = static_cast<MultiQueryThreadArgs*>(_args);
    
    // 动态获取下一个查询索引
    size_t q;
    while ((q = args->next_query->fetch_add(1)) < args->query_num) {
        float* query = args->queries + q * args->vecdim;
        
        // 使用串行IVF搜索处理单个查询
        (*args->results)[q] = flat_ivf_search(
            args->base, query, args->base_number, args->vecdim, args->k,
            args->centroids, args->nlist, *args->invlists, args->nprobe
        );
    }
    
    return nullptr;
}

// 多查询并行 - 使用pthread
// 每个线程动态获取查询任务，线程内部使用串行方法处理
std::vector<std::priority_queue<std::pair<float, uint32_t>>>
multi_ivf_search_ptd(
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
    int num_threads)
{
    // 结果数组
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_num);
    
    // 原子计数器，用于动态分配任务
    std::atomic<size_t> next_query(0);
    
    // 创建线程参数
    MultiQueryThreadArgs args;
    args.base = base;
    args.queries = queries;
    args.base_number = base_number;
    args.vecdim = vecdim;
    args.k = k;
    args.centroids = centroids;
    args.nlist = nlist;
    args.invlists = &invlists;
    args.nprobe = nprobe;
    args.query_num = query_num;
    args.results = &results;
    args.next_query = &next_query;
    
    // 创建线程
    std::vector<std::thread> threads(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads[t] = std::thread(multi_query_thread_fn, &args);
    }
    
    // 等待所有线程完成
    for (auto& th : threads) {
        th.join();
    }
    
    return results;
}

#endif // PTD_MULTI_IVF_H