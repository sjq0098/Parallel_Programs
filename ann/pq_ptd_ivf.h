#pragma once
#include <thread>
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include "pq_ivf.h" // 引入基本的IVFPQ索引定义

// 线程参数结构体
struct IVFPQThreadArgs {
    // 输入
    const IVFPQIndex* index;
    const std::vector<float>* pq_dist_table;
    const std::vector<uint32_t>* probe_list;
    size_t start_idx;
    size_t end_idx;
    size_t k;
    size_t per_list_cap;         // 每个簇的候选上限
    size_t per_thread_cap;       // 每个线程的候选上限
    bool do_rerank;
    
    // 输出 - 使用vector存储候选项，避免堆操作开销
    std::vector<std::pair<float, uint32_t>> candidates;
};

// 线程函数
void* ivfpq_thread_fn(void* _args) {
    IVFPQThreadArgs* args = (IVFPQThreadArgs*)_args;
    const auto& probe_list = *args->probe_list;
    const auto& pq_dist_table = *args->pq_dist_table;
    const auto* index = args->index;
    
    // 线程本地map: list_id -> heap(容量为per_list_cap)
    using ListHeap = std::priority_queue<std::pair<float, uint32_t>,
                                       std::vector<std::pair<float, uint32_t>>,
                                       std::less<std::pair<float, uint32_t>>>;
    std::unordered_map<uint32_t, ListHeap> local_list_heaps;
    
    // 第一阶段：每个簇局部筛选top-per_list_cap
    for (size_t pi = args->start_idx; pi < args->end_idx; ++pi) {
        uint32_t list_id = probe_list[pi];
        const auto& ids = index->invlists[list_id];
        const auto& codes = index->codes[list_id];
        
        // 确保map中有对应的heap
        if (local_list_heaps.find(list_id) == local_list_heaps.end()) {
            local_list_heaps.emplace(list_id, ListHeap());
        }
        
        auto& list_heap = local_list_heaps[list_id];
        
        for (size_t i = 0; i < ids.size(); ++i) {
            uint32_t id = ids[i];
            float sum_dist = 0;
            
            // 根据PQ编码查表计算近似距离，确保使用内积度量
            for (size_t m = 0; m < index->m; ++m) {
                uint8_t code = codes[i * index->m + m];
                sum_dist += pq_dist_table[m * index->ksub + code];
            }
            
            // 将IP距离转换为距离度量 (1-dot)，确保与后续精排保持一致
            float dist = 1.f - sum_dist;
            
            // 维护每个簇的局部top-per_list_cap
            if (list_heap.size() < args->per_list_cap) {
                list_heap.emplace(dist, id);
            } else if (dist < list_heap.top().first) {
                list_heap.pop();
                list_heap.emplace(dist, id);
            }
        }
    }
    
    // 第二阶段：线程合并其处理的所有簇候选
    std::vector<std::pair<float, uint32_t>> local_all_candidates;
    size_t total_local_candidates = 0;
    
    for (auto& kv : local_list_heaps) {
        total_local_candidates += kv.second.size();
    }
    local_all_candidates.reserve(total_local_candidates);
    
    for (auto& kv : local_list_heaps) {
        auto& list_heap = kv.second;
        while (!list_heap.empty()) {
            local_all_candidates.push_back(list_heap.top());
            list_heap.pop();
        }
    }
    
    // 线程局部筛选top-per_thread_cap
    auto cmp = [](const std::pair<float, uint32_t>& a, 
                  const std::pair<float, uint32_t>& b) {
        return a.first > b.first; // 小值优先
    };
    
    std::priority_queue<std::pair<float, uint32_t>,
                       std::vector<std::pair<float, uint32_t>>,
                       decltype(cmp)> thread_heap(cmp);
    
    for (const auto& cand : local_all_candidates) {
        if (thread_heap.size() < args->per_thread_cap) {
            thread_heap.push(cand);
        } else if (cand.first < thread_heap.top().first) {
            thread_heap.pop();
            thread_heap.push(cand);
        }
    }
    
    // 存储线程筛选结果
    args->candidates.reserve(thread_heap.size());
    while (!thread_heap.empty()) {
        args->candidates.push_back(thread_heap.top());
        thread_heap.pop();
    }
    
    return nullptr;
}

// 重排序线程参数结构体
struct RerankThreadArgs {
    // 输入
    const IVFPQIndex* index;
    const float* query;
    const std::vector<std::pair<float, uint32_t>>* candidates;
    size_t start_idx;
    size_t end_idx;
    size_t k;
    
    // 输出
    std::priority_queue<std::pair<float, uint32_t>> results;
};

// 重排序线程函数
void* rerank_thread_fn(void* _args) {
    RerankThreadArgs* args = (RerankThreadArgs*)_args;
    const auto& candidates = *args->candidates;
    const auto* index = args->index;
    const float* query = args->query;
    
    for (size_t i = args->start_idx; i < args->end_idx; ++i) {
        uint32_t id = candidates[i].second;
        const float* vec = index->raw_data.data() + id * index->d;
        
        // 计算精确内积，保持与PQ距离相同的度量
        float exact_ip = inner_product(query, vec, index->d);
        float exact_dist = 1.0f - exact_ip; // 保持使用1-dot作为距离度量
        
        if (args->results.size() < args->k) {
            args->results.emplace(exact_dist, id);
        } else if (exact_dist < args->results.top().first) {
            args->results.pop();
            args->results.emplace(exact_dist, id);
        }
    }
    
    return nullptr;
}

// pthread并行版IVFPQ搜索（带重排序选项）
std::priority_queue<std::pair<float, uint32_t>> ivfpq_search_ptd(
    const IVFPQIndex* index,
    float* query,
    size_t k,
    size_t nprobe,
    int num_threads,
    bool do_rerank = false,
    size_t rerank_candidates = 0)  // 改为0，表示自动计算
{
    // 动态计算重排序候选数量，如果未指定则使用 k * nprobe * 8
    if (rerank_candidates == 0) {
        rerank_candidates = k * nprobe * 8;
    }
    
    // 1. 找出与查询最近的nprobe个聚类中心（串行执行）
    struct CentDist { float dist; uint32_t idx; };
    std::vector<CentDist> cd(index->nlist);
    
    for (size_t i = 0; i < index->nlist; ++i) {
        const float* cptr = index->centroids.data() + i * index->d;
        float dot = 0;
        
        for (size_t d = 0; d < index->d; ++d) {
            dot += cptr[d] * query[d];
        }
        
        cd[i].dist = 1.f - dot;
        cd[i].idx = uint32_t(i);
    }
    
    if (nprobe < index->nlist) {
        std::nth_element(
            cd.begin(), cd.begin() + nprobe, cd.end(),
            [](const CentDist &a, const CentDist &b) {
                return a.dist < b.dist;
            }
        );
    }
    
    // 2. 计算PQ与查询向量的距离表（串行执行）
    std::vector<float> pq_dist_table = compute_pq_distance_table(index, query);
    
    // 准备nprobe个倒排列表的ID
    std::vector<uint32_t> probe_list(nprobe);
    for (size_t i = 0; i < nprobe && i < index->nlist; ++i) {
        probe_list[i] = cd[i].idx;
    }
    
    // 3. 并行处理倒排列表
    std::vector<IVFPQThreadArgs> thread_args(num_threads);
    std::vector<std::thread> threads(num_threads);
    
    // 均匀分配任务
    size_t per_thread = (probe_list.size() + num_threads - 1) / num_threads;
    
    // 计算每个簇的候选上限和每个线程的候选上限
    size_t per_list_cap = (rerank_candidates + nprobe - 1) / nprobe;
    size_t per_thread_cap = (rerank_candidates + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
        auto& arg = thread_args[t];
        arg.index = index;
        arg.pq_dist_table = &pq_dist_table;
        arg.probe_list = &probe_list;
        arg.k = k;
        arg.per_list_cap = per_list_cap;
        arg.per_thread_cap = per_thread_cap;
        arg.do_rerank = do_rerank && index->has_raw_data;
        arg.start_idx = std::min(static_cast<size_t>(t * per_thread), probe_list.size());
        arg.end_idx = std::min(static_cast<size_t>((t + 1) * per_thread), probe_list.size());
        
        // 启动线程
        threads[t] = std::thread(ivfpq_thread_fn, &arg);
    }
    
    // 等待所有线程完成
    for (int t = 0; t < num_threads; ++t) {
        threads[t].join();
    }
    
    // 4. 收集所有候选项并全局筛选
    std::vector<std::pair<float, uint32_t>> all_candidates;
    size_t total_candidates = 0;
    for (const auto& arg : thread_args) {
        total_candidates += arg.candidates.size();
    }
    all_candidates.reserve(total_candidates);
    
    for (auto& arg : thread_args) {
        all_candidates.insert(all_candidates.end(), 
                             arg.candidates.begin(), 
                             arg.candidates.end());
        // 释放内存
        std::vector<std::pair<float, uint32_t>>().swap(arg.candidates);
    }
    
    // 全局筛选top-rerank_candidates
    std::vector<std::pair<float, uint32_t>> final_candidates;
    
    if (do_rerank && index->has_raw_data && !all_candidates.empty()) {
        // 全局筛选，使用min-heap(greater比较函数)
        auto cmp = [](const std::pair<float, uint32_t>& a, 
                      const std::pair<float, uint32_t>& b) {
            return a.first > b.first; // 小值优先
        };
        
        std::priority_queue<std::pair<float, uint32_t>,
                           std::vector<std::pair<float, uint32_t>>,
                           decltype(cmp)> global_heap(cmp);
        
        for (const auto& cand : all_candidates) {
            if (global_heap.size() < rerank_candidates) {
                global_heap.push(cand);
            } else if (cand.first < global_heap.top().first) {
                global_heap.pop();
                global_heap.push(cand);
            }
        }
        
        // 转换为vector并排序
        final_candidates.reserve(global_heap.size());
        while (!global_heap.empty()) {
            final_candidates.push_back(global_heap.top());
            global_heap.pop();
        }
        
        // 按近似距离排序（从小到大）
        std::sort(final_candidates.begin(), final_candidates.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // 5. 进行重排序
        std::vector<RerankThreadArgs> rerank_args(num_threads);
        std::vector<std::thread> rerank_threads(num_threads);
        
        // 均匀分配候选项给线程
        size_t candidates_per_thread = (final_candidates.size() + num_threads - 1) / num_threads;
        
        for (int t = 0; t < num_threads; ++t) {
            auto& arg = rerank_args[t];
            arg.index = index;
            arg.query = query;
            arg.candidates = &final_candidates;
            arg.k = k;
            arg.start_idx = std::min(static_cast<size_t>(t * candidates_per_thread), final_candidates.size());
            arg.end_idx = std::min(static_cast<size_t>((t + 1) * candidates_per_thread), final_candidates.size());
            
            // 启动线程
            rerank_threads[t] = std::thread(rerank_thread_fn, &arg);
        }
        
        // 等待所有重排序线程完成
        for (int t = 0; t < num_threads; ++t) {
            rerank_threads[t].join();
        }
        
        // 合并所有线程的结果
        std::priority_queue<std::pair<float, uint32_t>> final_result;
        for (auto& arg : rerank_args) {
            while (!arg.results.empty()) {
                auto pr = arg.results.top();
                arg.results.pop();
                
                if (final_result.size() < k) {
                    final_result.push(pr);
                } else if (pr.first < final_result.top().first) {
                    final_result.pop();
                    final_result.push(pr);
                }
            }
        }
        
        return final_result;
    } 
    else {
        // 不进行重排序，直接使用PQ距离
        // 如果候选数量超过k，使用部分排序找出前k个
        std::priority_queue<std::pair<float, uint32_t>> result;
        
        if (all_candidates.size() > k) {
            std::nth_element(
                all_candidates.begin(), 
                all_candidates.begin() + k, 
                all_candidates.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; }
            );
            // 截断至k个
            all_candidates.resize(k);
        }
        
        // 构建返回的优先队列
        for (const auto& cand : all_candidates) {
            result.push(cand);
        }
        
        return result;
    }
} 