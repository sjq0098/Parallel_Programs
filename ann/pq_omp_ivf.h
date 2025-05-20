#pragma once
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <omp.h>
#include <unordered_map>
#include "pq_ivf.h" // 引入基本的IVFPQ索引定义

// OpenMP并行版IVFPQ搜索（带重排序选项）
std::priority_queue<std::pair<float, uint32_t>> ivfpq_search_omp(
    const IVFPQIndex* index,
    float* query,
    size_t k,
    size_t nprobe,
    int num_threads = omp_get_max_threads(),
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
    if (do_rerank && index->has_raw_data) {
        // 每个簇保留的候选数量（确保均衡贡献）
        size_t per_list_cap = (rerank_candidates + nprobe - 1) / nprobe;
        
        // 每个线程保留的候选数量
        size_t per_thread_cap = (rerank_candidates + num_threads - 1) / num_threads;
        
        // 使用vector存储线程处理结果
        std::vector<std::vector<std::pair<float, uint32_t>>> thread_candidates(num_threads);
        
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            
            // 线程本地map: list_id -> heap(容量为per_list_cap)
            using ListHeap = std::priority_queue<std::pair<float, uint32_t>,
                                               std::vector<std::pair<float, uint32_t>>,
                                               std::less<std::pair<float, uint32_t>>>;
            std::unordered_map<uint32_t, ListHeap> local_list_heaps;
            
            // 第一阶段：每线程每簇筛选top-per_list_cap
            #pragma omp for schedule(dynamic)
            for (int pi = 0; pi < static_cast<int>(probe_list.size()); ++pi) {
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
                    if (list_heap.size() < per_list_cap) {
                        list_heap.emplace(dist, id);
                    } else if (dist < list_heap.top().first) {
                        list_heap.pop();
                        list_heap.emplace(dist, id);
                    }
                }
            }
            
            // 第二阶段：每线程合并其处理的所有簇的候选
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
                if (thread_heap.size() < per_thread_cap) {
                    thread_heap.push(cand);
                } else if (cand.first < thread_heap.top().first) {
                    thread_heap.pop();
                    thread_heap.push(cand);
                }
            }
            
            // 存储线程筛选结果
            auto& thread_result = thread_candidates[tid];
            thread_result.reserve(thread_heap.size());
            
            while (!thread_heap.empty()) {
                thread_result.push_back(thread_heap.top());
                thread_heap.pop();
            }
        }
        
        // 第三阶段：主线程合并所有线程候选并全局筛选
        std::vector<std::pair<float, uint32_t>> all_candidates;
        size_t total_candidates = 0;
        
        for (const auto& thread_cands : thread_candidates) {
            total_candidates += thread_cands.size();
        }
        all_candidates.reserve(total_candidates);
        
        for (const auto& thread_cands : thread_candidates) {
            all_candidates.insert(all_candidates.end(), 
                                 thread_cands.begin(), 
                                 thread_cands.end());
        }
        
        // 全局筛选top-rerank_candidates
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
        std::vector<std::pair<float, uint32_t>> final_candidates;
        final_candidates.reserve(global_heap.size());
        
        while (!global_heap.empty()) {
            final_candidates.push_back(global_heap.top());
            global_heap.pop();
        }
        
        // 按近似距离排序（从小到大）
        std::sort(final_candidates.begin(), final_candidates.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // 第四阶段：使用OpenMP并行计算精确距离
        std::priority_queue<std::pair<float, uint32_t>> global_result;
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> rerank_results(num_threads);
        
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            auto& local_q = rerank_results[tid];
            
            #pragma omp for schedule(static)
            for (int i = 0; i < static_cast<int>(final_candidates.size()); ++i) {
                uint32_t id = final_candidates[i].second;
                const float* vec = index->raw_data.data() + id * index->d;
                
                // 计算精确内积，保持与PQ距离相同的度量
                float exact_ip = inner_product(query, vec, index->d);
                float exact_dist = 1.0f - exact_ip; // 保持使用1-dot作为距离度量
                
                if (local_q.size() < k) {
                    local_q.emplace(exact_dist, id);
                } else if (exact_dist < local_q.top().first) {
                    local_q.pop();
                    local_q.emplace(exact_dist, id);
                }
            }
        }
        
        // 合并各线程的重排序结果
        for (auto &local_q : rerank_results) {
            while (!local_q.empty()) {
                auto pr = local_q.top(); 
                local_q.pop();
                
                if (global_result.size() < k) {
                    global_result.push(pr);
                } else if (pr.first < global_result.top().first) {
                    global_result.push(pr);
                    global_result.pop();
                }
            }
        }
        
        return global_result;
    } else {
        // 不进行重排序，直接使用PQ近似距离
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_results(num_threads);
        
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            auto &local_q = local_results[tid];
            
            // 将probe_list均匀分配给各线程
            #pragma omp for schedule(static)
            for (int pi = 0; pi < static_cast<int>(probe_list.size()); ++pi) {
                uint32_t list_id = probe_list[pi];
                const auto& ids = index->invlists[list_id];
                const auto& codes = index->codes[list_id];
                
                for (size_t i = 0; i < ids.size(); ++i) {
                    uint32_t id = ids[i];
                    float sum_dist = 0;
                    
                    // 根据PQ编码查表计算近似距离
                    for (size_t m = 0; m < index->m; ++m) {
                        uint8_t code = codes[i * index->m + m];
                        sum_dist += pq_dist_table[m * index->ksub + code];
                    }
                    
                    // 将IP距离转换为L2距离
                    float dist = 1.f - sum_dist;
                    
                    // 使用近似距离
                    if (local_q.size() < k) {
                        local_q.emplace(dist, id);
                    } else if (dist < local_q.top().first) {
                        local_q.pop();
                        local_q.emplace(dist, id);
                    }
                }
            }
        }
        
        // 归并各线程的局部结果
        std::priority_queue<std::pair<float, uint32_t>> global_result;
        
        for (auto &local_q : local_results) {
            while (!local_q.empty()) {
                auto pr = local_q.top(); 
                local_q.pop();
                
                if (global_result.size() < k) {
                    global_result.push(pr);
                } else if (pr.first < global_result.top().first) {
                    global_result.push(pr);
                    global_result.pop();
                }
            }
        }
        
        return global_result;
    }
} 