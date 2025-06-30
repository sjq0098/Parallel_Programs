#pragma once

#include <mpi.h>
#include <vector>
#include <queue>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <memory>
#include <iostream>
#include <unordered_set>

#include "hnswlib/hnswlib/hnswlib.h"

// MPI并行化的HNSW索引管理类
class MPIHNSWIndex {
private:
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg;
    size_t dim;
    size_t max_elements;
    int rank, size;
    
public:
    MPIHNSWIndex(size_t dimension, const std::string& index_path, size_t max_elements_in_index)
        : dim(dimension), max_elements(max_elements_in_index)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        space.reset(new hnswlib::InnerProductSpace(dim));
        alg.reset(new hnswlib::HierarchicalNSW<float>(
            space.get(),
            index_path,
            false,  // 非 nmslib 格式
            max_elements,
            false   // 不替换已删除元素
        ));
    }

    // 设置搜索参数
    void setEf(size_t ef) {
        alg->setEf(ef);
    }

    // MPI并行化的KNN搜索
    std::priority_queue<std::pair<float, uint32_t>> mpi_hnsw_search(
        const float* query,
        size_t k) const
    {
        // 每个进程执行搜索，获取局部结果
        std::priority_queue<std::pair<float, hnswlib::labeltype>> local_result = 
            alg->searchKnn(query, k * 2); // 每个进程多搜索一些候选

        // 将局部结果转换为可传输的格式
        std::vector<float> local_dists;
        std::vector<uint32_t> local_ids;
        
        while (!local_result.empty()) {
            auto p = local_result.top();
            local_result.pop();
            local_dists.push_back(p.first);
            local_ids.push_back(static_cast<uint32_t>(p.second));
        }
        
        size_t local_count = local_dists.size();
        
        // 收集所有进程的结果数量
        std::vector<int> counts(size);
        int local_count_int = static_cast<int>(local_count);
        MPI_Allgather(&local_count_int, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // 计算位移
        std::vector<int> displs(size);
        int total_count = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = total_count;
            total_count += counts[i];
        }
        
        // 收集所有距离和ID
        std::vector<float> all_dists(total_count);
        std::vector<uint32_t> all_ids(total_count);
        
        MPI_Allgatherv(local_dists.data(), local_count_int, MPI_FLOAT,
                       all_dists.data(), counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(local_ids.data(), local_count_int, MPI_UNSIGNED,
                       all_ids.data(), counts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
        
        // 去重并选择top-k
        std::unordered_set<uint32_t> seen;
        std::vector<std::pair<float, uint32_t>> candidates;
        
        for (int i = 0; i < total_count; ++i) {
            if (seen.find(all_ids[i]) == seen.end()) {
                candidates.emplace_back(all_dists[i], all_ids[i]);
                seen.insert(all_ids[i]);
            }
        }
        
        // 按距离排序并取top-k
        std::sort(candidates.begin(), candidates.end());
        size_t final_k = std::min(k, candidates.size());
        
        std::priority_queue<std::pair<float, uint32_t>> final_result;
        for (size_t i = 0; i < final_k; ++i) {
            final_result.push(candidates[i]);
        }
        
        return final_result;
    }
    
    // 分布式构建索引
    void mpi_build_index(float* base, size_t base_number, size_t vecdim) {
        // 将数据分配给各个进程
        size_t elements_per_proc = (base_number + size - 1) / size;
        size_t start_idx = rank * elements_per_proc;
        size_t end_idx = std::min(start_idx + elements_per_proc, base_number);
        
        // 每个进程添加自己负责的数据点
        for (size_t i = start_idx; i < end_idx; ++i) {
            alg->addPoint(base + i * vecdim, i);
        }
        
        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "MPI HNSW index building completed on " << size << " processes" << std::endl;
        }
    }
};