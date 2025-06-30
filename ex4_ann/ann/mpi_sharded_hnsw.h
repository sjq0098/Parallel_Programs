#pragma once

#include <mpi.h>
#include <omp.h>
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <unordered_set>
#include <random>
#include <numeric>
#include <limits>
#include "hnswlib/hnswlib/hnswlib.h"

// 数据分片策略枚举
enum class ShardingStrategy {
    RANDOM,           // 随机分片
    ROUND_ROBIN,      // 轮询分片
    HASH_BASED,       // 基于哈希的分片
    KMEANS_BASED      // 基于K-means的分片
};

// 分片HNSW的MPI并行实现
class MPIShardedHNSWIndex {
private:
    int rank, size;
    size_t dim;
    size_t total_elements;
    
    // HNSW参数
    size_t M = 16;
    size_t efConstruction = 200;
    size_t efSearch = 50;
    
    // 每个进程的本地HNSW索引
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> local_hnsw;
    
    // 本地数据存储
    std::vector<float> local_data;
    std::vector<uint32_t> local_to_global_id;  // 本地ID到全局ID的映射
    
    ShardingStrategy strategy;

public:
    MPIShardedHNSWIndex(size_t dimension, size_t hnsw_M = 16, size_t hnsw_efC = 200, 
                       ShardingStrategy shard_strategy = ShardingStrategy::ROUND_ROBIN)
        : dim(dimension), M(hnsw_M), efConstruction(hnsw_efC), strategy(shard_strategy) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        space.reset(new hnswlib::InnerProductSpace(dim));
    }

    // 设置HNSW搜索参数
    void setEfSearch(size_t ef) {
        efSearch = ef;
        if (local_hnsw) {
            local_hnsw->setEf(ef);
        }
    }

    // 随机分片策略
    std::vector<int> random_sharding(size_t n) {
        std::vector<int> assignments(n);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, size - 1);
        
        for (size_t i = 0; i < n; ++i) {
            assignments[i] = dis(gen);
        }
        return assignments;
    }

    // 轮询分片策略
    std::vector<int> round_robin_sharding(size_t n) {
        std::vector<int> assignments(n);
        for (size_t i = 0; i < n; ++i) {
            assignments[i] = i % size;
        }
        return assignments;
    }

    // 基于哈希的分片策略
    std::vector<int> hash_based_sharding(const float* base_data, size_t n) {
        std::vector<int> assignments(n);
        
        for (size_t i = 0; i < n; ++i) {
            // 简单的哈希函数：对向量的前几个元素求和
            uint32_t hash = 0;
            size_t hash_dims = std::min(dim, size_t(8));
            for (size_t d = 0; d < hash_dims; ++d) {
                hash += static_cast<uint32_t>(base_data[i * dim + d] * 1000);
            }
            assignments[i] = hash % size;
        }
        return assignments;
    }

    // 基于K-means的分片策略（简化版）
    std::vector<int> kmeans_based_sharding(const float* base_data, size_t n) {
        std::vector<int> assignments(n);
        
        // 简化的K-means：随机选择size个中心点
        std::vector<float> centroids(size * dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n - 1);
        
        // 随机初始化中心点
        for (int i = 0; i < size; ++i) {
            size_t random_idx = dis(gen);
            std::copy(base_data + random_idx * dim,
                     base_data + (random_idx + 1) * dim,
                     centroids.begin() + i * dim);
        }
        
        // 分配每个向量到最近的中心
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            const float* vec = base_data + i * dim;
            int best_center = 0;
            float best_dist = std::numeric_limits<float>::max();
            
            for (int c = 0; c < size; ++c) {
                const float* centroid = centroids.data() + c * dim;
                float dist = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    float diff = vec[d] - centroid[d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_center = c;
                }
            }
            assignments[i] = best_center;
        }
        
        return assignments;
    }

    // 构建分片HNSW索引
    void build_index(const float* base_data, size_t n) {
        total_elements = n;
        
        // 步骤1：根据策略分配数据到各个进程
        std::vector<int> assignments;
        
        if (rank == 0) {
            // 主进程负责数据分片
            switch (strategy) {
                case ShardingStrategy::RANDOM:
                    assignments = random_sharding(n);
                    break;
                case ShardingStrategy::ROUND_ROBIN:
                    assignments = round_robin_sharding(n);
                    break;
                case ShardingStrategy::HASH_BASED:
                    assignments = hash_based_sharding(base_data, n);
                    break;
                case ShardingStrategy::KMEANS_BASED:
                    assignments = kmeans_based_sharding(base_data, n);
                    break;
            }
        }
        
        // 广播分配结果
        assignments.resize(n);
        MPI_Bcast(assignments.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

        // 步骤2：收集本进程应该处理的数据
        std::vector<size_t> my_indices;
        for (size_t i = 0; i < n; ++i) {
            if (assignments[i] == rank) {
                my_indices.push_back(i);
            }
        }
        
        size_t local_n = my_indices.size();
        local_data.resize(local_n * dim);
        local_to_global_id.resize(local_n);
        
        // 复制本地数据
        #pragma omp parallel for
        for (size_t i = 0; i < local_n; ++i) {
            size_t global_idx = my_indices[i];
            local_to_global_id[i] = global_idx;
            std::copy(base_data + global_idx * dim,
                     base_data + (global_idx + 1) * dim,
                     local_data.begin() + i * dim);
        }

        // 步骤3：为本地数据构建HNSW索引
        if (local_n > 0) {
            local_hnsw.reset(new hnswlib::HierarchicalNSW<float>(
                space.get(), local_n, M, efConstruction));
            
            // 添加第一个点
            local_hnsw->addPoint(local_data.data(), 0);
            
            // 并行添加剩余点
            #pragma omp parallel for
            for (size_t i = 1; i < local_n; ++i) {
                local_hnsw->addPoint(local_data.data() + i * dim, i);
            }
            
            local_hnsw->setEf(efSearch);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "MPI Sharded HNSW index building completed on " << size << " processes" << std::endl;
            // 暂时注释掉统计信息输出以避免潜在的MPI死锁
            // print_sharding_stats();
        }
    }

    // MPI并行搜索
    std::priority_queue<std::pair<float, uint32_t>>
    mpi_search(const float* query, size_t k) {
        std::vector<std::pair<float, uint32_t>> local_candidates;
        
        // 步骤1：在本地HNSW索引中搜索
        if (local_hnsw && local_to_global_id.size() > 0) {
            // 搜索更多候选以提高质量
            size_t search_k = std::min(k * 3, local_to_global_id.size());
            auto local_result = local_hnsw->searchKnn(query, search_k);
            
            // 转换为全局ID
            while (!local_result.empty()) {
                auto p = local_result.top();
                local_result.pop();
                
                size_t local_id = p.second;
                uint32_t global_id = local_to_global_id[local_id];
                local_candidates.emplace_back(p.first, global_id);
            }
        }

        // 步骤2：收集所有进程的候选结果
        int local_count = local_candidates.size();
        std::vector<int> counts(size);
        MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        int total_count = 0;
        std::vector<int> displs(size);
        for (int i = 0; i < size; ++i) {
            displs[i] = total_count;
            total_count += counts[i];
        }
        
        std::vector<float> all_dists(total_count);
        std::vector<uint32_t> all_ids(total_count);
        
        // 准备本地数据
        std::vector<float> local_dists(local_count);
        std::vector<uint32_t> local_ids(local_count);
        for (int i = 0; i < local_count; ++i) {
            local_dists[i] = local_candidates[i].first;
            local_ids[i] = local_candidates[i].second;
        }
        
        MPI_Allgatherv(local_dists.data(), local_count, MPI_FLOAT,
                       all_dists.data(), counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(local_ids.data(), local_count, MPI_UNSIGNED,
                       all_ids.data(), counts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);

        // 步骤3：选择全局top-k
        std::vector<std::pair<float, uint32_t>> all_candidates;
        for (int i = 0; i < total_count; ++i) {
            all_candidates.emplace_back(all_dists[i], all_ids[i]);
        }
        
        // 排序并取top-k
        std::sort(all_candidates.begin(), all_candidates.end());
        size_t final_k = std::min(k, all_candidates.size());
        
        std::priority_queue<std::pair<float, uint32_t>> result;
        for (size_t i = 0; i < final_k; ++i) {
            result.push(all_candidates[i]);
        }
        
        return result;
    }

    // 自适应搜索策略：根据查询特征动态调整各分片的搜索深度
    std::priority_queue<std::pair<float, uint32_t>>
    mpi_adaptive_search(const float* query, size_t k) {
        // 第一轮：快速搜索获取粗略结果
        size_t initial_k = std::max(size_t(1), k / 4);
        std::vector<std::pair<float, uint32_t>> local_candidates;
        
        if (local_hnsw && local_to_global_id.size() > 0) {
            size_t search_k = std::min(initial_k, local_to_global_id.size());
            auto local_result = local_hnsw->searchKnn(query, search_k);
            
            while (!local_result.empty()) {
                auto p = local_result.top();
                local_result.pop();
                
                size_t local_id = p.second;
                uint32_t global_id = local_to_global_id[local_id];
                local_candidates.emplace_back(p.first, global_id);
            }
        }

        // 收集第一轮结果
        int local_count = local_candidates.size();
        std::vector<int> counts(size);
        MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // 简化实现：如果第一轮结果足够，直接返回
        if (std::accumulate(counts.begin(), counts.end(), 0) >= static_cast<int>(k)) {
            return mpi_search(query, k);  // 使用标准搜索
        }
        
        // 否则进行更深入的搜索
        local_candidates.clear();
        if (local_hnsw && local_to_global_id.size() > 0) {
            size_t extended_k = std::min(k * 2, local_to_global_id.size());
            auto local_result = local_hnsw->searchKnn(query, extended_k);
            
            while (!local_result.empty()) {
                auto p = local_result.top();
                local_result.pop();
                
                size_t local_id = p.second;
                uint32_t global_id = local_to_global_id[local_id];
                local_candidates.emplace_back(p.first, global_id);
            }
        }

        // 收集并返回最终结果
        local_count = local_candidates.size();
        MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        int total_count = 0;
        std::vector<int> displs(size);
        for (int i = 0; i < size; ++i) {
            displs[i] = total_count;
            total_count += counts[i];
        }
        
        std::vector<float> all_dists(total_count);
        std::vector<uint32_t> all_ids(total_count);
        
        std::vector<float> local_dists(local_count);
        std::vector<uint32_t> local_ids(local_count);
        for (int i = 0; i < local_count; ++i) {
            local_dists[i] = local_candidates[i].first;
            local_ids[i] = local_candidates[i].second;
        }
        
        MPI_Allgatherv(local_dists.data(), local_count, MPI_FLOAT,
                       all_dists.data(), counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(local_ids.data(), local_count, MPI_UNSIGNED,
                       all_ids.data(), counts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);

        std::vector<std::pair<float, uint32_t>> all_candidates;
        for (int i = 0; i < total_count; ++i) {
            all_candidates.emplace_back(all_dists[i], all_ids[i]);
        }
        
        std::sort(all_candidates.begin(), all_candidates.end());
        size_t final_k = std::min(k, all_candidates.size());
        
        std::priority_queue<std::pair<float, uint32_t>> result;
        for (size_t i = 0; i < final_k; ++i) {
            result.push(all_candidates[i]);
        }
        
        return result;
    }

    // 打印分片统计信息
    void print_sharding_stats() {
        std::vector<int> local_sizes(size);
        int my_size = local_to_global_id.size();
        MPI_Gather(&my_size, 1, MPI_INT, local_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "Sharding Statistics:" << std::endl;
            std::cout << "Strategy: ";
            switch (strategy) {
                case ShardingStrategy::RANDOM: std::cout << "Random"; break;
                case ShardingStrategy::ROUND_ROBIN: std::cout << "Round Robin"; break;
                case ShardingStrategy::HASH_BASED: std::cout << "Hash Based"; break;
                case ShardingStrategy::KMEANS_BASED: std::cout << "K-means Based"; break;
            }
            std::cout << std::endl;
            
            for (int i = 0; i < size; ++i) {
                std::cout << "Process " << i << ": " << local_sizes[i] << " vectors" << std::endl;
            }
            
            // 计算负载均衡度
            double mean = static_cast<double>(total_elements) / size;
            double variance = 0.0;
            for (int i = 0; i < size; ++i) {
                double diff = local_sizes[i] - mean;
                variance += diff * diff;
            }
            variance /= size;
            double std_dev = std::sqrt(variance);
            
            std::cout << "Load balance - Mean: " << mean 
                      << ", Std Dev: " << std_dev 
                      << ", CV: " << (std_dev / mean) << std::endl;
        }
    }

    // 获取本地索引大小
    size_t get_local_size() const {
        return local_to_global_id.size();
    }
}; 