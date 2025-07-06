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
#include <chrono>
#include "hnswlib/hnswlib/hnswlib.h"

// 轻量级IVF+HNSW混合索引的MPI实现（用于对比测试）
class LightweightMPIIVFHNSWIndex {
private:
    int rank, size;
    size_t dim, nlist;
    std::vector<float> centroids;                    // 聚类中心
    std::vector<std::vector<uint32_t>> invlists;    // 倒排列表
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    const float* base_data_ptr;  // 基础数据指针
    size_t n_points;             // 数据点数量
    
    // HNSW参数
    size_t M = 16;
    size_t efConstruction = 200;
    size_t efSearch = 50;

public:
    LightweightMPIIVFHNSWIndex(size_t dimension, size_t num_lists, size_t hnsw_M = 16, size_t hnsw_efC = 200)
        : dim(dimension), nlist(num_lists), M(hnsw_M), efConstruction(hnsw_efC),
          base_data_ptr(nullptr), n_points(0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        centroids.resize(nlist * dim);
        invlists.resize(nlist);
        space.reset(new hnswlib::InnerProductSpace(dim));
    }

    // 加载预训练的聚类中心
    bool load_centroids(const std::string& file) {
        if (rank == 0) {
            std::ifstream in(file, std::ios::binary);
            if (!in) return false;
            in.read(reinterpret_cast<char*>(centroids.data()),
                    centroids.size() * sizeof(float));
        }
        MPI_Bcast(centroids.data(), centroids.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        return true;
    }

    // 设置HNSW搜索参数
    void setEfSearch(size_t ef) {
        efSearch = ef;
    }

    // 轻量级索引构建（避免为每个簇构建HNSW）
    void build_index(const float* base_data, size_t n) {
        base_data_ptr = base_data;
        n_points = n;
        
        if (rank == 0) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // 简化的IVF分配：只在rank 0上进行
            for (size_t i = 0; i < n; ++i) {
                const float* vec = base_data + i * dim;
                
                // 找到最近的聚类中心
                uint32_t best_cluster = 0;
                float best_dist = std::numeric_limits<float>::max();
                
                for (size_t c = 0; c < nlist; ++c) {
                    float dist = 0.0f;
                    const float* centroid = centroids.data() + c * dim;
                    for (size_t d = 0; d < dim; ++d) {
                        float diff = vec[d] - centroid[d];
                        dist += diff * diff;
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }
                
                invlists[best_cluster].push_back(i);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            std::cout << "轻量级IVF索引构建完成，耗时: " << duration << "ms" << std::endl;
        }
        
        // 广播倒排列表到所有进程
        broadcast_invlists();
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "MPI IVF+HNSW index building completed on " << size << " processes" << std::endl;
        }
    }

private:
    // 广播倒排列表
    void broadcast_invlists() {
        // 广播每个簇的大小
        std::vector<int> cluster_sizes(nlist);
        if (rank == 0) {
            for (size_t i = 0; i < nlist; ++i) {
                cluster_sizes[i] = invlists[i].size();
            }
        }
        MPI_Bcast(cluster_sizes.data(), nlist, MPI_INT, 0, MPI_COMM_WORLD);
        
        // 广播每个簇的内容
        for (size_t i = 0; i < nlist; ++i) {
            if (cluster_sizes[i] > 0) {
                if (rank != 0) {
                    invlists[i].resize(cluster_sizes[i]);
                }
                MPI_Bcast(invlists[i].data(), cluster_sizes[i], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            }
        }
    }

public:
    // 轻量级并行搜索（使用暴力搜索代替HNSW）
    std::priority_queue<std::pair<float, uint32_t>>
    mpi_search(const float* query, size_t k, size_t nprobe = 16) {
        // 步骤1：选择最近的nprobe个聚类
        std::vector<std::pair<float, uint32_t>> cluster_dists(nlist);
        for (size_t i = 0; i < nlist; ++i) {
            float dist = 0.0f;
            const float* centroid = centroids.data() + i * dim;
            for (size_t d = 0; d < dim; ++d) {
                float diff = query[d] - centroid[d];
                dist += diff * diff;
            }
            cluster_dists[i] = {dist, i};
        }
        
        std::partial_sort(cluster_dists.begin(), 
                         cluster_dists.begin() + nprobe,
                         cluster_dists.end());

        // 步骤2：MPI并行搜索 - 每个进程负责部分簇
        std::vector<std::pair<float, uint32_t>> local_candidates;
        
        for (size_t i = rank; i < nprobe; i += size) {
            uint32_t cluster_id = cluster_dists[i].second;
            
            if (!invlists[cluster_id].empty()) {
                // 对该簇中的所有点计算精确距离（暴力搜索）
                const auto& vec_ids = invlists[cluster_id];
                for (uint32_t vid : vec_ids) {
                    const float* vec = base_data_ptr + vid * dim;
                    float dot = 0.0f;
                    for (size_t d = 0; d < dim; ++d) {
                        dot += query[d] * vec[d];
                    }
                    float dist = 1.0f - dot;  // DEEP100K标准距离
                    local_candidates.emplace_back(dist, vid);
                }
            }
        }

        // 步骤3：收集所有进程的候选结果
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

        // 步骤4：去重并选择top-k
        std::unordered_set<uint32_t> seen;
        std::vector<std::pair<float, uint32_t>> unique_candidates;
        
        for (int i = 0; i < total_count; ++i) {
            if (seen.find(all_ids[i]) == seen.end()) {
                unique_candidates.emplace_back(all_dists[i], all_ids[i]);
                seen.insert(all_ids[i]);
            }
        }
        
        // 按距离排序并取top-k
        std::sort(unique_candidates.begin(), unique_candidates.end());
        size_t final_k = std::min(k, unique_candidates.size());
        
        std::priority_queue<std::pair<float, uint32_t>> result;
        for (size_t i = 0; i < final_k; ++i) {
            result.push(unique_candidates[i]);
        }
        
        return result;
    }

    // 获取索引统计信息
    void print_index_stats() {
        if (rank == 0) {
            std::cout << "轻量级IVF+HNSW索引统计:" << std::endl;
            std::cout << "聚类数: " << nlist << std::endl;
            
            size_t total_vectors = 0;
            size_t non_empty_clusters = 0;
            
            for (size_t i = 0; i < nlist; ++i) {
                if (!invlists[i].empty()) {
                    non_empty_clusters++;
                    total_vectors += invlists[i].size();
                }
            }
            
            std::cout << "非空聚类数: " << non_empty_clusters << std::endl;
            std::cout << "总向量数: " << total_vectors << std::endl;
            std::cout << "平均每簇向量数: " 
                      << (non_empty_clusters > 0 ? total_vectors / non_empty_clusters : 0) << std::endl;
        }
    }
}; 