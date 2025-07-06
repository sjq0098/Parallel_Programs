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
#include <unordered_map>
#include <random>
#include "hnswlib/hnswlib/hnswlib.h"

// HNSW+IVF混合索引的MPI实现（先HNSW定位，再IVF精化）
class MPIHNSWIVFIndex {
private:
    int rank, size;
    size_t dim, nlist;
    std::vector<float> centroids;                          // IVF聚类中心
    std::vector<uint32_t> point_to_cluster;              // 点到簇的映射
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> global_hnsw; // 全局HNSW索引
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    
    // HNSW参数
    size_t M = 16;
    size_t efConstruction = 200;
    size_t efSearch = 100;
    size_t hnsw_candidate_factor = 20;  // HNSW搜索候选倍数

public:
    MPIHNSWIVFIndex(size_t dimension, size_t num_lists, size_t hnsw_M = 16, size_t hnsw_efC = 200)
        : dim(dimension), nlist(num_lists), M(hnsw_M), efConstruction(hnsw_efC) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        centroids.resize(nlist * dim);
        space.reset(new hnswlib::InnerProductSpace(dim));
    }

    // 生成随机聚类中心
    void generate_centroids() {
        if (rank == 0) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dis(0.0f, 1.0f);
            
            for (size_t i = 0; i < centroids.size(); ++i) {
                centroids[i] = dis(gen);
            }
        }
        // 广播聚类中心到所有进程
        MPI_Bcast(centroids.data(), centroids.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // 设置HNSW搜索参数
    void setEfSearch(size_t ef) {
        efSearch = ef;
        if (global_hnsw) {
            global_hnsw->setEf(ef);
        }
    }

    // 设置HNSW候选倍数
    void setHNSWCandidateFactor(size_t factor) {
        hnsw_candidate_factor = factor;
    }

    // 构建HNSW+IVF混合索引
    void build_index(const float* base_data, size_t n) {
        // 步骤1：生成IVF聚类中心
        generate_centroids();

        // 步骤2：计算每个点所属的聚类
        assign_points_to_clusters(base_data, n);

        // 步骤3：构建全局HNSW索引
        build_global_hnsw(base_data, n);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "MPI HNSW+IVF index building completed on " << size << " processes" << std::endl;
        }
    }

private:
    // 将点分配到聚类
    void assign_points_to_clusters(const float* base_data, size_t n) {
        point_to_cluster.resize(n);
        
        // 每个进程处理一部分数据
        size_t elements_per_proc = (n + size - 1) / size;
        size_t start_idx = rank * elements_per_proc;
        size_t end_idx = std::min(start_idx + elements_per_proc, n);
        
        // 本地分配
        std::vector<uint32_t> local_assignments(n, 0);
        
        for (size_t i = start_idx; i < end_idx; ++i) {
            const float* vec = base_data + i * dim;
            
            // 找到最近的聚类中心（使用内积距离）
            uint32_t best_cluster = 0;
            float best_dist = std::numeric_limits<float>::lowest();
            
            for (size_t c = 0; c < nlist; ++c) {
                const float* centroid = centroids.data() + c * dim;
                float dot = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    dot += vec[d] * centroid[d];
                }
                if (dot > best_dist) {  // 内积越大越好
                    best_dist = dot;
                    best_cluster = c;
                }
            }
            
            local_assignments[i] = best_cluster;
        }
        
        // 汇总所有进程的分配结果
        MPI_Allreduce(local_assignments.data(), point_to_cluster.data(), 
                      n, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    }

    // 构建全局HNSW索引
    void build_global_hnsw(const float* base_data, size_t n) {
        if (rank == 0) {
            // 只在主进程上构建HNSW索引
            global_hnsw.reset(new hnswlib::HierarchicalNSW<float>(
                space.get(), n, M, efConstruction));
            
            // 添加第一个点
            global_hnsw->addPoint(base_data, 0);
            
            // 并行添加剩余点
            #pragma omp parallel for
            for (size_t i = 1; i < n; ++i) {
                global_hnsw->addPoint(base_data + i * dim, i);
            }
            
            global_hnsw->setEf(efSearch);
        }
        
        // 广播HNSW索引构建完成信号
        MPI_Barrier(MPI_COMM_WORLD);
    }

public:
    // MPI并行搜索
    std::priority_queue<std::pair<float, uint32_t>>
    mpi_search(const float* query, const float* base_data, size_t k, size_t nprobe = 16) {
        std::priority_queue<std::pair<float, uint32_t>> result;
        
        if (rank == 0) {
            // 步骤1：使用HNSW搜索大量候选点
            size_t hnsw_k = k * hnsw_candidate_factor;
            auto hnsw_result = global_hnsw->searchKnn(query, hnsw_k);
            
            std::vector<uint32_t> hnsw_candidates;
            while (!hnsw_result.empty()) {
                hnsw_candidates.push_back(hnsw_result.top().second);
                hnsw_result.pop();
            }
            
            // 步骤2：计算查询向量到各聚类中心的距离
            std::vector<std::pair<float, uint32_t>> cluster_dists(nlist);
            for (size_t i = 0; i < nlist; ++i) {
                const float* centroid = centroids.data() + i * dim;
                float dot = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    dot += query[d] * centroid[d];
                }
                cluster_dists[i] = {-dot, i};  // 负内积作为距离，用于排序
            }
            
            // 选择最近的nprobe个聚类
            std::partial_sort(cluster_dists.begin(), 
                             cluster_dists.begin() + nprobe,
                             cluster_dists.end());
            
            std::unordered_set<uint32_t> selected_clusters;
            for (size_t i = 0; i < nprobe; ++i) {
                selected_clusters.insert(cluster_dists[i].second);
            }
            
            // 步骤3：过滤HNSW候选点，只保留属于选定聚类的点
            std::vector<std::pair<float, uint32_t>> filtered_candidates;
            for (uint32_t candidate_id : hnsw_candidates) {
                if (selected_clusters.count(point_to_cluster[candidate_id]) > 0) {
                    // 计算精确距离
                    const float* candidate_vec = base_data + candidate_id * dim;
                    float dot = 0.0f;
                    for (size_t d = 0; d < dim; ++d) {
                        dot += query[d] * candidate_vec[d];
                    }
                    float dist = 1.0f - dot;  // DEEP100K标准距离
                    filtered_candidates.emplace_back(dist, candidate_id);
                }
            }
            
            // 步骤4：排序并选择top-k
            std::sort(filtered_candidates.begin(), filtered_candidates.end());
            size_t final_k = std::min(k, filtered_candidates.size());
            
            for (size_t i = 0; i < final_k; ++i) {
                result.push(filtered_candidates[i]);
            }
        }
        
        // 广播结果到所有进程
        broadcast_search_result(result, k);
        
        return result;
    }

private:
    // 广播搜索结果到所有进程
    void broadcast_search_result(std::priority_queue<std::pair<float, uint32_t>>& result, size_t k) {
        if (rank == 0) {
            // 主进程：准备数据
            std::vector<float> dists;
            std::vector<uint32_t> ids;
            
            auto temp_result = result;
            while (!temp_result.empty()) {
                dists.push_back(temp_result.top().first);
                ids.push_back(temp_result.top().second);
                temp_result.pop();
            }
            std::reverse(dists.begin(), dists.end());
            std::reverse(ids.begin(), ids.end());
            
            size_t actual_k = dists.size();
            MPI_Bcast(&actual_k, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(dists.data(), actual_k, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(ids.data(), actual_k, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        } else {
            // 其他进程：接收数据
            size_t actual_k;
            MPI_Bcast(&actual_k, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            
            std::vector<float> dists(actual_k);
            std::vector<uint32_t> ids(actual_k);
            MPI_Bcast(dists.data(), actual_k, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(ids.data(), actual_k, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            
            // 重建结果队列
            result = std::priority_queue<std::pair<float, uint32_t>>();
            for (size_t i = 0; i < actual_k; ++i) {
                result.emplace(dists[i], ids[i]);
            }
        }
    }

public:
    // 获取索引统计信息
    void print_index_stats() {
        if (rank == 0) {
            std::cout << "HNSW+IVF Index Statistics:" << std::endl;
            std::cout << "Number of IVF clusters: " << nlist << std::endl;
            std::cout << "HNSW parameters: M=" << M << ", efC=" << efConstruction << ", efS=" << efSearch << std::endl;
            std::cout << "HNSW candidate factor: " << hnsw_candidate_factor << std::endl;
            
            // 统计聚类分布
            std::vector<size_t> cluster_sizes(nlist, 0);
            for (size_t i = 0; i < point_to_cluster.size(); ++i) {
                cluster_sizes[point_to_cluster[i]]++;
            }
            
            size_t min_size = *std::min_element(cluster_sizes.begin(), cluster_sizes.end());
            size_t max_size = *std::max_element(cluster_sizes.begin(), cluster_sizes.end());
            size_t total_points = point_to_cluster.size();
            
            std::cout << "Cluster distribution - Min: " << min_size 
                      << ", Max: " << max_size 
                      << ", Avg: " << total_points / nlist << std::endl;
        }
    }
}; 