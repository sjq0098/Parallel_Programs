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

// 优化版HNSW+IVF混合索引的MPI实现
class OptimizedMPIHNSWIVFIndex {
private:
    int rank, size;
    size_t dim, nlist;
    std::vector<float> centroids;                          
    std::vector<uint32_t> point_to_cluster;              
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> global_hnsw; 
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    const float* base_data_ptr;  // 保存基础数据指针
    size_t n_points;             // 数据点数量
    
    // HNSW参数
    size_t M = 16;
    size_t efConstruction = 200;
    size_t efSearch = 100;

public:
    OptimizedMPIHNSWIVFIndex(size_t dimension, size_t num_lists, size_t hnsw_M = 16, size_t hnsw_efC = 200)
        : dim(dimension), nlist(num_lists), M(hnsw_M), efConstruction(hnsw_efC), 
          base_data_ptr(nullptr), n_points(0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        centroids.resize(nlist * dim);
        space.reset(new hnswlib::InnerProductSpace(dim));
    }

    // 生成K-means聚类中心（更好的聚类质量）
    void generate_kmeans_centroids(const float* base_data, size_t n, int max_iterations = 10) {
        if (rank == 0) {
            // 随机初始化聚类中心
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, n - 1);
            
            for (size_t i = 0; i < nlist; ++i) {
                size_t random_idx = dis(gen);
                std::copy(base_data + random_idx * dim, 
                         base_data + (random_idx + 1) * dim,
                         centroids.begin() + i * dim);
            }
            
            // 简化的K-means迭代
            for (int iter = 0; iter < max_iterations; ++iter) {
                std::vector<std::vector<float>> cluster_sums(nlist, std::vector<float>(dim, 0.0f));
                std::vector<size_t> cluster_counts(nlist, 0);
                
                // 分配点到最近聚类
                for (size_t i = 0; i < n; ++i) {
                    const float* vec = base_data + i * dim;
                    size_t best_cluster = 0;
                    float best_dot = std::numeric_limits<float>::lowest();
                    
                    for (size_t c = 0; c < nlist; ++c) {
                        const float* centroid = centroids.data() + c * dim;
                        float dot = 0.0f;
                        for (size_t d = 0; d < dim; ++d) {
                            dot += vec[d] * centroid[d];
                        }
                        if (dot > best_dot) {
                            best_dot = dot;
                            best_cluster = c;
                        }
                    }
                    
                    cluster_counts[best_cluster]++;
                    for (size_t d = 0; d < dim; ++d) {
                        cluster_sums[best_cluster][d] += vec[d];
                    }
                }
                
                // 更新聚类中心
                for (size_t c = 0; c < nlist; ++c) {
                    if (cluster_counts[c] > 0) {
                        for (size_t d = 0; d < dim; ++d) {
                            centroids[c * dim + d] = cluster_sums[c][d] / cluster_counts[c];
                        }
                    }
                }
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

    // 一次性构建索引（避免重复构建）
    void build_index_once(const float* base_data, size_t n) {
        base_data_ptr = base_data;
        n_points = n;
        
        // 生成高质量聚类中心
        generate_kmeans_centroids(base_data, n);

        // 计算每个点所属的聚类
        assign_points_to_clusters(base_data, n);

        // 构建全局HNSW索引（只构建一次）
        if (rank == 0) {
            auto build_start = std::chrono::high_resolution_clock::now();
            
            global_hnsw.reset(new hnswlib::HierarchicalNSW<float>(
                space.get(), n, M, efConstruction));
            
            global_hnsw->addPoint(base_data, 0);
            
            // OpenMP并行构建
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 1; i < n; ++i) {
                global_hnsw->addPoint(base_data + i * dim, i);
            }
            
            global_hnsw->setEf(efSearch);
            
            auto build_end = std::chrono::high_resolution_clock::now();
            auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                build_end - build_start).count();
            
            std::cout << "HNSW索引构建完成，耗时: " << build_time << "ms" << std::endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "优化版MPI HNSW+IVF索引构建完成" << std::endl;
        }
    }

private:
    // 将点分配到聚类（并行优化）
    void assign_points_to_clusters(const float* base_data, size_t n) {
        point_to_cluster.resize(n);
        
        // 每个进程处理一部分数据
        size_t elements_per_proc = (n + size - 1) / size;
        size_t start_idx = rank * elements_per_proc;
        size_t end_idx = std::min(start_idx + elements_per_proc, n);
        
        std::vector<uint32_t> local_assignments(n, 0);
        
        // OpenMP并行分配
        #pragma omp parallel for
        for (size_t i = start_idx; i < end_idx; ++i) {
            const float* vec = base_data + i * dim;
            
            uint32_t best_cluster = 0;
            float best_dot = std::numeric_limits<float>::lowest();
            
            for (size_t c = 0; c < nlist; ++c) {
                const float* centroid = centroids.data() + c * dim;
                float dot = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    dot += vec[d] * centroid[d];
                }
                if (dot > best_dot) {
                    best_dot = dot;
                    best_cluster = c;
                }
            }
            
            local_assignments[i] = best_cluster;
        }
        
        // 汇总所有进程的分配结果
        MPI_Allreduce(local_assignments.data(), point_to_cluster.data(), 
                      n, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    }

public:
    // 优化的搜索算法
    std::priority_queue<std::pair<float, uint32_t>>
    optimized_search(const float* query, size_t k, size_t nprobe = 16, 
                    size_t max_candidates = 1000) {
        std::priority_queue<std::pair<float, uint32_t>> result;
        
        if (rank == 0) {
            // 步骤1：选择目标聚类
            std::vector<std::pair<float, uint32_t>> cluster_dists(nlist);
            for (size_t i = 0; i < nlist; ++i) {
                const float* centroid = centroids.data() + i * dim;
                float dot = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    dot += query[d] * centroid[d];
                }
                cluster_dists[i] = {-dot, i};  // 负内积作为距离
            }
            
            std::partial_sort(cluster_dists.begin(), 
                             cluster_dists.begin() + nprobe,
                             cluster_dists.end());
            
            std::unordered_set<uint32_t> selected_clusters;
            for (size_t i = 0; i < nprobe; ++i) {
                selected_clusters.insert(cluster_dists[i].second);
            }
            
            // 步骤2：使用较小的候选数量进行HNSW搜索
            size_t search_k = std::min(max_candidates, k * 20);
            auto hnsw_result = global_hnsw->searchKnn(query, search_k);
            
            // 步骤3：过滤和重排序
            std::vector<std::pair<float, uint32_t>> filtered_candidates;
            filtered_candidates.reserve(search_k);
            
            while (!hnsw_result.empty()) {
                uint32_t candidate_id = hnsw_result.top().second;
                hnsw_result.pop();
                
                if (selected_clusters.count(point_to_cluster[candidate_id]) > 0) {
                    // 计算精确距离
                    const float* candidate_vec = base_data_ptr + candidate_id * dim;
                    float dot = 0.0f;
                    for (size_t d = 0; d < dim; ++d) {
                        dot += query[d] * candidate_vec[d];
                    }
                    float dist = 1.0f - dot;
                    filtered_candidates.emplace_back(dist, candidate_id);
                }
            }
            
            // 排序并选择top-k
            std::sort(filtered_candidates.begin(), filtered_candidates.end());
            size_t final_k = std::min(k, filtered_candidates.size());
            
            for (size_t i = 0; i < final_k; ++i) {
                result.push(filtered_candidates[i]);
            }
        }
        
        // 简化的结果广播
        broadcast_search_result(result, k);
        return result;
    }

private:
    // 优化的结果广播
    void broadcast_search_result(std::priority_queue<std::pair<float, uint32_t>>& result, size_t k) {
        if (rank == 0) {
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
            if (actual_k > 0) {
                MPI_Bcast(dists.data(), actual_k, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(ids.data(), actual_k, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            }
        } else {
            size_t actual_k;
            MPI_Bcast(&actual_k, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            
            if (actual_k > 0) {
                std::vector<float> dists(actual_k);
                std::vector<uint32_t> ids(actual_k);
                MPI_Bcast(dists.data(), actual_k, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(ids.data(), actual_k, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
                
                result = std::priority_queue<std::pair<float, uint32_t>>();
                for (size_t i = 0; i < actual_k; ++i) {
                    result.emplace(dists[i], ids[i]);
                }
            }
        }
    }

public:
    // 获取索引统计信息
    void print_index_stats() {
        if (rank == 0) {
            std::cout << "优化版HNSW+IVF索引统计:" << std::endl;
            std::cout << "IVF聚类数: " << nlist << std::endl;
            std::cout << "HNSW参数: M=" << M << ", efC=" << efConstruction << ", efS=" << efSearch << std::endl;
            
            std::vector<size_t> cluster_sizes(nlist, 0);
            for (size_t i = 0; i < point_to_cluster.size(); ++i) {
                cluster_sizes[point_to_cluster[i]]++;
            }
            
            size_t min_size = *std::min_element(cluster_sizes.begin(), cluster_sizes.end());
            size_t max_size = *std::max_element(cluster_sizes.begin(), cluster_sizes.end());
            size_t total_points = point_to_cluster.size();
            
            std::cout << "聚类分布 - 最小: " << min_size 
                      << ", 最大: " << max_size 
                      << ", 平均: " << total_points / nlist << std::endl;
                      
            // 计算聚类均衡度
            float mean_size = static_cast<float>(total_points) / nlist;
            float variance = 0.0f;
            for (size_t size : cluster_sizes) {
                float diff = size - mean_size;
                variance += diff * diff;
            }
            float cv = std::sqrt(variance / nlist) / mean_size;
            std::cout << "聚类均衡度(CV): " << std::fixed << std::setprecision(3) << cv << std::endl;
        }
    }
}; 