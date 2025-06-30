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

// 基础的MPI并行IVF索引实现
class MPIIVFIndex {
private:
    int rank, size;
    size_t dim, nlist;
    std::vector<float> centroids;                    // 聚类中心
    std::vector<std::vector<uint32_t>> invlists;    // 倒排列表
    std::vector<std::vector<float>> cluster_data;   // 每个簇的向量数据

public:
    MPIIVFIndex(size_t dimension, size_t num_lists) 
        : dim(dimension), nlist(num_lists) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        centroids.resize(nlist * dim);
        invlists.resize(nlist);
        cluster_data.resize(nlist);
    }

    // 加载预训练的聚类中心
    bool load_centroids(const std::string& file) {
        if (rank == 0) {
            std::ifstream in(file, std::ios::binary);
            if (!in) return false;
            in.read(reinterpret_cast<char*>(centroids.data()),
                    centroids.size() * sizeof(float));
        }
        // 广播聚类中心到所有进程
        MPI_Bcast(centroids.data(), centroids.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        return true;
    }

    // 构建IVF索引
    void build_index(const float* base_data, size_t n) {
        // 数据分配：每个进程处理一部分数据
        size_t elements_per_proc = (n + size - 1) / size;
        size_t start_idx = rank * elements_per_proc;
        size_t end_idx = std::min(start_idx + elements_per_proc, n);
        size_t local_n = end_idx - start_idx;

        // 本地数据分配到IVF簇（多线程并行）
        std::vector<std::vector<uint32_t>> local_invlists(nlist);
        std::vector<std::vector<std::vector<float>>> local_cluster_data(nlist);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < local_n; ++i) {
            size_t global_idx = start_idx + i;
            const float* vec = base_data + global_idx * dim;
            
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
            
            // 添加到对应簇（需要使用临界区保护）
            #pragma omp critical
            {
                local_invlists[best_cluster].push_back(global_idx);
                local_cluster_data[best_cluster].emplace_back(vec, vec + dim);
            }
        }

        // **优化后的MPI通信：批量收集所有簇数据**
        
        // 准备所有本地数据
        std::vector<uint32_t> all_local_ids;
        std::vector<float> all_local_data;
        std::vector<int> local_cluster_offsets(nlist + 1, 0);  // 每个簇在扁平化数组中的偏移
        
        for (size_t c = 0; c < nlist; ++c) {
            local_cluster_offsets[c] = all_local_ids.size();
            
            // 添加该簇的ID
            all_local_ids.insert(all_local_ids.end(), 
                                local_invlists[c].begin(), 
                                local_invlists[c].end());
            
            // 添加该簇的向量数据
            for (const auto& vec : local_cluster_data[c]) {
                all_local_data.insert(all_local_data.end(), vec.begin(), vec.end());
            }
        }
        local_cluster_offsets[nlist] = all_local_ids.size();
        
        // 收集每个进程的数据量
        int local_total_ids = all_local_ids.size();
        int local_total_data = all_local_data.size();
        
        std::vector<int> all_id_counts(size);
        std::vector<int> all_data_counts(size);
        
        MPI_Allgather(&local_total_ids, 1, MPI_INT, all_id_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&local_total_data, 1, MPI_INT, all_data_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // 计算全局偏移
        std::vector<int> id_displs(size), data_displs(size);
        int total_ids = 0, total_data = 0;
        for (int p = 0; p < size; ++p) {
            id_displs[p] = total_ids;
            data_displs[p] = total_data;
            total_ids += all_id_counts[p];
            total_data += all_data_counts[p];
        }
        
        // 一次性收集所有ID和数据
        std::vector<uint32_t> global_all_ids(total_ids);
        std::vector<float> global_all_data(total_data);
        
        MPI_Allgatherv(all_local_ids.data(), local_total_ids, MPI_UNSIGNED,
                       global_all_ids.data(), all_id_counts.data(), id_displs.data(),
                       MPI_UNSIGNED, MPI_COMM_WORLD);
        
        MPI_Allgatherv(all_local_data.data(), local_total_data, MPI_FLOAT,
                       global_all_data.data(), all_data_counts.data(), data_displs.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
        
        // 收集所有进程的簇偏移信息
        std::vector<int> all_cluster_offsets(size * (nlist + 1));
        MPI_Allgather(local_cluster_offsets.data(), nlist + 1, MPI_INT,
                      all_cluster_offsets.data(), nlist + 1, MPI_INT, MPI_COMM_WORLD);
        
        // 重建每个簇的数据
        invlists.assign(nlist, std::vector<uint32_t>());
        cluster_data.assign(nlist, std::vector<float>());
        
        for (size_t c = 0; c < nlist; ++c) {
            for (int p = 0; p < size; ++p) {
                int proc_offset = p * (nlist + 1);
                int cluster_start = all_cluster_offsets[proc_offset + c];
                int cluster_end = all_cluster_offsets[proc_offset + c + 1];
                int cluster_size = cluster_end - cluster_start;
                
                if (cluster_size > 0) {
                    // 复制ID
                    int global_id_start = id_displs[p] + cluster_start;
                    invlists[c].insert(invlists[c].end(),
                                      global_all_ids.begin() + global_id_start,
                                      global_all_ids.begin() + global_id_start + cluster_size);
                    
                    // 复制向量数据
                    int global_data_start = data_displs[p] + cluster_start * dim;
                    int data_size = cluster_size * dim;
                    cluster_data[c].insert(cluster_data[c].end(),
                                          global_all_data.begin() + global_data_start,
                                          global_all_data.begin() + global_data_start + data_size);
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "MPI IVF index building completed on " << size << " processes" << std::endl;
            print_index_stats();
        }
    }

    // MPI并行搜索
    std::priority_queue<std::pair<float, uint32_t>>
    mpi_search(const float* query, size_t k, size_t nprobe = 32) {
        // 步骤1：计算查询向量到各个聚类中心的距离
        std::vector<std::pair<float, uint32_t>> cluster_dists(nlist);
        
        #pragma omp parallel for
        for (size_t i = 0; i < nlist; ++i) {
            float dist = 0.0f;
            const float* centroid = centroids.data() + i * dim;
            for (size_t d = 0; d < dim; ++d) {
                float diff = query[d] - centroid[d];
                dist += diff * diff;
            }
            cluster_dists[i] = {dist, i};
        }
        
        // 选择最近的nprobe个聚类
        std::partial_sort(cluster_dists.begin(), 
                         cluster_dists.begin() + nprobe,
                         cluster_dists.end());

        // 步骤2：MPI分配搜索任务
        std::vector<uint32_t> my_clusters;
        for (size_t i = rank; i < nprobe; i += size) {
            my_clusters.push_back(cluster_dists[i].second);
        }

        // 步骤3：多线程搜索本进程负责的簇
        std::vector<std::vector<std::pair<float, uint32_t>>> thread_candidates(omp_get_max_threads());
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_candidates = thread_candidates[tid];
            
            #pragma omp for schedule(dynamic)
            for (size_t ci = 0; ci < my_clusters.size(); ++ci) {
                uint32_t cluster_id = my_clusters[ci];
                
                if (invlists[cluster_id].empty()) continue;
                
                const float* cluster_vecs = cluster_data[cluster_id].data();
                size_t cluster_size = invlists[cluster_id].size();
                
                // 计算每个向量与查询的距离
                for (size_t i = 0; i < cluster_size; ++i) {
                    const float* vec = cluster_vecs + i * dim;
                    float dot = 0.0f;
                    for (size_t d = 0; d < dim; ++d) {
                        dot += query[d] * vec[d];
                    }
                    float dist = 1.0f - dot;  // 转换为距离
                    
                    uint32_t global_id = invlists[cluster_id][i];
                    local_candidates.emplace_back(dist, global_id);
                }
            }
        }
        
        // 合并线程结果
        std::vector<std::pair<float, uint32_t>> local_candidates;
        for (const auto& thread_cand : thread_candidates) {
            local_candidates.insert(local_candidates.end(), 
                                  thread_cand.begin(), thread_cand.end());
        }
        
        // 本地Top-k筛选
        if (local_candidates.size() > k * 2) {
            std::nth_element(local_candidates.begin(), 
                           local_candidates.begin() + k * 2,
                           local_candidates.end());
            local_candidates.resize(k * 2);
        }

        // 步骤4：MPI收集所有进程的候选结果
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

        // 步骤5：去重并选择最终top-k
        std::unordered_set<uint32_t> seen;
        std::vector<std::pair<float, uint32_t>> unique_candidates;
        
        for (int i = 0; i < total_count; ++i) {
            if (seen.find(all_ids[i]) == seen.end()) {
                unique_candidates.emplace_back(all_dists[i], all_ids[i]);
                seen.insert(all_ids[i]);
            }
        }
        
        // 排序并取top-k
        std::sort(unique_candidates.begin(), unique_candidates.end());
        size_t final_k = std::min(k, unique_candidates.size());
        
        std::priority_queue<std::pair<float, uint32_t>> result;
        for (size_t i = 0; i < final_k; ++i) {
            result.push(unique_candidates[i]);
        }
        
        return result;
    }

    // 打印索引统计信息
    void print_index_stats() {
        if (rank == 0) {
            std::cout << "Basic IVF Index Statistics:" << std::endl;
            std::cout << "Number of clusters: " << nlist << std::endl;
            
            size_t total_vectors = 0;
            size_t non_empty_clusters = 0;
            
            for (size_t i = 0; i < nlist; ++i) {
                if (!invlists[i].empty()) {
                    non_empty_clusters++;
                    total_vectors += invlists[i].size();
                }
            }
            
            std::cout << "Non-empty clusters: " << non_empty_clusters << std::endl;
            std::cout << "Total vectors: " << total_vectors << std::endl;
            std::cout << "Average vectors per non-empty cluster: " 
                      << (non_empty_clusters > 0 ? total_vectors / non_empty_clusters : 0) << std::endl;
        }
    }

    // 获取簇数量
    size_t get_nlist() const { return nlist; }
    
    // 获取维度
    size_t get_dimension() const { return dim; }
};
