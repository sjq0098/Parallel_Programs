#pragma once

#include <mpi.h>
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>
#include <unordered_set>
#include <memory>

// MPI并行化的PQ-IVF实现
class MPIPQIVFIndex {
private:
    int rank, size;
    size_t dim, nlist, m, ksub, dsub;
    std::vector<float> pq_codebook;                  // PQ码本
    std::vector<uint8_t> pq_codes;                   // 所有向量的PQ编码
    std::vector<float> pq_centroids;                 // PQ编码空间的聚类中心
    std::vector<std::vector<uint32_t>> invlists;     // 倒排列表
    std::vector<uint32_t> id_map;                    // 保存全局ID映射，用于重排序
    std::vector<float> original_vectors;             // 保存原始向量副本用于重排序
    size_t n_vectors;
    bool use_residuals;                              // 是否使用残差计算

public:
    MPIPQIVFIndex(size_t dimension, size_t num_lists, size_t pq_m, size_t pq_ksub = 256, bool use_res = false)
        : dim(dimension), nlist(num_lists), m(pq_m), ksub(pq_ksub), n_vectors(0), use_residuals(use_res) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        dsub = dim / m;
        
        if (dim % m != 0) {
            if (rank == 0) {
                std::cerr << "警告: 维度 " << dim << " 不能被m=" << m << "整除" << std::endl;
            }
            dsub = dim / m + (dim % m > 0 ? 1 : 0);
        }
        
        pq_codebook.resize(m * ksub * dsub);
        pq_centroids.resize(nlist * m);  // 每个聚类中心是m维的PQ编码
        invlists.resize(nlist);
    }

    // 加载PQ码本
    bool load_pq_codebook(const std::string& file) {
        if (rank == 0) {
            std::ifstream in(file, std::ios::binary);
            if (!in) return false;
            in.read(reinterpret_cast<char*>(pq_codebook.data()),
                    pq_codebook.size() * sizeof(float));
        }
        MPI_Bcast(pq_codebook.data(), pq_codebook.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        return true;
    }

    // 加载PQ编码空间的聚类中心
    bool load_pq_centroids(const std::string& file) {
        if (rank == 0) {
            std::ifstream in(file, std::ios::binary);
            if (!in) return false;
            in.read(reinterpret_cast<char*>(pq_centroids.data()),
                    pq_centroids.size() * sizeof(float));
        }
        MPI_Bcast(pq_centroids.data(), pq_centroids.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        return true;
    }

    // 对单个向量进行PQ编码（使用内积距离）
    void encode_vector(const float* vec, uint8_t* code) {
        for (size_t i = 0; i < m; ++i) {
            size_t subvec_start = i * dsub;
            size_t subvec_end = std::min(subvec_start + dsub, dim);
            size_t actual_dsub = subvec_end - subvec_start;
            
            const float* subvec = vec + subvec_start;
            uint8_t best_code = 0;
            float best_dist = std::numeric_limits<float>::max();
            
            for (size_t j = 0; j < ksub; ++j) {
                const float* centroid = pq_codebook.data() + (i * ksub + j) * dsub;
                float dot = 0.0f;
                for (size_t d = 0; d < actual_dsub; ++d) {
                    dot += subvec[d] * centroid[d];  // 内积计算
                }
                float dist = 1.0f - dot;  // DEEP100K标准距离
                if (dist < best_dist) {
                    best_dist = dist;
                    best_code = j;
                }
            }
            code[i] = best_code;
        }
    }

    // 使用对称距离计算PQ编码之间的距离
    float compute_pq_symmetric_distance(const uint8_t* code1, const uint8_t* code2) {
        float dist = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            if (code1[i] != code2[i]) {
                const float* centroid1 = pq_codebook.data() + (i * ksub + code1[i]) * dsub;
                const float* centroid2 = pq_codebook.data() + (i * ksub + code2[i]) * dsub;
                
                size_t subvec_start = i * dsub;
                size_t subvec_end = std::min(subvec_start + dsub, dim);
                size_t actual_dsub = subvec_end - subvec_start;
                
                for (size_t d = 0; d < actual_dsub; ++d) {
                    float diff = centroid1[d] - centroid2[d];
                    dist += diff * diff;
                }
            }
        }
        return dist;
    }

    // 使用非对称距离计算查询向量与PQ编码的距离（使用内积）
    float compute_pq_asymmetric_distance(const float* query, const uint8_t* code) {
        float total_dot = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            size_t subvec_start = i * dsub;
            size_t subvec_end = std::min(subvec_start + dsub, dim);
            size_t actual_dsub = subvec_end - subvec_start;
            
            const float* subvec = query + subvec_start;
            const float* centroid = pq_codebook.data() + (i * ksub + code[i]) * dsub;
            
            for (size_t d = 0; d < actual_dsub; ++d) {
                total_dot += subvec[d] * centroid[d];  // 内积计算
            }
        }
        return 1.0f - total_dot;  // DEEP100K标准距离
    }

    // 预计算查询向量到所有PQ码字的距离表（使用内积距离）
    std::vector<float> compute_distance_table(const float* query) {
        std::vector<float> distance_table(m * ksub);
        for (size_t i = 0; i < m; ++i) {
            size_t subvec_start = i * dsub;
            size_t subvec_end = std::min(subvec_start + dsub, dim);
            size_t actual_dsub = subvec_end - subvec_start;
            
            const float* subvec = query + subvec_start;
            for (size_t j = 0; j < ksub; ++j) {
                const float* centroid = pq_codebook.data() + (i * ksub + j) * dsub;
                float dot = 0.0f;
                for (size_t d = 0; d < actual_dsub; ++d) {
                    dot += subvec[d] * centroid[d];  // 内积计算
                }
                distance_table[i * ksub + j] = 1.0f - dot;  // DEEP100K标准距离
            }
        }
        return distance_table;
    }

    // 计算使用预计算距离表的距离
    float compute_pq_distance_table(const std::vector<float>& distance_table, const uint8_t* code) {
        float dist = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            dist += distance_table[i * ksub + code[i]];
        }
        return dist;
    }

    // 构建PQ-IVF索引
    void build_index(const float* base_data, size_t n) {
        n_vectors = n;
        
        // 保存原始向量数据的副本（确保重排序时数据可用）
        original_vectors.resize(n * dim);
        std::copy(base_data, base_data + n * dim, original_vectors.begin());
        
        // 数据分配：每个进程处理一部分数据
        size_t elements_per_proc = (n + size - 1) / size;
        size_t start_idx = rank * elements_per_proc;
        size_t end_idx = std::min(start_idx + elements_per_proc, n);
        size_t local_n = end_idx - start_idx;

        // 步骤1：对所有向量进行PQ编码
        std::vector<uint8_t> local_pq_codes(local_n * m);
        for (size_t i = 0; i < local_n; ++i) {
            const float* vec = base_data + (start_idx + i) * dim;
            encode_vector(vec, local_pq_codes.data() + i * m);
        }

        // 收集所有PQ编码
        std::vector<int> counts(size);
        int local_count = local_n * m;
        MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        std::vector<int> displs(size);
        int total_codes = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = total_codes;
            total_codes += counts[i];
        }
        
        pq_codes.resize(total_codes);
        MPI_Allgatherv(local_pq_codes.data(), local_count, MPI_UNSIGNED_CHAR,
                       pq_codes.data(), counts.data(), displs.data(), MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

        // 步骤2：在PQ编码空间中建立IVF索引
        std::vector<std::vector<uint32_t>> local_invlists(nlist);
        
        for (size_t i = 0; i < local_n; ++i) {
            size_t global_idx = start_idx + i;
            const uint8_t* code = local_pq_codes.data() + i * m;
            
            // 找到最近的PQ聚类中心
            uint32_t best_cluster = 0;
            float best_dist = std::numeric_limits<float>::max();
            
            for (size_t c = 0; c < nlist; ++c) {
                // 计算编码到聚类中心的距离
                float dist = 0.0f;
                for (size_t j = 0; j < m; ++j) {
                    // 使用PQ聚类中心
                    float diff = static_cast<float>(code[j]) - pq_centroids[c * m + j];
                    dist += diff * diff;
                }
                
                if (dist < best_dist) {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            
            local_invlists[best_cluster].push_back(global_idx);
        }

        // 收集倒排列表
        for (size_t list_id = 0; list_id < nlist; ++list_id) {
            // 收集每个进程中该列表的大小
            std::vector<int> list_sizes(size);
            int local_size = local_invlists[list_id].size();
            MPI_Allgather(&local_size, 1, MPI_INT, list_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
            
            // 计算总大小和位移
            int total_size = 0;
            std::vector<int> list_displs(size);
            for (int p = 0; p < size; ++p) {
                list_displs[p] = total_size;
                total_size += list_sizes[p];
            }
            
            if (total_size > 0) {
                invlists[list_id].resize(total_size);
                MPI_Allgatherv(local_invlists[list_id].data(),
                              local_invlists[list_id].size(),
                              MPI_UNSIGNED,
                              invlists[list_id].data(),
                              list_sizes.data(),
                              list_displs.data(),
                              MPI_UNSIGNED,
                              MPI_COMM_WORLD);
            }
        }
        
        // 构建全局ID映射表，确保向量ID能够被重排序过程使用
        id_map.resize(n);
        for (size_t i = 0; i < nlist; ++i) {
            for (size_t j = 0; j < invlists[i].size(); ++j) {
                id_map[j] = invlists[i][j];
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "MPI PQ-IVF索引构建完成" << std::endl;
        }
    }

    // 计算精确的L2距离
    float compute_l2_distance(const float* query, const float* vec) {
        float dist = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float diff = query[i] - vec[i];
            dist += diff * diff;
        }
        return dist;
    }

    // 计算内积距离（DEEP100K标准：1-内积）
    float compute_inner_product_distance(const float* query, const float* vec) {
        float dot = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            dot += query[i] * vec[i];
        }
        return 1.0f - dot;  // DEEP100K标准距离计算
    }

    // MPI并行搜索（不带重排序）
    std::priority_queue<std::pair<float, uint32_t>>
    mpi_search_without_rerank(const float* query, size_t k, size_t nprobe = 32) {
        // 步骤1：对查询向量进行PQ编码
        std::vector<uint8_t> query_code(m);
        encode_vector(query, query_code.data());
        
        // 预计算距离表
        std::vector<float> distance_table = compute_distance_table(query);

        // 步骤2：计算查询编码到各个PQ聚类中心的距离
        std::vector<std::pair<float, uint32_t>> cluster_dists(nlist);
        for (size_t i = 0; i < nlist; ++i) {
            float dist = 0.0f;
            for (size_t j = 0; j < m; ++j) {
                float diff = static_cast<float>(query_code[j]) - pq_centroids[i * m + j];
                dist += diff * diff;
            }
            cluster_dists[i] = {dist, i};
        }
        
        // 选择最近的nprobe个聚类
        std::partial_sort(cluster_dists.begin(), 
                         cluster_dists.begin() + nprobe,
                         cluster_dists.end());

        // 步骤3：分配搜索任务给各进程
        std::vector<uint32_t> search_lists;
        for (size_t i = rank; i < nprobe; i += size) {
            search_lists.push_back(cluster_dists[i].second);
        }

        // 步骤4：每个进程搜索分配的倒排列表
        std::vector<std::pair<float, uint32_t>> local_candidates;
        
        for (uint32_t list_id : search_lists) {
            if (invlists[list_id].empty()) continue;
            
            for (uint32_t vec_id : invlists[list_id]) {
                const uint8_t* vec_code = pq_codes.data() + vec_id * m;
                
                // 使用预计算的距离表计算距离
                float dist = compute_pq_distance_table(distance_table, vec_code);
                local_candidates.emplace_back(dist, vec_id);
            }
        }

        // 步骤5：收集所有进程的候选结果
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

        // 步骤6：选择top-k结果
        std::vector<std::pair<float, uint32_t>> all_candidates;
        for (int i = 0; i < total_count; ++i) {
            all_candidates.emplace_back(all_dists[i], all_ids[i]);
        }
        
        // 去重并排序
        std::unordered_set<uint32_t> seen;
        std::vector<std::pair<float, uint32_t>> unique_candidates;
        for (const auto& cand : all_candidates) {
            if (seen.find(cand.second) == seen.end()) {
                unique_candidates.push_back(cand);
                seen.insert(cand.second);
            }
        }
        
        std::sort(unique_candidates.begin(), unique_candidates.end());
        
        size_t final_k = std::min(k, unique_candidates.size());
        std::priority_queue<std::pair<float, uint32_t>> result;
        for (size_t i = 0; i < final_k; ++i) {
            result.push(unique_candidates[i]);
        }
        
        return result;
    }

    // MPI并行搜索（带重排序）
    std::priority_queue<std::pair<float, uint32_t>>
    mpi_search(const float* query, size_t k, size_t nprobe = 32, size_t rerank_candidates = 500) {
        // 首先获取更多候选进行重排序
        auto candidates = mpi_search_without_rerank(query, rerank_candidates, nprobe);
        
        // 提取候选ID
        std::vector<uint32_t> candidate_ids;
        while (!candidates.empty()) {
            candidate_ids.push_back(candidates.top().second);
            candidates.pop();
        }
        
        // 只在第一次查询时输出重排序候选数量
        static bool first_rerank = true;
        if (rank == 0 && first_rerank) {
            std::cout << "重排序候选数量: " << candidate_ids.size() << std::endl;
            first_rerank = false;
        }
        
        // 重排序阶段：使用原始向量数据计算精确内积距离
        std::vector<std::pair<float, uint32_t>> rerank_results;
        size_t candidates_per_proc = (candidate_ids.size() + size - 1) / size;
        size_t start_idx = rank * candidates_per_proc;
        size_t end_idx = std::min(start_idx + candidates_per_proc, candidate_ids.size());
        
        for (size_t i = start_idx; i < end_idx; ++i) {
            uint32_t id = candidate_ids[i];
            
            // 确保ID在有效范围内
            if (id >= n_vectors) {
                if (rank == 0) {
                    std::cerr << "警告: 无效的向量ID " << id << " >= " << n_vectors << std::endl;
                }
                continue;
            }
            
            const float* vec = original_vectors.data() + id * dim;
            
            // 计算精确内积距离
            float dist = compute_inner_product_distance(query, vec);
            
            rerank_results.emplace_back(dist, id);
        }
        
        // 收集所有重排序结果
        int local_count = rerank_results.size();
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
        
        std::vector<float> local_dists(local_count);
        std::vector<uint32_t> local_ids(local_count);
        for (int i = 0; i < local_count; ++i) {
            local_dists[i] = rerank_results[i].first;
            local_ids[i] = rerank_results[i].second;
        }
        
        MPI_Allgatherv(local_dists.data(), local_count, MPI_FLOAT,
                       all_dists.data(), counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(local_ids.data(), local_count, MPI_UNSIGNED,
                       all_ids.data(), counts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
        
        // 合并并选择最终结果
        std::vector<std::pair<float, uint32_t>> final_candidates;
        for (int i = 0; i < total_count; ++i) {
            final_candidates.emplace_back(all_dists[i], all_ids[i]);
        }
        
        std::sort(final_candidates.begin(), final_candidates.end());
        
        size_t final_k = std::min(k, final_candidates.size());
        std::priority_queue<std::pair<float, uint32_t>> result;
        for (size_t i = 0; i < final_k; ++i) {
            result.push(final_candidates[i]);
        }
        
        return result;
    }
};