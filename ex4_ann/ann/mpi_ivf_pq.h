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

// MPI并行化的IVF-PQ实现
class MPIIVFPQIndex {
private:
    int rank, size;
    size_t dim, nlist, m, ksub, dsub;
    std::vector<float> centroids;                    // 聚类中心
    std::vector<std::vector<uint32_t>> invlists;    // 倒排列表
    std::vector<float> codebook;                     // PQ码本
    std::vector<std::vector<uint8_t>> pq_codes;     // 每个倒排列表的PQ编码
    std::vector<float> original_vectors;             // 保存原始向量副本用于重排序
    bool use_inner_product;                          // 是否使用内积距离而非L2

public:
    MPIIVFPQIndex(size_t dimension, size_t num_lists, size_t pq_m, size_t pq_ksub = 256, bool use_ip = false)
        : dim(dimension), nlist(num_lists), m(pq_m), ksub(pq_ksub), use_inner_product(use_ip) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        dsub = dim / m;
        
        if (dim % m != 0) {
            if (rank == 0) {
                std::cerr << "警告: 维度 " << dim << " 不能被m=" << m << "整除" << std::endl;
            }
            dsub = dim / m + (dim % m > 0 ? 1 : 0);
        }
        
        centroids.resize(nlist * dim);
        invlists.resize(nlist);
        pq_codes.resize(nlist);
        codebook.resize(m * ksub * dsub);
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

    // 加载PQ码本
    bool load_codebook(const std::string& file) {
        if (rank == 0) {
            std::ifstream in(file, std::ios::binary);
            if (!in) return false;
            in.read(reinterpret_cast<char*>(codebook.data()),
                    codebook.size() * sizeof(float));
        }
        // 广播码本到所有进程
        MPI_Bcast(codebook.data(), codebook.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        return true;
    }

    // 构建倒排索引和PQ编码
    void build_index(const float* base_data, size_t n) {
        // 保存原始向量数据的副本，用于重排序
        original_vectors.resize(n * dim);
        std::copy(base_data, base_data + n * dim, original_vectors.begin());
        
        // 数据分配：每个进程处理一部分数据
        size_t elements_per_proc = (n + size - 1) / size;
        size_t start_idx = rank * elements_per_proc;
        size_t end_idx = std::min(start_idx + elements_per_proc, n);
        size_t local_n = end_idx - start_idx;

        // 分配数据到倒排列表
        std::vector<std::vector<uint32_t>> local_invlists(nlist);
        std::vector<std::vector<std::vector<float>>> local_residuals(nlist);

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
            
            // 添加到倒排列表
            local_invlists[best_cluster].push_back(global_idx);
            
            // 计算残差向量
            std::vector<float> residual(dim);
            const float* centroid = centroids.data() + best_cluster * dim;
            for (size_t d = 0; d < dim; ++d) {
                residual[d] = vec[d] - centroid[d];
            }
            local_residuals[best_cluster].push_back(residual);
        }

        // 收集所有进程的倒排列表大小
        std::vector<int> local_list_sizes(nlist);
        for (size_t i = 0; i < nlist; ++i) {
            local_list_sizes[i] = local_invlists[i].size();
        }
        
        std::vector<int> all_list_sizes(nlist * size);
        MPI_Allgather(local_list_sizes.data(), nlist, MPI_INT,
                      all_list_sizes.data(), nlist, MPI_INT, MPI_COMM_WORLD);

        // 合并倒排列表并进行PQ编码
        for (size_t list_id = 0; list_id < nlist; ++list_id) {
            // 计算总大小和位移
            int total_size = 0;
            std::vector<int> recv_counts(size);
            std::vector<int> displs(size);
            
            for (int p = 0; p < size; ++p) {
                recv_counts[p] = all_list_sizes[p * nlist + list_id];
                displs[p] = total_size;
                total_size += recv_counts[p];
            }
            
            if (total_size > 0) {
                // 收集ID
                invlists[list_id].resize(total_size);
                MPI_Allgatherv(local_invlists[list_id].data(),
                              local_invlists[list_id].size(),
                              MPI_UNSIGNED,
                              invlists[list_id].data(),
                              recv_counts.data(),
                              displs.data(),
                              MPI_UNSIGNED,
                              MPI_COMM_WORLD);
                
                // 收集残差向量
                std::vector<float> all_residuals(total_size * dim);
                std::vector<int> residual_counts(size);
                std::vector<int> residual_displs(size);
                
                for (int p = 0; p < size; ++p) {
                    residual_counts[p] = recv_counts[p] * dim;
                    residual_displs[p] = displs[p] * dim;
                }
                
                // 准备本地残差数据
                std::vector<float> local_residual_data(local_residuals[list_id].size() * dim);
                        for (size_t i = 0; i < local_residuals[list_id].size(); ++i) {
                            std::copy(local_residuals[list_id][i].begin(),
                                    local_residuals[list_id][i].end(),
                             local_residual_data.begin() + i * dim);
                        }
                
                MPI_Allgatherv(local_residual_data.data(),
                              local_residual_data.size(),
                              MPI_FLOAT,
                              all_residuals.data(),
                              residual_counts.data(),
                              residual_displs.data(),
                              MPI_FLOAT,
                              MPI_COMM_WORLD);
                
                // 进行PQ编码
                pq_codes[list_id].resize(total_size * m);
                encode_residuals(all_residuals.data(), total_size, pq_codes[list_id].data());
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "MPI IVF-PQ索引构建完成" << std::endl;
        }
    }

    // 对残差向量进行PQ编码
    void encode_residuals(const float* residuals, size_t n, uint8_t* codes) {
        for (size_t i = 0; i < n; ++i) {
            const float* vec = residuals + i * dim;
            uint8_t* code = codes + i * m;
            
            for (size_t j = 0; j < m; ++j) {
                size_t subvec_start = j * dsub;
                size_t subvec_end = std::min(subvec_start + dsub, dim);
                size_t actual_dsub = subvec_end - subvec_start;
                
                const float* subvec = vec + subvec_start;
                uint8_t best_code = 0;
                float best_dist = std::numeric_limits<float>::max();
                
                for (size_t k = 0; k < ksub; ++k) {
                    const float* centroid = codebook.data() + (j * ksub + k) * dsub;
                    float dot = 0.0f;
                    for (size_t d = 0; d < actual_dsub; ++d) {
                        dot += subvec[d] * centroid[d];  // 内积计算
                    }
                    float dist = 1.0f - dot;  // DEEP100K标准距离
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_code = k;
                    }
                }
                code[j] = best_code;
            }
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

    // 计算内积距离（用于余弦相似度）
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
        // 步骤1：计算查询向量到各个聚类中心的距离
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
        
        // 选择最近的nprobe个聚类
        std::partial_sort(cluster_dists.begin(), 
                         cluster_dists.begin() + nprobe,
                         cluster_dists.end());
        
        // 步骤2：分配搜索任务给各进程
        std::vector<uint32_t> search_lists;
        for (size_t i = rank; i < nprobe; i += size) {
            search_lists.push_back(cluster_dists[i].second);
        }
        
        // 步骤3：每个进程搜索分配的倒排列表
        std::vector<std::pair<float, uint32_t>> local_candidates;
        
        for (uint32_t list_id : search_lists) {
            if (invlists[list_id].empty()) continue;
            
            // 计算查询残差
            std::vector<float> query_residual(dim);
            const float* centroid = centroids.data() + list_id * dim;
            for (size_t d = 0; d < dim; ++d) {
                query_residual[d] = query[d] - centroid[d];
            }
            
            // 构建距离表（对每个PQ子空间，计算查询残差与码字的内积距离）
            std::vector<float> distance_table(m * ksub);
            for (size_t i = 0; i < m; ++i) {
                size_t subvec_start = i * dsub;
                size_t subvec_end = std::min(subvec_start + dsub, dim);
                size_t actual_dsub = subvec_end - subvec_start;
                
                const float* sub_query = query_residual.data() + subvec_start;
                for (size_t j = 0; j < ksub; ++j) {
                    const float* sub_centroid = codebook.data() + (i * ksub + j) * dsub;
                    float dot = 0.0f;
                    for (size_t d = 0; d < actual_dsub; ++d) {
                        dot += sub_query[d] * sub_centroid[d];  // 内积计算
                    }
                    distance_table[i * ksub + j] = 1.0f - dot;  // DEEP100K标准距离
                }
            }
            
            // 计算每个向量的近似距离
            for (size_t i = 0; i < invlists[list_id].size(); ++i) {
                uint32_t vec_id = invlists[list_id][i];
                const uint8_t* code = pq_codes[list_id].data() + i * m;
                
                float approx_dist = 0.0f;
                for (size_t j = 0; j < m; ++j) {
                    approx_dist += distance_table[j * ksub + code[j]];
                }
                
                local_candidates.emplace_back(approx_dist, vec_id);
            }
        }
        
        // 步骤4：收集所有进程的候选结果
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
        
        // 步骤5：选择top-k结果
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
        
        std::sort(unique_candidates.begin(), unique_candidates.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        
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
        // 第一阶段：使用近似搜索获取更多候选集
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
        
        // 第二阶段：使用原始向量计算精确内积距离重新排序
        std::vector<std::pair<float, uint32_t>> rerank_results;
        
        // 分配rerank任务给各进程
        size_t candidates_per_proc = (candidate_ids.size() + size - 1) / size;
        size_t start_idx = rank * candidates_per_proc;
        size_t end_idx = std::min(start_idx + candidates_per_proc, candidate_ids.size());
        
        for (size_t i = start_idx; i < end_idx; ++i) {
            uint32_t id = candidate_ids[i];
            
            // 确保ID在有效范围内
            if (id >= original_vectors.size() / dim) {
                if (rank == 0) {
                    std::cerr << "警告: 无效的向量ID " << id << std::endl;
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