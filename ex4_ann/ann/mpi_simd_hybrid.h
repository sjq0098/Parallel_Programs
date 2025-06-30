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
#include <thread>
#include <atomic>

// 根据编译环境选择SIMD指令集
#ifdef __AVX2__
#include <immintrin.h>
#define SIMD_WIDTH 8
typedef __m256 simd_vec;
#elif defined(__SSE2__)
#include <emmintrin.h>
#define SIMD_WIDTH 4
typedef __m128 simd_vec;
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_WIDTH 4
typedef float32x4_t simd_vec;
#else
#define SIMD_WIDTH 1
typedef float simd_vec;
#endif

// SIMD优化的距离计算函数
class SIMDDistanceComputer {
public:
    // SIMD优化的内积计算
    static float compute_inner_product_simd(const float* a, const float* b, size_t dim) {
#ifdef __AVX2__
        __m256 sum = _mm256_setzero_ps();
        size_t simd_end = (dim / 8) * 8;
        
        for (size_t i = 0; i < simd_end; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
        
        // 提取结果
        float result[8];
        _mm256_storeu_ps(result, sum);
        float total = result[0] + result[1] + result[2] + result[3] + 
                     result[4] + result[5] + result[6] + result[7];
        
        // 处理剩余元素
        for (size_t i = simd_end; i < dim; ++i) {
            total += a[i] * b[i];
        }
        
        return total;
        
#elif defined(__SSE2__)
        __m128 sum = _mm_setzero_ps();
        size_t simd_end = (dim / 4) * 4;
        
        for (size_t i = 0; i < simd_end; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
        }
        
        // 提取结果
        float result[4];
        _mm_storeu_ps(result, sum);
        float total = result[0] + result[1] + result[2] + result[3];
        
        // 处理剩余元素
        for (size_t i = simd_end; i < dim; ++i) {
            total += a[i] * b[i];
        }
        
        return total;
        
#elif defined(__ARM_NEON)
        float32x4_t sum = vdupq_n_f32(0.0f);
        size_t simd_end = (dim / 4) * 4;
        
        for (size_t i = 0; i < simd_end; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            sum = vmlaq_f32(sum, va, vb);
        }
        
        // 提取结果
        float result[4];
        vst1q_f32(result, sum);
        float total = result[0] + result[1] + result[2] + result[3];
        
        // 处理剩余元素
        for (size_t i = simd_end; i < dim; ++i) {
            total += a[i] * b[i];
        }
        
        return total;
        
#else
        // 标量版本
        float total = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            total += a[i] * b[i];
        }
        return total;
#endif
    }
    
    // SIMD优化的L2距离计算（保留，以防需要）
    static float compute_l2_distance_simd(const float* a, const float* b, size_t dim) {
#ifdef __AVX2__
        __m256 sum = _mm256_setzero_ps();
        size_t simd_end = (dim / 8) * 8;
        
        for (size_t i = 0; i < simd_end; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        
        // 提取结果
        float result[8];
        _mm256_storeu_ps(result, sum);
        float total = result[0] + result[1] + result[2] + result[3] + 
                     result[4] + result[5] + result[6] + result[7];
        
        // 处理剩余元素
        for (size_t i = simd_end; i < dim; ++i) {
            float diff = a[i] - b[i];
            total += diff * diff;
        }
        
        return total;
        
#elif defined(__SSE2__)
        __m128 sum = _mm_setzero_ps();
        size_t simd_end = (dim / 4) * 4;
        
        for (size_t i = 0; i < simd_end; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 diff = _mm_sub_ps(va, vb);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        
        // 提取结果
        float result[4];
        _mm_storeu_ps(result, sum);
        float total = result[0] + result[1] + result[2] + result[3];
        
        // 处理剩余元素
        for (size_t i = simd_end; i < dim; ++i) {
            float diff = a[i] - b[i];
            total += diff * diff;
        }
        
        return total;
        
#elif defined(__ARM_NEON)
        float32x4_t sum = vdupq_n_f32(0.0f);
        size_t simd_end = (dim / 4) * 4;
        
        for (size_t i = 0; i < simd_end; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t diff = vsubq_f32(va, vb);
            sum = vmlaq_f32(sum, diff, diff);
        }
        
        // 提取结果
        float result[4];
        vst1q_f32(result, sum);
        float total = result[0] + result[1] + result[2] + result[3];
        
        // 处理剩余元素
        for (size_t i = simd_end; i < dim; ++i) {
            float diff = a[i] - b[i];
            total += diff * diff;
        }
        
        return total;
        
#else
        // 标量版本
        float total = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            total += diff * diff;
        }
        return total;
#endif
    }
    
    // 内积距离计算（内积越大，距离越小）
    static float inner_product_to_distance(float inner_product) {
        return -inner_product;  // 负内积作为距离，内积越大距离越小
    }
};

// 流水线搜索的异步结果结构
struct AsyncSearchResult {
    std::vector<std::pair<float, uint32_t>> candidates;
    std::atomic<bool> ready{false};
    uint32_t cluster_id;
};

// MPI+多线程+SIMD混合并行的IVF索引
class MPISIMDHybridIndex {
private:
    int rank, size;
    size_t dim, nlist;
    std::vector<float> centroids;                    // 聚类中心
    std::vector<std::vector<uint32_t>> invlists;    // 倒排列表
    std::vector<std::vector<float>> cluster_data;   // 每个簇的向量数据

public:
    MPISIMDHybridIndex(size_t dimension, size_t num_lists) : dim(dimension), nlist(num_lists) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        centroids.resize(nlist * dim);
        invlists.resize(nlist);
        cluster_data.resize(nlist);
    }

    // 加载聚类中心
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

    // 构建索引（统一使用内积距离）
    void build_index(const float* base_data, size_t n) {
        // 数据分配到各进程
        size_t elements_per_proc = (n + size - 1) / size;
        size_t start_idx = rank * elements_per_proc;
        size_t end_idx = std::min(start_idx + elements_per_proc, n);
        size_t local_n = end_idx - start_idx;

        // 本地数据分配到IVF簇（多线程并行）
        std::vector<std::vector<uint32_t>> local_invlists(nlist);
        std::vector<std::vector<std::vector<float>>> local_cluster_data(nlist);
        
        // 为每个线程分配本地存储
        int num_threads = omp_get_max_threads();
        std::vector<std::vector<std::vector<uint32_t>>> thread_invlists(num_threads);
        std::vector<std::vector<std::vector<std::vector<float>>>> thread_cluster_data(num_threads);
        
        for (int t = 0; t < num_threads; ++t) {
            thread_invlists[t].resize(nlist);
            thread_cluster_data[t].resize(nlist);
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            
            #pragma omp for schedule(static)
            for (size_t i = 0; i < local_n; ++i) {
                size_t global_idx = start_idx + i;
                const float* vec = base_data + global_idx * dim;
                
                // 使用SIMD加速找到最近的聚类中心（使用内积距离）
                uint32_t best_cluster = 0;
                float best_inner_product = -std::numeric_limits<float>::max();
                
                for (size_t c = 0; c < nlist; ++c) {
                    const float* centroid = centroids.data() + c * dim;
                    float inner_product = SIMDDistanceComputer::compute_inner_product_simd(vec, centroid, dim);
                    
                    if (inner_product > best_inner_product) {
                        best_inner_product = inner_product;
                        best_cluster = c;
                    }
                }
                
                // 添加到线程本地存储
                thread_invlists[tid][best_cluster].push_back(global_idx);
                thread_cluster_data[tid][best_cluster].emplace_back(vec, vec + dim);
            }
        }
        
        // 合并各线程的结果
        for (int t = 0; t < num_threads; ++t) {
            for (size_t c = 0; c < nlist; ++c) {
                local_invlists[c].insert(local_invlists[c].end(),
                                       thread_invlists[t][c].begin(),
                                       thread_invlists[t][c].end());
                local_cluster_data[c].insert(local_cluster_data[c].end(),
                                           thread_cluster_data[t][c].begin(),
                                           thread_cluster_data[t][c].end());
            }
        }

        // **优化后的MPI通信：批量收集所有簇数据**
        
        // 步骤1：准备所有本地数据
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
        
        // 步骤2：收集每个进程的数据量
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
        
        // 步骤3：一次性收集所有ID和数据
        std::vector<uint32_t> global_all_ids(total_ids);
        std::vector<float> global_all_data(total_data);
        
        MPI_Allgatherv(all_local_ids.data(), local_total_ids, MPI_UNSIGNED,
                       global_all_ids.data(), all_id_counts.data(), id_displs.data(),
                       MPI_UNSIGNED, MPI_COMM_WORLD);
        
        MPI_Allgatherv(all_local_data.data(), local_total_data, MPI_FLOAT,
                       global_all_data.data(), all_data_counts.data(), data_displs.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
        
        // 步骤4：收集所有进程的簇偏移信息
        std::vector<int> all_cluster_offsets(size * (nlist + 1));
        MPI_Allgather(local_cluster_offsets.data(), nlist + 1, MPI_INT,
                      all_cluster_offsets.data(), nlist + 1, MPI_INT, MPI_COMM_WORLD);
        
        // 步骤5：重建每个簇的数据
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
            std::cout << "MPI+SIMD Hybrid index building completed on " << size << " processes" << std::endl;
            
            // 打印簇统计信息
            size_t total_vectors = 0;
            size_t non_empty_clusters = 0;
            for (size_t c = 0; c < nlist; ++c) {
                if (!invlists[c].empty()) {
                    non_empty_clusters++;
                    total_vectors += invlists[c].size();
                }
            }
            std::cout << "Non-empty clusters: " << non_empty_clusters << "/" << nlist << std::endl;
            std::cout << "Total vectors: " << total_vectors << std::endl;
        }
    }

    // 流水线搜索单个簇的函数
    void pipeline_search_cluster(const float* query, uint32_t cluster_id, 
                                AsyncSearchResult& result) {
        if (invlists[cluster_id].empty()) {
            result.ready = true;
            return;
        }
        
        const float* cluster_vecs = cluster_data[cluster_id].data();
        size_t cluster_size = invlists[cluster_id].size();
        
        result.candidates.reserve(cluster_size);
        result.cluster_id = cluster_id;
        
        // 使用SIMD加速计算每个向量与查询的距离
        for (size_t i = 0; i < cluster_size; ++i) {
            const float* vec = cluster_vecs + i * dim;
            float inner_product = SIMDDistanceComputer::compute_inner_product_simd(query, vec, dim);
            float dist = SIMDDistanceComputer::inner_product_to_distance(inner_product);
            
            uint32_t global_id = invlists[cluster_id][i];
            result.candidates.emplace_back(dist, global_id);
        }
        
        result.ready = true;
    }

    // MPI+多线程+SIMD并行搜索（带流水线优化）
    std::priority_queue<std::pair<float, uint32_t>>
    mpi_simd_search(const float* query, size_t k, size_t nprobe = 32) {
        // 步骤1：使用SIMD计算查询向量到各个聚类中心的距离（使用内积）
        std::vector<std::pair<float, uint32_t>> cluster_dists(nlist);
        
        #pragma omp parallel for
        for (size_t i = 0; i < nlist; ++i) {
            const float* centroid = centroids.data() + i * dim;
            float inner_product = SIMDDistanceComputer::compute_inner_product_simd(query, centroid, dim);
            float dist = SIMDDistanceComputer::inner_product_to_distance(inner_product);
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

        // 步骤3：流水线并行搜索 - 重叠计算和通信
        std::vector<AsyncSearchResult> async_results(my_clusters.size());
        std::vector<std::thread> search_threads;
        
        // 启动异步搜索线程
        for (size_t ci = 0; ci < my_clusters.size(); ++ci) {
            search_threads.emplace_back(
                [this, query, ci, &my_clusters, &async_results]() {
                    this->pipeline_search_cluster(query, my_clusters[ci], async_results[ci]);
                }
            );
        }
        
        // 边搜索边收集结果（流水线优化）
        std::vector<std::pair<float, uint32_t>> local_candidates;
        std::vector<bool> collected(my_clusters.size(), false);
        
        // 流水线收集：不等所有搜索完成，边完成边收集
        bool all_collected = false;
        while (!all_collected) {
            all_collected = true;
            
            for (size_t i = 0; i < async_results.size(); ++i) {
                if (!collected[i] && async_results[i].ready.load()) {
                    // 收集这个簇的结果
                    local_candidates.insert(local_candidates.end(),
                                          async_results[i].candidates.begin(),
                                          async_results[i].candidates.end());
                    collected[i] = true;
                }
                
                if (!collected[i]) {
                    all_collected = false;
                }
            }
            
            // 短暂休眠避免忙等待
            if (!all_collected) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
        
        // 等待所有搜索线程完成
        for (auto& t : search_threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        
        // 本地Top-k筛选
        if (local_candidates.size() > k * 2) {
            std::nth_element(local_candidates.begin(), 
                           local_candidates.begin() + k * 2,
                           local_candidates.end());
            local_candidates.resize(k * 2);
        }

        // 步骤4：流水线MPI通信 - 重叠收集和处理
        int local_count = local_candidates.size();
        std::vector<int> counts(size);
        
        // 异步启动MPI通信
        MPI_Request count_req;
        MPI_Iallgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 
                       MPI_COMM_WORLD, &count_req);
        
        // 在等待通信完成的同时，准备本地数据
        std::vector<float> local_dists(local_count);
        std::vector<uint32_t> local_ids(local_count);
        for (int i = 0; i < local_count; ++i) {
            local_dists[i] = local_candidates[i].first;
            local_ids[i] = local_candidates[i].second;
        }
        
        // 等待count通信完成
        MPI_Wait(&count_req, MPI_STATUS_IGNORE);
        
        // 计算偏移
        int total_count = 0;
        std::vector<int> displs(size);
        for (int i = 0; i < size; ++i) {
            displs[i] = total_count;
            total_count += counts[i];
        }
        
        // 异步收集所有数据
        std::vector<float> all_dists(total_count);
        std::vector<uint32_t> all_ids(total_count);
        
        MPI_Request dist_req, id_req;
        MPI_Iallgatherv(local_dists.data(), local_count, MPI_FLOAT,
                        all_dists.data(), counts.data(), displs.data(), MPI_FLOAT, 
                        MPI_COMM_WORLD, &dist_req);
        MPI_Iallgatherv(local_ids.data(), local_count, MPI_UNSIGNED,
                        all_ids.data(), counts.data(), displs.data(), MPI_UNSIGNED, 
                        MPI_COMM_WORLD, &id_req);
        
        // 等待数据通信完成
        MPI_Wait(&dist_req, MPI_STATUS_IGNORE);
        MPI_Wait(&id_req, MPI_STATUS_IGNORE);

        // 步骤5：去重并选择最终top-k
        std::unordered_set<uint32_t> seen;
        std::vector<std::pair<float, uint32_t>> unique_candidates;
        unique_candidates.reserve(total_count);
        
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

    // 性能统计
    void print_performance_info() {
        if (rank == 0) {
            std::cout << "SIMD Optimization: ";
#ifdef __AVX2__
            std::cout << "AVX2 (256-bit, 8 floats)" << std::endl;
#elif defined(__SSE2__)
            std::cout << "SSE2 (128-bit, 4 floats)" << std::endl;
#elif defined(__ARM_NEON)
            std::cout << "ARM NEON (128-bit, 4 floats)" << std::endl;
#else
            std::cout << "Scalar (no SIMD)" << std::endl;
#endif
            std::cout << "MPI processes: " << size << std::endl;
            std::cout << "OpenMP threads per process: " << omp_get_max_threads() << std::endl;
            std::cout << "Pipeline optimization: ENABLED (overlapped compute+communication)" << std::endl;
            std::cout << "Distance metric: Inner Product (consistent throughout)" << std::endl;
        }
    }
};