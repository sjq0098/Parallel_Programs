#pragma once
#include <queue>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <omp.h>
#include <iostream>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

class SIMDParallelLSH {
private:
    size_t dimension;
    size_t num_tables;
    size_t hash_size;
    std::vector<std::vector<std::vector<float>>> hash_functions;
    std::vector<std::unordered_map<size_t, std::vector<uint32_t>>> hash_tables;
    
    size_t search_radius;
    size_t min_candidates;
    
    // SIMD优化的内积距离计算
    float simd_inner_product_distance(const float* a, const float* b, size_t dim) {
        const size_t simd_width = 8;  // AVX可以并行处理8个float
        size_t simd_end = (dim / simd_width) * simd_width;
        
        __m256 sum_vec = _mm256_setzero_ps();
        
        // SIMD并行计算
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 a_vec = _mm256_loadu_ps(&a[i]);
            __m256 b_vec = _mm256_loadu_ps(&b[i]);
            __m256 mul_vec = _mm256_mul_ps(a_vec, b_vec);
            sum_vec = _mm256_add_ps(sum_vec, mul_vec);
        }
        
        // 水平加法获得总和
        float result[8];
        _mm256_storeu_ps(result, sum_vec);
        float ip = result[0] + result[1] + result[2] + result[3] + 
                   result[4] + result[5] + result[6] + result[7];
        
        // 处理剩余元素
        for (size_t i = simd_end; i < dim; ++i) {
            ip += a[i] * b[i];
        }
        
        return 1 - ip;  // 与flat_scan.h保持一致
    }
    
    // SIMD优化的哈希投影计算
    size_t compute_hash_simd(const float* point, int table_id) {
        size_t hash_value = 0;
        const size_t simd_width = 8;
        
        for (size_t i = 0; i < hash_size; ++i) {
            size_t simd_end = (dimension / simd_width) * simd_width;
            __m256 sum_vec = _mm256_setzero_ps();
            
            // SIMD计算投影
            for (size_t d = 0; d < simd_end; d += simd_width) {
                __m256 point_vec = _mm256_loadu_ps(&point[d]);
                __m256 hash_vec = _mm256_loadu_ps(&hash_functions[table_id][i][d]);
                __m256 mul_vec = _mm256_mul_ps(point_vec, hash_vec);
                sum_vec = _mm256_add_ps(sum_vec, mul_vec);
            }
            
            // 水平加法
            float result[8];
            _mm256_storeu_ps(result, sum_vec);
            float projection = result[0] + result[1] + result[2] + result[3] + 
                              result[4] + result[5] + result[6] + result[7];
            
            // 处理剩余元素
            for (size_t d = simd_end; d < dimension; ++d) {
                projection += point[d] * hash_functions[table_id][i][d];
            }
            
            // 符号随机投影
            if (projection > 0) {
                hash_value |= (1ULL << i);
            }
        }
        return hash_value;
    }
    
    // 并行生成邻居哈希值
    void generate_neighbors_parallel(size_t hash_value, size_t dist, 
                                   std::vector<size_t>& neighbors) {
        if (dist == 0) {
            neighbors.push_back(hash_value);
            return;
        }
        
        std::vector<size_t> temp_neighbors;
        
        // 串行生成（由于递归特性，并行化复杂）
        for (size_t bit = 0; bit < hash_size; ++bit) {
            generate_neighbors_recursive(hash_value ^ (1ULL << bit), dist - 1, 
                                       bit + 1, temp_neighbors);
        }
        
        neighbors.insert(neighbors.end(), temp_neighbors.begin(), temp_neighbors.end());
    }
    
    void generate_neighbors_recursive(size_t hash_value, size_t dist, size_t start_bit,
                                    std::vector<size_t>& neighbors) {
        if (dist == 0) {
            neighbors.push_back(hash_value);
            return;
        }
        
        for (size_t bit = start_bit; bit < hash_size && bit <= start_bit + dist; ++bit) {
            size_t neighbor = hash_value ^ (1ULL << bit);
            generate_neighbors_recursive(neighbor, dist - 1, bit + 1, neighbors);
        }
    }
    
    // 并行候选点收集
    std::unordered_set<uint32_t> collect_candidates_parallel(float* query) {
        std::unordered_set<uint32_t> candidates;
        
        // 并行计算每个表的哈希值
        std::vector<size_t> query_hashes(num_tables);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < static_cast<int>(num_tables); ++t) {
            query_hashes[t] = compute_hash_simd(query, t);
        }
        
        // 多级搜索策略
        for (size_t radius = 0; radius <= search_radius; ++radius) {
            std::vector<std::unordered_set<uint32_t>> thread_candidates(omp_get_max_threads());
            
            #pragma omp parallel for schedule(dynamic)
            for (int t = 0; t < static_cast<int>(num_tables); ++t) {
                int thread_id = omp_get_thread_num();
                size_t query_hash = query_hashes[t];
                
                if (radius == 0) {
                    // 精确匹配
                    if (hash_tables[t].find(query_hash) != hash_tables[t].end()) {
                        for (uint32_t idx : hash_tables[t][query_hash]) {
                            thread_candidates[thread_id].insert(idx);
                        }
                    }
                } else {
                    // 汉明距离为radius的邻居
                    std::vector<size_t> neighbors;
                    generate_neighbors_parallel(query_hash, radius, neighbors);
                    
                    for (size_t neighbor_hash : neighbors) {
                        if (hash_tables[t].find(neighbor_hash) != hash_tables[t].end()) {
                            for (uint32_t idx : hash_tables[t][neighbor_hash]) {
                                thread_candidates[thread_id].insert(idx);
                                if (thread_candidates[thread_id].size() >= min_candidates / num_tables) {
                                    break;
                                }
                            }
                            if (thread_candidates[thread_id].size() >= min_candidates / num_tables) {
                                break;
                            }
                        }
                    }
                }
            }
            
            // 合并线程结果
            for (const auto& thread_set : thread_candidates) {
                candidates.insert(thread_set.begin(), thread_set.end());
            }
            
            // 如果已经找到足够的候选点，提前退出
            if (candidates.size() >= min_candidates) {
                break;
            }
        }
        
        return candidates;
    }
    
    // 并行扩展候选点收集
    void extend_candidates_parallel(float* query, std::unordered_set<uint32_t>& candidates) {
        if (candidates.size() >= min_candidates * 2) return;
        
        std::vector<size_t> query_hashes(num_tables);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < static_cast<int>(num_tables); ++t) {
            query_hashes[t] = compute_hash_simd(query, t);
        }
        
        std::vector<std::unordered_set<uint32_t>> thread_candidates(omp_get_max_threads());
        
        #pragma omp parallel for schedule(dynamic)
        for (int t = 0; t < static_cast<int>(num_tables); ++t) {
            int thread_id = omp_get_thread_num();
            size_t query_hash = query_hashes[t];
            
            for (size_t radius = search_radius + 1; radius <= search_radius + 6; ++radius) {
                std::vector<size_t> neighbors;
                generate_neighbors_parallel(query_hash, radius, neighbors);
                
                for (size_t neighbor_hash : neighbors) {
                    if (hash_tables[t].find(neighbor_hash) != hash_tables[t].end()) {
                        for (uint32_t idx : hash_tables[t][neighbor_hash]) {
                            thread_candidates[thread_id].insert(idx);
                            if (thread_candidates[thread_id].size() >= min_candidates) {
                                goto next_table;
                            }
                        }
                    }
                }
                next_table:;
            }
        }
        
        // 合并结果
        for (const auto& thread_set : thread_candidates) {
            candidates.insert(thread_set.begin(), thread_set.end());
        }
    }
    
public:
    SIMDParallelLSH(size_t dim, size_t tables = 120, size_t hash_bits = 14, 
                   size_t radius = 10, size_t min_cand = 2000) 
        : dimension(dim), num_tables(tables), hash_size(hash_bits), 
          search_radius(radius), min_candidates(min_cand) {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0, 1.0);
        
        hash_functions.resize(num_tables);
        hash_tables.resize(num_tables);
        
        // 并行初始化随机投影向量
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < static_cast<int>(num_tables); ++t) {
            std::mt19937 local_gen(rd() + t);  // 每个线程独立的随机数生成器
            std::normal_distribution<float> local_dist(0.0, 1.0);
            
            hash_functions[t].resize(hash_size);
            for (size_t h = 0; h < hash_size; ++h) {
                hash_functions[t][h].resize(dimension);
                for (size_t d = 0; d < dimension; ++d) {
                    hash_functions[t][h][d] = local_dist(local_gen);
                }
            }
        }
    }
    
    void insert(float* base, size_t base_number) {
        // 并行插入所有点
        #pragma omp parallel for schedule(dynamic, 100)
        for (int i = 0; i < static_cast<int>(base_number); ++i) {
            float* point = base + i * dimension;
            
            for (size_t t = 0; t < num_tables; ++t) {
                size_t hash_value = compute_hash_simd(point, t);
                
                #pragma omp critical
                {
                    hash_tables[t][hash_value].push_back(i);
                }
            }
        }
    }
    
    std::priority_queue<std::pair<float, uint32_t>> search(
        float* base, float* query, size_t k) {
        
        // 使用并行候选点收集
        std::unordered_set<uint32_t> candidates = collect_candidates_parallel(query);
        
        // 如果候选点不够，并行扩展
        if (candidates.size() < k * 20) {
            extend_candidates_parallel(query, candidates);
        }
        
        // 如果候选点还是太少，随机添加
        if (candidates.size() < k * 10) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint32_t> dis(0, 99999);
            
            while (candidates.size() < k * 30) {
                candidates.insert(dis(gen));
            }
        }
        
        // 并行计算所有候选点的距离
        std::vector<uint32_t> candidate_vec(candidates.begin(), candidates.end());
        std::vector<std::pair<float, uint32_t>> all_candidates(candidate_vec.size());
        
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(candidate_vec.size()); ++i) {
            uint32_t idx = candidate_vec[i];
            float* point = base + idx * dimension;
            float dist = simd_inner_product_distance(point, query, dimension);
            all_candidates[i] = std::make_pair(dist, idx);
        }
        
        // 并行排序（使用OpenMP支持的并行算法）
        std::sort(all_candidates.begin(), all_candidates.end());
        
        // 构建结果
        std::priority_queue<std::pair<float, uint32_t>> result;
        size_t result_size = (k < all_candidates.size()) ? k : all_candidates.size();
        for (size_t i = 0; i < result_size; ++i) {
            result.push(all_candidates[i]);
        }
        
        // 如果结果不够k个，用最好的结果填充
        while (result.size() < k && !all_candidates.empty()) {
            result.push(all_candidates[result.size() % all_candidates.size()]);
        }
        
        return result;
    }
    
    void set_search_params(size_t radius, size_t min_cand) {
        search_radius = radius;
        min_candidates = min_cand;
    }
    
    // 获取统计信息
    void get_stats() {
        size_t total_points = 0;
        size_t max_bucket_size = 0;
        size_t non_empty_buckets = 0;
        
        #pragma omp parallel for reduction(+:total_points,non_empty_buckets) reduction(max:max_bucket_size)
        for (int t = 0; t < static_cast<int>(num_tables); ++t) {
            non_empty_buckets += hash_tables[t].size();
            for (const auto& bucket : hash_tables[t]) {
                total_points += bucket.second.size();
                if (bucket.second.size() > max_bucket_size) {
                    max_bucket_size = bucket.second.size();
                }
            }
        }
        
        std::cout << "SIMD并行LSH统计信息:" << std::endl;
        std::cout << "  哈希表数量: " << num_tables << std::endl;
        std::cout << "  哈希位数: " << hash_size << std::endl;
        std::cout << "  非空桶数量: " << non_empty_buckets << std::endl;
        std::cout << "  平均桶大小: " << (float)total_points / non_empty_buckets << std::endl;
        std::cout << "  最大桶大小: " << max_bucket_size << std::endl;
        std::cout << "  OpenMP线程数: " << omp_get_max_threads() << std::endl;
    }
};

// 接口函数
std::priority_queue<std::pair<float, uint32_t>> lsh_simd_parallel_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    static SIMDParallelLSH* lsh_index = nullptr;
    static bool initialized = false;
    
    if (!initialized) {
        lsh_index = new SIMDParallelLSH(vecdim, 120, 14, 10, 2000);
        // lsh_index->get_stats();  // 可以取消注释来查看统计信息
        lsh_index->insert(base, base_number);
        initialized = true;
    }
    
    return lsh_index->search(base, query, k);
} 