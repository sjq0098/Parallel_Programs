#pragma once
#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <queue>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <cassert>
#include "simd_utils.h"
#include "pq_simd_ops.h"

// 乘积量化索引
class ProductQuantizer {
public:
    ProductQuantizer() = default;
    
    // 从文件加载码本
    bool load_codebook(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return false;
        
        // 读取元数据
        in.read(reinterpret_cast<char*>(&m), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&ksub), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&dimension), sizeof(size_t));
        
        // 计算每个子空间的维度
        dsub = dimension / m;
        
        // 分配并读取码本
        codebook.resize(m * ksub * dsub);
        in.read(reinterpret_cast<char*>(codebook.data()), codebook.size() * sizeof(float));
        
        // 优化：转置码本布局，提高缓存局部性
        transpose_codebook();
        
        // 优化：预计算查表索引
        precompute_table_offsets();
        
        in.close();
        return true;
    }
    
    // 从文件加载量化后的编码
    bool load_codes(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return false;
        
        // 读取向量数量
        in.read(reinterpret_cast<char*>(&n_vectors), sizeof(size_t));
        
        // 分配并读取编码
        codes.resize(n_vectors * m);
        in.read(reinterpret_cast<char*>(codes.data()), codes.size());
        
        // 优化：按每8个向量对齐重排代码
        if (m >= 8) {
            reorder_codes_for_simd();
        }
        
        in.close();
        return true;
    }
    
    // 从文件加载旋转矩阵(OPQ使用)
    bool load_rotation(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return false;
        
        // 读取维度
        size_t dim;
        in.read(reinterpret_cast<char*>(&dim), sizeof(size_t));
        
        if (dim != dimension) {
            in.close();
            return false;
        }
        
        // 分配并读取旋转矩阵
        rotation_matrix.resize(dimension * dimension);
        in.read(reinterpret_cast<char*>(rotation_matrix.data()), rotation_matrix.size() * sizeof(float));
        
        has_rotation = true;
        in.close();
        return true;
    }
    
    // 使用旋转矩阵(OPQ)对查询向量进行变换
    std::vector<float> rotate_query(const float* query) const {
        if (!has_rotation) {
            // 如果没有旋转矩阵，直接返回原查询向量的拷贝
            return std::vector<float>(query, query + dimension);
        }
        
        // 优化：使用AVX加速旋转矩阵乘法
        std::vector<float> rotated(dimension, 0);
        
        // 对于大矩阵，使用分块矩阵乘法
        constexpr size_t BLOCK_SIZE = 8; // AVX块大小
        
        for (size_t i = 0; i < dimension; i += BLOCK_SIZE) {
            const size_t block_end = std::min(i + BLOCK_SIZE, dimension);
            __m256 sum = _mm256_setzero_ps();
            
            for (size_t j = 0; j < dimension; j++) {
                __m256 q_val = _mm256_set1_ps(query[j]);
                __m256 r_row = _mm256_loadu_ps(&rotation_matrix[j * dimension + i]);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(q_val, r_row));
            }
            
            _mm256_storeu_ps(&rotated[i], sum);
        }
        
        return rotated;
    }
    
    // 为查询向量计算距离表
    std::vector<float> compute_distance_table(const float* query) const {
        // 对查询向量应用旋转（如果有）
        std::vector<float> rotated_query = rotate_query(query);
        const float* q = rotated_query.data();
        
        // 分配距离表空间: m个子空间，每个子空间ksub个聚类
        std::vector<float> distance_table(m * ksub);
        
        // 优化：使用SIMD加速距离表计算
        compute_distance_table_simd(q, distance_table.data());
        
        return distance_table;
    }
    
    // 使用AVX2加速表查找的距离计算
    float compute_distance_table_lookup_avx2(const std::vector<float>& distance_table, size_t idx) const {
        assert(idx < n_vectors);
        
        const uint8_t* code = &codes[idx * m];
        
        // 优化：使用预取减少内存访问延迟
        _mm_prefetch((const char*)(code + m), _MM_HINT_T0);
        
        // 优化：使用固定大小的本地缓冲区
        alignas(32) float local_distances[32] = {0};
        
        // 使用优化的查表函数
        return lookup_distance_optimized(code, distance_table.data(), local_distances);
    }
    
    // 快速距离计算(未使用AVX加速，用于小m值)
    float compute_distance_table_lookup(const std::vector<float>& distance_table, size_t idx) const {
        const uint8_t* code = &codes[idx * m];
        float distance = 0;
        
        // 优化：循环展开，减少分支预测失败
        size_t i = 0;
        for (; i + 3 < m; i += 4) {
            distance += distance_table[i * ksub + code[i]];
            distance += distance_table[(i+1) * ksub + code[i+1]];
            distance += distance_table[(i+2) * ksub + code[i+2]];
            distance += distance_table[(i+3) * ksub + code[i+3]];
        }
        
        // 处理剩余元素
        for (; i < m; i++) {
            distance += distance_table[i * ksub + code[i]];
        }
        
        return distance;
    }
    
    // 搜索最近邻
    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k) const {
        
        // 计算距离表
        std::vector<float> distance_table = compute_distance_table(query);
        
        // 创建结果队列
        std::priority_queue<std::pair<float, uint32_t>> result;
        
        // 优化：使用堆排序减少堆操作
        if (k > 10 && n_vectors > 1000) {
            return optimized_heap_search(distance_table, k);
        }
        
        // 搜索每个向量
        for (size_t i = 0; i < n_vectors; i++) {
            float dist;
            
            // 根据子空间数量选择不同的距离计算方法
            if (m >= 8) {
                dist = compute_distance_table_lookup_avx2(distance_table, i);
            } else {
                dist = compute_distance_table_lookup(distance_table, i);
            }
            
            if (result.size() < k) {
                result.push({dist, i});
            } else if (dist < result.top().first) {
                result.pop();
                result.push({dist, i});
            }
        }
        
        return result;
    }
    
    // 附加精确重排序的搜索
    std::priority_queue<std::pair<float, uint32_t>> search_with_rerank(
        const float* query, const float* base_data, size_t k, size_t rerank_k) const {
        
        // 第一阶段：使用PQ进行粗搜索，获取rerank_k个候选项
        auto candidates = search(query, rerank_k);
        
        // 第二阶段：精确重排序
        std::vector<std::pair<float, uint32_t>> candidates_vec;
        while (!candidates.empty()) {
            candidates_vec.push_back(candidates.top());
            candidates.pop();
        }
        
        // 使用精确向量计算距离并重排序
        std::priority_queue<std::pair<float, uint32_t>> result;
        
        for (const auto& candidate : candidates_vec) {
            uint32_t idx = candidate.second;
            
            // 使用AVX计算精确内积
            float ip = dot_product_avx(query, base_data + idx * dimension, dimension);
            float dist = 1.0f - ip;
            
            if (result.size() < k) {
                result.push({dist, idx});
            } else if (dist < result.top().first) {
                result.pop();
                result.push({dist, idx});
            }
        }
        
        return result;
    }
    
private:
    size_t n_vectors = 0;   // 向量数量
    size_t dimension = 0;   // 向量维度
    size_t m = 0;           // 子空间数量
    size_t dsub = 0;        // 每个子空间的维度 (dimension / m)
    size_t ksub = 256;      // 每个子空间的聚类数量(固定为256，使用uint8_t存储)
    
    std::vector<float> codebook;      // 码本: m * ksub * dsub
    std::vector<float> codebook_transposed; // 优化布局的码本
    std::vector<uint8_t> codes;       // 量化后的编码: n_vectors * m
    std::vector<uint8_t> codes_aligned; // 按SIMD对齐的编码
    
    bool has_rotation = false;              // 是否有旋转矩阵(OPQ)
    std::vector<float> rotation_matrix;     // 旋转矩阵: dimension * dimension
    
    // 优化相关成员
    std::vector<size_t> table_offsets;      // 预计算的表索引偏移
    
    // 优化：转置码本布局，提高缓存局部性
    void transpose_codebook() {
        // 将码本从(m, ksub, dsub)转置为(dsub, m, ksub)布局
        codebook_transposed.resize(m * ksub * dsub);
        
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < ksub; j++) {
                for (size_t d = 0; d < dsub; d++) {
                    // 原布局索引: (i * ksub + j) * dsub + d
                    // 新布局索引: d * (m * ksub) + i * ksub + j
                    codebook_transposed[d * (m * ksub) + i * ksub + j] = 
                        codebook[(i * ksub + j) * dsub + d];
                }
            }
        }
    }
    
    // 优化：预计算查表索引
    void precompute_table_offsets() {
        table_offsets.resize(m * ksub);
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < ksub; j++) {
                table_offsets[i * ksub + j] = i * ksub + j;
            }
        }
    }
    
    // 优化：按每8个向量对齐重排代码
    void reorder_codes_for_simd() {
        if (n_vectors < 8) return;
        
        codes_aligned.resize(n_vectors * m);
        
        // 按8个向量一组重排
        for (size_t i = 0; i < n_vectors; i += 8) {
            for (size_t j = 0; j < m; j++) {
                for (size_t k = 0; k < 8 && i+k < n_vectors; k++) {
                    // 转置小块：从按向量连续改为按子空间连续
                    codes_aligned[(j * 8) + k + (i / 8) * (m * 8)] = codes[(i + k) * m + j];
                }
            }
        }
    }
    
    // 优化：使用SIMD加速距离表计算
    void compute_distance_table_simd(const float* query, float* distance_table) const {
        // 调用优化的距离表计算
        pq_simd::compute_multispace_distance_table(query, codebook.data(), distance_table, m, dsub, ksub);
    }
    
    // 优化：使用固定大小缓冲区的查表函数
    float lookup_distance_optimized(const uint8_t* code, const float* distance_table, 
                                   float* local_distances) const {
        // 根据子空间数量选择最佳优化
        if (m <= 8) {
            // 对于小M值使用专门优化的函数
            return pq_simd::compute_small_m_distance(code, distance_table, m);
        }
        else {
            // 对于较大M值使用预取优化版本
            return pq_simd::compute_distance_with_prefetch(code, distance_table, m, ksub);
        }
    }
    
    // 优化：使用堆排序减少堆操作
    std::priority_queue<std::pair<float, uint32_t>> optimized_heap_search(
        const std::vector<float>& distance_table, size_t k) const {
        
        // 先计算所有距离，减少堆操作次数
        std::vector<std::pair<float, uint32_t>> distances(n_vectors);
        
        #pragma omp parallel for
        for (size_t i = 0; i < n_vectors; i++) {
            float dist;
            if (m >= 8) {
                alignas(32) float local_distances[32] = {0};
                const uint8_t* code = &codes[i * m];
                dist = lookup_distance_optimized(code, distance_table.data(), local_distances);
            } else {
                dist = compute_distance_table_lookup(distance_table, i);
            }
            distances[i] = {dist, i};
        }
        
        // 部分排序，只找前k个
        std::partial_sort(distances.begin(), distances.begin() + std::min(k, n_vectors),
                         distances.end());
        
        // 创建结果队列
        std::priority_queue<std::pair<float, uint32_t>> result;
        for (size_t i = 0; i < std::min(k, n_vectors); i++) {
            result.push(distances[i]);
        }
        
        return result;
    }
}; 