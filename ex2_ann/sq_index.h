#pragma once
#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <cassert>
#include <cmath>
#include <string>
#include <fstream>
#include "simd_utils.h"

// 标量量化索引
class ScalarQuantizer {
public:
    ScalarQuantizer() = default;
    
    // 构建量化索引
    void build(const float* base, size_t n, size_t dim) {
        n_vectors = n;
        dimension = dim;
        
        // 分配存储空间
        quantized_base.resize(n * dim);
        scales.resize(n);
        offsets.resize(n);
        
        // 对每个向量进行量化
        for (size_t i = 0; i < n; i++) {
            quantize_vector(base + i * dim, i);
        }
    }
    
    // 保存索引到文件
    void save(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        
        // 写入元数据
        out.write(reinterpret_cast<const char*>(&n_vectors), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(&dimension), sizeof(size_t));
        
        // 写入量化数据
        out.write(reinterpret_cast<const char*>(quantized_base.data()), quantized_base.size());
        out.write(reinterpret_cast<const char*>(scales.data()), scales.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(offsets.data()), offsets.size() * sizeof(float));
        
        out.close();
    }
    
    // 从文件加载索引
    bool load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return false;
        
        // 读取元数据
        in.read(reinterpret_cast<char*>(&n_vectors), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&dimension), sizeof(size_t));
        
        // 分配空间
        quantized_base.resize(n_vectors * dimension);
        scales.resize(n_vectors);
        offsets.resize(n_vectors);
        
        // 读取量化数据
        in.read(reinterpret_cast<char*>(quantized_base.data()), quantized_base.size());
        in.read(reinterpret_cast<char*>(scales.data()), scales.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(offsets.data()), offsets.size() * sizeof(float));
        
        in.close();
        return true;
    }
    
    // 搜索最近邻
    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k) const {
        
        std::priority_queue<std::pair<float, uint32_t>> result;
        
        // 搜索每个向量
        for (size_t i = 0; i < n_vectors; i++) {
            float dist = compute_distance(query, i);
            
            if (result.size() < k) {
                result.push({dist, i});
            } else if (dist < result.top().first) {
                result.pop();
                result.push({dist, i});
            }
        }
        
        return result;
    }
    
private:
    size_t n_vectors = 0;     // 向量数量
    size_t dimension = 0;     // 向量维度
    
    std::vector<uint8_t> quantized_base;  // 量化后的向量数据
    std::vector<float> scales;            // 每个向量的缩放因子
    std::vector<float> offsets;           // 每个向量的偏移量
    
    // 量化单个向量
    void quantize_vector(const float* vec, size_t idx) {
        // 找到向量中的最小值和最大值
        float min_val = vec[0];
        float max_val = vec[0];
        
        for (size_t j = 1; j < dimension; j++) {
            min_val = std::min(min_val, vec[j]);
            max_val = std::max(max_val, vec[j]);
        }
        
        // 计算缩放和偏移
        float scale = (max_val - min_val) / 255.0f;
        if (scale == 0) scale = 1.0f;  // 避免除零
        
        // 保存缩放和偏移
        scales[idx] = scale;
        offsets[idx] = min_val;
        
        // 量化向量
        for (size_t j = 0; j < dimension; j++) {
            float normalized = (vec[j] - min_val) / scale;
            uint8_t quantized = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, normalized)));
            quantized_base[idx * dimension + j] = quantized;
        }
    }
    
    // 计算量化后向量与查询向量的距离
    float compute_distance(const float* query, size_t idx) const {
        float scale = scales[idx];
        float offset = offsets[idx];
        const uint8_t* code = &quantized_base[idx * dimension];
        
        return compute_distance_avx2(query, code, scale, offset, dimension);
    }
    
    // 使用AVX2加速量化向量与查询向量的内积计算
    float compute_distance_avx2(const float* query, const uint8_t* code,
                               float scale, float offset, size_t dim) const {
        size_t i = 0;
        __m256 sum256 = _mm256_setzero_ps();
        __m256 vscale = _mm256_set1_ps(scale);
        __m256 voffset = _mm256_set1_ps(offset);
        
        // 每次处理8个元素
        for (; i + 15 < dim; i += 16) {
            // 加载16个量化值并转换为两个__m256i
            __m128i chunk = _mm_loadu_si128((__m128i*)(code + i));
            
            // 低8个字节
            __m256i lowq = _mm256_cvtepu8_epi32(chunk);
            // 高8个字节
            __m128i high_chunk = _mm_bsrli_si128(chunk, 8);
            __m256i highq = _mm256_cvtepu8_epi32(high_chunk);
            
            // 转换为浮点数
            __m256 vlow = _mm256_cvtepi32_ps(lowq);
            __m256 vhigh = _mm256_cvtepi32_ps(highq);
            
            // 反量化: q * scale + offset
            __m256 flow = _mm256_add_ps(_mm256_mul_ps(vlow, vscale), voffset);
            __m256 fhigh = _mm256_add_ps(_mm256_mul_ps(vhigh, vscale), voffset);
            
            // 加载查询向量
            __m256 vquery1 = _mm256_loadu_ps(query + i);
            __m256 vquery2 = _mm256_loadu_ps(query + i + 8);
            
            // 计算内积
            __m256 vmul1 = _mm256_mul_ps(flow, vquery1);
            __m256 vmul2 = _mm256_mul_ps(fhigh, vquery2);
            
            // 累加结果
            sum256 = _mm256_add_ps(sum256, vmul1);
            sum256 = _mm256_add_ps(sum256, vmul2);
        }
        
        // 处理剩余的8个元素
        if (i + 7 < dim) {
            __m128i chunk = _mm_loadl_epi64((__m128i*)(code + i));
            __m256i vq = _mm256_cvtepu8_epi32(chunk);
            __m256 vf = _mm256_cvtepi32_ps(vq);
            __m256 values = _mm256_add_ps(_mm256_mul_ps(vf, vscale), voffset);
            __m256 vquery = _mm256_loadu_ps(query + i);
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(values, vquery));
            i += 8;
        }
        
        // 水平相加获取内积
        float ip = 0;
        
        // 提取sum256的结果
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum256);
        for (int j = 0; j < 8; j++) {
            ip += sum_array[j];
        }
        
        // 处理剩余元素
        for (; i < dim; i++) {
            float val = scale * code[i] + offset;
            ip += val * query[i];
        }
        
        // 返回1.0-内积作为距离
        return 1.0f - ip;
    }
}; 