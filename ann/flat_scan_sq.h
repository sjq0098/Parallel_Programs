#pragma once
#include <immintrin.h>  // 包含x86 SIMD指令集
#include <queue>
#include <cassert>
#include "flat_scan_simd.h"

// SQ量化数据结构
struct SQData {
    std::vector<uint8_t> codes;  // 量化后的数据
    std::vector<float> min_vals; // 每维度最小值
    std::vector<float> scales;   // 每维度缩放因子
    size_t num_vectors;
    size_t dim;

    SQData(size_t n, size_t d) : codes(n * d), min_vals(d), scales(d), 
                                 num_vectors(n), dim(d) {}
};

// 使用SSE优化的量化函数
inline SQData quantize_base_sse(const float* base, size_t n, size_t d) {
    SQData sq(n, d);
    
    // 分块处理以提高缓存效率
    constexpr size_t BLOCK_SIZE = 32;
    
    for(size_t dim = 0; dim < d; dim += 4) {
        // 使用SSE加速查找最大最小值
        __m128 vmin = _mm_loadu_ps(base + dim);
        __m128 vmax = vmin;
        
        // 分块计算最大最小值
        for(size_t i = 0; i < n; i += BLOCK_SIZE) {
            size_t block_end = std::min(i + BLOCK_SIZE, n);
            for(size_t j = i; j < block_end; j++) {
                __m128 vcur = _mm_loadu_ps(base + j*d + dim);
                vmin = _mm_min_ps(vmin, vcur);
                vmax = _mm_max_ps(vmax, vcur);
            }
        }
        
        // 存储最小值和范围
        float min_vals[4], max_vals[4];
        _mm_storeu_ps(min_vals, vmin);
        _mm_storeu_ps(max_vals, vmax);
        
        for(int j = 0; j < 4 && dim + j < d; j++) {
            sq.min_vals[dim + j] = min_vals[j];
            float range = max_vals[j] - min_vals[j];
            sq.scales[dim + j] = range > 0 ? range / 255.0f : 0;
        }
        
        // 量化数据 - 使用分块处理
        for(size_t i = 0; i < n; i += BLOCK_SIZE) {
            size_t block_end = std::min(i + BLOCK_SIZE, n);
            for(size_t j = i; j < block_end; j++) {
                for(int k = 0; k < 4 && dim + k < d; k++) {
                    float val = base[j*d + dim + k];
                    float normalized = sq.scales[dim + k] > 0 ? 
                        (val - sq.min_vals[dim + k]) / sq.scales[dim + k] : 0;
                    sq.codes[j*d + dim + k] = static_cast<uint8_t>(normalized);
                }
            }
        }
    }
    return sq;
}

// SSE优化的量化内积计算
inline float dot_product_sq_sse(const uint8_t* codes, const float* mins,
                               const float* scales, const float* query,
                               size_t idx, size_t dim) {
    __m128 sum = _mm_setzero_ps();
    const uint8_t* code_ptr = codes + idx * dim;
    
    for(size_t d = 0; d < dim; d += 4) {
        // 加载4个uint8值并转换为4个float
        __m128i vcodes_i = _mm_setr_epi32(
            code_ptr[d], code_ptr[d+1], 
            code_ptr[d+2], code_ptr[d+3]
        );
        __m128i vcodes_u32 = _mm_and_si128(vcodes_i, _mm_set1_epi32(0xFF));
        __m128 vf = _mm_cvtepi32_ps(vcodes_u32);
        
        __m128 vmins = _mm_loadu_ps(mins + d);
        __m128 vscales = _mm_loadu_ps(scales + d);
        __m128 vdecode = _mm_add_ps(vmins, _mm_mul_ps(vf, vscales));
        __m128 vquery = _mm_loadu_ps(query + d);
        sum = _mm_add_ps(sum, _mm_mul_ps(vdecode, vquery));
    }
    
    // 水平求和
    __m128 shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// SQ优化的暴力搜索
inline std::priority_queue<std::pair<float, uint32_t>> 
flat_search_sq_sse(const uint8_t* quantized_base, const float* min_vals,
                    const float* scales, const float* query,
                    size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    
    // 分块处理以提高缓存效率
    constexpr size_t BLOCK_SIZE = 32;
    
    for(size_t i = 0; i < base_number; i += BLOCK_SIZE) {
        size_t block_end = std::min(i + BLOCK_SIZE, base_number);
        
        for(size_t j = i; j < block_end; j++) {
            float dis = dot_product_sq_sse(quantized_base, min_vals, scales,
                                          query, j, vecdim);
            dis = 1.0f - dis;
            
            if(q.size() < k) {
                q.push({dis, j});
            } else if(dis < q.top().first) {
                q.pop();
                q.push({dis, j});
            }
        }
    }
    
    return q;
}