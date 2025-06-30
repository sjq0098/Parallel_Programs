#pragma once
#include <arm_neon.h>
#include <queue>
#include <cassert>
#include "flat_scan_neon.h"

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

// 使用NEON优化的量化函数
inline SQData quantize_base_neon(const float* base, size_t n, size_t d) {
    SQData sq(n, d);
    
    // 分块处理以提高缓存效率
    constexpr size_t BLOCK_SIZE = 32;
    
    for(size_t dim = 0; dim < d; dim += 4) {
        // 使用NEON加速查找最大最小值
        float32x4_t vmin = vld1q_f32(base + dim);
        float32x4_t vmax = vmin;
        
        // 分块计算最大最小值
        for(size_t i = 0; i < n; i += BLOCK_SIZE) {
            size_t block_end = std::min(i + BLOCK_SIZE, n);
            for(size_t j = i; j < block_end; j++) {
                float32x4_t vcur = vld1q_f32(base + j*d + dim);
                vmin = vminq_f32(vmin, vcur);
                vmax = vmaxq_f32(vmax, vcur);
            }
        }
        
        // 存储最小值和范围
        float min_vals[4], max_vals[4];
        vst1q_f32(min_vals, vmin);
        vst1q_f32(max_vals, vmax);
        
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

// NEON优化的量化内积计算
inline float dot_product_sq_neon(const uint8_t* codes, const float* mins,
                               const float* scales, const float* query,
                               size_t idx, size_t dim) {
    float32x4_t sum = vdupq_n_f32(0);
    const uint8_t* code_ptr = codes + idx * dim;
    
    for(size_t d = 0; d < dim; d += 4) {
        uint8x8_t vcodes = vld1_u8(code_ptr + d);
        uint16x8_t vcodes_u16 = vmovl_u8(vcodes);
        uint32x4_t vcodes_u32 = vmovl_u16(vget_low_u16(vcodes_u16));
        
        float32x4_t vf = vcvtq_f32_u32(vcodes_u32);
        float32x4_t vmins = vld1q_f32(mins + d);
        float32x4_t vscales = vld1q_f32(scales + d);
        float32x4_t vdecode = vmlaq_f32(vmins, vf, vscales);
        float32x4_t vquery = vld1q_f32(query + d);
        sum = vmlaq_f32(sum, vdecode, vquery);
    }
    
    float32x2_t r = vadd_f32(vget_high_f32(sum), vget_low_f32(sum));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}

// SQ优化的暴力搜索
inline std::priority_queue<std::pair<float, uint32_t>> 
flat_search_sq_neon(const uint8_t* quantized_base, const float* min_vals,
                    const float* scales, const float* query,
                    size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    
    // 分块处理以提高缓存效率
    constexpr size_t BLOCK_SIZE = 32;
    
    for(size_t i = 0; i < base_number; i += BLOCK_SIZE) {
        size_t block_end = std::min(i + BLOCK_SIZE, base_number);
        
        for(size_t j = i; j < block_end; j++) {
            float dis = dot_product_sq_neon(quantized_base, min_vals, scales,
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