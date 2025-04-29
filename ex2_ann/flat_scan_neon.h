#pragma once
#include <arm_neon.h>
#include <queue>
#include <cassert>

// Neon向量操作封装
struct neon4f32 {
    float32x4_t v;
    neon4f32() = default;
    neon4f32(float32x4_t v) : v(v) {}
    explicit neon4f32(const float* p) : v(vld1q_f32(p)) {}
    
    neon4f32 operator+(const neon4f32& o) const { 
        return vaddq_f32(v, o.v); 
    }
    
    neon4f32 operator*(const neon4f32& o) const { 
        return vmulq_f32(v, o.v); 
    }
    
    void store(float* p) const { 
        vst1q_f32(p, v); 
    }
    
    float horizontal_sum() const {
        float32x2_t sum = vadd_f32(vget_high_f32(v), vget_low_f32(v));
        return vget_lane_f32(vpadd_f32(sum, sum), 0);
    }
};

// 优化的内积计算
inline float dot_neon(const float* a, const float* b, int d) {
    assert(d % 4 == 0 && "Dimension must be multiple of 4");
    
    neon4f32 sum(vdupq_n_f32(0.0f));
    // 使用预取提高性能
    __builtin_prefetch(a + 16);
    __builtin_prefetch(b + 16);
    
    for(int i = 0; i < d; i += 4) {
        neon4f32 va(a + i);
        neon4f32 vb(b + i);
        sum = sum + (va * vb);
        // 预取下一个块
        if (i + 20 < d) {
            __builtin_prefetch(a + i + 20);
            __builtin_prefetch(b + i + 20);
        }
    }
    return sum.horizontal_sum();
}

// 优化的暴力搜索
std::priority_queue<std::pair<float, uint32_t>> flat_search_neon(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    std::priority_queue<std::pair<float, uint32_t>> q;
    constexpr size_t BLOCK_SIZE = 32;  // 缓存友好的块大小
    
    // 分块处理以提高缓存效率
    float block_distances[BLOCK_SIZE];
    
    for(size_t i = 0; i < base_number; i += BLOCK_SIZE) {
        size_t block_end = std::min(i + BLOCK_SIZE, base_number);
        size_t block_size = block_end - i;
        
        // 预计算block内的所有距离
        for(size_t j = 0; j < block_size; j++) {
            float* base_vec = base + (i + j) * vecdim;
            block_distances[j] = 1.0f - dot_neon(base_vec, query, vecdim);
        }
        
        // 批量处理block内的结果
        for(size_t j = 0; j < block_size; j++) {
            float dis = block_distances[j];
            if(q.size() < k) {
                q.push({dis, i + j});
            } else if(dis < q.top().first) {
                q.pop();
                q.push({dis, i + j});
            }
        }
    }
    
    return q;
}



