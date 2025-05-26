#pragma once
#include <immintrin.h>  // 包含x86 SIMD指令集
#include <queue>
#include <cassert>

// SSE向量操作封装
struct sse4f32 {
    __m128 v;
    sse4f32() = default;
    sse4f32(__m128 v) : v(v) {}
    explicit sse4f32(const float* p) : v(_mm_loadu_ps(p)) {}
    
    sse4f32 operator+(const sse4f32& o) const { 
        return _mm_add_ps(v, o.v); 
    }
    
    sse4f32 operator*(const sse4f32& o) const { 
        return _mm_mul_ps(v, o.v); 
    }
    
    void store(float* p) const { 
        _mm_storeu_ps(p, v); 
    }
    
    float horizontal_sum() const {
        __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(v, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }
};

// 优化的内积计算
inline float dot_sse(const float* a, const float* b, int d) {
    assert(d % 4 == 0 && "Dimension must be multiple of 4");
    
    sse4f32 sum(_mm_setzero_ps());
    // 使用预取提高性能
    _mm_prefetch((const char*)(a + 16), _MM_HINT_T0);
    _mm_prefetch((const char*)(b + 16), _MM_HINT_T0);
    
    for(int i = 0; i < d; i += 4) {
        sse4f32 va(a + i);
        sse4f32 vb(b + i);
        sum = sum + (va * vb);
        // 预取下一个块
        if (i + 20 < d) {
            _mm_prefetch((const char*)(a + i + 20), _MM_HINT_T0);
            _mm_prefetch((const char*)(b + i + 20), _MM_HINT_T0);
        }
    }
    return sum.horizontal_sum();
}

// 优化的暴力搜索
std::priority_queue<std::pair<float, uint32_t>> flat_search_sse(
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
            block_distances[j] = 1.0f - dot_sse(base_vec, query, vecdim);
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



