#pragma once
#include <immintrin.h>  // SSE, AVX, AVX2

// 128位SSE向量封装(4个float)
struct simd4f32 {
    __m128 v;
    
    simd4f32() = default;
    simd4f32(__m128 val) : v(val) {}
    explicit simd4f32(const float* p) : v(_mm_loadu_ps(p)) {}
    explicit simd4f32(float val) : v(_mm_set1_ps(val)) {}
    
    simd4f32 operator+(const simd4f32& o) const { return simd4f32(_mm_add_ps(v, o.v)); }
    simd4f32 operator-(const simd4f32& o) const { return simd4f32(_mm_sub_ps(v, o.v)); }
    simd4f32 operator*(const simd4f32& o) const { return simd4f32(_mm_mul_ps(v, o.v)); }
    
    void store(float* p) const { _mm_storeu_ps(p, v); }
    
    static simd4f32 zero() { return simd4f32(_mm_setzero_ps()); }
};

// 256位AVX向量封装(8个float)
struct simd8f32 {
    __m256 v;
    
    simd8f32() = default;
    simd8f32(__m256 val) : v(val) {}
    explicit simd8f32(const float* p) : v(_mm256_loadu_ps(p)) {}
    explicit simd8f32(float val) : v(_mm256_set1_ps(val)) {}
    
    simd8f32 operator+(const simd8f32& o) const { return simd8f32(_mm256_add_ps(v, o.v)); }
    simd8f32 operator-(const simd8f32& o) const { return simd8f32(_mm256_sub_ps(v, o.v)); }
    simd8f32 operator*(const simd8f32& o) const { return simd8f32(_mm256_mul_ps(v, o.v)); }
    
    void store(float* p) const { _mm256_storeu_ps(p, v); }
    
    static simd8f32 zero() { return simd8f32(_mm256_setzero_ps()); }
    
    // 水平相加获取8个元素的和
    float hsum() const {
        // 先把256位分成两个128位
        __m128 sum128 = _mm_add_ps(
            _mm256_extractf128_ps(v, 0),
            _mm256_extractf128_ps(v, 1)
        );
        // 水平相加128位向量
        __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x1));
        float result;
        _mm_store_ss(&result, sum32);
        return result;
    }
};

// 使用SSE计算内积距离
inline float dot_product_sse(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    simd4f32 sum = simd4f32::zero();
    
    // 每次处理4个元素
    for (; i + 3 < dim; i += 4) {
        simd4f32 va(a + i);
        simd4f32 vb(b + i);
        sum = sum + (va * vb);
    }
    
    float result[4];
    sum.store(result);
    float dot = result[0] + result[1] + result[2] + result[3];
    
    // 处理剩余元素
    for (; i < dim; i++) {
        dot += a[i] * b[i];
    }
    
    return dot;
}

// 使用AVX计算内积距离
inline float dot_product_avx(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    simd8f32 sum = simd8f32::zero();
    
    // 每次处理8个元素
    for (; i + 7 < dim; i += 8) {
        simd8f32 va(a + i);
        simd8f32 vb(b + i);
        sum = sum + (va * vb);
    }
    
    float dot = sum.hsum();
    
    // 处理剩余元素
    for (; i < dim; i++) {
        dot += a[i] * b[i];
    }
    
    return dot;
}

// 计算内积距离 1.0 - ip
inline float inner_product_distance(float ip) {
    return 1.0f - ip;
} 