#include <immintrin.h>

// AVX2优化的8维浮点向量封装
struct simd8float32 {
    __m256 data;

    simd8float32() = default;

    // 从连续内存加载8个float
    explicit simd8float32(const float* ptr) : data(_mm256_loadu_ps(ptr)) {}

    // 从两个128位内存块加载（用于子空间处理）
    simd8float32(const float* x, const float* y) 
        : data(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(x)), _mm_loadu_ps(y), 1)) {}

    // 广播单个值到所有通道
    explicit simd8float32(float value) : data(_mm256_set1_ps(value)) {}

    // 算术运算
    simd8float32 operator*(const simd8float32& other) const {
        return simd8float32(_mm256_mul_ps(data, other.data));
    }

    simd8float32 operator+(const simd8float32& other) const {
        return simd8float32(_mm256_add_ps(data, other.data));
    }

    simd8float32 operator-(const simd8float32& other) const {
        return simd8float32(_mm256_sub_ps(data, other.data));
    }

    // 存储操作
    void storeu(float* ptr) const {
        _mm256_storeu_ps(ptr, data);
    }

    void store(float* ptr) const {
        _mm256_store_ps(ptr, data);
    }

    // 水平求和（用于距离计算）
    float horizontal_sum() const {
        __m128 vlow = _mm256_castps256_ps128(data);
        __m128 vhigh = _mm256_extractf128_ps(data, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        __m128 shuf = _mm_movehdup_ps(vlow);
        __m128 sums = _mm_add_ps(vlow, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

private:
    // 内部构造函数
    explicit simd8float32(__m256 val) : data(val) {}
};

// 专门用于PQ距离计算的函数
inline simd8float32 pq_l2_distance(const simd8float32& a, const simd8float32& b) {
    simd8float32 diff = a - b;
    return diff * diff; // 平方差
}

// OPQ旋转后的内积计算
inline simd8float32 opq_inner_product(const simd8float32& query, 
                                    const simd8float32* codebook_ptr,
                                    int m) {
    simd8float32 sum(0.0f);
    for (int i = 0; i < m; ++i) {
        sum = sum + query * codebook_ptr[i];
    }
    return sum;
}