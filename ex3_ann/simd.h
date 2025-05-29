#pragma once
#include <immintrin.h>  // AVX2 intrinsics
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <mutex>
#include <condition_variable>
// SIMD-optimized L2 distance
static inline float l2_dist_avx2(const float* q, const float* p, size_t d) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 15 < d; i += 16) {
        __m256 q0 = _mm256_loadu_ps(q + i);
        __m256 p0 = _mm256_loadu_ps(p + i);
        __m256 diff0 = _mm256_sub_ps(q0, p0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
        __m256 q1 = _mm256_loadu_ps(q + i + 8);
        __m256 p1 = _mm256_loadu_ps(p + i + 8);
        __m256 diff1 = _mm256_sub_ps(q1, p1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    float sum = buf[0]+buf[1]+buf[2]+buf[3] + buf[4]+buf[5]+buf[6]+buf[7];
    for (; i < d; ++i) {
        float diff = q[i] - p[i];
        sum += diff * diff;
    }
    return sum;
}

// SIMD-based fast PQ table lookup (AVX2 + gather)
static inline float pq_dist_simd(const float* pq_dist, const uint8_t* codes, size_t m, size_t ksub) {
    // For simplicity use gather (lower performance than PSHUFB, but simpler)
    __m256 acc0 = _mm256_setzero_ps();
    for (size_t c = 0; c < m; ++c) {
        // gather each distance
        __m256 d = _mm256_set1_ps(pq_dist[c * ksub + codes[c]]);
        acc0 = _mm256_add_ps(acc0, d);
    }
    float buf[8];
    _mm256_storeu_ps(buf, acc0);
    float sum = buf[0]+buf[1]+buf[2]+buf[3] + buf[4]+buf[5]+buf[6]+buf[7];
    return sum;
}
