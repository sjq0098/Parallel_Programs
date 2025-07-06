#pragma once
#include <queue>
#include <omp.h>
#include <vector>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

// SIMD优化的内积距离计算
float simd_inner_product_distance_flat(const float* a, const float* b, size_t dim) {
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
    
    return 1 - ip;  // 与原始flat_scan.h保持一致
}

std::priority_queue<std::pair<float, uint32_t>> flat_search_simd_parallel(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    // 使用线程局部存储来避免竞争
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> thread_results(omp_get_max_threads());
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& local_pq = thread_results[thread_id];
        
        #pragma omp for schedule(static)
        for (int i = 0; i < static_cast<int>(base_number); ++i) {
            float* point = base + i * vecdim;
            float dis = simd_inner_product_distance_flat(point, query, vecdim);
            
            if (local_pq.size() < k) {
                local_pq.push({dis, static_cast<uint32_t>(i)});
            } else if (dis < local_pq.top().first) {
                local_pq.pop();
                local_pq.push({dis, static_cast<uint32_t>(i)});
            }
        }
    }
    
    // 合并所有线程的结果
    std::vector<std::pair<float, uint32_t>> all_results;
    for (auto& pq : thread_results) {
        while (!pq.empty()) {
            all_results.push_back(pq.top());
            pq.pop();
        }
    }
    
    // 排序并选择最好的k个
    std::sort(all_results.begin(), all_results.end());
    
    std::priority_queue<std::pair<float, uint32_t>> final_result;
    for (size_t i = 0; i < std::min(k, all_results.size()); ++i) {
        final_result.push(all_results[i]);
    }
    
    return final_result;
} 