#pragma once
#include <vector>
#include <queue>
#include <cstdint>
#include <immintrin.h>
#include <omp.h>
#include <fstream>
#include <iostream>

// AVX2 向量封装：8×float32
struct simd8float32 {
    __m256 data;

    simd8float32() = default;
    // 从连续内存加载 8 个 float（不要求 32 字节对齐）
    explicit simd8float32(const float* x) 
        : data(_mm256_loadu_ps(x)) {}

    // 用同一个值填充
    explicit simd8float32(float value) 
        : data(_mm256_set1_ps(value)) {}

    // 加法
    simd8float32 operator+(const simd8float32& o) const {
        return simd8float32{ _mm256_add_ps(data, o.data) };
    }

    // 乘法
    simd8float32 operator*(const simd8float32& o) const {
        return simd8float32{ _mm256_mul_ps(data, o.data) };
    }

    // 存回内存（不要求对齐）
    void store(float* x) const {
        _mm256_storeu_ps(x, data);
    }

private:
    // 私有构造：直接用 __m256 初始化
    explicit simd8float32(__m256 v) : data(v) {}
};

class PQIndexAvx2 {
public:
    bool load_codebook(const std::string& file) {
        std::ifstream in(file, std::ios::binary);
        if (!in) return false;
        in.read(reinterpret_cast<char*>(&m), sizeof(m));
        in.read(reinterpret_cast<char*>(&ksub), sizeof(ksub));
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        dsub = dim / m;
        codebook.resize(m * ksub * dsub);
        in.read(reinterpret_cast<char*>(codebook.data()),
                codebook.size() * sizeof(float));
        return true;
    }

    bool load_codes(const std::string& file) {
        std::ifstream in(file, std::ios::binary);
        if (!in) return false;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        codes.resize(n * m);
        in.read(reinterpret_cast<char*>(codes.data()), codes.size());
        return true;
    }

    std::priority_queue<std::pair<float,uint32_t>> search_with_rerank(
        const float* q, const float* data, size_t k, size_t rerank_k = 100) const {
        
        // 阶段一：距离表
        std::vector<float> dt(m * ksub);
        #pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            const float* subq = q + i * dsub;
            float* dst = dt.data() + i * ksub;

            for (size_t j = 0; j < ksub; ++j) {
                const float* subc = codebook.data() + (i * ksub + j) * dsub;
                simd8float32 sum(0.0f);
                for (size_t d = 0; d < dsub; d += 8) {
                    simd8float32 vq(subq + d);
                    simd8float32 vc(subc + d);
                    sum = sum + (vq * vc);
                }
                float tmp[8];
                sum.store(tmp);
                // 求和
                dst[j] = tmp[0] + tmp[1] + tmp[2] + tmp[3]
                       + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            }
        }

        // 阶段一 top rerank_k
        std::priority_queue<std::pair<float,uint32_t>> cand;
        #pragma omp parallel
        {
            std::priority_queue<std::pair<float,uint32_t>> local;
            #pragma omp for nowait
            for (size_t i = 0; i < n; ++i) {
                const uint8_t* code = &codes[i * m];
                float dist = 0;
                for (size_t j = 0; j < m; ++j) {
                    dist += dt[j * ksub + code[j]];
                }
                float score = 1.0f - dist;
                if (local.size() < rerank_k) {
                    local.push({score, (uint32_t)i});
                } else if (score < local.top().first) {
                    local.pop();
                    local.push({score, (uint32_t)i});
                }
            }
            #pragma omp critical
            while (!local.empty()) {
                cand.push(local.top());
                local.pop();
            }
        }

        // 阶段二重排序
        std::priority_queue<std::pair<float,uint32_t>> res;
        while (!cand.empty()) {
            uint32_t idx = cand.top().second;
            cand.pop();
            const float* vec = data + idx * dim;
            simd8float32 sum(0.0f);
            for (size_t i = 0; i < dim; i += 8) {
                simd8float32 vq(q + i);
                simd8float32 vd(vec + i);
                sum = sum + (vq * vd);
            }
            float tmp[8];
            sum.store(tmp);
            float dsum = tmp[0] + tmp[1] + tmp[2] + tmp[3]
                       + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            float score = 1.0f - dsum;
            if (res.size() < k) {
                res.push({score, idx});
            } else if (score < res.top().first) {
                res.pop();
                res.push({score, idx});
            }
        }
        return res;
    }

private:
    size_t n = 0, dim = 0, m = 0, dsub = 0, ksub = 256;
    std::vector<float> codebook;
    std::vector<uint8_t> codes;
};

inline std::priority_queue<std::pair<float,uint32_t>> 
pq16_rerank_avx2(float* base, float* query, size_t base_n,
                 size_t vecdim, size_t k) {
    static PQIndexAvx2 idx;
    static bool inited = false;
    if (!inited) {
        if (!idx.load_codebook("files/pq16_codebook.bin") ||
            !idx.load_codes("files/pq16_codes.bin")) {
            std::cerr << "无法加载PQ索引文件\n";
            return {};
        }
        inited = true;
    }
    return idx.search_with_rerank(query, base, k);
}
