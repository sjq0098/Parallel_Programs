#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include <arm_neon.h>
#include <fstream>
#include <iostream>
#include <string>

// SIMD 向量封装
struct simd8float32 {
    float32x4x2_t data;

    simd8float32() = default;

    simd8float32(const float* x, const float* y) {
        data.val[0] = vld1q_f32(x);
        data.val[1] = vld1q_f32(y);
    }

    explicit simd8float32(const float* x) {
        data.val[0] = vld1q_f32(x);
        data.val[1] = vld1q_f32(x + 4);
    }

    explicit simd8float32(float value) {
        data.val[0] = vdupq_n_f32(value);
        data.val[1] = vdupq_n_f32(value);
    }

    simd8float32 operator*(const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vmulq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vmulq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    simd8float32 operator+(const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    void store(float* x) const {
        vst1q_f32(x,       data.val[0]);
        vst1q_f32(x + 4,   data.val[1]);
    }
};

// 第一阶段候选排序的比较函数对象
struct CandidateCmp {
    bool operator()(const std::pair<float, uint32_t>& a,
                    const std::pair<float, uint32_t>& b) const {
        // 我们希望分数小的靠优先出栈，用小顶堆实现时这里返回 a.first < b.first
        return a.first < b.first;
    }
};

class PQIndexNeon {
public:
    bool load_codebook(const std::string& file) {
        std::ifstream in(file, std::ios::binary);
        if (!in) return false;
        in.read(reinterpret_cast<char*>(&m), sizeof(m));
        in.read(reinterpret_cast<char*>(&ksub), sizeof(ksub));
        in.read(reinterpret_cast<char*>(&dim),   sizeof(dim));
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
        in.read(reinterpret_cast<char*>(codes.data()),
                codes.size() * sizeof(uint8_t));
        return true;
    }

    // search_with_rerank：第一阶段粗排 + 第二阶段精排
    std::priority_queue<std::pair<float,uint32_t>>
    search_with_rerank(const float* q,
                       const float* data,
                       size_t k,
                       size_t rerank_k = 400) const
    {
        // 构建距离表 dt：m × ksub
        std::vector<float> dt(m * ksub);
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
                dst[j] = tmp[0] + tmp[1] + tmp[2] + tmp[3]
                       + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            }
        }

        // 第一阶段：粗排，取 rerank_k 个候选
        std::priority_queue<
            std::pair<float,uint32_t>,
            std::vector<std::pair<float,uint32_t>>,
            CandidateCmp
        > cand;
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* code = codes.data() + i * m;
            double dist = 0.0;
            for (size_t j = 0; j < m; ++j) {
                dist += dt[j * ksub + code[j]];
            }
            float score = 1.0f - static_cast<float>(dist);
            if (cand.size() < rerank_k) {
                cand.push({score, static_cast<uint32_t>(i)});
            } else if (score < cand.top().first) {
                cand.pop();
                cand.push({score, static_cast<uint32_t>(i)});
            }
        }

        // 第二阶段：对候选进行精排
        std::priority_queue<std::pair<float,uint32_t>> result;
        std::vector<std::pair<float,uint32_t>> cand_list;
        while (!cand.empty()) {
            cand_list.push_back(cand.top());
            cand.pop();
        }
        // 反向遍历保持稳定性
        for (auto it = cand_list.rbegin(); it != cand_list.rend(); ++it) {
            uint32_t idx = it->second;
            const float* vec = data + size_t(idx) * dim;
            double dist = 0.0;
            for (size_t off = 0; off < dim; off += 8) {
                simd8float32 vq(q + off);
                simd8float32 vd(vec + off);
                simd8float32 prod = vq * vd;
                float tmp[8];
                prod.store(tmp);
                for (int j = 0; j < 8 && off + j < dim; ++j) {
                    dist += tmp[j];
                }
            }
            float score = 1.0f - static_cast<float>(dist);
            if (result.size() < k) {
                result.push({score, idx});
            } else if (score < result.top().first) {
                result.pop();
                result.push({score, idx});
            }
        }
        return result;
    }

private:
    size_t n = 0, dim = 0, m = 0, dsub = 0, ksub = 256;
    std::vector<float>      codebook;
    std::vector<uint8_t>    codes;
};

// 顶层接口函数
inline std::priority_queue<std::pair<float,uint32_t>>
pq16_rerank_neon(float* base,
                 float* query,
                 size_t base_n,
                 size_t vecdim,
                 size_t k)
{
    static PQIndexNeon idx;
    static bool inited = false;
    if (!inited) {
        if (!idx.load_codebook("files/pq16_codebook.bin") ||
            !idx.load_codes   ("files/pq16_codes.bin")) {
            std::cerr << "无法加载 PQ 索引文件" << std::endl;
            return std::priority_queue<std::pair<float,uint32_t>>();
        }
        inited = true;
    }
    return idx.search_with_rerank(query, base, k);
}

