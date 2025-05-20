#pragma once
#include <vector>
#include <queue>
#include <cstdint>
#include <arm_neon.h>
#include <fstream>

// SIMD向量封装
struct simd8float32 {
    float32x4x2_t data;

    simd8float32() = default;
    simd8float32(const float* x, const float* y) : data{vld1q_f32(x), vld1q_f32(y)} {}
    explicit simd8float32(const float* x) : data{vld1q_f32(x), vld1q_f32(x+4)} {}
    explicit simd8float32(float value) {
        data.val[0] = vdupq_n_f32(value);
        data.val[1] = vdupq_n_f32(value);
    }

    simd8float32 operator*(const simd8float32& other) const{
        simd8float32 result;
        result.data.val[0] = vmulq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vmulq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    simd8float32 operator+(const simd8float32& other) const{
        simd8float32 result;
        result.data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    void store(float* x) const{
        vst1q_f32(x, data.val[0]);
        vst1q_f32(x + 4, data.val[1]);
    }
};

class PQIndexNeon {
public:
    bool load_codebook(const std::string& file) {
        std::ifstream in(file, std::ios::binary);
        if (!in) return false;
        in.read(reinterpret_cast<char*>(&m), sizeof(m));
        in.read(reinterpret_cast<char*>(&ksub), sizeof(ksub));
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        dsub = dim / m;
        codebook.resize(m * ksub * dsub);
        in.read(reinterpret_cast<char*>(codebook.data()), codebook.size() * sizeof(float));
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
        const float* q, const float* data, size_t k, size_t rerank_k = 400) const {  // 增加重排序候选数
        
        // 更精确的距离表计算
        std::vector<float> dt(m * ksub);
        for (size_t i = 0; i < m; i++) {
            const float* subq = q + i * dsub;
            float* dst = dt.data() + i * ksub;
            
            for (size_t j = 0; j < ksub; j++) {
                const float* subc = codebook.data() + (i * ksub + j) * dsub;
                simd8float32 sum(0.0f);
                
                // 每8个元素一组计算，减少累积误差
                for (size_t d = 0; d < dsub; d += 8) {
                    simd8float32 vq(subq + d);
                    simd8float32 vc(subc + d);
                    sum = sum + (vq * vc);
                }
                
                float tmp[8];
                sum.store(tmp);
                dst[j] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + 
                         tmp[4] + tmp[5] + tmp[6] + tmp[7];
            }
        }

        // 第一阶段搜索 - 单独处理每个向量以提高精度
        auto cmp = [](const std::pair<float,uint32_t>& a, const std::pair<float,uint32_t>& b) {
            return a.first < b.first;
        };
        std::priority_queue<std::pair<float,uint32_t>,
                           std::vector<std::pair<float,uint32_t>>,
                           decltype(cmp)> cand(cmp);
        
        for (size_t i = 0; i < n; i++) {
            const uint8_t* code = codes.data() + i * m;
            double dist = 0.0;  // 使用double提高精度
            
            // 逐个累加以减少舍入误差
            for (size_t j = 0; j < m; j++) {
                dist += dt[j * ksub + code[j]];
            }
            
            float score = 1.0f - static_cast<float>(dist);
            if (cand.size() < rerank_k) {
                cand.push({score, i});
            } else if (score < cand.top().first) {
                cand.pop();
                cand.push({score, i});
            }
        }

        // 第二阶段重排序 - 使用更精确的计算
        std::priority_queue<std::pair<float,uint32_t>> res;
        std::vector<std::pair<float,uint32_t>> candidates;
        while (!cand.empty()) {
            candidates.push_back(cand.top());
            cand.pop();
        }
        
        // 反向处理候选项，保持稳定性
        for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
            uint32_t idx = it->second;
            const float* vec = data + idx * dim;
            double dist = 0.0;  // 使用double提高精度
            
            // 使用SIMD但保持精确累加
            for (size_t i = 0; i < dim; i += 8) {
                simd8float32 vq(q + i);
                simd8float32 vd(vec + i);
                simd8float32 prod = vq * vd;
                float tmp[8];
                prod.store(tmp);
                for (int j = 0; j < 8 && i + j < dim; j++) {
                    dist += tmp[j];
                }
            }
            
            float score = 1.0f - static_cast<float>(dist);
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

inline std::priority_queue<std::pair<float,uint32_t>> pq16_rerank_neon(
    float* base, float* query, size_t base_n, size_t vecdim, size_t k) {
    static PQIndexNeon idx;
    static bool L = false;
    if (!L) {
        if (!idx.load_codebook("files/pq16_codebook.bin") || 
            !idx.load_codes("files/pq16_codes.bin")) {
            std::cerr << "无法加载PQ索引文件" << std::endl;
            return std::priority_queue<std::pair<float,uint32_t>>();
        }
        L = true;
    }
    return idx.search_with_rerank(query, base, k);
}
