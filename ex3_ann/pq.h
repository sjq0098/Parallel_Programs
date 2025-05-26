#pragma once
#include <vector>
#include <queue>
#include <cstdint>
#include <immintrin.h>  // 包含x86 SIMD指令集
#include <fstream>
#include <iostream>
#include <algorithm> // std::min, std::fill, std::sort
#include <cfloat>    // FLT_MAX

// SIMD向量封装
struct simd8float32 {
    __m128 data[2];  // 使用两个__m128代替float32x4x2_t

    simd8float32() = default;
    simd8float32(const float* x, const float* y) {
        data[0] = _mm_loadu_ps(x);
        data[1] = _mm_loadu_ps(y);
    }
    explicit simd8float32(const float* x) {
        data[0] = _mm_loadu_ps(x);
        data[1] = _mm_loadu_ps(x+4);
    }
    explicit simd8float32(float value) {
        data[0] = _mm_set1_ps(value);
        data[1] = _mm_set1_ps(value);
    }

    simd8float32 operator*(const simd8float32& other) const{
        simd8float32 result;
        result.data[0] = _mm_mul_ps(data[0], other.data[0]);
        result.data[1] = _mm_mul_ps(data[1], other.data[1]);
        return result;
    }

    simd8float32 operator+(const simd8float32& other) const{
        simd8float32 result;
        result.data[0] = _mm_add_ps(data[0], other.data[0]);
        result.data[1] = _mm_add_ps(data[1], other.data[1]);
        return result;
    }

    void store(float* x) const{
        _mm_storeu_ps(x, data[0]);
        _mm_storeu_ps(x + 4, data[1]);
    }
};

// PQ索引实现，支持不同参数
template<int M_val> // Renamed template parameter to avoid conflict with member m
class PQIndexSSE {
public:
    bool load_codebook(const std::string& file) {
        std::ifstream in(file, std::ios::binary);
        if (!in) return false;
        in.read(reinterpret_cast<char*>(&m), sizeof(m));       // m is num_subvectors
        in.read(reinterpret_cast<char*>(&ksub), sizeof(ksub)); // ksub is num_centroids_per_subvector (usually 256)
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));   // dim is original_dimension
        if (m == 0) { /* handle error or assert */ return false; }
        dsub = dim / m; // dim_per_subvector
        codebook.resize(m * ksub * dsub);
        in.read(reinterpret_cast<char*>(codebook.data()), codebook.size() * sizeof(float));
        return true;
    }

    bool load_codes(const std::string& file) {
        std::ifstream in(file, std::ios::binary);
        if (!in) return false;
        in.read(reinterpret_cast<char*>(&n), sizeof(n)); // n is num_database_vectors
        codes.resize(n * m); // Each vector has m codes
        in.read(reinterpret_cast<char*>(codes.data()), codes.size() * sizeof(uint8_t)); // codes are uint8_t
        return true;
    }

    // 不带重排序的搜索方法
    std::priority_queue<std::pair<float,uint32_t>> search(
        const float* q, size_t k) const {  
        
        std::vector<float> dt(m * ksub); // Distance table
        for (size_t i = 0; i < m; i++) {
            const float* subq = q + i * dsub;
            float* dst = dt.data() + i * ksub;
            for (size_t j = 0; j < ksub; j++) {
                const float* subc = codebook.data() + (i * ksub + j) * dsub;
                float sum_dist = 0.0f; // Actually dot product, not distance yet
                for (size_t d_idx = 0; d_idx < dsub; d_idx += 8) {
                    size_t N_elem = std::min(static_cast<size_t>(8), dsub - d_idx);
                    if (N_elem >= 8) {
                        simd8float32 vq_s(subq + d_idx);
                        simd8float32 vc_s(subc + d_idx);
                        simd8float32 prod_s = vq_s * vc_s;
                        float tmp_s[8];
                        prod_s.store(tmp_s);
                        for (int r = 0; r < 8; r++) sum_dist += tmp_s[r];
                    } else {
                        for (size_t r = 0; r < N_elem; r++) sum_dist += (subq + d_idx)[r] * (subc + d_idx)[r];
                    }
                }
                dst[j] = sum_dist; // Store dot product
            }
        }

        std::priority_queue<std::pair<float,uint32_t>> res; // Max-heap for scores
        for (size_t i = 0; i < n; i++) {
            const uint8_t* current_code = codes.data() + i * m;
            float total_dot_product = 0.0f;
            for (size_t j = 0; j < m; j++) {
                total_dot_product += dt[j * ksub + current_code[j]];
            }
            float score = total_dot_product; // Using dot product directly as score (higher is better)
                                          // Or 1.0f - L2_approx if dt stored squared L2 distances
                                          // Assuming higher dot product means higher similarity for now.
                                          // The original code used 1.0f - dist, implying dist was sum of dot products.
                                          // So, score = total_dot_product is consistent if we want similarity. 

            if (res.size() < k) {
                res.push({score, i});
            } else if (score > res.top().first) { // Keep k largest scores (if res is min-heap)
                                                // If res is max-heap, this logic is for replacing top if even larger score found
                                                // This needs to be: if score > smallest_of_the_k_largest. 
                                                // Correct logic for top-K largest with MIN PQ: if score > pq.top(), pq.pop, pq.push
                                                // Correct logic for top-K largest with MAX PQ (current `res`): 
                                                // push all, then trim. OR, if new_score > (what would be popped if full and this replaces an item)
                                                // The current code's `score < res.top().first` for MAX PQ was for keeping k-SMALLEST scores.
                                                // To keep k-LARGEST scores with MAX PQ: if (res.size() < k) res.push; else if (score > ???) - this is hard with maxPQ
                                                // Simplest fix: use a min-PQ internally for search, then convert.
                                                // For now, let's assume the provided NEON code's final stage for `res` was intended for similarity.
                                                // The user's NEON final res PQ (max-heap): `if (score < res.top().first)` implies keeping k smallest scores (if score=similarity)
                                                // This is very confusing. Let's ensure `res` is a min-PQ for search and then convert. 
                // This 'search' part also had the inverted logic. Fixing it to keep LARGEST scores:
                // To keep K largest scores in a MAX PQ `res`: only push if res has < K elements, or if score is larger than the current Kth largest (which is not directly accessible)
                // Alternative: if we maintain `res` as a MIN-PQ of K elements: 
                // if (res.size() < k) res.push({score,i}); else if (score > res.top().first) {res.pop(); res.push({score,i});}
                // Let's make `res` a min-PQ temporarily for this logic.
                 if (res.top().first < score) { // If current score is better than the smallest of the top-k
                    res.pop();
                    res.push({score, i});
                 }
            }
        }
        // If `res` was truly a min-PQ, need to convert back. But it's a max-PQ by default.
        // The logic `else if (score < res.top().first)` for a MAX PQ was indeed for keeping K smallest.
        // For K largest: if (res.size() < k) res.push({score,i}); else if (score > /* smallest score currently in res */ ) { /* pop smallest, push new */ }
        // This means the original default max-PQ for `res` in `search` must be managed carefully or use temp min-PQ.
        // The simplest is just to populate a min-PQ and convert.
        // Let's stick to the current `std::priority_queue<std::pair<float,uint32_t>> res;` (max PQ)
        // and fix the logic. To keep K largest items with a MAX PQ: insert if size < K. If size == K, only insert if new_item is larger than the smallest item currently held.
        // This is not directly supported. So, use temp min PQ.
        
        // Corrected logic for `search` using temporary min-PQ to find K largest scores
        std::priority_queue<std::pair<float,uint32_t>, std::vector<std::pair<float,uint32_t>>, std::greater<std::pair<float,uint32_t>>> temp_min_pq;
        for (size_t i = 0; i < n; i++) {
            const uint8_t* current_code = codes.data() + i * m;
            float total_dot_product = 0.0f;
            for (size_t j = 0; j < m; j++) {
                total_dot_product += dt[j * ksub + current_code[j]];
            }
            float score = total_dot_product; 

            if (temp_min_pq.size() < k) {
                temp_min_pq.push({score, i});
            } else if (score > temp_min_pq.top().first) {
                temp_min_pq.pop();
                temp_min_pq.push({score, i});
            }
        }
        std::priority_queue<std::pair<float,uint32_t>> final_res_max_pq;
        while(!temp_min_pq.empty()){
            final_res_max_pq.push(temp_min_pq.top());
            temp_min_pq.pop();
        }
        return final_res_max_pq;
    }

    // 优化的重排序搜索
    std::priority_queue<std::pair<float,uint32_t>> search_with_rerank(
        const float* q, const float* data, size_t base_n_passed, size_t vecdim_passed, size_t k, size_t rerank_k = 100) const {  
        
        if (dim == 0 || m == 0) { 
            std::cerr << "PQIndex not loaded or invalid parameters." << std::endl; 
            return {}; 
        }
        if (dim != vecdim_passed) {
            std::cerr << "Dimension mismatch: index dim " << dim << " vs passed vecdim " << vecdim_passed << std::endl;
            return {};
        }
        if (this->n == 0) {
             std::cerr << "No codes loaded in PQIndex." << std::endl; 
            return {};
        }

        std::vector<float> dt(m * ksub); 
        for (size_t i = 0; i < m; i++) {
            const float* subq = q + i * dsub;
            float* dst = dt.data() + i * ksub;
            for (size_t j = 0; j < ksub; j++) {
                const float* subc = codebook.data() + (i * ksub + j) * dsub;
                float approx_dot_product = 0.0f;
                for (size_t d_idx = 0; d_idx < dsub; d_idx += 8) {
                    size_t N_elem = std::min(static_cast<size_t>(8), dsub - d_idx);
                    if (N_elem >= 8) {
                        simd8float32 vq_s(subq + d_idx);
                        simd8float32 vc_s(subc + d_idx);
                        simd8float32 prod_s = vq_s * vc_s;
                        float tmp_s[8];
                        prod_s.store(tmp_s);
                        for (int r = 0; r < 8; r++) approx_dot_product += tmp_s[r];
                    } else {
                        for (size_t r = 0; r < N_elem; r++) approx_dot_product += (subq + d_idx)[r] * (subc + d_idx)[r];
                    }
                }
                dst[j] = approx_dot_product; 
            }
        }

        // Stage 1: Candidate selection using approximate scores (dot products)
        // We want rerank_k candidates with the HIGHEST approximate dot products.
        // Use a min-priority_queue of size rerank_k.
        std::priority_queue<std::pair<float, uint32_t>, 
                            std::vector<std::pair<float, uint32_t>>, 
                            std::greater<std::pair<float, uint32_t>>> candidate_min_pq;

        for (size_t i = 0; i < this->n; i++) { // this->n is the number of database vectors
            const uint8_t* current_code = codes.data() + i * m;
            float total_approx_dot_product = 0.0f;
            for (size_t j = 0; j < m; j++) {
                total_approx_dot_product += dt[j * ksub + current_code[j]];
            }
            // float approx_score = total_approx_dot_product; // Higher is better

            if (candidate_min_pq.size() < rerank_k) {
                candidate_min_pq.push({total_approx_dot_product, i});
            } else if (total_approx_dot_product > candidate_min_pq.top().first) {
                candidate_min_pq.pop();
                candidate_min_pq.push({total_approx_dot_product, i});
            }
        }
        
        // Stage 2: Rerank candidates using exact dot products and select top k
        // `data` is the original full float base vectors, `q` is the query vector
        std::priority_queue<std::pair<float, uint32_t>, 
                            std::vector<std::pair<float, uint32_t>>, 
                            std::greater<std::pair<float, uint32_t>>> final_top_k_min_pq; // Min-PQ for final k results

        while(!candidate_min_pq.empty()){
            std::pair<float, uint32_t> cand_entry = candidate_min_pq.top();
            candidate_min_pq.pop();
            uint32_t original_idx = cand_entry.second;
            // float approx_score_val = cand_entry.first; // Not needed for reranking

            const float* database_vector_for_exact_dist = data + original_idx * vecdim_passed;
            float exact_dot_product = 0.0f;
            for (size_t d_idx = 0; d_idx < vecdim_passed; d_idx += 8) {
                size_t N_elem = std::min(static_cast<size_t>(8), vecdim_passed - d_idx);
                if (N_elem >= 8) {
                    simd8float32 vq_simd(q + d_idx);
                    simd8float32 vd_simd(database_vector_for_exact_dist + d_idx);
                    simd8float32 prod_simd = vq_simd * vd_simd;
                    float tmp_exact[8];
                    prod_simd.store(tmp_exact);
                    for (int r = 0; r < 8; ++r) exact_dot_product += tmp_exact[r];
                } else {
                    for (size_t r = 0; r < N_elem; ++r) exact_dot_product += (q + d_idx)[r] * (database_vector_for_exact_dist + d_idx)[r];
                }
            }
            // float exact_score = exact_dot_product; // Higher is better

            if (final_top_k_min_pq.size() < k) {
                final_top_k_min_pq.push({exact_dot_product, original_idx});
            } else if (exact_dot_product > final_top_k_min_pq.top().first) {
                final_top_k_min_pq.pop();
                final_top_k_min_pq.push({exact_dot_product, original_idx});
            }
        }

        // Convert the temporary min-PQ to the final max-PQ result type (as per original function signature)
        std::priority_queue<std::pair<float, uint32_t>> final_result_max_pq;
        while (!final_top_k_min_pq.empty()) {
            final_result_max_pq.push(final_top_k_min_pq.top());
            final_top_k_min_pq.pop();
        }
        return final_result_max_pq;
    }

private:
    size_t n = 0;    // Number of database vectors
    size_t dim = 0;  // Original dimension of vectors
    size_t m = 0;    // Number of subvectors / code segments
    size_t dsub = 0; // Dimension of each subvector (dim / m)
    size_t ksub = 0; // Number of centroids per subvector (typically 256 for uint8_t codes)
    std::vector<float> codebook; // Flattened codebook: m * ksub * dsub
    std::vector<uint8_t> codes;  // Flattened codes: n * m
};

// Helper functions to call the templated PQIndexSSE
// PQ4
inline std::priority_queue<std::pair<float,uint32_t>> pq4_search_sse(
    float* base, float* query, size_t base_n, size_t vecdim, size_t k) {
    static PQIndexSSE<4> idx; // M_val = 4, actual m will be read from codebook
    static bool loaded = false;
    if (!loaded) {
        if (!idx.load_codebook("files/pq4_codebook.bin") || 
            !idx.load_codes("files/pq4_codes.bin")) {
            std::cerr << "无法加载PQ4索引文件" << std::endl; return {}; }
        loaded = true;    }
    return idx.search(query, k);
}
inline std::priority_queue<std::pair<float,uint32_t>> pq4_rerank_sse(
    float* base, float* query, size_t base_n, size_t vecdim, size_t k) {
    static PQIndexSSE<4> idx;
    static bool loaded = false;
    if (!loaded) {
        if (!idx.load_codebook("files/pq4_codebook.bin") || 
            !idx.load_codes("files/pq4_codes.bin")) {
            std::cerr << "无法加载PQ4索引文件" << std::endl; return {}; }
        loaded = true;    }
    return idx.search_with_rerank(query, base, base_n, vecdim, k);
}

// PQ8
inline std::priority_queue<std::pair<float,uint32_t>> pq8_search_sse(
    float* base, float* query, size_t base_n, size_t vecdim, size_t k) {
    static PQIndexSSE<8> idx;
    static bool loaded = false;
    if (!loaded) {
        if (!idx.load_codebook("files/pq8_codebook.bin") || 
            !idx.load_codes("files/pq8_codes.bin")) {
            std::cerr << "无法加载PQ8索引文件" << std::endl; return {}; }
        loaded = true;    }
    return idx.search(query, k);
}
inline std::priority_queue<std::pair<float,uint32_t>> pq8_rerank_sse(
    float* base, float* query, size_t base_n, size_t vecdim, size_t k) {
    static PQIndexSSE<8> idx;
    static bool loaded = false;
    if (!loaded) {
        if (!idx.load_codebook("files/pq8_codebook.bin") || 
            !idx.load_codes("files/pq8_codes.bin")) {
            std::cerr << "无法加载PQ8索引文件" << std::endl; return {}; }
        loaded = true;    }
    return idx.search_with_rerank(query, base, base_n, vecdim, k);
}

// PQ16
inline std::priority_queue<std::pair<float,uint32_t>> pq16_search_sse(
    float* base, float* query, size_t base_n, size_t vecdim, size_t k) {
    static PQIndexSSE<16> idx;
    static bool loaded = false;
    if (!loaded) {
        if (!idx.load_codebook("files/pq16_codebook.bin") || 
            !idx.load_codes("files/pq16_codes.bin")) {
            std::cerr << "无法加载PQ16索引文件" << std::endl; return {}; }
        loaded = true;    }
    return idx.search(query, k);
}
inline std::priority_queue<std::pair<float,uint32_t>> pq16_rerank_sse(
    float* base, float* query, size_t base_n, size_t vecdim, size_t k) {
    static PQIndexSSE<16> idx;
    static bool loaded = false;
    if (!loaded) {
        if (!idx.load_codebook("files/pq16_codebook.bin") || 
            !idx.load_codes("files/pq16_codes.bin")) {
            std::cerr << "无法加载PQ16索引文件" << std::endl; return {}; }
        loaded = true;    }
    return idx.search_with_rerank(query, base, base_n, vecdim, k);
}

// PQ32
inline std::priority_queue<std::pair<float,uint32_t>> pq32_search_sse(
    float* base, float* query, size_t base_n, size_t vecdim, size_t k) {
    static PQIndexSSE<32> idx;
    static bool loaded = false;
    if (!loaded) {
        if (!idx.load_codebook("files/pq32_codebook.bin") || 
            !idx.load_codes("files/pq32_codes.bin")) {
            std::cerr << "无法加载PQ32索引文件" << std::endl; return {}; }
        loaded = true;    }
    return idx.search(query, k);
}
inline std::priority_queue<std::pair<float,uint32_t>> pq32_rerank_sse(
    float* base, float* query, size_t base_n, size_t vecdim, size_t k) {
    static PQIndexSSE<32> idx;
    static bool loaded = false;
    if (!loaded) {
        if (!idx.load_codebook("files/pq32_codebook.bin") || 
            !idx.load_codes("files/pq32_codes.bin")) {
            std::cerr << "无法加载PQ32索引文件" << std::endl; return {}; }
        loaded = true;    }
    return idx.search_with_rerank(query, base, base_n, vecdim, k);
}
