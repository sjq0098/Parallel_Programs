#pragma once
#include <vector>
#include <cstdint>
#include <queue>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <cassert>
#include <arm_neon.h>

// Neon向量操作封装(4个float32)
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
    
    // 水平相加
    float horizontal_sum() const {
        float32x2_t sum = vadd_f32(vget_high_f32(v), vget_low_f32(v));
        return vget_lane_f32(vpadd_f32(sum, sum), 0);
    }
};

// 使用float32x4x2_t的8通道SIMD操作封装
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
    
    simd8float32 operator-(const simd8float32& other) const{
        simd8float32 result;
        result.data.val[0] = vsubq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vsubq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    void storeu(float* x) const{
        vst1q_f32(x, data.val[0]);
        vst1q_f32(x + 4, data.val[1]);
    }
    
    void store(float* x) const{
        vst1q_f32(x, data.val[0]);
        vst1q_f32(x + 4, data.val[1]);
    }
};

// Neon优化的内积计算
inline float dot_neon(const float* a, const float* b, int d) {
    assert(d % 4 == 0 && "Dimension must be multiple of 4");
    
    neon4f32 sum(vdupq_n_f32(0.0f));
    for(int i = 0; i < d; i += 4) {
        neon4f32 va(a + i);
        neon4f32 vb(b + i);
        sum = sum + (va * vb);
    }
    return sum.horizontal_sum();
}

// SIMD Neon优化的内积计算(使用8通道)
inline float InnerProductSIMDNeon(const float* b1, const float* b2, size_t vecdim) {
    assert(vecdim % 8 == 0);
    float zeros[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    simd8float32 sum(zeros);
    for (size_t i = 0; i < vecdim; i += 8) {
        simd8float32 a1(b1 + i);
        simd8float32 a2(b2 + i);
        simd8float32 prod = a1 * a2;
        sum = sum + prod;
    }
    float tmp[8];
    sum.storeu(tmp);
    float dis = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    return 1-dis;
}

// 使用Neon指令计算多个子空间的距离表
inline void compute_multispace_distance_table(
    const float* query,          // 查询向量
    const float* codebook,       // 码本
    float* distance_table,       // 输出距离表
    size_t m,                    // 子空间数量
    size_t dsub,                 // 每个子空间的维度
    size_t ksub) {               // 每个子空间的聚类数量
    
    #pragma omp parallel for
    for (size_t subspace = 0; subspace < m; subspace++) {
        const float* subq = query + subspace * dsub;
        float* subdist = distance_table + subspace * ksub;
        
        // 对于每个子空间，计算与所有码字的距离
        for (size_t cluster = 0; cluster < ksub; cluster++) {
            const float* subcode = codebook + (subspace * ksub + cluster) * dsub;
            
            // 使用Neon计算内积
            float ip = 0;
            
            if (dsub % 8 == 0) {
                // 使用8通道SIMD Neon加速
                float dis = InnerProductSIMDNeon(subq, subcode, dsub);
                ip = 1.0f - dis;  // InnerProductSIMDNeon已经返回1-内积，这里需要转回内积
            } else if (dsub % 4 == 0) {
                // 使用4通道Neon加速
                ip = dot_neon(subq, subcode, dsub);
            } else {
                // 标量回退
                for (size_t d = 0; d < dsub; d++) {
                    ip += subq[d] * subcode[d];
                }
            }
            
            // 存储距离
            subdist[cluster] = 1.0f - ip;
        }
    }
}

// 针对16位整数的距离表查找优化
inline void compute_16bit_table_lookup(
    const uint8_t* codes,        // 编码数据
    const uint16_t* table_16bit, // 16位预计算距离表
    float* distances,            // 输出距离
    size_t n,                    // 向量数量
    size_t m) {                  // 子空间数量
    
    constexpr size_t BLOCK_SIZE = 16; // 处理的向量块大小
    
    // 按块处理向量，提高数据局部性
    #pragma omp parallel for
    for (size_t i = 0; i < n; i += BLOCK_SIZE) {
        const size_t block_end = std::min(i + BLOCK_SIZE, n);
        
        for (size_t j = i; j < block_end; j++) {
            const uint8_t* code = codes + j * m;
            float dist = 0;
            
            // 批量查表
            for (size_t k = 0; k < m; k++) {
                dist += table_16bit[k * 256 + code[k]] / 65535.0f;
            }
            
            distances[j] = dist;
        }
    }
}

// 使用Neon优化的距离计算
inline float compute_distance_with_prefetch(
    const uint8_t* code,         // 编码
    const float* distance_table, // 距离表
    size_t m,                    // 子空间数量
    size_t ksub) {               // 每个子空间的聚类数量
    
    // ARM平台上的预取指令
    __builtin_prefetch(code + m, 0, 3);
    
    // 使用临时数组存储中间结果
    alignas(16) float sums[4] = {0, 0, 0, 0};
    
    // 每次处理4个子空间
    size_t i = 0;
    for (; i + 3 < m; i += 4) {
        __builtin_prefetch(code + i + 8, 0, 3);
        
        // 查表并累加
        sums[0] += distance_table[i * ksub + code[i]];
        sums[1] += distance_table[(i+1) * ksub + code[i+1]];
        sums[2] += distance_table[(i+2) * ksub + code[i+2]];
        sums[3] += distance_table[(i+3) * ksub + code[i+3]];
    }
    
    // 处理剩余子空间
    float distance = 0;
    for (int j = 0; j < 4; j++) {
        distance += sums[j];
    }
    
    for (; i < m; i++) {
        distance += distance_table[i * ksub + code[i]];
    }
    
    return distance;
}

// 针对较小的向量(如M=4,8)的优化计算
inline float compute_small_m_distance(
    const uint8_t* code,         // 编码
    const float* distance_table, // 距离表
    size_t m) {                  // 子空间数量
    
    assert(m <= 8);
    
    // 对于小M值，展开循环比使用SIMD更高效
    float dist = 0;
    
    switch (m) {
        case 8:
            dist += distance_table[7 * 256 + code[7]];
            [[fallthrough]];
        case 7:
            dist += distance_table[6 * 256 + code[6]];
            [[fallthrough]];
        case 6:
            dist += distance_table[5 * 256 + code[5]];
            [[fallthrough]];
        case 5:
            dist += distance_table[4 * 256 + code[4]];
            [[fallthrough]];
        case 4:
            dist += distance_table[3 * 256 + code[3]];
            [[fallthrough]];
        case 3:
            dist += distance_table[2 * 256 + code[2]];
            [[fallthrough]];
        case 2:
            dist += distance_table[1 * 256 + code[1]];
            [[fallthrough]];
        case 1:
            dist += distance_table[0 * 256 + code[0]];
            break;
        default:
            break;
    }
    
    return dist;
}

// 批量处理8个向量的距离计算
inline void compute_batch_distances(
    const uint8_t* codes,        // 批量编码
    const float* distance_tables, // 距离表
    float* distances,            // 输出距离
    size_t m,                    // 子空间数量
    size_t batch_size) {         // 批处理大小
    
    for (size_t i = 0; i < batch_size; i++) {
        const uint8_t* code = codes + i * m;
        float dist = 0;
        
        // 使用循环展开处理
        size_t j = 0;
        for (; j + 3 < m; j += 4) {
            dist += distance_tables[j * 256 + code[j]];
            dist += distance_tables[(j+1) * 256 + code[j+1]];
            dist += distance_tables[(j+2) * 256 + code[j+2]];
            dist += distance_tables[(j+3) * 256 + code[j+3]];
        }
        
        // 处理剩余子空间
        for (; j < m; j++) {
            dist += distance_tables[j * 256 + code[j]];
        }
        
        distances[i] = dist;
    }
}

// 应用PQ16+精确重排序进行搜索
inline std::priority_queue<std::pair<float, uint32_t>> pq16_rerank_search(
    const float* base,              // 原始向量库
    const float* query,             // 查询向量
    const uint8_t* codes,           // PQ编码数据
    const float* codebook,          // 码本数据
    size_t n_vectors,               // 向量数量
    size_t dimension,               // 向量维度
    size_t m,                       // 子空间数量
    size_t k,                       // 返回的结果数量
    size_t rerank_k) {              // 重排序候选数量
    
    // 步骤1: 计算距离表
    const size_t dsub = dimension / m;   // 每个子空间的维度
    const size_t ksub = 256;             // 每个子空间的聚类数量
    
    std::vector<float> distance_table(m * ksub);
    compute_multispace_distance_table(
        query, codebook, distance_table.data(), m, dsub, ksub
    );
    
    // 步骤2: 第一阶段PQ搜索
    std::priority_queue<std::pair<float, uint32_t>> candidates;
    
    for (size_t i = 0; i < n_vectors; i++) {
        float dist;
        const uint8_t* code = &codes[i * m];
        
        // 选择合适的距离计算方法
        if (m <= 8) {
            dist = compute_small_m_distance(code, distance_table.data(), m);
        } else {
            dist = compute_distance_with_prefetch(code, distance_table.data(), m, ksub);
        }
        
        if (candidates.size() < rerank_k) {
            candidates.push({dist, i});
        } else if (dist < candidates.top().first) {
            candidates.pop();
            candidates.push({dist, i});
        }
    }
    
    // 步骤3: 第二阶段精确重排序
    std::vector<std::pair<float, uint32_t>> candidates_vec;
    while (!candidates.empty()) {
        candidates_vec.push_back(candidates.top());
        candidates.pop();
    }
    
    std::priority_queue<std::pair<float, uint32_t>> result;
    
    for (const auto& candidate : candidates_vec) {
        uint32_t idx = candidate.second;
        
        // 使用SIMD Neon计算精确内积
        float dist = InnerProductSIMDNeon(query, base + idx * dimension, dimension);
        
        if (result.size() < k) {
            result.push({dist, idx});
        } else if (dist < result.top().first) {
            result.pop();
            result.push({dist, idx});
        }
    }
    
    return result;
}

// ARM Neon优化的乘积量化索引
class PQIndexNeon {
public:
    PQIndexNeon() = default;
    
    // 从文件加载码本
    bool load_codebook(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return false;
        
        // 读取元数据
        in.read(reinterpret_cast<char*>(&m), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&ksub), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&dimension), sizeof(size_t));
        
        // 计算每个子空间的维度
        dsub = dimension / m;
        
        // 分配并读取码本
        codebook.resize(m * ksub * dsub);
        in.read(reinterpret_cast<char*>(codebook.data()), codebook.size() * sizeof(float));
        
        in.close();
        return true;
    }
    
    // 从文件加载量化后的编码
    bool load_codes(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return false;
        
        // 读取向量数量
        in.read(reinterpret_cast<char*>(&n_vectors), sizeof(size_t));
        
        // 分配并读取编码
        codes.resize(n_vectors * m);
        in.read(reinterpret_cast<char*>(codes.data()), codes.size());
        
        in.close();
        return true;
    }
    
    // 为查询向量计算距离表
    std::vector<float> compute_distance_table(const float* query) const {
        // 分配距离表空间: m个子空间，每个子空间ksub个聚类
        std::vector<float> distance_table(m * ksub);
        
        // 使用Neon加速距离表计算
        compute_multispace_distance_table(
            query, codebook.data(), distance_table.data(), m, dsub, ksub
        );
        
        return distance_table;
    }
    
    // 计算单个向量的距离
    float compute_distance(const std::vector<float>& distance_table, size_t idx) const {
        assert(idx < n_vectors);
        const uint8_t* code = &codes[idx * m];
        
        // 针对不同子空间数选择不同的计算方法
        if (m <= 8) {
            return compute_small_m_distance(code, distance_table.data(), m);
        } else {
            return compute_distance_with_prefetch(code, distance_table.data(), m, ksub);
        }
    }
    
    // PQ16搜索
    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k) const {
        
        // 计算距离表
        std::vector<float> distance_table = compute_distance_table(query);
        
        // 创建结果队列
        std::priority_queue<std::pair<float, uint32_t>> result;
        
        // 搜索每个向量
        for (size_t i = 0; i < n_vectors; i++) {
            float dist = compute_distance(distance_table, i);
            
            if (result.size() < k) {
                result.push({dist, i});
            } else if (dist < result.top().first) {
                result.pop();
                result.push({dist, i});
            }
        }
        
        return result;
    }
    
    // PQ16+精确重排序搜索
    std::priority_queue<std::pair<float, uint32_t>> search_with_rerank(
        const float* query, const float* base_data, size_t k, size_t rerank_k) const {
        
        // 直接使用优化后的pq16_rerank_search函数
        return pq16_rerank_search(
            base_data,                   // 原始向量库
            query,                       // 查询向量
            codes.data(),               // PQ编码数据
            codebook.data(),            // 码本数据
            n_vectors,                  // 向量数量
            dimension,                  // 向量维度
            m,                          // 子空间数量
            k,                          // 返回的结果数量
            rerank_k                    // 重排序候选数量
        );
    }
    
    // 批量搜索优化
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> batch_search(
        const float* queries, size_t n_queries, size_t k) const {
        
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(n_queries);
        
        #pragma omp parallel for
        for (size_t i = 0; i < n_queries; i++) {
            results[i] = search(queries + i * dimension, k);
        }
        
        return results;
    }
    
    // 批量搜索+精确重排序优化
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> batch_search_with_rerank(
        const float* queries, size_t n_queries, const float* base_data, 
        size_t k, size_t rerank_k) const {
        
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(n_queries);
        
        #pragma omp parallel for
        for (size_t i = 0; i < n_queries; i++) {
            results[i] = search_with_rerank(
                queries + i * dimension, base_data, k, rerank_k
            );
        }
        
        return results;
    }
    
    // 获取索引信息
    size_t get_dimension() const { return dimension; }
    size_t get_n_vectors() const { return n_vectors; }
    size_t get_subspaces() const { return m; }
    
private:
    size_t n_vectors = 0;   // 向量数量
    size_t dimension = 0;   // 向量维度
    size_t m = 0;           // 子空间数量
    size_t dsub = 0;        // 每个子空间的维度 (dimension / m)
    size_t ksub = 256;      // 每个子空间的聚类数量(固定为256，使用uint8_t存储)
    
    std::vector<float> codebook;       // 码本数据
    std::vector<uint8_t> codes;        // 量化编码
};

// 提供一个与flat_search类似的简单接口函数
// 用于在ARM平台上执行PQ16+精确重排序搜索
inline std::priority_queue<std::pair<float, uint32_t>> pq16_rerank_neon(
    float* base,             // 原始向量库
    float* query,            // 查询向量
    size_t base_number,      // 向量数量
    size_t vecdim,           // 向量维度
    size_t k) {              // 返回的结果数量
    
    static PQIndexNeon pq_index;
    static bool loaded = false;
    
    // 如果索引尚未加载，加载它
    if (!loaded) {
        // 使用与x86版本相同的索引文件
        if (!pq_index.load_codebook("files/pq16_codebook.bin")) {
            std::cerr << "无法加载码本，检查文件路径\n";
            return std::priority_queue<std::pair<float, uint32_t>>();
        }
        
        if (!pq_index.load_codes("files/pq16_codes.bin")) {
            std::cerr << "无法加载编码，检查文件路径\n";
            return std::priority_queue<std::pair<float, uint32_t>>();
        }
        
        loaded = true;
    }
    
    // 使用100个候选向量进行重排序(也可根据需要调整)
    const size_t rerank_k = 100;
    
    // 调用PQ16+精确重排序搜索
    return pq_index.search_with_rerank(query, base, k, rerank_k);
} 