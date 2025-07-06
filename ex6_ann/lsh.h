#pragma once
#include <queue>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <cmath>
#include <cstdint>

class LSH {
private:
    size_t dimension;
    size_t num_tables;
    size_t hash_size;
    std::vector<std::vector<std::vector<float>>> hash_functions;  // [table][hash_func][dim]
    std::vector<std::unordered_map<size_t, std::vector<uint32_t>>> hash_tables;
    
    float inner_product_distance(const float* a, const float* b, size_t dim) {
        float ip = 0;
        for (size_t i = 0; i < dim; ++i) {
            ip += a[i] * b[i];
        }
        return 1 - ip;  // 与flat_scan.h保持一致
    }
    
    size_t compute_hash(const float* point, int table_id) {
        size_t hash_value = 0;
        for (size_t i = 0; i < hash_size; ++i) {
            float projection = 0;
            for (size_t d = 0; d < dimension; ++d) {
                projection += point[d] * hash_functions[table_id][i][d];
            }
            // 符号随机投影：>0为1，<=0为0
            if (projection > 0) {
                hash_value |= (1ULL << i);
            }
        }
        return hash_value;
    }
    
public:
    LSH(size_t dim, size_t tables = 10, size_t hash_bits = 16) 
        : dimension(dim), num_tables(tables), hash_size(hash_bits) {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0, 1.0);
        
        hash_functions.resize(num_tables);
        hash_tables.resize(num_tables);
        
        // 初始化随机投影向量
        for (size_t t = 0; t < num_tables; ++t) {
            hash_functions[t].resize(hash_size);
            for (size_t h = 0; h < hash_size; ++h) {
                hash_functions[t][h].resize(dimension);
                for (size_t d = 0; d < dimension; ++d) {
                    hash_functions[t][h][d] = dist(gen);
                }
            }
        }
    }
    
    void insert(float* base, size_t base_number) {
        for (size_t i = 0; i < base_number; ++i) {
            float* point = base + i * dimension;
            for (size_t t = 0; t < num_tables; ++t) {
                size_t hash_value = compute_hash(point, t);
                hash_tables[t][hash_value].push_back(i);
            }
        }
    }
    
    std::priority_queue<std::pair<float, uint32_t>> search(
        float* base, float* query, size_t k) {
        
        std::unordered_set<uint32_t> candidates;
        
        // 收集候选点
        for (size_t t = 0; t < num_tables; ++t) {
            size_t hash_value = compute_hash(query, t);
            if (hash_tables[t].find(hash_value) != hash_tables[t].end()) {
                for (uint32_t idx : hash_tables[t][hash_value]) {
                    candidates.insert(idx);
                }
            }
        }
        
        // 如果候选点太少，使用更多的桶
        if (candidates.size() < k * 2) {
            for (size_t t = 0; t < num_tables; ++t) {
                size_t query_hash = compute_hash(query, t);
                // 检查相邻的哈希桶（汉明距离为1）
                for (size_t bit = 0; bit < hash_size; ++bit) {
                    size_t neighbor_hash = query_hash ^ (1ULL << bit);
                    if (hash_tables[t].find(neighbor_hash) != hash_tables[t].end()) {
                        for (uint32_t idx : hash_tables[t][neighbor_hash]) {
                            candidates.insert(idx);
                        }
                    }
                }
            }
        }
        
        // 计算距离并返回top-k
        std::priority_queue<std::pair<float, uint32_t>> result;
        for (uint32_t idx : candidates) {
            float* point = base + idx * dimension;
            float dist = inner_product_distance(point, query, dimension);
            
            if (result.size() < k) {
                result.push({dist, idx});
            } else if (dist < result.top().first) {
                result.pop();
                result.push({dist, idx});
            }
        }
        
        return result;
    }
};

// 为了与flat_search接口保持一致
std::priority_queue<std::pair<float, uint32_t>> lsh_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    static LSH* lsh_index = nullptr;
    static bool initialized = false;
    
    if (!initialized) {
        lsh_index = new LSH(vecdim, 15, 18);  // 15个表，每个18位哈希
        lsh_index->insert(base, base_number);
        initialized = true;
    }
    
    return lsh_index->search(base, query, k);
} 