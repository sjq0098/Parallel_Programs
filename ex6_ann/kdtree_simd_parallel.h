#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <omp.h>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

struct SIMDKDNode {
    std::vector<float> point;
    uint32_t index;
    int split_dim;
    SIMDKDNode* left;
    SIMDKDNode* right;
    
    SIMDKDNode(const std::vector<float>& p, uint32_t idx) 
        : point(p), index(idx), split_dim(-1), left(nullptr), right(nullptr) {}
    
    ~SIMDKDNode() {
        delete left;
        delete right;
    }
};

struct SIMDSearchNode {
    SIMDKDNode* node;
    float dist_to_query;
    float bound_dist;
    
    SIMDSearchNode(SIMDKDNode* n, float d, float b) : node(n), dist_to_query(d), bound_dist(b) {}
    
    bool operator<(const SIMDSearchNode& other) const {
        return bound_dist > other.bound_dist;
    }
};

class SIMDParallelKDTree {
private:
    SIMDKDNode* root;
    size_t dimension;
    size_t max_search_nodes;
    
    SIMDKDNode* build_tree(std::vector<std::pair<std::vector<float>, uint32_t>>& points, int depth) {
        if (points.empty()) return nullptr;
        
        int split_dim = depth % dimension;
        
        // 并行排序优化
        std::sort(points.begin(), points.end(), 
                 [split_dim](const auto& a, const auto& b) {
                     return a.first[split_dim] < b.first[split_dim];
                 });
        
        size_t median = points.size() / 2;
        SIMDKDNode* node = new SIMDKDNode(points[median].first, points[median].second);
        node->split_dim = split_dim;
        
        std::vector<std::pair<std::vector<float>, uint32_t>> left_points(
            points.begin(), points.begin() + median);
        std::vector<std::pair<std::vector<float>, uint32_t>> right_points(
            points.begin() + median + 1, points.end());
        
        // 并行构建左右子树
        #pragma omp parallel sections if(points.size() > 1000)
        {
            #pragma omp section
            {
                node->left = build_tree(left_points, depth + 1);
            }
            #pragma omp section
            {
                node->right = build_tree(right_points, depth + 1);
            }
        }
        
        return node;
    }
    
    // SIMD优化的内积距离计算
    float simd_inner_product_distance(const std::vector<float>& a, const std::vector<float>& b) {
        const size_t simd_width = 8;  // AVX可以并行处理8个float
        size_t simd_end = (dimension / simd_width) * simd_width;
        
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
        for (size_t i = simd_end; i < dimension; ++i) {
            ip += a[i] * b[i];
        }
        
        return 1 - ip;  // 与flat_scan.h保持一致
    }
    
    // 并行搜索多个节点
    void parallel_search_knn(const std::vector<float>& query, size_t k,
                            std::priority_queue<std::pair<float, uint32_t>>& best,
                            std::vector<SIMDKDNode*>& nodes_to_search) {
        
        std::vector<std::pair<float, uint32_t>> local_candidates;
        
        // 并行计算所有节点的距离
        #pragma omp parallel
        {
            std::vector<std::pair<float, uint32_t>> thread_candidates;
            
            #pragma omp for schedule(dynamic, 32)
            for (int i = 0; i < static_cast<int>(nodes_to_search.size()); ++i) {
                SIMDKDNode* node = nodes_to_search[i];
                if (node) {
                    float dist = simd_inner_product_distance(node->point, query);
                    thread_candidates.emplace_back(dist, node->index);
                }
            }
            
            // 合并线程结果
            #pragma omp critical
            {
                local_candidates.insert(local_candidates.end(), 
                                      thread_candidates.begin(), 
                                      thread_candidates.end());
            }
        }
        
        // 选择最好的k个结果
        std::sort(local_candidates.begin(), local_candidates.end());
        for (size_t i = 0; i < std::min(k, local_candidates.size()); ++i) {
            if (best.size() < k) {
                best.push(local_candidates[i]);
            } else if (local_candidates[i].first < best.top().first) {
                best.pop();
                best.push(local_candidates[i]);
            }
        }
    }
    
    void search_knn_simd_parallel(const std::vector<float>& query, size_t k,
                                 std::priority_queue<std::pair<float, uint32_t>>& best) {
        if (!root) return;
        
        std::priority_queue<SIMDSearchNode> search_queue;
        std::vector<SIMDKDNode*> leaf_nodes;
        size_t nodes_visited = 0;
        
        // 收集叶子节点进行并行处理
        float init_dist = simd_inner_product_distance(root->point, query);
        search_queue.emplace(root, init_dist, 0.0f);
        
        while (!search_queue.empty() && nodes_visited < max_search_nodes) {
            SIMDSearchNode current = search_queue.top();
            search_queue.pop();
            nodes_visited++;
            
            SIMDKDNode* node = current.node;
            if (!node) continue;
            
            // 如果是叶子节点，添加到并行处理列表
            if (!node->left && !node->right) {
                leaf_nodes.push_back(node);
                continue;
            }
            
            // 计算到当前节点的距离
            float dist = simd_inner_product_distance(node->point, query);
            
            // 更新最佳结果
            if (best.size() < k) {
                best.push({dist, node->index});
            } else if (dist < best.top().first) {
                best.pop();
                best.push({dist, node->index});
            }
            
            // 继续搜索子节点
            int split_dim = node->split_dim;
            float split_val = node->point[split_dim];
            float query_val = query[split_dim];
            float axis_dist = std::abs(query_val - split_val);
            
            SIMDKDNode* near_child = (query_val <= split_val) ? node->left : node->right;
            SIMDKDNode* far_child = (query_val <= split_val) ? node->right : node->left;
            
            // 添加子节点到搜索队列
            if (near_child) {
                float near_dist = simd_inner_product_distance(near_child->point, query);
                search_queue.emplace(near_child, near_dist, 0.0f);
            }
            
            if (far_child) {
                bool should_search_far = false;
                if (best.size() < k) {
                    should_search_far = true;
                } else {
                    float bound_factor = 7.0f;
                    should_search_far = (axis_dist * bound_factor < best.top().first);
                }
                
                if (should_search_far) {
                    float far_dist = simd_inner_product_distance(far_child->point, query);
                    search_queue.emplace(far_child, far_dist, axis_dist);
                }
            }
        }
        
        // 并行处理收集到的叶子节点
        if (!leaf_nodes.empty()) {
            parallel_search_knn(query, k, best, leaf_nodes);
        }
    }
    
public:
    SIMDParallelKDTree(float* base, size_t base_number, size_t vecdim, size_t max_nodes = 8000) 
        : dimension(vecdim), max_search_nodes(max_nodes) {
        
        std::vector<std::pair<std::vector<float>, uint32_t>> points;
        points.reserve(base_number);
        
        // 并行数据预处理
        points.resize(base_number);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(base_number); ++i) {
            std::vector<float> point(vecdim);
            for (size_t d = 0; d < vecdim; ++d) {
                point[d] = base[d + i * vecdim];
            }
            points[i] = std::make_pair(std::move(point), i);
        }
        
        root = build_tree(points, 0);
    }
    
    ~SIMDParallelKDTree() {
        delete root;
    }
    
    std::priority_queue<std::pair<float, uint32_t>> search(float* query, size_t k) {
        std::vector<float> query_vec(dimension);
        for (size_t d = 0; d < dimension; ++d) {
            query_vec[d] = query[d];
        }
        
        std::priority_queue<std::pair<float, uint32_t>> result;
        search_knn_simd_parallel(query_vec, k, result);
        return result;
    }
    
    void set_max_search_nodes(size_t max_nodes) {
        max_search_nodes = max_nodes;
    }
};

// 接口函数
std::priority_queue<std::pair<float, uint32_t>> kdtree_simd_parallel_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    static SIMDParallelKDTree* tree = nullptr;
    static bool initialized = false;
    
    if (!initialized) {
        tree = new SIMDParallelKDTree(base, base_number, vecdim, 8000);
        initialized = true;
    }
    
    return tree->search(query, k);
} 