#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <omp.h>
#include <random>
#include <atomic>
#include <thread>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

struct EnsembleKDNode {
    std::vector<float> point;
    uint32_t index;
    int split_dim;
    EnsembleKDNode* left;
    EnsembleKDNode* right;
    
    EnsembleKDNode(const std::vector<float>& p, uint32_t idx) 
        : point(p), index(idx), split_dim(-1), left(nullptr), right(nullptr) {}
    
    ~EnsembleKDNode() {
        delete left;
        delete right;
    }
};

// 线程安全的结果收集器
class ThreadSafeResultCollector {
private:
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> thread_results;
    std::atomic<int> completed_threads{0};
    int total_threads;
    
public:
    ThreadSafeResultCollector(int num_threads) : total_threads(num_threads) {
        thread_results.resize(num_threads);
    }
    
    void add_result(int thread_id, float dist, uint32_t idx, size_t k) {
        auto& pq = thread_results[thread_id];
        if (pq.size() < k) {
            pq.push({dist, idx});
        } else if (dist < pq.top().first) {
            pq.pop();
            pq.push({dist, idx});
        }
    }
    
    std::priority_queue<std::pair<float, uint32_t>> merge_results(size_t k) {
        std::vector<std::pair<float, uint32_t>> all_results;
        
        // 收集所有线程的结果
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
};

class EnsembleParallelKDTree {
private:
    std::vector<EnsembleKDNode*> forest;  // 多棵树的森林
    size_t dimension;
    size_t max_search_nodes;
    size_t num_trees;
    std::mt19937 rng;
    
    // SIMD优化的距离计算
    float simd_inner_product_distance(const std::vector<float>& a, const std::vector<float>& b) {
        const size_t simd_width = 8;
        size_t simd_end = (dimension / simd_width) * simd_width;
        
        __m256 sum_vec = _mm256_setzero_ps();
        
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 a_vec = _mm256_loadu_ps(&a[i]);
            __m256 b_vec = _mm256_loadu_ps(&b[i]);
            __m256 mul_vec = _mm256_mul_ps(a_vec, b_vec);
            sum_vec = _mm256_add_ps(sum_vec, mul_vec);
        }
        
        // 水平加法
        float result[8];
        _mm256_storeu_ps(result, sum_vec);
        float ip = result[0] + result[1] + result[2] + result[3] + 
                   result[4] + result[5] + result[6] + result[7];
        
        // 处理剩余元素
        for (size_t i = simd_end; i < dimension; ++i) {
            ip += a[i] * b[i];
        }
        
        return 1 - ip;
    }
    
    // 随机化维度选择构建多样化的树
    EnsembleKDNode* build_randomized_tree(std::vector<std::pair<std::vector<float>, uint32_t>>& points, 
                                         int depth, std::mt19937& local_rng) {
        if (points.empty()) return nullptr;
        
        // 随机选择分割维度（从几个候选维度中选择）
        std::uniform_int_distribution<int> dim_dist(0, dimension - 1);
        std::vector<int> candidate_dims;
        
        // 选择5个候选维度
        for (int i = 0; i < 5 && i < static_cast<int>(dimension); ++i) {
            candidate_dims.push_back(dim_dist(local_rng));
        }
        
        // 从候选维度中选择方差最大的
        int best_dim = candidate_dims[0];
        float max_variance = 0;
        
        for (int dim : candidate_dims) {
            float mean = 0, variance = 0;
            for (const auto& p : points) {
                mean += p.first[dim];
            }
            mean /= points.size();
            
            for (const auto& p : points) {
                float diff = p.first[dim] - mean;
                variance += diff * diff;
            }
            variance /= points.size();
            
            if (variance > max_variance) {
                max_variance = variance;
                best_dim = dim;
            }
        }
        
        int split_dim = best_dim;
        
        // 排序
        std::sort(points.begin(), points.end(), 
                 [split_dim](const auto& a, const auto& b) {
                     return a.first[split_dim] < b.first[split_dim];
                 });
        
        size_t median = points.size() / 2;
        EnsembleKDNode* node = new EnsembleKDNode(points[median].first, points[median].second);
        node->split_dim = split_dim;
        
        std::vector<std::pair<std::vector<float>, uint32_t>> left_points(
            points.begin(), points.begin() + median);
        std::vector<std::pair<std::vector<float>, uint32_t>> right_points(
            points.begin() + median + 1, points.end());
        
        // 并行构建子树
        #pragma omp parallel sections if(points.size() > 1000)
        {
            #pragma omp section
            {
                node->left = build_randomized_tree(left_points, depth + 1, local_rng);
            }
            #pragma omp section
            {
                node->right = build_randomized_tree(right_points, depth + 1, local_rng);
            }
        }
        
        return node;
    }
    
    // 单棵树的搜索（优化版本）
    void search_single_tree(EnsembleKDNode* root, const std::vector<float>& query, 
                           size_t k, ThreadSafeResultCollector& collector, int thread_id) {
        if (!root) return;
        
        std::priority_queue<std::pair<float, EnsembleKDNode*>, 
                           std::vector<std::pair<float, EnsembleKDNode*>>,
                           std::greater<std::pair<float, EnsembleKDNode*>>> node_queue;
        
        size_t nodes_visited = 0;
        
        // 初始化搜索
        float init_dist = simd_inner_product_distance(root->point, query);
        node_queue.push({0.0f, root});
        
        while (!node_queue.empty() && nodes_visited < max_search_nodes) {
            auto [priority, node] = node_queue.top();
            node_queue.pop();
            nodes_visited++;
            
            if (!node) continue;
            
            // 计算到当前节点的实际距离
            float dist = simd_inner_product_distance(node->point, query);
            collector.add_result(thread_id, dist, node->index, k);
            
            // 如果是叶子节点，跳过
            if (!node->left && !node->right) continue;
            
            int split_dim = node->split_dim;
            float split_val = node->point[split_dim];
            float query_val = query[split_dim];
            float axis_dist = std::abs(query_val - split_val);
            
            EnsembleKDNode* near_child = (query_val <= split_val) ? node->left : node->right;
            EnsembleKDNode* far_child = (query_val <= split_val) ? node->right : node->left;
            
            // 总是访问近端
            if (near_child) {
                node_queue.push({0.0f, near_child});
            }
            
            // 根据启发式决定是否访问远端
            if (far_child) {
                // 使用更激进的剪枝策略
                float bound_factor = 7.0f;
                node_queue.push({axis_dist / bound_factor, far_child});
            }
        }
    }
    
public:
    EnsembleParallelKDTree(float* base, size_t base_number, size_t vecdim, 
                          size_t max_nodes = 8000, size_t trees = 8) 
        : dimension(vecdim), max_search_nodes(max_nodes), num_trees(trees), rng(std::random_device{}()) {
        
        forest.resize(num_trees);
        
        std::cout << "构建 " << num_trees << " 棵随机化KDTree..." << std::endl;
        
        // 并行构建多棵随机化的树
        #pragma omp parallel for schedule(dynamic)
        for (int tree_id = 0; tree_id < static_cast<int>(num_trees); ++tree_id) {
            std::mt19937 local_rng(rng() + tree_id);
            
            std::vector<std::pair<std::vector<float>, uint32_t>> points;
            points.reserve(base_number);
            
            // 为每棵树准备数据
            for (size_t i = 0; i < base_number; ++i) {
                std::vector<float> point(vecdim);
                for (size_t d = 0; d < vecdim; ++d) {
                    point[d] = base[d + i * vecdim];
                }
                points.emplace_back(std::move(point), i);
            }
            
            // 为每棵树随机打乱数据顺序，增加多样性
            std::shuffle(points.begin(), points.end(), local_rng);
            
            forest[tree_id] = build_randomized_tree(points, 0, local_rng);
        }
        
        std::cout << "KDTree森林构建完成！" << std::endl;
    }
    
    ~EnsembleParallelKDTree() {
        for (auto tree : forest) {
            delete tree;
        }
    }
    
    // 批量查询优化
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> 
    batch_search(std::vector<float*>& queries, size_t k) {
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(queries.size());
        
        #pragma omp parallel for schedule(dynamic, 1)
        for (int q = 0; q < static_cast<int>(queries.size()); ++q) {
            results[q] = search(queries[q], k);
        }
        
        return results;
    }
    
    // 单个查询（使用多树并行搜索）
    std::priority_queue<std::pair<float, uint32_t>> search(float* query, size_t k) {
        std::vector<float> query_vec(dimension);
        for (size_t d = 0; d < dimension; ++d) {
            query_vec[d] = query[d];
        }
        
        // 创建线程安全的结果收集器
        ThreadSafeResultCollector collector(num_trees);
        
        // 并行搜索所有树
        #pragma omp parallel for schedule(static)
        for (int tree_id = 0; tree_id < static_cast<int>(num_trees); ++tree_id) {
            search_single_tree(forest[tree_id], query_vec, k * 2, collector, tree_id);  // 每棵树收集更多候选
        }
        
        // 合并所有树的结果
        return collector.merge_results(k);
    }
    
    // 动态调整参数
    void tune_parameters(float target_recall) {
        if (target_recall > 0.8) {
            max_search_nodes = 12000;  // 增加搜索节点
        } else if (target_recall > 0.6) {
            max_search_nodes = 8000;
        } else {
            max_search_nodes = 4000;   // 减少搜索节点以提高速度
        }
    }
    
    // 获取统计信息
    void get_stats() {
        std::cout << "KDTree森林统计信息:" << std::endl;
        std::cout << "  树的数量: " << num_trees << std::endl;
        std::cout << "  数据维度: " << dimension << std::endl;
        std::cout << "  最大搜索节点: " << max_search_nodes << std::endl;
        std::cout << "  OpenMP线程数: " << omp_get_max_threads() << std::endl;
    }
};

// 接口函数
std::priority_queue<std::pair<float, uint32_t>> kdtree_ensemble_parallel_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    static EnsembleParallelKDTree* tree = nullptr;
    static bool initialized = false;
    
    if (!initialized) {
        tree = new EnsembleParallelKDTree(base, base_number, vecdim, 8000, 12);  // 12棵树
        // tree->get_stats();
        initialized = true;
    }
    
    return tree->search(query, k);
} 