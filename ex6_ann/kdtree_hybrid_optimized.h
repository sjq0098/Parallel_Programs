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
#include <memory>
#include <iostream>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

struct HybridKDNode {
    std::vector<float> point;
    uint32_t index;
    int split_dim;
    std::unique_ptr<HybridKDNode> left;
    std::unique_ptr<HybridKDNode> right;
    
    // 缓存友好性优化：预计算常用信息
    float split_value;
    
    HybridKDNode(const std::vector<float>& p, uint32_t idx) 
        : point(p), index(idx), split_dim(-1), split_value(0.0f) {}
};

// 改进的线程安全结果收集器
class OptimizedResultCollector {
private:
    struct alignas(64) ThreadLocalResult {  // 缓存行对齐避免false sharing
        std::priority_queue<std::pair<float, uint32_t>> pq;
        char padding[64];  // 填充避免false sharing
    };
    
    std::vector<ThreadLocalResult> thread_results;
    int num_threads;
    
public:
    OptimizedResultCollector(int threads) : num_threads(threads) {
        thread_results.resize(threads);
    }
    
    void add_result(int thread_id, float dist, uint32_t idx, size_t k) {
        auto& pq = thread_results[thread_id].pq;
        if (pq.size() < k) {
            pq.push({dist, idx});
        } else if (dist < pq.top().first) {
            pq.pop();
            pq.push({dist, idx});
        }
    }
    
    std::priority_queue<std::pair<float, uint32_t>> merge_results(size_t k) {
        std::vector<std::pair<float, uint32_t>> all_results;
        all_results.reserve(k * num_threads);  // 预分配内存
        
        for (auto& tr : thread_results) {
            while (!tr.pq.empty()) {
                all_results.push_back(tr.pq.top());
                tr.pq.pop();
            }
        }
        
        // 使用nth_element优化选择
        if (all_results.size() > k) {
            std::nth_element(all_results.begin(), all_results.begin() + k, all_results.end());
            all_results.resize(k);
        }
        std::sort(all_results.begin(), all_results.end());
        
        std::priority_queue<std::pair<float, uint32_t>> final_result;
        for (const auto& result : all_results) {
            final_result.push(result);
        }
        
        return final_result;
    }
};

class HybridOptimizedKDTree {
private:
    std::vector<std::unique_ptr<HybridKDNode>> forest;
    size_t dimension;
    size_t num_trees;
    std::mt19937 rng;
    
    // 自适应参数
    size_t base_search_nodes;
    float recall_target;
    std::atomic<size_t> current_search_nodes{5000};
    
    // 内存池优化
    static constexpr size_t MEMORY_ALIGNMENT = 32;
    
    // 高度优化的SIMD距离计算
    inline float simd_distance_optimized(const std::vector<float>& a, const std::vector<float>& b) {
        const size_t simd_width = 8;
        size_t simd_end = (dimension / simd_width) * simd_width;
        
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        
        // 循环展开提高ILP (指令级并行)
        size_t i = 0;
        for (; i + 16 <= simd_end; i += 16) {
            __m256 a1 = _mm256_loadu_ps(&a[i]);
            __m256 b1 = _mm256_loadu_ps(&b[i]);
            __m256 a2 = _mm256_loadu_ps(&a[i + 8]);
            __m256 b2 = _mm256_loadu_ps(&b[i + 8]);
            
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);  // FMA指令
            sum2 = _mm256_fmadd_ps(a2, b2, sum2);
        }
        
        // 处理剩余的SIMD块
        for (; i < simd_end; i += 8) {
            __m256 a_vec = _mm256_loadu_ps(&a[i]);
            __m256 b_vec = _mm256_loadu_ps(&b[i]);
            sum1 = _mm256_fmadd_ps(a_vec, b_vec, sum1);
        }
        
        // 水平求和
        sum1 = _mm256_add_ps(sum1, sum2);
        float result[8];
        _mm256_storeu_ps(result, sum1);
        float ip = result[0] + result[1] + result[2] + result[3] + 
                   result[4] + result[5] + result[6] + result[7];
        
        // 处理剩余标量元素
        for (; i < dimension; ++i) {
            ip += a[i] * b[i];
        }
        
        return 1.0f - ip;
    }
    
    // 改进的随机化树构建，使用更好的启发式
    std::unique_ptr<HybridKDNode> build_optimized_tree(
        std::vector<std::pair<std::vector<float>, uint32_t>>& points, 
        int depth, std::mt19937& local_rng, int tree_id) {
        
        if (points.empty()) return nullptr;
        
        // 智能维度选择：结合随机性和方差分析
        std::uniform_int_distribution<int> dim_dist(0, dimension - 1);
        std::vector<int> candidate_dims;
        
        // 每棵树使用不同的随机化策略增加多样性
        int num_candidates = 3 + (tree_id % 5);  // 3-7个候选维度
        for (int i = 0; i < num_candidates && i < static_cast<int>(dimension); ++i) {
            candidate_dims.push_back(dim_dist(local_rng));
        }
        
        // 选择最佳分割维度
        int best_dim = candidate_dims[0];
        float max_variance = 0;
        
        for (int dim : candidate_dims) {
            // 快速方差估算（采样方式）
            size_t sample_size = std::min(points.size(), size_t(1000));
            std::uniform_int_distribution<size_t> sample_dist(0, points.size() - 1);
            
            float sum = 0, sum_sq = 0;
            for (size_t s = 0; s < sample_size; ++s) {
                size_t idx = (points.size() <= 1000) ? s : sample_dist(local_rng);
                float val = points[idx].first[dim];
                sum += val;
                sum_sq += val * val;
            }
            
            float mean = sum / sample_size;
            float variance = (sum_sq / sample_size) - (mean * mean);
            
            if (variance > max_variance) {
                max_variance = variance;
                best_dim = dim;
            }
        }
        
        // 排序（对小数据集使用插入排序优化）
        if (points.size() < 50) {
            std::sort(points.begin(), points.end(), 
                     [best_dim](const auto& a, const auto& b) {
                         return a.first[best_dim] < b.first[best_dim];
                     });
        } else {
            // 使用快速选择找到中位数，避免完全排序
            size_t median_idx = points.size() / 2;
            std::nth_element(points.begin(), points.begin() + median_idx, points.end(),
                           [best_dim](const auto& a, const auto& b) {
                               return a.first[best_dim] < b.first[best_dim];
                           });
        }
        
        size_t median = points.size() / 2;
        auto node = std::make_unique<HybridKDNode>(points[median].first, points[median].second);
        node->split_dim = best_dim;
        node->split_value = points[median].first[best_dim];  // 缓存分割值
        
        std::vector<std::pair<std::vector<float>, uint32_t>> left_points(
            points.begin(), points.begin() + median);
        std::vector<std::pair<std::vector<float>, uint32_t>> right_points(
            points.begin() + median + 1, points.end());
        
        // 自适应并行策略
        bool use_parallel = points.size() > 500 && depth < 10;
        
        if (use_parallel) {
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    node->left = build_optimized_tree(left_points, depth + 1, local_rng, tree_id);
                }
                #pragma omp section
                {
                    node->right = build_optimized_tree(right_points, depth + 1, local_rng, tree_id);
                }
            }
        } else {
            node->left = build_optimized_tree(left_points, depth + 1, local_rng, tree_id);
            node->right = build_optimized_tree(right_points, depth + 1, local_rng, tree_id);
        }
        
        return node;
    }
    
    // 高度优化的单树搜索
    void search_optimized_tree(HybridKDNode* root, const std::vector<float>& query, 
                              size_t k, OptimizedResultCollector& collector, 
                              int thread_id, size_t max_nodes) {
        if (!root) return;
        
        // 使用自定义堆栈避免递归开销
        struct SearchNode {
            HybridKDNode* node;
            float priority;
            
            bool operator>(const SearchNode& other) const {
                return priority > other.priority;
            }
        };
        
        std::priority_queue<SearchNode, std::vector<SearchNode>, 
                           std::greater<SearchNode>> node_queue;
        
        size_t nodes_visited = 0;
        node_queue.push({root, 0.0f});
        
        // 预分配向量避免动态分配
        std::vector<SearchNode> next_nodes;
        next_nodes.reserve(64);
        
        while (!node_queue.empty() && nodes_visited < max_nodes) {
            SearchNode current = node_queue.top();
            node_queue.pop();
            nodes_visited++;
            
            HybridKDNode* node = current.node;
            if (!node) continue;
            
            // 计算距离并更新结果
            float dist = simd_distance_optimized(node->point, query);
            collector.add_result(thread_id, dist, node->index, k);
            
            // 如果是叶子节点，继续
            if (!node->left && !node->right) continue;
            
            // 使用预缓存的split_value
            float query_val = query[node->split_dim];
            float axis_dist = std::abs(query_val - node->split_value);
            
            HybridKDNode* near_child = (query_val <= node->split_value) ? 
                                      node->left.get() : node->right.get();
            HybridKDNode* far_child = (query_val <= node->split_value) ? 
                                     node->right.get() : node->left.get();
            
            // 智能剪枝策略
            if (near_child) {
                node_queue.push({near_child, 0.0f});
            }
            
            if (far_child) {
                // 自适应边界因子
                float bound_factor = 5.0f + (nodes_visited * 0.001f);  // 动态调整
                float priority = axis_dist / bound_factor;
                
                if (priority < 1.0f) {  // 只有在足够接近时才添加
                    node_queue.push({far_child, priority});
                }
            }
        }
    }
    
public:
    HybridOptimizedKDTree(float* base, size_t base_number, size_t vecdim, 
                         size_t search_nodes = 8000, size_t trees = 10,
                         float target_recall = 0.9) 
        : dimension(vecdim), num_trees(trees), base_search_nodes(search_nodes),
          recall_target(target_recall), rng(std::random_device{}()) {
        
        forest.resize(num_trees);
        current_search_nodes.store(search_nodes);
        
        std::cout << "构建智能混合KDTree森林 (" << num_trees << "棵树)..." << std::endl;
        
        // 并行构建多棵优化的树
        #pragma omp parallel for schedule(dynamic)
        for (int tree_id = 0; tree_id < static_cast<int>(num_trees); ++tree_id) {
            std::mt19937 local_rng(rng() + tree_id * 12345);
            
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
            
            // 每棵树使用不同的随机化程度
            if (tree_id % 3 == 0) {
                std::shuffle(points.begin(), points.end(), local_rng);
            } else if (tree_id % 3 == 1) {
                // 部分随机化
                for (size_t i = 0; i < points.size(); i += 10) {
                    size_t end = std::min(i + 10, points.size());
                    std::shuffle(points.begin() + i, points.begin() + end, local_rng);
                }
            }
            // tree_id % 3 == 2 保持原始顺序
            
            forest[tree_id] = build_optimized_tree(points, 0, local_rng, tree_id);
        }
        
        std::cout << "智能混合KDTree森林构建完成！" << std::endl;
    }
    
    // 自适应搜索
    std::priority_queue<std::pair<float, uint32_t>> search(float* query, size_t k) {
        std::vector<float> query_vec(dimension);
        for (size_t d = 0; d < dimension; ++d) {
            query_vec[d] = query[d];
        }
        
        OptimizedResultCollector collector(num_trees);
        size_t search_nodes = current_search_nodes.load();
        
        // 并行搜索所有树
        #pragma omp parallel for schedule(static)
        for (int tree_id = 0; tree_id < static_cast<int>(num_trees); ++tree_id) {
            // 不同树使用略微不同的搜索参数增加多样性
            size_t tree_search_nodes = search_nodes + (tree_id * 200);
            search_optimized_tree(forest[tree_id].get(), query_vec, k * 2, 
                                collector, tree_id, tree_search_nodes);
        }
        
        return collector.merge_results(k);
    }
    
    // 动态参数调整
    void tune_performance(float observed_recall) {
        if (observed_recall < recall_target - 0.05f) {
            // 召回率过低，增加搜索节点
            size_t new_nodes = current_search_nodes.load() * 1.2f;
            current_search_nodes.store(std::min(new_nodes, size_t(20000)));
        } else if (observed_recall > recall_target + 0.05f) {
            // 召回率过高，可以减少搜索节点提高速度
            size_t new_nodes = current_search_nodes.load() * 0.9f;
            current_search_nodes.store(std::max(new_nodes, size_t(2000)));
        }
    }
    
    void get_status() {
        std::cout << "混合优化KDTree状态:" << std::endl;
        std::cout << "  森林大小: " << num_trees << " 棵树" << std::endl;
        std::cout << "  当前搜索节点: " << current_search_nodes.load() << std::endl;
        std::cout << "  目标召回率: " << recall_target << std::endl;
        std::cout << "  OpenMP线程: " << omp_get_max_threads() << std::endl;
    }
};

// 接口函数
std::priority_queue<std::pair<float, uint32_t>> kdtree_hybrid_optimized_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    static HybridOptimizedKDTree* tree = nullptr;
    static bool initialized = false;
    
    if (!initialized) {
        tree = new HybridOptimizedKDTree(base, base_number, vecdim, 8000, 10, 0.95);
        initialized = true;
    }
    
    return tree->search(query, k);
} 