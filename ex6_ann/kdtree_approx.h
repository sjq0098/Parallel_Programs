#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>

struct ApproxKDNode {
    std::vector<float> point;
    uint32_t index;
    int split_dim;
    ApproxKDNode* left;
    ApproxKDNode* right;
    
    ApproxKDNode(const std::vector<float>& p, uint32_t idx) 
        : point(p), index(idx), split_dim(-1), left(nullptr), right(nullptr) {}
    
    ~ApproxKDNode() {
        delete left;
        delete right;
    }
};

struct SearchNode {
    ApproxKDNode* node;
    float dist_to_query;
    float bound_dist;  // 到分割超平面的距离
    
    SearchNode(ApproxKDNode* n, float d, float b) : node(n), dist_to_query(d), bound_dist(b) {}
    
    bool operator<(const SearchNode& other) const {
        return bound_dist > other.bound_dist;  // 优先队列按bound_dist升序
    }
};

class ApproxKDTree {
private:
    ApproxKDNode* root;
    size_t dimension;
    size_t max_search_nodes;  // 最大搜索节点数，控制精度vs速度权衡
    
    ApproxKDNode* build_tree(std::vector<std::pair<std::vector<float>, uint32_t>>& points, int depth) {
        if (points.empty()) return nullptr;
        
        int split_dim = depth % dimension;
        
        // 按当前维度排序
        std::sort(points.begin(), points.end(), 
                 [split_dim](const auto& a, const auto& b) {
                     return a.first[split_dim] < b.first[split_dim];
                 });
        
        size_t median = points.size() / 2;
        ApproxKDNode* node = new ApproxKDNode(points[median].first, points[median].second);
        node->split_dim = split_dim;
        
        std::vector<std::pair<std::vector<float>, uint32_t>> left_points(
            points.begin(), points.begin() + median);
        std::vector<std::pair<std::vector<float>, uint32_t>> right_points(
            points.begin() + median + 1, points.end());
        
        node->left = build_tree(left_points, depth + 1);
        node->right = build_tree(right_points, depth + 1);
        
        return node;
    }
    
    float inner_product_distance(const std::vector<float>& a, const std::vector<float>& b) {
        float ip = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            ip += a[i] * b[i];
        }
        return 1 - ip;  // 与flat_scan.h保持一致
    }
    
    void search_knn_approx(const std::vector<float>& query, size_t k,
                          std::priority_queue<std::pair<float, uint32_t>>& best) {
        if (!root) return;
        
        // 使用best-bin-first搜索策略
        std::priority_queue<SearchNode> search_queue;
        size_t nodes_visited = 0;
        
        // 初始化搜索队列
        float init_dist = inner_product_distance(root->point, query);
        search_queue.emplace(root, init_dist, 0.0f);
        
        while (!search_queue.empty() && nodes_visited < max_search_nodes) {
            SearchNode current = search_queue.top();
            search_queue.pop();
            nodes_visited++;
            
            ApproxKDNode* node = current.node;
            if (!node) continue;
            
            // 计算到当前节点的距离
            float dist = inner_product_distance(node->point, query);
            
            // 更新最佳结果
            if (best.size() < k) {
                best.push({dist, node->index});
            } else if (dist < best.top().first) {
                best.pop();
                best.push({dist, node->index});
            }
            
            // 如果是叶子节点，跳过
            if (!node->left && !node->right) continue;
            
            int split_dim = node->split_dim;
            float split_val = node->point[split_dim];
            float query_val = query[split_dim];
            float axis_dist = std::abs(query_val - split_val);
            
            // 决定搜索顺序
            ApproxKDNode* near_child = (query_val <= split_val) ? node->left : node->right;
            ApproxKDNode* far_child = (query_val <= split_val) ? node->right : node->left;
            
            // 总是搜索近侧
            if (near_child) {
                float near_dist = inner_product_distance(near_child->point, query);
                search_queue.emplace(near_child, near_dist, 0.0f);
            }
            
            // 根据启发式决定是否搜索远侧
            bool should_search_far = false;
            if (far_child) {
                if (best.size() < k) {
                    should_search_far = true;
                } else {
                    // 使用最宽松的边界检查来冲击90%召回率  
                    float bound_factor = 7.0f;  // 最大程度放松边界
                    should_search_far = (axis_dist * bound_factor < best.top().first);
                }
                
                if (should_search_far) {
                    float far_dist = inner_product_distance(far_child->point, query);
                    search_queue.emplace(far_child, far_dist, axis_dist);
                }
            }
        }
    }
    
public:
    ApproxKDTree(float* base, size_t base_number, size_t vecdim, size_t max_nodes = 100) 
        : dimension(vecdim), max_search_nodes(max_nodes) {
        std::vector<std::pair<std::vector<float>, uint32_t>> points;
        points.reserve(base_number);
        
        for (size_t i = 0; i < base_number; ++i) {
            std::vector<float> point(vecdim);
            for (size_t d = 0; d < vecdim; ++d) {
                point[d] = base[d + i * vecdim];
            }
            points.emplace_back(std::move(point), i);
        }
        
        root = build_tree(points, 0);
    }
    
    ~ApproxKDTree() {
        delete root;
    }
    
    std::priority_queue<std::pair<float, uint32_t>> search(float* query, size_t k) {
        std::vector<float> query_vec(dimension);
        for (size_t d = 0; d < dimension; ++d) {
            query_vec[d] = query[d];
        }
        
        std::priority_queue<std::pair<float, uint32_t>> result;
        search_knn_approx(query_vec, k, result);
        return result;
    }
    
    void set_max_search_nodes(size_t max_nodes) {
        max_search_nodes = max_nodes;
    }
};

// 为了与其他算法接口保持一致
std::priority_queue<std::pair<float, uint32_t>> kdtree_approx_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    static ApproxKDTree* tree = nullptr;
    static bool initialized = false;
    
    if (!initialized) {
        tree = new ApproxKDTree(base, base_number, vecdim, 15000);  // 最终尝试15000节点冲击90%召回率
        initialized = true;
    }
    
    return tree->search(query, k);
} 