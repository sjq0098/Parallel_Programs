#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>

struct KDNode {
    std::vector<float> point;
    uint32_t index;
    int split_dim;
    KDNode* left;
    KDNode* right;
    
    KDNode(const std::vector<float>& p, uint32_t idx) 
        : point(p), index(idx), split_dim(-1), left(nullptr), right(nullptr) {}
    
    ~KDNode() {
        delete left;
        delete right;
    }
};

class KDTree {
private:
    KDNode* root;
    size_t dimension;
    
    KDNode* build_tree(std::vector<std::pair<std::vector<float>, uint32_t>>& points, int depth) {
        if (points.empty()) return nullptr;
        
        int split_dim = depth % dimension;
        
        // 按当前维度排序
        std::sort(points.begin(), points.end(), 
                 [split_dim](const auto& a, const auto& b) {
                     return a.first[split_dim] < b.first[split_dim];
                 });
        
        size_t median = points.size() / 2;
        KDNode* node = new KDNode(points[median].first, points[median].second);
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
    
    void search_knn(KDNode* node, const std::vector<float>& query, size_t k,
                   std::priority_queue<std::pair<float, uint32_t>>& best) {
        if (!node) return;
        
        float dist = inner_product_distance(node->point, query);
        
        if (best.size() < k) {
            best.push({dist, node->index});
        } else if (dist < best.top().first) {
            best.pop();
            best.push({dist, node->index});
        }
        
        int split_dim = node->split_dim;
        KDNode* first_branch = nullptr;
        KDNode* second_branch = nullptr;
        
        if (query[split_dim] <= node->point[split_dim]) {
            first_branch = node->left;
            second_branch = node->right;
        } else {
            first_branch = node->right;
            second_branch = node->left;
        }
        
        search_knn(first_branch, query, k, best);
        
        // 检查是否需要搜索另一边
        float axis_dist = std::abs(query[split_dim] - node->point[split_dim]);
        if (best.size() < k || axis_dist < best.top().first) {
            search_knn(second_branch, query, k, best);
        }
    }
    
public:
    KDTree(float* base, size_t base_number, size_t vecdim) : dimension(vecdim) {
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
    
    ~KDTree() {
        delete root;
    }
    
    std::priority_queue<std::pair<float, uint32_t>> search(float* query, size_t k) {
        std::vector<float> query_vec(dimension);
        for (size_t d = 0; d < dimension; ++d) {
            query_vec[d] = query[d];
        }
        
        std::priority_queue<std::pair<float, uint32_t>> result;
        search_knn(root, query_vec, k, result);
        return result;
    }
};

// 为了与flat_search接口保持一致
std::priority_queue<std::pair<float, uint32_t>> kdtree_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    KDTree tree(base, base_number, vecdim);
    return tree.search(query, k);
} 