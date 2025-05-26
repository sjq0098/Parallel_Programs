#pragma once
#include <omp.h>
#include <vector>
#include <queue>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <memory>

#include "hnswlib/hnswlib/hnswlib.h"

// HNSW索引管理类
class HNSWIndex_omp {
private:
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg;
    size_t dim;
    
public:
    HNSWIndex_omp(size_t dimension, const std::string& index_path, size_t max_elements) 
        : dim(dimension) {
        try {
            space = std::make_unique<hnswlib::InnerProductSpace>(dim);
            alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                space.get(), 
                index_path,
                false,  // 非nmslib格式
                max_elements,
                false   // 不替换已删除元素
            );
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("无法加载HNSW索引 '" + index_path + 
                                   "'。错误: " + e.what());
        } catch (...) {
            throw std::runtime_error("加载HNSW索引 '" + index_path + 
                                   "' 时发生未知错误。");
        }
    }

    // 设置搜索参数
    void setEf(size_t ef) {
        alg->setEf(ef);
    }

    // 执行KNN搜索
    std::priority_queue<std::pair<float, uint32_t>> flat_hnsw_search(
        const float* query,
        size_t k) const {
        try {
            auto result = alg->searchKnn_omp(query, k);
            
            // 转换结果类型（避免中间vector）
            std::priority_queue<std::pair<float, uint32_t>> final_result;
            while (!result.empty()) {
                auto [dist, label] = result.top();
                if (label > std::numeric_limits<uint32_t>::max()) {
                    throw std::overflow_error("HNSW索引标签(ID)过大，无法容纳于uint32_t。");
                }
                final_result.push({dist, static_cast<uint32_t>(label)});
                result.pop();
            }
            return final_result;
            
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("HNSW搜索期间出错: ") + e.what());
        }
    }
};
