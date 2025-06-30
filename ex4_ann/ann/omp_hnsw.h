#pragma once

#include<omp.h>
#include <vector>
#include <queue>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <memory>

#include "hnswlib/hnswlib/hnswlib.h"

// HNSW索引管理类，适配C++11
class HNSWIndex {
private:
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg;
    size_t dim;
    size_t max_elements;

public:
    HNSWIndex(size_t dimension, const std::string& index_path, size_t max_elements_in_index)
        : dim(dimension), max_elements(max_elements_in_index)
    {
        space.reset(new hnswlib::InnerProductSpace(dim));
        alg.reset(new hnswlib::HierarchicalNSW<float>(
            space.get(),
            index_path,
            false,  // 非 nmslib 格式
            max_elements,
            false   // 不替换已删除元素
        ));
    }

    // 设置搜索参数
    void setEf(size_t ef) {
        alg->setEf(ef);
    }

    // 执行KNN搜索
    std::priority_queue<std::pair<float, uint32_t>> omp_hnsw_search(
        const float* query,
        size_t k) const
    {
        // 调用 hnswlib 的 searchKnn
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg->searchKnn_omp(query, k);

        std::priority_queue<std::pair<float, uint32_t>> final_result;
        while (!result.empty()) {
            std::pair<float, hnswlib::labeltype> p = result.top();
            float dist = p.first;
            hnswlib::labeltype label = p.second;
            if (label > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("HNSW索引标签(ID)过大，无法容纳于uint32_t。");
            }
            final_result.push(std::make_pair(dist, static_cast<uint32_t>(label)));
            result.pop();
        }
        return final_result;
    }
};
