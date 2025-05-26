#pragma once
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <utility>  // 添加utility头文件以支持std::pair
#include <limits>   // 添加limits头文件以支持std::numeric_limits

// IVF PQ索引结构体定义
struct IVFPQIndex {
    size_t nlist;     // 聚类中心数量
    size_t d;         // 原始特征维度
    size_t m;         // PQ子向量组数
    size_t ksub;      // 每个子空间的聚类中心数量(通常为256)
    size_t dsub;      // 每个子空间的维度 (d/m)
    
    std::vector<float> centroids;            // IVF聚类中心，大小为nlist*d
    std::vector<std::vector<uint32_t>> invlists; // 倒排列表
    std::vector<std::vector<uint8_t>> codes;     // PQ编码，每个倒排列表对应的向量PQ编码
    std::vector<float> pq_codebooks;         // PQ码本，大小为m*ksub*dsub
    
    IVFPQIndex() : nlist(0), d(0), m(0), ksub(0), dsub(0) {}
};

// 从文件加载IVFPQ索引
std::unique_ptr<IVFPQIndex> load_ivfpq_index(const std::string& filename) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("无法打开索引文件: " + filename);
    }
    
    auto index = std::make_unique<IVFPQIndex>();
    
    try {
        // 读取索引参数
        int32_t nlist, d, m, ksub;
        fin.read(reinterpret_cast<char*>(&nlist), sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(&m), sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(&ksub), sizeof(int32_t));
        
        if (nlist <= 0 || d <= 0 || m <= 0 || ksub <= 0 || d % m != 0) {
            throw std::runtime_error("无效的索引参数: nlist=" + std::to_string(nlist) + 
                                    ", d=" + std::to_string(d) + 
                                    ", m=" + std::to_string(m) + 
                                    ", ksub=" + std::to_string(ksub));
        }
        
        index->nlist = nlist;
        index->d = d;
        index->m = m;
        index->ksub = ksub;
        index->dsub = d / m;
        
        //std::cerr << "加载IVFPQ索引: nlist=" << nlist << ", d=" << d 
        //          << ", m=" << m << ", ksub=" << ksub << std::endl;
        
        // 预先分配内存以避免动态扩展
        try {
            // 读取IVF聚类中心
            const size_t centroids_size = nlist * d;
            index->centroids.resize(centroids_size);
            fin.read(reinterpret_cast<char*>(index->centroids.data()), sizeof(float) * centroids_size);
            
            if (!fin) {
                throw std::runtime_error("读取聚类中心失败");
            }
            
            // 读取PQ码本
            const size_t codebook_size = m * ksub * index->dsub;
            index->pq_codebooks.resize(codebook_size);
            fin.read(reinterpret_cast<char*>(index->pq_codebooks.data()), sizeof(float) * codebook_size);
            
            if (!fin) {
                throw std::runtime_error("读取PQ码本失败");
            }
            
            // 预分配倒排列表和PQ编码
            index->invlists.resize(nlist);
            index->codes.resize(nlist);
            
            // 读取倒排列表和PQ编码
            for (size_t i = 0; i < nlist; ++i) {
                int32_t list_size;
                fin.read(reinterpret_cast<char*>(&list_size), sizeof(int32_t));
                
                if (list_size < 0 || !fin) {
                    throw std::runtime_error("读取倒排列表大小失败, 列表 " + std::to_string(i));
                }
                
                if (list_size > 0) {
                    try {
                        // 读取倒排列表ID
                        index->invlists[i].resize(list_size);
                        fin.read(reinterpret_cast<char*>(index->invlists[i].data()), 
                               sizeof(uint32_t) * list_size);
                        
                        // 读取PQ编码
                        index->codes[i].resize(list_size * m);
                        fin.read(reinterpret_cast<char*>(index->codes[i].data()), 
                               sizeof(uint8_t) * list_size * m);
                        
                        if (!fin) {
                            throw std::runtime_error("读取倒排列表或PQ编码失败, 列表 " + std::to_string(i));
                        }
                    } catch (const std::exception& e) {
                        throw std::runtime_error("处理倒排列表 " + std::to_string(i) + 
                                              " 时发生错误: " + e.what());
                    }
                }
            }
        } catch (const std::bad_alloc&) {
            throw std::runtime_error("内存分配失败，无法加载索引");
        }
        
        if (!fin) {
            throw std::runtime_error("读取索引文件失败: " + filename);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("加载IVFPQ索引失败: " + std::string(e.what()));
    }
    
    return index;
}

// 计算PQ与查询向量的查找表
std::vector<float> compute_pq_distance_table(const IVFPQIndex* index, const float* query) {
    std::vector<float> distance_table(index->m * index->ksub);
    
   // L2距离平方
for (size_t m = 0; m < index->m; ++m) {
    const float* query_sub = query + m * index->dsub;
    for (size_t k = 0; k < index->ksub; ++k) {
        const float* cb = index->pq_codebooks.data() + (m*index->ksub + k)*index->dsub;
        float dist2 = 0;
        for (size_t d = 0; d < index->dsub; ++d) {
            float diff = query_sub[d] - cb[d];
            dist2 += diff * diff;
        }
        distance_table[m*index->ksub + k] = dist2;
    }
}

    return distance_table;
}

// 串行版IVFPQ搜索
// 串行版 IVFPQ 搜索（使用 ADC，即 L2 距离平方）
std::priority_queue<std::pair<float, uint32_t>> ivfpq_search(
    const IVFPQIndex* index,
    float* query,
    size_t k,
    size_t nprobe)
{
    // 1. 找出与查询最近的 nprobe 个聚类中心（基于 L2 距离平方）
    struct CentDist { float dist2; uint32_t idx; };
    std::vector<CentDist> cd(index->nlist);
    
    for (size_t i = 0; i < index->nlist; ++i) {
        const float* cptr = index->centroids.data() + i * index->d;
        float dist2 = 0;
        for (size_t d = 0; d < index->d; ++d) {
            float diff = query[d] - cptr[d];
            dist2 += diff * diff;
        }
        cd[i].dist2 = dist2;
        cd[i].idx   = uint32_t(i);
    }
    
    // 部分排序，找出前 nprobe 个最小 dist2
    if (nprobe < index->nlist) {
        std::nth_element(
            cd.begin(), cd.begin() + nprobe, cd.end(),
            [](auto &a, auto &b) {
                return a.dist2 < b.dist2;
            }
        );
    }
    
    // 2. 计算 PQ 与查询向量在每个子空间的 L2 距离表（已由用户提供的 compute_pq_distance_table 实现）
    std::vector<float> pq_dist_table = compute_pq_distance_table(index, query);
    
    // 3. 对选定的倒排列表中的向量计算近似 L2 距离，并用一个 max-heap 保持 top-k 最小距离
    std::priority_queue<std::pair<float, uint32_t>> result;
    
    for (size_t pi = 0; pi < nprobe && pi < index->nlist; ++pi) {
        uint32_t list_id = cd[pi].idx;
        const auto& ids   = index->invlists[list_id];
        const auto& codes = index->codes[list_id];
        
        for (size_t i = 0; i < ids.size(); ++i) {
            uint32_t id = ids[i];
            float dist2 = 0;
            
            // 累加每个子空间的 L2 距离平方
            for (size_t m = 0; m < index->m; ++m) {
                uint8_t code = codes[i * index->m + m];
                dist2 += pq_dist_table[m * index->ksub + code];
            }
            
            // 维护一个大小为 k 的 max-heap，heap.top() 保持当前最大的 dist2
            if (result.size() < k) {
                result.emplace(dist2, id);
            } else if (dist2 < result.top().first) {
                result.pop();
                result.emplace(dist2, id);
            }
        }
    }
    
    return result;
}

// 串行版本的重排序函数
std::vector<std::pair<float, uint32_t>> rerank_with_serial(
    const float* base,
    const float* query,
    const std::vector<std::pair<float, uint32_t>>& candidates,
    size_t d,
    size_t k)
{
    std::vector<std::pair<float, uint32_t>> result(candidates);
    
    // 对每个候选计算精确L2距离
    for (size_t i = 0; i < candidates.size(); ++i) {
        uint32_t id = candidates[i].second;
        const float* vec = base + id * d;
        float dist2 = 0;
        for (size_t j = 0; j < d; ++j) {
            float diff = query[j] - vec[j];
            dist2 += diff * diff;
        }
        result[i].first = dist2;
    }
    
    // 部分排序，只保留前k个结果
    if (k < result.size()) {
        std::partial_sort(result.begin(), result.begin() + k, result.end(),
            [](auto &a, auto &b) { return a.first < b.first; });
        result.resize(k);
    } else {
        std::sort(result.begin(), result.end(),
            [](auto &a, auto &b) { return a.first < b.first; });
    }
    
    return result;
}

// 带重排序的串行IVFPQ搜索
std::priority_queue<std::pair<float, uint32_t>> ivfpq_search_rerank(
    const IVFPQIndex* index,
    const float* base,
    float* query,
    size_t k,
    size_t nprobe,
    size_t L)
{
    // 1. 找出与查询最近的 nprobe 个聚类中心（基于 L2 距离平方）
    struct CentDist { float dist2; uint32_t idx; };
    std::vector<CentDist> cd(index->nlist);
    
    for (size_t i = 0; i < index->nlist; ++i) {
        const float* cptr = index->centroids.data() + i * index->d;
        float dist2 = 0;
        for (size_t d = 0; d < index->d; ++d) {
            float diff = query[d] - cptr[d];
            dist2 += diff * diff;
        }
        cd[i].dist2 = dist2;
        cd[i].idx   = uint32_t(i);
    }
    
    // 部分排序，找出前 nprobe 个最小 dist2
    if (nprobe < index->nlist) {
        std::nth_element(
            cd.begin(), cd.begin() + nprobe, cd.end(),
            [](auto &a, auto &b) {
                return a.dist2 < b.dist2;
            }
        );
    }
    
    // 2. 计算 PQ 与查询向量在每个子空间的 L2 距离表
    std::vector<float> pq_dist_table = compute_pq_distance_table(index, query);
    
    // 3. 对选定的倒排列表中的向量计算近似 L2 距离，生成候选
    std::vector<std::pair<float, uint32_t>> candidates;
    
    for (size_t pi = 0; pi < nprobe && pi < index->nlist; ++pi) {
        uint32_t list_id = cd[pi].idx;
        const auto& ids   = index->invlists[list_id];
        const auto& codes = index->codes[list_id];
        
        for (size_t i = 0; i < ids.size(); ++i) {
            uint32_t id = ids[i];
            float dist2 = 0;
            
            // 累加每个子空间的 L2 距离平方
            for (size_t m = 0; m < index->m; ++m) {
                uint8_t code = codes[i * index->m + m];
                dist2 += pq_dist_table[m * index->ksub + code];
            }
            
            candidates.emplace_back(dist2, id);
        }
    }
    
    // 4. 部分排序，选取Top-L个候选
    size_t rerank_size = (std::min)(L, candidates.size());
    if (rerank_size < candidates.size()) {
        std::nth_element(
            candidates.begin(), 
            candidates.begin() + rerank_size, 
            candidates.end(),
            [](auto &a, auto &b) { return a.first < b.first; }
        );
        candidates.resize(rerank_size);
    }
    
    // 5. Re-rank: 对Top-L个候选进行精确距离计算
    auto reranked = rerank_with_serial(base, query, candidates, index->d, k);
    
    // 6. 转换为优先队列返回
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (const auto& p : reranked) {
        result.push(p);
    }
    
    return result;
}
