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
    
    // 用于rerank的原始数据
    std::vector<float> raw_data;             // 用于rerank的原始数据，按ID索引
    bool has_raw_data;                      // 是否包含原始数据
    
    IVFPQIndex() : nlist(0), d(0), m(0), ksub(0), dsub(0), has_raw_data(false) {}
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
        
        std::cerr << "加载IVFPQ索引: nlist=" << nlist << ", d=" << d 
                  << ", m=" << m << ", ksub=" << ksub << std::endl;
        
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

// 加载原始数据用于重排序
bool load_raw_data_for_rerank(IVFPQIndex* index, const std::string& raw_data_file) {
    try {
        std::ifstream fin(raw_data_file, std::ios::binary);
        if (!fin) {
            std::cerr << "无法打开原始数据文件: " << raw_data_file << std::endl;
            return false;
        }
        
        // 读取数据维度和数量
        uint32_t n, d;
        fin.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
        fin.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        
        if (d != index->d) {
            std::cerr << "原始数据维度 (" << d << ") 与索引维度 (" << index->d << ") 不匹配" << std::endl;
            return false;
        }
        
        // 读取所有原始数据
        index->raw_data.resize(n * d);
        fin.read(reinterpret_cast<char*>(index->raw_data.data()), sizeof(float) * n * d);
        
        if (!fin) {
            std::cerr << "读取原始数据失败" << std::endl;
            return false;
        }
        
        index->has_raw_data = true;
        std::cerr << "成功加载原始数据用于重排序: " << n << " 向量" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "加载原始数据失败: " << e.what() << std::endl;
        return false;
    }
}

// 计算两向量间的内积
inline float inner_product(const float* a, const float* b, size_t d) {
    float result = 0;
    for (size_t i = 0; i < d; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// 计算PQ与查询向量的查找表
std::vector<float> compute_pq_distance_table(const IVFPQIndex* index, const float* query) {
    std::vector<float> distance_table(index->m * index->ksub);
    
    for (size_t m = 0; m < index->m; ++m) {
        const float* query_sub = query + m * index->dsub;
        
        for (size_t k = 0; k < index->ksub; ++k) {
            const float* cb = index->pq_codebooks.data() + (m * index->ksub + k) * index->dsub;
            float dot = 0;
            
            for (size_t d = 0; d < index->dsub; ++d) {
                dot += query_sub[d] * cb[d];
            }
            
            distance_table[m * index->ksub + k] = dot;
        }
    }
    
    return distance_table;
}

// 串行版IVFPQ搜索（带重排序选项）
std::priority_queue<std::pair<float, uint32_t>> ivfpq_search(
    const IVFPQIndex* index,
    float* query,
    size_t k,
    size_t nprobe,
    bool do_rerank = false,
    size_t rerank_candidates = 0)  // 改为0，表示自动计算
{
    // 动态计算重排序候选数量，如果未指定则使用 k * nprobe * 8
    if (rerank_candidates == 0) {
        rerank_candidates = k * nprobe * 8;
    }
    
    // 1. 找出与查询最近的nprobe个聚类中心
    struct CentDist { float dist; uint32_t idx; };
    std::vector<CentDist> cd(index->nlist);
    
    for (size_t i = 0; i < index->nlist; ++i) {
        const float* cptr = index->centroids.data() + i * index->d;
        float dot = 0;
        
        for (size_t d = 0; d < index->d; ++d) {
            dot += cptr[d] * query[d];
        }
        
        cd[i].dist = 1.f - dot;
        cd[i].idx = uint32_t(i);
    }
    
    if (nprobe < index->nlist) {
        std::nth_element(
            cd.begin(), cd.begin() + nprobe, cd.end(),
            [](const CentDist &a, const CentDist &b) {
                return a.dist < b.dist;
            }
        );
    }
    
    // 2. 计算PQ与查询向量的距离表
    std::vector<float> pq_dist_table = compute_pq_distance_table(index, query);
    
    // 3. 对选定的倒排列表中的向量计算近似距离
    std::priority_queue<std::pair<float, uint32_t>> result;
    
    // 如果需要重排序，进行多层级筛选
    if (do_rerank && index->has_raw_data) {
        // 每个簇保留的最大候选数量（确保均衡贡献）
        size_t per_list_cap = (rerank_candidates + nprobe - 1) / nprobe;
        
        // 保存每个簇的局部候选
        std::vector<std::vector<std::pair<float, uint32_t>>> per_list_candidates(nprobe);
        
        // 第一阶段：对每个簇局部筛选top-per_list_cap
        for (size_t pi = 0; pi < nprobe && pi < index->nlist; ++pi) {
            uint32_t list_id = cd[pi].idx;
            const auto& ids = index->invlists[list_id];
            const auto& codes = index->codes[list_id];
            
            // 使用max-heap维护每个簇的top-per_list_cap
            std::priority_queue<std::pair<float, uint32_t>, 
                               std::vector<std::pair<float, uint32_t>>, 
                               std::less<std::pair<float, uint32_t>>> list_heap;
            
            for (size_t i = 0; i < ids.size(); ++i) {
                uint32_t id = ids[i];
                float sum_dist = 0;
                
                // 根据PQ编码查表计算近似距离，确保使用内积度量
                for (size_t m = 0; m < index->m; ++m) {
                    uint8_t code = codes[i * index->m + m];
                    sum_dist += pq_dist_table[m * index->ksub + code];
                }
                
                // 将IP距离转换为距离度量 (1-dot)，确保与后续精排保持一致
                float dist = 1.f - sum_dist;
                
                // 维护每个簇的局部top-per_list_cap
                if (list_heap.size() < per_list_cap) {
                    list_heap.emplace(dist, id);
                } else if (dist < list_heap.top().first) {
                    list_heap.pop();
                    list_heap.emplace(dist, id);
                }
            }
            
            // 将簇的局部候选保存
            auto& list_candidates = per_list_candidates[pi];
            list_candidates.reserve(list_heap.size());
            while (!list_heap.empty()) {
                list_candidates.push_back(list_heap.top());
                list_heap.pop();
            }
        }
        
        // 第二阶段：合并所有簇的候选并全局筛选top-rerank_candidates
        std::vector<std::pair<float, uint32_t>> all_candidates;
        size_t total_candidates = 0;
        for (const auto& list_cands : per_list_candidates) {
            total_candidates += list_cands.size();
        }
        all_candidates.reserve(total_candidates);
        
        for (const auto& list_cands : per_list_candidates) {
            all_candidates.insert(all_candidates.end(), list_cands.begin(), list_cands.end());
        }
        
        // 全局筛选top-rerank_candidates
        std::priority_queue<std::pair<float, uint32_t>, 
                           std::vector<std::pair<float, uint32_t>>, 
                           std::greater<std::pair<float, uint32_t>>> global_heap;
        
        for (const auto& candidate : all_candidates) {
            if (global_heap.size() < rerank_candidates) {
                global_heap.push(candidate);
            } else if (candidate.first < global_heap.top().first) {
                global_heap.pop();
                global_heap.push(candidate);
            }
        }
        
        // 转换为vector并排序
        std::vector<std::pair<float, uint32_t>> final_candidates;
        final_candidates.reserve(global_heap.size());
        while (!global_heap.empty()) {
            final_candidates.push_back(global_heap.top());
            global_heap.pop();
        }
        
        // 按近似距离排序（从小到大）
        std::sort(final_candidates.begin(), final_candidates.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // 第三阶段：使用原始向量重新计算精确距离
        for (const auto& candidate : final_candidates) {
            uint32_t id = candidate.second;
            const float* vec = index->raw_data.data() + id * index->d;
            
            // 计算精确内积，保持与PQ距离相同的度量（1-dot）
            float exact_ip = inner_product(query, vec, index->d);
            float exact_dist = 1.0f - exact_ip;
            
            if (result.size() < k) {
                result.emplace(exact_dist, id);
            } else if (exact_dist < result.top().first) {
                result.pop();
                result.emplace(exact_dist, id);
            }
        }
    } else {
        // 不进行重排序，使用PQ近似距离
        for (size_t pi = 0; pi < nprobe && pi < index->nlist; ++pi) {
            uint32_t list_id = cd[pi].idx;
            const auto& ids = index->invlists[list_id];
            const auto& codes = index->codes[list_id];
            
            for (size_t i = 0; i < ids.size(); ++i) {
                uint32_t id = ids[i];
                float sum_dist = 0;
                
                // 根据PQ编码查表计算近似距离
                for (size_t m = 0; m < index->m; ++m) {
                    uint8_t code = codes[i * index->m + m];
                    sum_dist += pq_dist_table[m * index->ksub + code];
                }
                
                // 将IP距离转换为L2距离
                float dist = 1.f - sum_dist;
                
                if (result.size() < k) {
                    result.emplace(dist, id);
                } else if (dist < result.top().first) {
                    result.pop();
                    result.emplace(dist, id);
                }
            }
        }
    }
    
    return result;
} 