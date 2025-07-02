#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <cmath>
#include <random>
#include <queue>
#include <omp.h>
#include "gpu_ivf.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct GroupingResult {
    std::string strategy_name;
    size_t batch_size;
    double avg_time_us;
    double avg_recall;
    double cluster_overlap_ratio;
    size_t total_matrix_ops;
};

// 计算内积距离
float inner_product_distance(const float* a, const float* b, size_t dim) {
    float dot_product = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot_product += a[i] * b[i];
    }
    return 1.0f - dot_product;
}

// 计算向量到簇中心的距离并返回最近的nprobe个簇
std::vector<uint32_t> find_nearest_clusters(const float* query, const std::vector<float>& centroids,
                                           size_t vecdim, size_t nlist, size_t nprobe) {
    std::vector<std::pair<float, uint32_t>> distances;
    distances.reserve(nlist);
    
    for (size_t i = 0; i < nlist; ++i) {
        float dot_product = 0.0f;
        for (size_t j = 0; j < vecdim; ++j) {
            dot_product += query[j] * centroids[i * vecdim + j];
        }
        float dist = 1.0f - dot_product;
        distances.push_back({dist, i});
    }
    
    std::partial_sort(distances.begin(), distances.begin() + nprobe, 
                     distances.end(), [](const std::pair<float, uint32_t>& a, 
                                       const std::pair<float, uint32_t>& b) {
        return a.first < b.first;
    });
    
    std::vector<uint32_t> result;
    result.reserve(nprobe);
    for (size_t i = 0; i < nprobe; ++i) {
        result.push_back(distances[i].second);
    }
    
    return result;
}

// 基准策略：无分组
std::vector<std::vector<size_t>> strategy_baseline(size_t query_count, size_t batch_size) {
    std::vector<std::vector<size_t>> batches;
    
    for (size_t i = 0; i < query_count; i += batch_size) {
        std::vector<size_t> batch;
        for (size_t j = i; j < std::min(i + batch_size, query_count); ++j) {
            batch.push_back(j);
        }
        batches.push_back(batch);
    }
    
    return batches;
}

// 新策略1: 自适应批大小策略
std::vector<std::vector<size_t>> strategy_adaptive_batch_size(
    float* queries, size_t query_count, size_t base_batch_size, size_t vecdim) {
    
    std::vector<std::vector<size_t>> batches;
    std::vector<bool> used(query_count, false);
    
    for (size_t start = 0; start < query_count; ) {
        if (used[start]) {
            start++;
            continue;
        }
        
        std::vector<size_t> current_batch;
        current_batch.push_back(start);
        used[start] = true;
        
        // 计算起始查询的"复杂度"（与其他查询的平均相似度）
        float avg_similarity = 0.0f;
        size_t sample_size = std::min(20UL, query_count - start - 1);
        
        for (size_t i = 1; i <= sample_size; ++i) {
            if (start + i < query_count) {
                avg_similarity += inner_product_distance(queries + start * vecdim, 
                                                        queries + (start + i) * vecdim, vecdim);
            }
        }
        if (sample_size > 0) avg_similarity /= sample_size;
        
        // 根据复杂度调整批大小
        size_t dynamic_batch_size;
        if (avg_similarity < 0.1) {  // 高相似性
            dynamic_batch_size = base_batch_size * 2;
        } else if (avg_similarity < 0.3) {  // 中等相似性
            dynamic_batch_size = base_batch_size;
        } else {  // 低相似性
            dynamic_batch_size = base_batch_size / 2;
        }
        dynamic_batch_size = std::max(dynamic_batch_size, 32UL);  // 最小批大小
        
        // 填充当前batch
        for (size_t i = start + 1; i < query_count && current_batch.size() < dynamic_batch_size; ++i) {
            if (!used[i]) {
                current_batch.push_back(i);
                used[i] = true;
            }
        }
        
        batches.push_back(current_batch);
        
        // 寻找下一个未使用的查询
        while (start < query_count && used[start]) {
            start++;
        }
    }
    
    return batches;
}

// 新策略2: 负载均衡分组
std::vector<std::vector<size_t>> strategy_load_balanced(
    float* queries, size_t query_count, size_t batch_size, size_t vecdim,
    const std::vector<float>& centroids, size_t nlist, size_t nprobe) {
    
    // 计算每个查询的"复杂度"（访问的簇数量）
    std::vector<std::pair<size_t, size_t>> query_complexity; // (complexity, query_idx)
    
    for (size_t i = 0; i < query_count; ++i) {
        auto clusters = find_nearest_clusters(queries + i * vecdim, centroids, vecdim, nlist, nprobe);
        size_t complexity = clusters.size();
        query_complexity.push_back({complexity, i});
    }
    
    // 按复杂度排序
    std::sort(query_complexity.begin(), query_complexity.end());
    
    // 使用贪心算法分配到batch中
    std::vector<std::vector<size_t>> batches;
    std::vector<size_t> batch_loads;
    
    for (const auto& [complexity, query_idx] : query_complexity) {
        // 找到当前负载最小的batch
        size_t min_load_batch = 0;
        if (!batches.empty()) {
            for (size_t i = 1; i < batches.size(); ++i) {
                if (batch_loads[i] < batch_loads[min_load_batch]) {
                    min_load_batch = i;
                }
            }
        }
        
        // 如果最小负载的batch已满，创建新batch
        if (batches.empty() || batches[min_load_batch].size() >= batch_size) {
            batches.push_back({query_idx});
            batch_loads.push_back(complexity);
        } else {
            batches[min_load_batch].push_back(query_idx);
            batch_loads[min_load_batch] += complexity;
        }
    }
    
    return batches;
}

// 新策略3: 局部性感知分组
std::vector<std::vector<size_t>> strategy_locality_aware(
    float* queries, size_t query_count, size_t batch_size, size_t vecdim) {
    
    std::vector<std::vector<size_t>> batches;
    std::vector<bool> used(query_count, false);
    
    for (size_t start = 0; start < query_count; ) {
        if (used[start]) {
            start++;
            continue;
        }
        
        std::vector<size_t> current_batch;
        current_batch.push_back(start);
        used[start] = true;
        
        // 在局部邻域内寻找相似查询
        size_t search_radius = std::min(batch_size * 4, query_count - start);
        std::vector<std::pair<float, size_t>> similarities;
        
        for (size_t i = start + 1; i < start + search_radius && i < query_count; ++i) {
            if (used[i]) continue;
            
            float sim = inner_product_distance(queries + start * vecdim, 
                                             queries + i * vecdim, vecdim);
            similarities.push_back({sim, i});
        }
        
        // 按相似度排序，选择最相似的查询
        std::sort(similarities.begin(), similarities.end());
        
        for (auto& [sim, idx] : similarities) {
            if (current_batch.size() >= batch_size) break;
            if (used[idx]) continue;
            
            current_batch.push_back(idx);
            used[idx] = true;
        }
        
        batches.push_back(current_batch);
        
        // 寻找下一个未使用的查询
        while (start < query_count && used[start]) {
            start++;
        }
    }
    
    return batches;
}

// 新策略4: 分层分组策略
std::vector<std::vector<size_t>> strategy_hierarchical_grouping(
    float* queries, size_t query_count, size_t batch_size, size_t vecdim,
    const std::vector<float>& centroids, size_t nlist, size_t nprobe) {
    
    // 第一层：按主要簇分组
    std::map<uint32_t, std::vector<size_t>> primary_groups;
    
    for (size_t i = 0; i < query_count; ++i) {
        auto clusters = find_nearest_clusters(queries + i * vecdim, centroids, vecdim, nlist, nprobe);
        uint32_t primary_cluster = clusters[0];  // 使用最近的簇
        primary_groups[primary_cluster].push_back(i);
    }
    
    // 第二层：在每个主要组内按相似度细分
    std::vector<std::vector<size_t>> batches;
    
    for (auto& [cluster_id, group_queries] : primary_groups) {
        if (group_queries.size() <= batch_size) {
            // 组足够小，直接作为一个batch
            batches.push_back(group_queries);
        } else {
            // 组太大，需要细分
            std::vector<bool> used(group_queries.size(), false);
            
            for (size_t start = 0; start < group_queries.size(); ) {
                if (used[start]) {
                    start++;
                    continue;
                }
                
                std::vector<size_t> current_batch;
                current_batch.push_back(group_queries[start]);
                used[start] = true;
                
                // 在当前组内寻找相似查询
                std::vector<std::pair<float, size_t>> similarities;
                for (size_t i = start + 1; i < group_queries.size(); ++i) {
                    if (used[i]) continue;
                    
                    float sim = inner_product_distance(queries + group_queries[start] * vecdim, 
                                                     queries + group_queries[i] * vecdim, vecdim);
                    similarities.push_back({sim, i});
                }
                
                std::sort(similarities.begin(), similarities.end());
                
                for (auto& [sim, local_idx] : similarities) {
                    if (current_batch.size() >= batch_size) break;
                    if (used[local_idx]) continue;
                    
                    current_batch.push_back(group_queries[local_idx]);
                    used[local_idx] = true;
                }
                
                batches.push_back(current_batch);
                
                while (start < group_queries.size() && used[start]) {
                    start++;
                }
            }
        }
    }
    
    return batches;
}

// 新策略5: 时间感知自适应分组
std::vector<std::vector<size_t>> strategy_time_aware_adaptive(
    float* queries, size_t query_count, size_t batch_size, size_t vecdim) {
    
    // 简化版：基于查询向量的方差来估计计算复杂度
    std::vector<std::pair<float, size_t>> query_variance; // (variance, query_idx)
    
    for (size_t i = 0; i < query_count; ++i) {
        float mean = 0.0f;
        for (size_t j = 0; j < vecdim; ++j) {
            mean += queries[i * vecdim + j];
        }
        mean /= vecdim;
        
        float variance = 0.0f;
        for (size_t j = 0; j < vecdim; ++j) {
            float diff = queries[i * vecdim + j] - mean;
            variance += diff * diff;
        }
        variance /= vecdim;
        
        query_variance.push_back({variance, i});
    }
    
    // 按方差排序
    std::sort(query_variance.begin(), query_variance.end());
    
    // 将相似方差的查询分到同一batch
    std::vector<std::vector<size_t>> batches;
    std::vector<size_t> current_batch;
    
    for (const auto& [variance, query_idx] : query_variance) {
        current_batch.push_back(query_idx);
        
        if (current_batch.size() >= batch_size) {
            batches.push_back(current_batch);
            current_batch.clear();
        }
    }
    
    if (!current_batch.empty()) {
        batches.push_back(current_batch);
    }
    
    return batches;
}

// 计算批次的簇重叠率
double calculate_cluster_overlap_ratio(const std::vector<size_t>& batch_indices, 
                                     float* queries, size_t vecdim,
                                     const std::vector<float>& centroids, 
                                     size_t nlist, size_t nprobe) {
    if (batch_indices.empty()) return 0.0;
    
    std::set<uint32_t> all_clusters;
    size_t total_clusters = 0;
    
    for (size_t idx : batch_indices) {
        auto clusters = find_nearest_clusters(queries + idx * vecdim, centroids, 
                                            vecdim, nlist, nprobe);
        for (uint32_t cluster : clusters) {
            all_clusters.insert(cluster);
        }
        total_clusters += nprobe;
    }
    
    double overlap_ratio = 1.0 - (double)all_clusters.size() / total_clusters;
    return overlap_ratio;
}

// 运行单个分组策略的测试
GroupingResult run_grouping_strategy_test(
    const std::string& strategy_name,
    const std::vector<std::vector<size_t>>& batches,
    float* base, float* queries, size_t base_number, size_t vecdim,
    size_t k, const std::vector<float>& centroids, size_t nlist,
    const std::vector<std::vector<uint32_t>>& invlists, size_t nprobe,
    int* test_gt, size_t test_gt_d, int repeat_count) {
    
    double total_time = 0.0;
    double total_recall = 0.0;
    double total_overlap = 0.0;
    size_t total_matrix_ops = 0;
    size_t total_queries = 0;
    
    for (int rep = 0; rep < repeat_count; ++rep) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (const auto& batch : batches) {
            if (batch.empty()) continue;
            
            std::vector<float> batch_queries(batch.size() * vecdim);
            for (size_t i = 0; i < batch.size(); ++i) {
                memcpy(batch_queries.data() + i * vecdim, 
                      queries + batch[i] * vecdim, vecdim * sizeof(float));
            }
            
            auto results = ivf_search_gpu(base, batch_queries.data(), base_number, vecdim,
                                        batch.size(), k, centroids, nlist, invlists, nprobe);
            
            if (rep == 0) {
                for (size_t i = 0; i < batch.size(); ++i) {
                    size_t query_idx = batch[i];
                    std::set<uint32_t> gtset;
                    for (int j = 0; j < k; ++j) {
                        gtset.insert(test_gt[j + query_idx * test_gt_d]);
                    }
                    
                    size_t acc = 0;
                    auto pq = results[i];
                    while (!pq.empty()) {
                        int x = pq.top().second;
                        if (gtset.find(x) != gtset.end()) {
                            ++acc;
                        }
                        pq.pop();
                    }
                    total_recall += (float)acc / k;
                }
                
                total_overlap += calculate_cluster_overlap_ratio(batch, queries, vecdim, 
                                                              centroids, nlist, nprobe);
                total_matrix_ops += batch.size();
                total_queries += batch.size();
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        total_time += duration.count();
    }
    
    GroupingResult result;
    result.strategy_name = strategy_name;
    result.batch_size = batches.empty() ? 0 : batches[0].size();
    result.avg_time_us = total_time / repeat_count / total_queries;
    result.avg_recall = total_recall / total_queries;
    result.cluster_overlap_ratio = total_overlap / batches.size();
    result.total_matrix_ops = total_matrix_ops;
    
    return result;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "用法: " << argv[0] << " <数据路径> <查询数> <重复次数> <暖机次数>" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];
    size_t query_count = std::stoi(argv[2]);
    int repeat_count = std::stoi(argv[3]);
    int warm_up_count = std::stoi(argv[4]);

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    size_t nlist = 256;
    size_t nprobe = 8;
    const size_t k = 10;
    
    std::string ivf_path = "file/";
    std::vector<float> centroids;
    std::vector<std::vector<uint32_t>> invlists;
    
    try {
        centroids = gpu_load_ivf_centroids(ivf_path + "ivf_flat_centroids_256.fbin", nlist, vecdim);
        invlists = gpu_load_ivf_invlists(ivf_path + "ivf_flat_invlists_256.bin", nlist);
        std::cout << "成功加载IVF数据，nlist=" << nlist << ", nprobe=" << nprobe << std::endl;
    } catch (const std::exception& e) {
        std::cout << "无法加载IVF数据: " << e.what() << std::endl;
        return 1;
    }

    query_count = std::min(query_count, test_number);
    
    // 测试关键的批大小
    std::vector<size_t> batch_sizes = {256, 512, 1024};
    
    std::ofstream out_file("advanced_grouping_results.csv");
    out_file << "分组策略,批大小,平均时间(us),平均召回率,簇重叠率,矩阵操作数,batch数量" << std::endl;
    
    std::cout << "开始高级分组策略实验..." << std::endl;
    std::cout << "查询数量: " << query_count << ", 重复次数: " << repeat_count << std::endl;
    
    // 暖机
    std::cout << "暖机 " << warm_up_count << " 条查询..." << std::endl;
    for (int i = 0; i < warm_up_count; ++i) {
        auto warm_result = ivf_search_gpu(base, test_query + i * vecdim, base_number, vecdim,
                                        1, k, centroids, nlist, invlists, nprobe);
    }
    
    for (size_t batch_size : batch_sizes) {
        std::cout << "\n测试批大小: " << batch_size << std::endl;
        
        // 基准策略
        std::cout << "  测试策略: 基准无分组..." << std::endl;
        auto batches_0 = strategy_baseline(query_count, batch_size);
        auto result_0 = run_grouping_strategy_test("基准无分组", batches_0, base, test_query,
                                                  base_number, vecdim, k, centroids, nlist,
                                                  invlists, nprobe, test_gt, test_gt_d, repeat_count);
        
        // 自适应批大小
        std::cout << "  测试策略: 自适应批大小..." << std::endl;
        auto batches_1 = strategy_adaptive_batch_size(test_query, query_count, batch_size, vecdim);
        auto result_1 = run_grouping_strategy_test("自适应批大小", batches_1, base, test_query,
                                                  base_number, vecdim, k, centroids, nlist,
                                                  invlists, nprobe, test_gt, test_gt_d, repeat_count);
        
        // 负载均衡
        std::cout << "  测试策略: 负载均衡..." << std::endl;
        auto batches_2 = strategy_load_balanced(test_query, query_count, batch_size,
                                               vecdim, centroids, nlist, nprobe);
        auto result_2 = run_grouping_strategy_test("负载均衡", batches_2, base, test_query,
                                                  base_number, vecdim, k, centroids, nlist,
                                                  invlists, nprobe, test_gt, test_gt_d, repeat_count);
        
        // 局部性感知
        std::cout << "  测试策略: 局部性感知..." << std::endl;
        auto batches_3 = strategy_locality_aware(test_query, query_count, batch_size, vecdim);
        auto result_3 = run_grouping_strategy_test("局部性感知", batches_3, base, test_query,
                                                  base_number, vecdim, k, centroids, nlist,
                                                  invlists, nprobe, test_gt, test_gt_d, repeat_count);
        
        // 分层分组
        std::cout << "  测试策略: 分层分组..." << std::endl;
        auto batches_4 = strategy_hierarchical_grouping(test_query, query_count, batch_size,
                                                       vecdim, centroids, nlist, nprobe);
        auto result_4 = run_grouping_strategy_test("分层分组", batches_4, base, test_query,
                                                  base_number, vecdim, k, centroids, nlist,
                                                  invlists, nprobe, test_gt, test_gt_d, repeat_count);
        
        // 时间感知自适应
        std::cout << "  测试策略: 时间感知自适应..." << std::endl;
        auto batches_5 = strategy_time_aware_adaptive(test_query, query_count, batch_size, vecdim);
        auto result_5 = run_grouping_strategy_test("时间感知自适应", batches_5, base, test_query,
                                                  base_number, vecdim, k, centroids, nlist,
                                                  invlists, nprobe, test_gt, test_gt_d, repeat_count);
        
        // 输出结果
        std::vector<std::pair<GroupingResult, size_t>> batch_results = {
            {result_0, batches_0.size()}, {result_1, batches_1.size()}, 
            {result_2, batches_2.size()}, {result_3, batches_3.size()},
            {result_4, batches_4.size()}, {result_5, batches_5.size()}
        };
        
        for (const auto& [result, num_batches] : batch_results) {
            std::cout << "    " << result.strategy_name 
                      << ": 时间=" << std::fixed << std::setprecision(2) << result.avg_time_us << "us"
                      << ", 召回率=" << std::setprecision(4) << result.avg_recall
                      << ", 重叠率=" << std::setprecision(3) << result.cluster_overlap_ratio 
                      << ", batch数=" << num_batches << std::endl;
            
            out_file << result.strategy_name << "," << batch_size << ","
                     << result.avg_time_us << "," << result.avg_recall << ","
                     << result.cluster_overlap_ratio << "," << result.total_matrix_ops 
                     << "," << num_batches << std::endl;
        }
    }
    
    out_file.close();
    std::cout << "\n高级分组策略实验完成！结果已保存到 advanced_grouping_results.csv" << std::endl;
    
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    return 0;
} 