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
    double cluster_overlap_ratio;  // 簇重叠率
    size_t total_matrix_ops;       // 总矩阵运算次数
};

// 计算两个向量的内积距离 (1 - dot_product)
float inner_product_distance(const float* a, const float* b, size_t dim) {
    float dot_product = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot_product += a[i] * b[i];
    }
    return 1.0f - dot_product;  // 内积距离，值越小表示越相似
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
        float dist = 1.0f - dot_product;  // 内积距离
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

// 分组策略0: 无分组 (基准方法)
std::vector<std::vector<size_t>> strategy_no_grouping(size_t query_count, size_t batch_size) {
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

// 分组策略1: 基于簇相似性的分组
std::vector<std::vector<size_t>> strategy_cluster_based_grouping(
    float* queries, size_t query_count, size_t batch_size, size_t vecdim,
    const std::vector<float>& centroids, size_t nlist, size_t nprobe) {
    
    // 为每个查询找到最近的簇
    std::vector<std::vector<uint32_t>> query_clusters(query_count);
    for (size_t i = 0; i < query_count; ++i) {
        query_clusters[i] = find_nearest_clusters(queries + i * vecdim, centroids, 
                                                 vecdim, nlist, nprobe);
    }
    
    // 按照主要簇ID分组
    std::map<uint32_t, std::vector<size_t>> cluster_groups;
    for (size_t i = 0; i < query_count; ++i) {
        uint32_t primary_cluster = query_clusters[i][0];  // 使用最近的簇作为主要簇
        cluster_groups[primary_cluster].push_back(i);
    }
    
    // 将分组结果组织成batch
    std::vector<std::vector<size_t>> batches;
    for (auto& pair : cluster_groups) {
        auto& queries_in_cluster = pair.second;
        
        // 将该簇的查询分成多个batch
        for (size_t i = 0; i < queries_in_cluster.size(); i += batch_size) {
            std::vector<size_t> batch;
            for (size_t j = i; j < std::min(i + batch_size, queries_in_cluster.size()); ++j) {
                batch.push_back(queries_in_cluster[j]);
            }
            batches.push_back(batch);
        }
    }
    
    return batches;
}

// 分组策略2: 基于查询向量相似性的分组
std::vector<std::vector<size_t>> strategy_query_similarity_grouping(
    float* queries, size_t query_count, size_t batch_size, size_t vecdim) {
    
    // 使用K-means对查询向量进行聚类
    size_t num_clusters = std::max(1UL, query_count / batch_size);
    std::vector<size_t> assignments(query_count);
    
    // 简化的K-means聚类
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, query_count - 1);
    
    // 随机选择初始聚类中心
    std::vector<std::vector<float>> cluster_centers(num_clusters, std::vector<float>(vecdim));
    for (size_t i = 0; i < num_clusters; ++i) {
        size_t random_idx = dis(gen);
        memcpy(cluster_centers[i].data(), queries + random_idx * vecdim, vecdim * sizeof(float));
    }
    
    // 运行几轮K-means
    for (int iter = 0; iter < 5; ++iter) {
        // 分配查询到最近的簇中心
        for (size_t i = 0; i < query_count; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            size_t best_cluster = 0;
            
            for (size_t j = 0; j < num_clusters; ++j) {
                float dist = inner_product_distance(queries + i * vecdim, 
                                                   cluster_centers[j].data(), vecdim);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
        }
        
        // 更新簇中心
        std::vector<std::vector<float>> new_centers(num_clusters, std::vector<float>(vecdim, 0.0f));
        std::vector<size_t> counts(num_clusters, 0);
        
        for (size_t i = 0; i < query_count; ++i) {
            size_t cluster_id = assignments[i];
            for (size_t j = 0; j < vecdim; ++j) {
                new_centers[cluster_id][j] += queries[i * vecdim + j];
            }
            counts[cluster_id]++;
        }
        
        for (size_t i = 0; i < num_clusters; ++i) {
            if (counts[i] > 0) {
                for (size_t j = 0; j < vecdim; ++j) {
                    cluster_centers[i][j] = new_centers[i][j] / counts[i];
                }
            }
        }
    }
    
    // 根据聚类结果分组
    std::vector<std::vector<size_t>> cluster_queries(num_clusters);
    for (size_t i = 0; i < query_count; ++i) {
        cluster_queries[assignments[i]].push_back(i);
    }
    
    // 将聚类结果组织成batch
    std::vector<std::vector<size_t>> batches;
    for (auto& cluster : cluster_queries) {
        for (size_t i = 0; i < cluster.size(); i += batch_size) {
            std::vector<size_t> batch;
            for (size_t j = i; j < std::min(i + batch_size, cluster.size()); ++j) {
                batch.push_back(cluster[j]);
            }
            batches.push_back(batch);
        }
    }
    
    return batches;
}

// 分组策略3: 混合策略 (簇相似性 + 查询相似性)
std::vector<std::vector<size_t>> strategy_hybrid_grouping(
    float* queries, size_t query_count, size_t batch_size, size_t vecdim,
    const std::vector<float>& centroids, size_t nlist, size_t nprobe) {
    
    // 首先按簇分组
    auto cluster_batches = strategy_cluster_based_grouping(queries, query_count, batch_size * 2, 
                                                          vecdim, centroids, nlist, nprobe);
    
    // 然后在每个簇内按查询相似性细分
    std::vector<std::vector<size_t>> final_batches;
    
    for (auto& cluster_batch : cluster_batches) {
        if (cluster_batch.size() <= batch_size) {
            final_batches.push_back(cluster_batch);
        } else {
            // 在该簇内进行查询相似性分组
            std::vector<float> cluster_queries(cluster_batch.size() * vecdim);
            for (size_t i = 0; i < cluster_batch.size(); ++i) {
                memcpy(cluster_queries.data() + i * vecdim, 
                      queries + cluster_batch[i] * vecdim, vecdim * sizeof(float));
            }
            
            auto sub_batches = strategy_query_similarity_grouping(
                cluster_queries.data(), cluster_batch.size(), batch_size, vecdim);
            
            // 将子批次的索引映射回原始查询索引
            for (auto& sub_batch : sub_batches) {
                std::vector<size_t> mapped_batch;
                for (size_t idx : sub_batch) {
                    mapped_batch.push_back(cluster_batch[idx]);
                }
                final_batches.push_back(mapped_batch);
            }
        }
    }
    
    return final_batches;
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
    
    // 重叠率 = 1 - (唯一簇数 / 总簇数)
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
            
            // 准备批次查询数据
            std::vector<float> batch_queries(batch.size() * vecdim);
            for (size_t i = 0; i < batch.size(); ++i) {
                memcpy(batch_queries.data() + i * vecdim, 
                      queries + batch[i] * vecdim, vecdim * sizeof(float));
            }
            
            // 执行GPU IVF搜索
            auto results = ivf_search_gpu(base, batch_queries.data(), base_number, vecdim,
                                        batch.size(), k, centroids, nlist, invlists, nprobe);
            
            // 计算recall (只在第一次重复时计算)
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
                
                // 计算簇重叠率
                total_overlap += calculate_cluster_overlap_ratio(batch, queries, vecdim, 
                                                              centroids, nlist, nprobe);
                total_matrix_ops += batch.size();  // 简化的矩阵操作计数
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

    // 加载数据
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    // 加载IVF数据
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

    // 限制查询数量
    query_count = std::min(query_count, test_number);
    
    // 测试不同批大小
    std::vector<size_t> batch_sizes = {64, 128, 256, 512, 1024};
    
    // 创建结果文件
    std::ofstream out_file("grouping_strategy_results.csv");
    out_file << "分组策略,批大小,平均时间(us),平均召回率,簇重叠率,矩阵操作数" << std::endl;
    
    std::cout << "开始分组策略实验..." << std::endl;
    std::cout << "查询数量: " << query_count << ", 重复次数: " << repeat_count << std::endl;
    
    // 暖机
    std::cout << "暖机 " << warm_up_count << " 条查询..." << std::endl;
    for (int i = 0; i < warm_up_count; ++i) {
        auto warm_result = ivf_search_gpu(base, test_query + i * vecdim, base_number, vecdim,
                                        1, k, centroids, nlist, invlists, nprobe);
    }
    
    // 测试各种分组策略
    for (size_t batch_size : batch_sizes) {
        std::cout << "\n测试批大小: " << batch_size << std::endl;
        
        // 策略0: 无分组
        std::cout << "  测试策略: 无分组..." << std::endl;
        auto batches_0 = strategy_no_grouping(query_count, batch_size);
        auto result_0 = run_grouping_strategy_test("无分组", batches_0, base, test_query,
                                                  base_number, vecdim, k, centroids, nlist,
                                                  invlists, nprobe, test_gt, test_gt_d, repeat_count);
        
        // 策略1: 簇相似性分组
        std::cout << "  测试策略: 簇相似性分组..." << std::endl;
        auto batches_1 = strategy_cluster_based_grouping(test_query, query_count, batch_size,
                                                        vecdim, centroids, nlist, nprobe);
        auto result_1 = run_grouping_strategy_test("簇相似性分组", batches_1, base, test_query,
                                                  base_number, vecdim, k, centroids, nlist,
                                                  invlists, nprobe, test_gt, test_gt_d, repeat_count);
        
        // 策略2: 查询相似性分组
        std::cout << "  测试策略: 查询相似性分组..." << std::endl;
        auto batches_2 = strategy_query_similarity_grouping(test_query, query_count, batch_size, vecdim);
        auto result_2 = run_grouping_strategy_test("查询相似性分组", batches_2, base, test_query,
                                                  base_number, vecdim, k, centroids, nlist,
                                                  invlists, nprobe, test_gt, test_gt_d, repeat_count);
        
        // 策略3: 混合策略
        std::cout << "  测试策略: 混合策略..." << std::endl;
        auto batches_3 = strategy_hybrid_grouping(test_query, query_count, batch_size,
                                                 vecdim, centroids, nlist, nprobe);
        auto result_3 = run_grouping_strategy_test("混合策略", batches_3, base, test_query,
                                                  base_number, vecdim, k, centroids, nlist,
                                                  invlists, nprobe, test_gt, test_gt_d, repeat_count);
        
        // 输出结果
        std::vector<GroupingResult> batch_results = {result_0, result_1, result_2, result_3};
        for (const auto& result : batch_results) {
            std::cout << "    " << result.strategy_name 
                      << ": 时间=" << std::fixed << std::setprecision(2) << result.avg_time_us << "us"
                      << ", 召回率=" << std::setprecision(4) << result.avg_recall
                      << ", 重叠率=" << std::setprecision(3) << result.cluster_overlap_ratio << std::endl;
            
            out_file << result.strategy_name << "," << batch_size << ","
                     << result.avg_time_us << "," << result.avg_recall << ","
                     << result.cluster_overlap_ratio << "," << result.total_matrix_ops << std::endl;
        }
    }
    
    out_file.close();
    
    std::cout << "\n实验完成！结果已保存到 grouping_strategy_results.csv" << std::endl;
    
    // 清理内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    return 0;
} 