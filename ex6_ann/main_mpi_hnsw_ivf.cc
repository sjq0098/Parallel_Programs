#include <mpi.h>
#include <omp.h>
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include "mpi_hnsw_ivf.h"

using namespace std;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d, int rank)
{
    T* data = nullptr;
    
    if (rank == 0) {
        std::ifstream fin;
        fin.open(data_path, std::ios::in | std::ios::binary);
        if (!fin) {
            std::cerr << "无法打开文件: " << data_path << std::endl;
            n = 0; d = 0;
            return nullptr;
        }
        
        fin.read((char*)&n, 4);
        fin.read((char*)&d, 4);
        data = new T[n * d];
        int sz = sizeof(T);
        for(size_t i = 0; i < n; ++i){
            fin.read(((char*)data + i*d*sz), d*sz);
        }
        fin.close();
        
        std::cerr << "加载数据 " << data_path << std::endl;
        std::cerr << "维度: " << d << "  数量: " << n << "  每元素大小: " << sizeof(T) << std::endl;
    }
    
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        data = new T[n * d];
    }
    
    if (std::is_same<T, float>::value) {
        MPI_Bcast(data, n * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else if (std::is_same<T, int>::value) {
        MPI_Bcast(data, n * d, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    return data;
}

struct TestResult {
    size_t nlist;
    size_t nprobe;
    size_t M;
    size_t efConstruction;
    size_t efSearch;
    size_t candidate_factor;
    float recall;
    int64_t latency_us;
    int64_t build_time_ms;
    int mpi_processes;
    int omp_threads;
};

// 计算召回率
float calculate_recall(const std::vector<uint32_t>& result, const int* gt, size_t k, size_t gt_d) {
    std::set<int> gt_set;
    for (size_t i = 0; i < gt_d && i < k; ++i) {
        gt_set.insert(gt[i]);
    }
    
    int correct = 0;
    for (size_t i = 0; i < std::min(k, result.size()); ++i) {
        if (gt_set.count(result[i])) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / std::min(k, gt_d);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== MPI HNSW+IVF混合算法参数测试 ===" << std::endl;
        std::cout << "MPI进程数: " << size << std::endl;
        std::cout << "OpenMP线程数: " << omp_get_max_threads() << std::endl;
    }
    
    // 加载数据
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim, rank);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d, rank);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim, rank);
    
    if (!test_query || !test_gt || !base) {
        if (rank == 0) {
            std::cerr << "数据加载失败!" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    test_number = std::min(test_number, size_t(500));  // 限制测试数量
    const size_t k = 10;
    
    // 参数组合定义
    std::vector<size_t> nlist_values = {64, 128, 256};
    std::vector<size_t> nprobe_values = {4, 8, 16, 32};
    std::vector<size_t> M_values = {8, 16, 24};
    std::vector<size_t> efConstruction_values = {100, 200};
    std::vector<size_t> efSearch_values = {50, 100, 200};
    std::vector<size_t> candidate_factor_values = {10, 20, 30};
    
    std::vector<TestResult> results;
    
    if (rank == 0) {
        std::cout << "开始参数测试..." << std::endl;
        std::cout << "nlist,nprobe,M,efC,efS,candidate_factor,recall,latency_us,build_time_ms,mpi_processes,omp_threads" << std::endl;
    }
    
    for (size_t nlist : nlist_values) {
        for (size_t M : M_values) {
            for (size_t efConstruction : efConstruction_values) {
                for (size_t efSearch : efSearch_values) {
                    for (size_t candidate_factor : candidate_factor_values) {
                        // 构建索引
                        auto build_start = std::chrono::high_resolution_clock::now();
                        
                        MPIHNSWIVFIndex index(vecdim, nlist, M, efConstruction);
                        index.setEfSearch(efSearch);
                        index.setHNSWCandidateFactor(candidate_factor);
                        index.build_index(base, base_number);
                        
                        auto build_end = std::chrono::high_resolution_clock::now();
                        int64_t build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            build_end - build_start).count();
                        
                        if (rank == 0) {
                            std::cout << "参数组合: nlist=" << nlist << ", M=" << M 
                                      << ", efC=" << efConstruction << ", efS=" << efSearch
                                      << ", candidate_factor=" << candidate_factor << std::endl;
                        }
                        
                        // 对每个nprobe值进行测试
                        for (size_t nprobe : nprobe_values) {
                            std::vector<float> query_recalls;
                            std::vector<int64_t> query_latencies;
                            
                            // 暖机
                            if (rank == 0) {
                                for (size_t i = 0; i < std::min(size_t(30), test_number); ++i) {
                                    auto result = index.mpi_search(test_query + i * vecdim, base, k, nprobe);
                                }
                            }
                            MPI_Barrier(MPI_COMM_WORLD);
                            
                            // 实际测试
                            for (size_t i = 0; i < test_number; ++i) {
                                auto start = std::chrono::high_resolution_clock::now();
                                auto result = index.mpi_search(test_query + i * vecdim, base, k, nprobe);
                                auto end = std::chrono::high_resolution_clock::now();
                                
                                int64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(
                                    end - start).count();
                                
                                if (rank == 0) {
                                    // 转换结果格式
                                    std::vector<uint32_t> result_ids;
                                    auto temp_result = result;
                                    while (!temp_result.empty()) {
                                        result_ids.push_back(temp_result.top().second);
                                        temp_result.pop();
                                    }
                                    std::reverse(result_ids.begin(), result_ids.end());
                                    
                                    float recall = calculate_recall(result_ids, test_gt + i * test_gt_d, k, test_gt_d);
                                    query_recalls.push_back(recall);
                                    query_latencies.push_back(latency);
                                }
                            }
                            
                            if (rank == 0) {
                                // 计算平均召回率和延迟
                                float avg_recall = 0.0f;
                                int64_t avg_latency = 0;
                                for (size_t i = 0; i < query_recalls.size(); ++i) {
                                    avg_recall += query_recalls[i];
                                    avg_latency += query_latencies[i];
                                }
                                avg_recall /= query_recalls.size();
                                avg_latency /= query_latencies.size();
                                
                                // 输出结果
                                std::cout << nlist << "," << nprobe << "," << M << "," 
                                          << efConstruction << "," << efSearch << "," << candidate_factor << ","
                                          << std::fixed << std::setprecision(4) << avg_recall << "," 
                                          << avg_latency << "," << build_time << "," 
                                          << size << "," << omp_get_max_threads() << std::endl;
                                
                                // 保存详细结果
                                TestResult result_record;
                                result_record.nlist = nlist;
                                result_record.nprobe = nprobe;
                                result_record.M = M;
                                result_record.efConstruction = efConstruction;
                                result_record.efSearch = efSearch;
                                result_record.candidate_factor = candidate_factor;
                                result_record.recall = avg_recall;
                                result_record.latency_us = avg_latency;
                                result_record.build_time_ms = build_time;
                                result_record.mpi_processes = size;
                                result_record.omp_threads = omp_get_max_threads();
                                results.push_back(result_record);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 输出最佳结果摘要
    if (rank == 0) {
        if (!results.empty()) {
            std::cout << "\n=== 最佳性能配置 ===" << std::endl;
            
            // 按召回率排序
            auto best_recall = *std::max_element(results.begin(), results.end(),
                [](const TestResult& a, const TestResult& b) {
                    return a.recall < b.recall;
                });
            
            std::cout << "最高召回率: " << std::fixed << std::setprecision(4) << best_recall.recall 
                      << " (nlist=" << best_recall.nlist << ", nprobe=" << best_recall.nprobe 
                      << ", M=" << best_recall.M << ", efS=" << best_recall.efSearch
                      << ", candidate_factor=" << best_recall.candidate_factor 
                      << ", 延迟=" << best_recall.latency_us << "μs)" << std::endl;
            
            // 在高召回率中找最快的
            auto high_recall_results = results;
            high_recall_results.erase(
                std::remove_if(high_recall_results.begin(), high_recall_results.end(),
                    [](const TestResult& r) { return r.recall < 0.9; }),
                high_recall_results.end());
            
            if (!high_recall_results.empty()) {
                auto best_latency = *std::min_element(high_recall_results.begin(), high_recall_results.end(),
                    [](const TestResult& a, const TestResult& b) {
                        return a.latency_us < b.latency_us;
                    });
                
                std::cout << "高召回率(≥0.9)最低延迟: " << best_latency.latency_us << "μs "
                          << "(召回率=" << std::fixed << std::setprecision(4) << best_latency.recall 
                          << ", nlist=" << best_latency.nlist << ", nprobe=" << best_latency.nprobe 
                          << ", M=" << best_latency.M << ", efS=" << best_latency.efSearch
                          << ", candidate_factor=" << best_latency.candidate_factor << ")" << std::endl;
            }
        }
    }
    
    // 清理内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    MPI_Finalize();
    return 0;
} 