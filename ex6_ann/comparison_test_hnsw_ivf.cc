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
#include <numeric>
#include "mpi_hnsw_ivf_optimized.h"

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
        
        if (rank == 0) {
            std::cerr << "加载数据 " << data_path << std::endl;
            std::cerr << "维度: " << d << "  数量: " << n << "  每元素大小: " << sizeof(T) << std::endl;
        }
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

struct TestResult {
    string algorithm;
    size_t nlist;
    size_t nprobe;
    size_t M;
    size_t efConstruction;
    size_t efSearch;
    size_t max_candidates;
    float recall_mean;
    float recall_std;
    double latency_us_mean;
    double latency_us_std;
    int64_t build_time_ms;
    int repeat_count;
};

// 计算标准差
double calculate_std(const std::vector<double>& values, double mean) {
    if (values.size() <= 1) return 0.0;
    double sum_sq_diff = 0.0;
    for (double v : values) {
        double diff = v - mean;
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / (values.size() - 1));
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== HNSW+IVF 优化版对比测试程序 ===" << std::endl;
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
    
    test_number = std::min(test_number, size_t(1000));  // 使用1000条查询保证统计意义
    const size_t k = 10;
    const size_t warmup_queries = 20;  // 减少暖机查询数 (从100改为20)
    const int repeat_runs = 3;  // 减少重复运行次数 (从5改为3)
    
    // 测试配置参数
    struct TestConfig {
        size_t nlist, nprobe, M, efConstruction, efSearch, max_candidates;
        string name;
    };
    
    std::vector<TestConfig> configs = {
        {128, 8, 16, 150, 100, 500, "快速配置"},
        {128, 16, 16, 150, 150, 800, "平衡配置"},
        {256, 32, 24, 200, 200, 1000, "高精度配置"}
    };
    
    std::vector<TestResult> results;
    
    if (rank == 0) {
        std::cout << "\n=== 开始 HNSW+IVF 优化版测试 ===" << std::endl;
        std::cout << "算法,配置,nlist,nprobe,M,efC,efS,max_candidates,recall_mean,recall_std,latency_us_mean,latency_us_std,build_time_ms,repeat_count" << std::endl;
    }
    
    // 构建索引（只构建一次）
    auto total_build_start = std::chrono::high_resolution_clock::now();
    OptimizedMPIHNSWIVFIndex index(vecdim, 128, 16, 150);
    index.build_index_once(base, base_number);
    auto total_build_end = std::chrono::high_resolution_clock::now();
    int64_t total_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_build_end - total_build_start).count();
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 对每个配置进行测试
    for (const auto& config : configs) {
        if (rank == 0) {
            std::cout << "\n测试配置: " << config.name << std::endl;
        }
        
        index.setEfSearch(config.efSearch);
        
        // 暖机阶段
        if (rank == 0) {
            std::cout << "暖机中..." << std::endl;
            for (size_t i = 0; i < warmup_queries; ++i) {
                auto result = index.optimized_search(test_query + i * vecdim, k, 
                                                   config.nprobe, config.max_candidates);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        // 正式测试 - 多次重复
        std::vector<double> run_recalls;
        std::vector<double> run_latencies;
        
        for (int run = 0; run < repeat_runs; ++run) {
            if (rank == 0) {
                std::cout << "第 " << (run + 1) << "/" << repeat_runs << " 轮测试..." << std::endl;
            }
            
            std::vector<float> query_recalls;
            std::vector<double> query_latencies;
            
            // 单轮测试
            for (size_t i = 0; i < test_number; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                auto result = index.optimized_search(test_query + i * vecdim, k, 
                                                   config.nprobe, config.max_candidates);
                auto end = std::chrono::high_resolution_clock::now();
                
                double latency = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - start).count() / 1000.0;  // 转换为微秒
                
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
                // 计算本轮平均值
                double run_recall = std::accumulate(query_recalls.begin(), query_recalls.end(), 0.0f) / query_recalls.size();
                double run_latency = std::accumulate(query_latencies.begin(), query_latencies.end(), 0.0) / query_latencies.size();
                
                run_recalls.push_back(run_recall);
                run_latencies.push_back(run_latency);
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
        }
        
        if (rank == 0) {
            // 计算多轮统计
            double recall_mean = std::accumulate(run_recalls.begin(), run_recalls.end(), 0.0) / run_recalls.size();
            double latency_mean = std::accumulate(run_latencies.begin(), run_latencies.end(), 0.0) / run_latencies.size();
            
            double recall_std = calculate_std(run_recalls, recall_mean);
            double latency_std = calculate_std(run_latencies, latency_mean);
            
            // 输出结果
            std::cout << "HNSW+IVF优化版," << config.name << "," 
                      << config.nlist << "," << config.nprobe << "," << config.M << "," 
                      << config.efConstruction << "," << config.efSearch << "," << config.max_candidates << ","
                      << std::fixed << std::setprecision(4) << recall_mean << ","
                      << std::fixed << std::setprecision(4) << recall_std << ","
                      << std::fixed << std::setprecision(2) << latency_mean << ","
                      << std::fixed << std::setprecision(2) << latency_std << ","
                      << total_build_time << "," << repeat_runs << std::endl;
            
            // 保存详细结果
            TestResult result_record;
            result_record.algorithm = "HNSW+IVF优化版";
            result_record.nlist = config.nlist;
            result_record.nprobe = config.nprobe;
            result_record.M = config.M;
            result_record.efConstruction = config.efConstruction;
            result_record.efSearch = config.efSearch;
            result_record.max_candidates = config.max_candidates;
            result_record.recall_mean = recall_mean;
            result_record.recall_std = recall_std;
            result_record.latency_us_mean = latency_mean;
            result_record.latency_us_std = latency_std;
            result_record.build_time_ms = total_build_time;
            result_record.repeat_count = repeat_runs;
            results.push_back(result_record);
        }
    }
    
    if (rank == 0) {
        std::cout << "\n=== HNSW+IVF 优化版测试完成 ===" << std::endl;
        
        // 输出最佳配置分析
        if (!results.empty()) {
            auto best_recall = *std::max_element(results.begin(), results.end(),
                [](const TestResult& a, const TestResult& b) {
                    return a.recall_mean < b.recall_mean;
                });
            
            auto best_latency = *std::min_element(results.begin(), results.end(),
                [](const TestResult& a, const TestResult& b) {
                    return a.latency_us_mean < b.latency_us_mean;
                });
            
            std::cout << "\n最高召回率: " << std::fixed << std::setprecision(4) 
                      << best_recall.recall_mean << "±" << best_recall.recall_std
                      << " (延迟: " << std::setprecision(1) << best_recall.latency_us_mean << "μs)" << std::endl;
            
            std::cout << "最低延迟: " << std::fixed << std::setprecision(1) 
                      << best_latency.latency_us_mean << "±" << best_latency.latency_us_std 
                      << "μs (召回率: " << std::setprecision(4) << best_latency.recall_mean << ")" << std::endl;
        }
    }
    
    // 清理内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    MPI_Finalize();
    return 0;
} 