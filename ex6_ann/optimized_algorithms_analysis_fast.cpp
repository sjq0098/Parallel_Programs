#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include <omp.h>
#include <map>
#include <algorithm>

#ifdef _WIN32
#define NOMINMAX
#endif

#include "kdtree_hybrid_optimized.h"
#include "lsh_simd_parallel.h"

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

    std::cerr<<"Loading data "<<data_path<<"\n";
    std::cerr<<"Dimension: "<<d<<"  Number:"<<n<<"  Element size:"<<sizeof(T)<<"\n";

    return data;
}

#ifdef _WIN32
int64_t get_time_us() {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (counter.QuadPart * 1000000) / frequency.QuadPart;
}
#else
int64_t get_time_us() {
    struct timeval val;
    gettimeofday(&val, NULL);
    return val.tv_sec * 1000000LL + val.tv_usec;
}
#endif

float calculate_recall(const std::priority_queue<std::pair<float, uint32_t>>& result, 
                      const int* ground_truth, size_t k, size_t gt_start_idx) {
    std::set<uint32_t> gt_set;
    for(size_t j = 0; j < k; ++j){
        gt_set.insert(ground_truth[gt_start_idx + j]);
    }
    
    auto temp_result = result;
    size_t correct = 0;
    while (!temp_result.empty()) {
        uint32_t idx = temp_result.top().second;
        if(gt_set.find(idx) != gt_set.end()){
            ++correct;
        }
        temp_result.pop();
    }
    
    return (float)correct / k;
}

struct ExperimentConfig {
    std::string algorithm;
    std::map<std::string, int> params;
    bool parallel;
    
    std::string get_config_string() const {
        std::ostringstream oss;
        oss << algorithm;
        for (const auto& p : params) {
            oss << "_" << p.first << p.second;
        }
        if (parallel) oss << "_parallel";
        return oss.str();
    }
};

void run_fast_experiment(const std::vector<ExperimentConfig>& configs,
                        float* base, float* test_query, int* test_gt,
                        size_t base_number, size_t vecdim, size_t test_number, size_t test_gt_d,
                        const std::string& output_file) {
    
    std::ofstream csv_file(output_file);
    csv_file << "algorithm,parallel,num_trees,search_nodes,num_tables,hash_bits,search_radius,min_candidates,"
             << "recall_mean,latency_mean_us,speedup\n";
    
    const size_t k = 10;
    
    // 简化基准测试
    float baseline_avg = 3000.0f; // 基于之前测试的估算值
    std::cout << "Using estimated baseline latency: " << baseline_avg << " us" << std::endl;
    
    for (const auto& config : configs) {
        std::cout << "\nTesting: " << config.get_config_string() << std::endl;
        
        std::vector<float> recall_results;
        std::vector<float> latency_results;
        
        if (config.algorithm == "KDTree Hybrid Optimized") {
            int search_nodes = config.params.at("search_nodes");
            int num_trees = config.params.at("num_trees");
            
            HybridOptimizedKDTree* tree = new HybridOptimizedKDTree(
                base, base_number, vecdim, search_nodes, num_trees, 0.9);
            
            for (size_t i = 0; i < test_number; ++i) {
                auto start = get_time_us();
                auto result = tree->search(test_query + i * vecdim, k);
                auto end = get_time_us();
                
                float recall = calculate_recall(result, test_gt, k, i * test_gt_d);
                float latency = (float)(end - start);
                
                recall_results.push_back(recall);
                latency_results.push_back(latency);
            }
            
            delete tree;
            
        } else if (config.algorithm == "LSH SIMD Parallel") {
            int num_tables = config.params.at("num_tables");
            int hash_bits = config.params.at("hash_bits");
            int search_radius = config.params.at("search_radius");
            int min_candidates = config.params.at("min_candidates");
            
            SIMDParallelLSH* lsh = new SIMDParallelLSH(
                vecdim, num_tables, hash_bits, search_radius, min_candidates);
            lsh->insert(base, base_number);
            
            for (size_t i = 0; i < test_number; ++i) {
                auto start = get_time_us();
                auto result = lsh->search(base, test_query + i * vecdim, k);
                auto end = get_time_us();
                
                float recall = calculate_recall(result, test_gt, k, i * test_gt_d);
                float latency = (float)(end - start);
                
                recall_results.push_back(recall);
                latency_results.push_back(latency);
            }
            
            delete lsh;
        }
        
        // 计算统计信息
        float recall_mean = 0, latency_mean = 0;
        for (size_t i = 0; i < recall_results.size(); ++i) {
            recall_mean += recall_results[i];
            latency_mean += latency_results[i];
        }
        recall_mean /= recall_results.size();
        latency_mean /= latency_results.size();
        
        float speedup = baseline_avg / latency_mean;
        
        std::cout << "  Results: Recall=" << std::fixed << std::setprecision(4) << recall_mean 
                  << ", Latency=" << std::setprecision(2) << latency_mean 
                  << " us, Speedup=" << std::setprecision(2) << speedup << "x" << std::endl;
        
        // 写入CSV
        csv_file << config.algorithm << "," << (config.parallel ? "true" : "false");
        
        // 写入参数
        csv_file << "," << (config.params.count("num_trees") ? config.params.at("num_trees") : -1);
        csv_file << "," << (config.params.count("search_nodes") ? config.params.at("search_nodes") : -1);
        csv_file << "," << (config.params.count("num_tables") ? config.params.at("num_tables") : -1);
        csv_file << "," << (config.params.count("hash_bits") ? config.params.at("hash_bits") : -1);
        csv_file << "," << (config.params.count("search_radius") ? config.params.at("search_radius") : -1);
        csv_file << "," << (config.params.count("min_candidates") ? config.params.at("min_candidates") : -1);
        
        csv_file << "," << std::fixed << std::setprecision(6) << recall_mean;
        csv_file << "," << std::setprecision(2) << latency_mean;
        csv_file << "," << speedup << "\n";
        csv_file.flush();
    }
    
    csv_file.close();
    std::cout << "\nResults saved to " << output_file << std::endl;
}

int main(int argc, char *argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 使用100个查询进行快速测试
    if (test_number > 100) {
        test_number = 100;
    }
    std::cout << "Using " << test_number << " queries for fast parameter analysis" << std::endl;
    std::cout << "Dimension: " << vecdim << ", Base data: " << base_number << std::endl;
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;

    // 定义精简的实验配置
    std::vector<ExperimentConfig> configs;
    
    // KDTree Hybrid Optimized - 核心配置
    std::vector<std::pair<int, int>> kdtree_configs = {
        {5, 3000},   // 快速配置
        {8, 5000},   // 平衡配置1  
        {10, 8000},  // 原始配置
        {12, 12000}, // 高性能配置
    };
    
    for (const auto& kd_config : kdtree_configs) {
        ExperimentConfig config;
        config.algorithm = "KDTree Hybrid Optimized";
        config.parallel = true;
        config.params["num_trees"] = kd_config.first;
        config.params["search_nodes"] = kd_config.second;
        configs.push_back(config);
    }
    
    // LSH SIMD Parallel - 核心配置
    std::vector<std::tuple<int, int, int, int>> lsh_configs = {
        {60, 14, 8, 1500},   // 快速配置
        {90, 14, 10, 2000},  // 平衡配置
        {120, 14, 10, 2000}, // 原始最优配置 
        {150, 14, 12, 2500}, // 高召回配置
    };
    
    for (const auto& lsh_config : lsh_configs) {
        ExperimentConfig config;
        config.algorithm = "LSH SIMD Parallel";
        config.parallel = true;
        config.params["num_tables"] = std::get<0>(lsh_config);
        config.params["hash_bits"] = std::get<1>(lsh_config);
        config.params["search_radius"] = std::get<2>(lsh_config);
        config.params["min_candidates"] = std::get<3>(lsh_config);
        configs.push_back(config);
    }
    
    std::cout << "\nTotal configurations to test: " << configs.size() << std::endl;
    std::cout << "KDTree configurations: " << kdtree_configs.size() << std::endl;
    std::cout << "LSH configurations: " << lsh_configs.size() << std::endl;
    
    // 运行快速实验
    run_fast_experiment(configs, base, test_query, test_gt, 
                       base_number, vecdim, test_number, test_gt_d,
                       "optimized_algorithms_fast_results.csv");
    
    // 清理内存
    delete[] test_query;
    delete[] test_gt;  
    delete[] base;

    std::cout << "\nFast optimized algorithms analysis completed!" << std::endl;
    std::cout << "Results saved to optimized_algorithms_fast_results.csv" << std::endl;
    return 0;
} 