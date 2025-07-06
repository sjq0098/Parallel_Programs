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
#include <cmath>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include <omp.h>

#include "flat_scan.h"
#include "kdtree_approx.h"
#include "lsh_improved.h"
#include "kdtree_simd_parallel.h"
#include "lsh_simd_parallel.h"

// 实验结果结构
struct ExperimentResult {
    std::string algorithm_name;
    std::map<std::string, std::string> parameters;
    double avg_recall;
    double std_recall;
    double avg_latency_us;
    double std_latency_us;
    double speedup;
    int build_time_ms;
    int num_runs;
};

// 参数组合结构
struct KDTreeParams {
    size_t max_search_nodes;
    size_t num_trees;
    float bound_factor;
    
    std::string to_string() const {
        return "nodes:" + std::to_string(max_search_nodes) + 
               "_trees:" + std::to_string(num_trees) + 
               "_bound:" + std::to_string(bound_factor);
    }
};

struct LSHParams {
    size_t num_tables;
    size_t hash_bits;
    size_t search_radius;
    size_t min_candidates;
    
    std::string to_string() const {
        return "tables:" + std::to_string(num_tables) + 
               "_bits:" + std::to_string(hash_bits) + 
               "_radius:" + std::to_string(search_radius) + 
               "_candidates:" + std::to_string(min_candidates);
    }
};

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

// 计算统计信息
void calculate_stats(const std::vector<double>& values, double& mean, double& std_dev) {
    mean = 0.0;
    for (double v : values) mean += v;
    mean /= values.size();
    
    double variance = 0.0;
    for (double v : values) {
        variance += (v - mean) * (v - mean);
    }
    variance /= values.size();
    std_dev = std::sqrt(variance);
}

// 暖机函数
void warmup_system(float* base, float* test_query, size_t base_number, size_t vecdim, size_t test_number) {
    std::cout << "系统暖机中..." << std::endl;
    
    // 暖机Flat Search
    for (int warmup = 0; warmup < 3; ++warmup) {
        for (size_t i = 0; i < std::min(test_number, size_t(100)); ++i) {
            auto result = flat_search(base, test_query + i * vecdim, base_number, vecdim, 10);
        }
    }
    
    // 暖机SIMD指令
    #pragma omp parallel for
    for (int i = 0; i < 1000; ++i) {
        volatile float sum = 0;
        for (size_t j = 0; j < vecdim; j += 8) {
            sum += test_query[i % test_number * vecdim + j];
        }
    }
    
    std::cout << "暖机完成!" << std::endl;
}

// 基准测试函数
double measure_baseline_latency(float* base, float* test_query, int* test_gt,
                               size_t base_number, size_t vecdim, size_t test_number, 
                               size_t k, size_t test_gt_d, int num_runs) {
    
    std::vector<double> latencies;
    
    for (int run = 0; run < num_runs; ++run) {
        auto start_time = get_time_us();
        
        for (size_t i = 0; i < test_number; ++i) {
            auto result = flat_search(base, test_query + i * vecdim, base_number, vecdim, k);
        }
        
        auto end_time = get_time_us();
        double avg_latency = double(end_time - start_time) / test_number;
        latencies.push_back(avg_latency);
    }
    
    double mean, std_dev;
    calculate_stats(latencies, mean, std_dev);
    return mean;
}

// KDTree参数测试
ExperimentResult test_kdtree_params(const KDTreeParams& params, bool use_parallel,
                                   float* base, float* test_query, int* test_gt,
                                   size_t base_number, size_t vecdim, size_t test_number, 
                                   size_t k, size_t test_gt_d, int num_runs, double baseline_latency) {
    
    ExperimentResult result;
    result.algorithm_name = use_parallel ? "KDTree SIMD Parallel" : "KDTree Approx";
    result.parameters["max_search_nodes"] = std::to_string(params.max_search_nodes);
    result.parameters["num_trees"] = std::to_string(params.num_trees);
    result.parameters["bound_factor"] = std::to_string(params.bound_factor);
    result.parameters["parallel"] = use_parallel ? "true" : "false";
    result.num_runs = num_runs;
    
    std::vector<double> recalls, latencies;
    
    // 构建索引
    auto build_start = std::chrono::high_resolution_clock::now();
    
    if (use_parallel) {
        SIMDParallelKDTree* tree = new SIMDParallelKDTree(base, base_number, vecdim, params.max_search_nodes);
        
        auto build_end = std::chrono::high_resolution_clock::now();
        result.build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
        
        // 多次运行测试
        for (int run = 0; run < num_runs; ++run) {
            std::vector<double> run_recalls, run_latencies;
            
            for (size_t i = 0; i < test_number; ++i) {
                auto query_start = get_time_us();
                auto search_result = tree->search(test_query + i * vecdim, k);
                auto query_end = get_time_us();
                
                float recall = calculate_recall(search_result, test_gt, k, i * test_gt_d);
                double latency = double(query_end - query_start);
                
                run_recalls.push_back(recall);
                run_latencies.push_back(latency);
            }
            
            double run_avg_recall = 0, run_avg_latency = 0;
            for (size_t i = 0; i < run_recalls.size(); ++i) {
                run_avg_recall += run_recalls[i];
                run_avg_latency += run_latencies[i];
            }
            run_avg_recall /= run_recalls.size();
            run_avg_latency /= run_latencies.size();
            
            recalls.push_back(run_avg_recall);
            latencies.push_back(run_avg_latency);
        }
        
        delete tree;
    } else {
        ApproxKDTree* tree = new ApproxKDTree(base, base_number, vecdim, params.max_search_nodes);
        
        auto build_end = std::chrono::high_resolution_clock::now();
        result.build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
        
        // 多次运行测试
        for (int run = 0; run < num_runs; ++run) {
            std::vector<double> run_recalls, run_latencies;
            
            for (size_t i = 0; i < test_number; ++i) {
                auto query_start = get_time_us();
                auto search_result = tree->search(test_query + i * vecdim, k);
                auto query_end = get_time_us();
                
                float recall = calculate_recall(search_result, test_gt, k, i * test_gt_d);
                double latency = double(query_end - query_start);
                
                run_recalls.push_back(recall);
                run_latencies.push_back(latency);
            }
            
            double run_avg_recall = 0, run_avg_latency = 0;
            for (size_t i = 0; i < run_recalls.size(); ++i) {
                run_avg_recall += run_recalls[i];
                run_avg_latency += run_latencies[i];
            }
            run_avg_recall /= run_recalls.size();
            run_avg_latency /= run_latencies.size();
            
            recalls.push_back(run_avg_recall);
            latencies.push_back(run_avg_latency);
        }
        
        delete tree;
    }
    
    calculate_stats(recalls, result.avg_recall, result.std_recall);
    calculate_stats(latencies, result.avg_latency_us, result.std_latency_us);
    result.speedup = baseline_latency / result.avg_latency_us;
    
    return result;
}

// LSH参数测试
ExperimentResult test_lsh_params(const LSHParams& params, bool use_parallel,
                                float* base, float* test_query, int* test_gt,
                                size_t base_number, size_t vecdim, size_t test_number, 
                                size_t k, size_t test_gt_d, int num_runs, double baseline_latency) {
    
    ExperimentResult result;
    result.algorithm_name = use_parallel ? "LSH SIMD Parallel" : "LSH Improved";
    result.parameters["num_tables"] = std::to_string(params.num_tables);
    result.parameters["hash_bits"] = std::to_string(params.hash_bits);
    result.parameters["search_radius"] = std::to_string(params.search_radius);
    result.parameters["min_candidates"] = std::to_string(params.min_candidates);
    result.parameters["parallel"] = use_parallel ? "true" : "false";
    result.num_runs = num_runs;
    
    std::vector<double> recalls, latencies;
    
    // 构建索引
    auto build_start = std::chrono::high_resolution_clock::now();
    
    if (use_parallel) {
        SIMDParallelLSH* lsh = new SIMDParallelLSH(vecdim, params.num_tables, params.hash_bits, 
                                                  params.search_radius, params.min_candidates);
        lsh->insert(base, base_number);
        
        auto build_end = std::chrono::high_resolution_clock::now();
        result.build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
        
        // 多次运行测试
        for (int run = 0; run < num_runs; ++run) {
            std::vector<double> run_recalls, run_latencies;
            
            for (size_t i = 0; i < test_number; ++i) {
                auto query_start = get_time_us();
                auto search_result = lsh->search(base, test_query + i * vecdim, k);
                auto query_end = get_time_us();
                
                float recall = calculate_recall(search_result, test_gt, k, i * test_gt_d);
                double latency = double(query_end - query_start);
                
                run_recalls.push_back(recall);
                run_latencies.push_back(latency);
            }
            
            double run_avg_recall = 0, run_avg_latency = 0;
            for (size_t i = 0; i < run_recalls.size(); ++i) {
                run_avg_recall += run_recalls[i];
                run_avg_latency += run_latencies[i];
            }
            run_avg_recall /= run_recalls.size();
            run_avg_latency /= run_latencies.size();
            
            recalls.push_back(run_avg_recall);
            latencies.push_back(run_avg_latency);
        }
        
        delete lsh;
    } else {
        ImprovedLSH* lsh = new ImprovedLSH(vecdim, params.num_tables, params.hash_bits, 
                                          params.search_radius, params.min_candidates);
        lsh->insert(base, base_number);
        
        auto build_end = std::chrono::high_resolution_clock::now();
        result.build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
        
        // 多次运行测试
        for (int run = 0; run < num_runs; ++run) {
            std::vector<double> run_recalls, run_latencies;
            
            for (size_t i = 0; i < test_number; ++i) {
                auto query_start = get_time_us();
                auto search_result = lsh->search(base, test_query + i * vecdim, k);
                auto query_end = get_time_us();
                
                float recall = calculate_recall(search_result, test_gt, k, i * test_gt_d);
                double latency = double(query_end - query_start);
                
                run_recalls.push_back(recall);
                run_latencies.push_back(latency);
            }
            
            double run_avg_recall = 0, run_avg_latency = 0;
            for (size_t i = 0; i < run_recalls.size(); ++i) {
                run_avg_recall += run_recalls[i];
                run_avg_latency += run_latencies[i];
            }
            run_avg_recall /= run_recalls.size();
            run_avg_latency /= run_latencies.size();
            
            recalls.push_back(run_avg_recall);
            latencies.push_back(run_avg_latency);
        }
        
        delete lsh;
    }
    
    calculate_stats(recalls, result.avg_recall, result.std_recall);
    calculate_stats(latencies, result.avg_latency_us, result.std_latency_us);
    result.speedup = baseline_latency / result.avg_latency_us;
    
    return result;
}

// 结果输出
void print_experiment_results(const std::vector<ExperimentResult>& results) {
    std::cout << "\n" << std::string(120, '=') << std::endl;
    std::cout << "参数分析实验结果总结" << std::endl;
    std::cout << std::string(120, '=') << std::endl;
    
    // 按算法分组
    std::map<std::string, std::vector<ExperimentResult>> grouped_results;
    for (const auto& result : results) {
        grouped_results[result.algorithm_name].push_back(result);
    }
    
    for (const auto& group : grouped_results) {
        std::cout << "\n【" << group.first << "】参数分析:" << std::endl;
        std::cout << std::string(120, '-') << std::endl;
        
        std::cout << std::left << std::setw(30) << "参数组合" 
                  << std::setw(12) << "召回率%" 
                  << std::setw(12) << "召回率std"
                  << std::setw(15) << "延迟(us)" 
                  << std::setw(12) << "延迟std"
                  << std::setw(12) << "加速比" 
                  << std::setw(15) << "构建时间(ms)" 
                  << std::setw(8) << "运行次数" << std::endl;
        std::cout << std::string(120, '-') << std::endl;
        
        for (const auto& result : group.second) {
            std::string param_str;
            for (const auto& param : result.parameters) {
                if (param.first != "parallel") {
                    param_str += param.first.substr(0, 4) + ":" + param.second + " ";
                }
            }
            if (param_str.length() > 28) param_str = param_str.substr(0, 28) + "..";
            
            std::cout << std::left << std::setw(30) << param_str
                      << std::setw(12) << std::fixed << std::setprecision(2) << (result.avg_recall * 100)
                      << std::setw(12) << std::fixed << std::setprecision(4) << result.std_recall
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.avg_latency_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.std_latency_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.speedup
                      << std::setw(15) << result.build_time_ms
                      << std::setw(8) << result.num_runs << std::endl;
        }
    }
}

// 保存结果到CSV
void save_results_to_csv(const std::vector<ExperimentResult>& results, const std::string& filename) {
    std::ofstream csv_file(filename);
    
    // 写入CSV头
    csv_file << "algorithm,parallel,recall_mean,recall_std,latency_mean_us,latency_std_us,speedup,build_time_ms,num_runs";
    
    // 获取所有参数键
    std::set<std::string> all_param_keys;
    for (const auto& result : results) {
        for (const auto& param : result.parameters) {
            if (param.first != "parallel") {
                all_param_keys.insert(param.first);
            }
        }
    }
    
    for (const auto& key : all_param_keys) {
        csv_file << "," << key;
    }
    csv_file << "\n";
    
    // 写入数据
    for (const auto& result : results) {
        csv_file << result.algorithm_name << ","
                 << result.parameters.at("parallel") << ","
                 << result.avg_recall << ","
                 << result.std_recall << ","
                 << result.avg_latency_us << ","
                 << result.std_latency_us << ","
                 << result.speedup << ","
                 << result.build_time_ms << ","
                 << result.num_runs;
        
        for (const auto& key : all_param_keys) {
            csv_file << ",";
            auto it = result.parameters.find(key);
            if (it != result.parameters.end()) {
                csv_file << it->second;
            }
        }
        csv_file << "\n";
    }
    
    csv_file.close();
    std::cout << "\n结果已保存到: " << filename << std::endl;
}

int main(int argc, char *argv[])
{
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
    
    // 限制测试数量以加速实验
    test_number = std::min(test_number, size_t(500));  // 使用500个查询进行参数分析
    const size_t k = 10;
    const int num_runs = 5;  // 每个参数组合运行5次
    
    std::cout << "=== 参数分析实验 ===" << std::endl;
    std::cout << "测试查询数量: " << test_number << std::endl;
    std::cout << "数据维度: " << vecdim << ", 基础数据量: " << base_number << std::endl;
    std::cout << "每个参数组合运行次数: " << num_runs << std::endl;
    std::cout << "使用线程数: " << omp_get_max_threads() << std::endl;
    
    // 系统暖机
    warmup_system(base, test_query, base_number, vecdim, test_number);
    
    // 测量基准性能
    std::cout << "\n测量基准算法性能..." << std::endl;
    double baseline_latency = measure_baseline_latency(base, test_query, test_gt, 
                                                      base_number, vecdim, test_number, 
                                                      k, test_gt_d, num_runs);
    std::cout << "基准平均延迟: " << std::fixed << std::setprecision(2) << baseline_latency << " us" << std::endl;
    
    std::vector<ExperimentResult> all_results;
    
    // KDTree参数网格搜索
    std::cout << "\n开始KDTree参数分析..." << std::endl;
    std::vector<KDTreeParams> kdtree_param_grid = {
        // 搜索节点数对比
        {2000, 1, 7.0},   // 低搜索节点
        {5000, 1, 7.0},   // 中等搜索节点  
        {8000, 1, 7.0},   // 高搜索节点
        {12000, 1, 7.0},  // 很高搜索节点
        
        // 树数量对比
        {8000, 1, 7.0},   // 单树
        {8000, 4, 7.0},   // 4棵树
        {8000, 8, 7.0},   // 8棵树
        {8000, 12, 7.0},  // 12棵树
        
        // 边界因子对比
        {8000, 1, 3.0},   // 严格剪枝
        {8000, 1, 5.0},   // 中等剪枝
        {8000, 1, 7.0},   // 宽松剪枝
        {8000, 1, 10.0},  // 很宽松剪枝
        
        // 综合最优组合测试
        {6000, 6, 6.0},   // 平衡配置1
        {10000, 8, 8.0},  // 高性能配置
        {4000, 4, 5.0},   // 快速配置
    };
    
    int kdtree_count = 0;
    for (const auto& params : kdtree_param_grid) {
        std::cout << "测试KDTree参数组合 " << (++kdtree_count) << "/" << kdtree_param_grid.size() 
                  << ": " << params.to_string() << std::endl;
        
        // 测试非并行版本
        auto result_seq = test_kdtree_params(params, false, base, test_query, test_gt,
                                           base_number, vecdim, test_number, k, test_gt_d, 
                                           num_runs, baseline_latency);
        all_results.push_back(result_seq);
        
        // 测试并行版本
        auto result_par = test_kdtree_params(params, true, base, test_query, test_gt,
                                           base_number, vecdim, test_number, k, test_gt_d, 
                                           num_runs, baseline_latency);
        all_results.push_back(result_par);
        
        std::cout << "  非并行: 召回率=" << std::fixed << std::setprecision(3) << (result_seq.avg_recall * 100) 
                  << "%, 延迟=" << std::setprecision(1) << result_seq.avg_latency_us << "us, 加速=" 
                  << std::setprecision(2) << result_seq.speedup << "x" << std::endl;
        std::cout << "  并行版: 召回率=" << std::fixed << std::setprecision(3) << (result_par.avg_recall * 100) 
                  << "%, 延迟=" << std::setprecision(1) << result_par.avg_latency_us << "us, 加速=" 
                  << std::setprecision(2) << result_par.speedup << "x" << std::endl;
    }
    
    // LSH参数网格搜索
    std::cout << "\n开始LSH参数分析..." << std::endl;
    std::vector<LSHParams> lsh_param_grid = {
        // 哈希表数量对比
        {20, 14, 5, 1000},   // 少表数
        {60, 14, 5, 1000},   // 中等表数
        {100, 14, 5, 1000},  // 多表数
        {150, 14, 5, 1000},  // 很多表数
        
        // 哈希位数对比  
        {80, 12, 5, 1000},   // 少位数
        {80, 14, 5, 1000},   // 中等位数
        {80, 16, 5, 1000},   // 多位数
        {80, 18, 5, 1000},   // 很多位数
        
        // 搜索半径对比
        {80, 14, 3, 1000},   // 小半径
        {80, 14, 5, 1000},   // 中等半径
        {80, 14, 8, 1000},   // 大半径
        {80, 14, 12, 1000},  // 很大半径
        
        // 候选点数量对比
        {80, 14, 5, 500},    // 少候选点
        {80, 14, 5, 1000},   // 中等候选点
        {80, 14, 5, 2000},   // 多候选点
        {80, 14, 5, 3000},   // 很多候选点
        
        // 综合最优组合测试
        {100, 13, 6, 1500},  // 平衡配置1
        {120, 14, 8, 2000},  // 高召回率配置
        {60, 12, 4, 800},    // 快速配置
    };
    
    int lsh_count = 0;
    for (const auto& params : lsh_param_grid) {
        std::cout << "测试LSH参数组合 " << (++lsh_count) << "/" << lsh_param_grid.size() 
                  << ": " << params.to_string() << std::endl;
        
        // 测试非并行版本
        auto result_seq = test_lsh_params(params, false, base, test_query, test_gt,
                                        base_number, vecdim, test_number, k, test_gt_d, 
                                        num_runs, baseline_latency);
        all_results.push_back(result_seq);
        
        // 测试并行版本
        auto result_par = test_lsh_params(params, true, base, test_query, test_gt,
                                        base_number, vecdim, test_number, k, test_gt_d, 
                                        num_runs, baseline_latency);
        all_results.push_back(result_par);
        
        std::cout << "  非并行: 召回率=" << std::fixed << std::setprecision(3) << (result_seq.avg_recall * 100) 
                  << "%, 延迟=" << std::setprecision(1) << result_seq.avg_latency_us << "us, 加速=" 
                  << std::setprecision(2) << result_seq.speedup << "x" << std::endl;
        std::cout << "  并行版: 召回率=" << std::fixed << std::setprecision(3) << (result_par.avg_recall * 100) 
                  << "%, 延迟=" << std::setprecision(1) << result_par.avg_latency_us << "us, 加速=" 
                  << std::setprecision(2) << result_par.speedup << "x" << std::endl;
    }
    
    // 输出结果
    print_experiment_results(all_results);
    
    // 保存到CSV文件
    save_results_to_csv(all_results, "parameter_analysis_results.csv");
    
    // 找出最佳配置
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "最佳配置分析:" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // 按召回率排序找最高召回率
    auto best_recall = *std::max_element(all_results.begin(), all_results.end(),
        [](const ExperimentResult& a, const ExperimentResult& b) {
            return a.avg_recall < b.avg_recall;
        });
    
    // 按加速比排序找最高加速比
    auto best_speedup = *std::max_element(all_results.begin(), all_results.end(),
        [](const ExperimentResult& a, const ExperimentResult& b) {
            return a.speedup < b.speedup;
        });
    
    // 找性能平衡点 (召回率>80%且加速比最高)
    auto good_results = all_results;
    good_results.erase(std::remove_if(good_results.begin(), good_results.end(),
        [](const ExperimentResult& r) { return r.avg_recall < 0.8; }), good_results.end());
    
    if (!good_results.empty()) {
        auto best_balanced = *std::max_element(good_results.begin(), good_results.end(),
            [](const ExperimentResult& a, const ExperimentResult& b) {
                return a.speedup < b.speedup;
            });
        
        std::cout << "最佳平衡配置 (召回率>80%): " << best_balanced.algorithm_name << std::endl;
        std::cout << "  召回率: " << std::fixed << std::setprecision(3) << (best_balanced.avg_recall * 100) << "%" << std::endl;
        std::cout << "  加速比: " << std::setprecision(2) << best_balanced.speedup << "x" << std::endl;
        std::cout << "  参数: ";
        for (const auto& param : best_balanced.parameters) {
            if (param.first != "parallel") {
                std::cout << param.first << "=" << param.second << " ";
            }
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n最高召回率配置: " << best_recall.algorithm_name << std::endl;
    std::cout << "  召回率: " << std::fixed << std::setprecision(3) << (best_recall.avg_recall * 100) << "%" << std::endl;
    std::cout << "  加速比: " << std::setprecision(2) << best_recall.speedup << "x" << std::endl;
    
    std::cout << "\n最高加速比配置: " << best_speedup.algorithm_name << std::endl;
    std::cout << "  召回率: " << std::fixed << std::setprecision(3) << (best_speedup.avg_recall * 100) << "%" << std::endl;
    std::cout << "  加速比: " << std::setprecision(2) << best_speedup.speedup << "x" << std::endl;
    
    // 清理内存
    delete[] test_query;
    delete[] test_gt;  
    delete[] base;

    std::cout << "\n参数分析实验完成!" << std::endl;
    return 0;
} 