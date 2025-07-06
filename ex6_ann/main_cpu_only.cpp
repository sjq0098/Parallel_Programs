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

#include "flat_scan.h"              // 原始的flat search
#include "kdtree.h"                 // KDTree CPU版本
#include "lsh.h"                    // LSH CPU版本
#include "kdtree_approx.h"          // 近似KDTree版本
#include "lsh_improved.h"           // 改进LSH版本
#include "kdtree_simd_parallel.h"   // SIMD并行KDTree版本
#include "lsh_simd_parallel.h"      // SIMD并行LSH版本
#include "kdtree_ensemble_parallel.h" // 集成并行KDTree版本
#include "flat_scan_simd_parallel.h"  // SIMD并行Flat Search版本
#include "kdtree_hybrid_optimized.h"  // 智能混合优化KDTree版本  

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

    std::cerr<<"加载数据 "<<data_path<<"\n";
    std::cerr<<"维度: "<<d<<"  数量:"<<n<<"  每元素大小:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
    std::string algorithm_name;
};

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

template<typename SearchFunc>
void test_algorithm(const std::string& algo_name,
                   SearchFunc search_func,
                   float* base, float* test_query, int* test_gt,
                   size_t base_number, size_t vecdim, size_t test_number, size_t k, size_t test_gt_d,
                   std::vector<SearchResult>& all_results) {
    
    std::cout << "\n=== 测试 " << algo_name << " ===" << std::endl;
    
    std::vector<SearchResult> results(test_number);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for(size_t i = 0; i < test_number; ++i) {
        if (i % 200 == 0) {
            std::cout << "处理查询 " << i << "/" << test_number << std::endl;
        }
        
        // 只测量查询时间，不包括其他开销
        auto query_start = get_time_us();
        auto res = search_func(test_query + i*vecdim, k);
        auto query_end = get_time_us();
        
        int64_t diff = query_end - query_start;

        float recall = calculate_recall(res, test_gt, k, i * test_gt_d);
        
        results[i] = {recall, diff, algo_name};
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 计算统计信息
    float avg_recall = 0, avg_latency = 0;
    for(size_t i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }
    avg_recall /= test_number;
    avg_latency /= test_number;
    
    std::cout << algo_name << " 结果:" << std::endl;
    std::cout << "  平均召回率: " << std::fixed << std::setprecision(6) << avg_recall << std::endl;
    std::cout << "  平均查询延迟 (us): " << std::fixed << std::setprecision(2) << avg_latency << std::endl;
    std::cout << "  总时间 (ms): " << total_time.count() << std::endl;
    std::cout << "  QPS: " << std::fixed << std::setprecision(2) << (test_number * 1000.0 / total_time.count()) << std::endl;
    
    // 添加到总体结果
    all_results.insert(all_results.end(), results.begin(), results.end());
}

void print_summary(const std::vector<SearchResult>& all_results, size_t test_number) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "算法性能总结 (纯查询时间对比)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::map<std::string, std::pair<float, float>> algo_stats; // recall, latency
    std::map<std::string, int> algo_counts;
    
    for (const auto& result : all_results) {
        algo_stats[result.algorithm_name].first += result.recall;
        algo_stats[result.algorithm_name].second += result.latency;
        algo_counts[result.algorithm_name]++;
    }
    
    std::cout << std::left << std::setw(20) << "算法" 
              << std::setw(15) << "平均召回率" 
              << std::setw(18) << "平均查询延迟(us)" 
              << std::setw(15) << "相对加速比" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    float baseline_latency = 0;
    for (const auto& entry : algo_stats) {
        const std::string& algo_name = entry.first;
        const std::pair<float, float>& stats = entry.second;
        
        float avg_recall = stats.first / algo_counts[algo_name];
        float avg_latency = stats.second / algo_counts[algo_name];
        
        if (algo_name == "Flat Search") {
            baseline_latency = avg_latency;
        }
        
        float speedup = (baseline_latency > 0) ? baseline_latency / avg_latency : 1.0f;
        
        std::cout << std::left << std::setw(20) << algo_name
                  << std::setw(15) << std::fixed << std::setprecision(6) << avg_recall
                  << std::setw(18) << std::fixed << std::setprecision(2) << avg_latency
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
}

int main(int argc, char *argv[])
{
#ifdef _WIN32
    // 设置Windows控制台编码为UTF-8
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 只测试前2000条查询以加快测试速度
    test_number = (test_number < 2000) ? test_number : 2000;
    std::cout << "实际测试查询数量: " << test_number << std::endl;
    std::cout << "数据维度: " << vecdim << ", 基础数据量: " << base_number << std::endl;

    const size_t k = 10;
    std::vector<SearchResult> all_results;

    // 预先构建索引
    std::cout << "\n开始构建索引..." << std::endl;
    
    // 构建KDTree索引
    auto build_start = std::chrono::high_resolution_clock::now();
    KDTree* kdtree = new KDTree(base, base_number, vecdim);
    auto build_end = std::chrono::high_resolution_clock::now();
    auto kdtree_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);
    std::cout << "KDTree构建时间: " << kdtree_build_time.count() << " ms" << std::endl;
    
    // 构建LSH索引
    build_start = std::chrono::high_resolution_clock::now();
    LSH* lsh = new LSH(vecdim, 15, 18);
    lsh->insert(base, base_number);
    build_end = std::chrono::high_resolution_clock::now();
    auto lsh_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);
    std::cout << "LSH构建时间: " << lsh_build_time.count() << " ms" << std::endl;
    
    // 构建近似KDTree索引(冲刺90%版本)
    build_start = std::chrono::high_resolution_clock::now();
    ApproxKDTree* approx_kdtree = new ApproxKDTree(base, base_number, vecdim, 15000);  // 最终尝试15000节点冲击90%召回率
    build_end = std::chrono::high_resolution_clock::now();
    auto approx_kdtree_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);
    std::cout << "近似KDTree构建时间(冲刺90%): " << approx_kdtree_build_time.count() << " ms" << std::endl;
    
    // 构建改进LSH索引(冲刺90%版本)
    build_start = std::chrono::high_resolution_clock::now();
    ImprovedLSH* improved_lsh = new ImprovedLSH(vecdim, 120, 14, 10, 2000);  // 最终冲刺90%：120表，14位，半径10，2000候选点
    improved_lsh->insert(base, base_number);
    build_end = std::chrono::high_resolution_clock::now();
    auto improved_lsh_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);
    std::cout << "改进LSH构建时间(冲刺90%): " << improved_lsh_build_time.count() << " ms" << std::endl;
    
    // 构建SIMD并行KDTree索引
    build_start = std::chrono::high_resolution_clock::now();
    SIMDParallelKDTree* simd_kdtree = new SIMDParallelKDTree(base, base_number, vecdim, 8000);
    build_end = std::chrono::high_resolution_clock::now();
    auto simd_kdtree_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);
    std::cout << "SIMD并行KDTree构建时间: " << simd_kdtree_build_time.count() << " ms" << std::endl;
    
    // 构建SIMD并行LSH索引
    build_start = std::chrono::high_resolution_clock::now();
    SIMDParallelLSH* simd_lsh = new SIMDParallelLSH(vecdim, 120, 14, 10, 2000);
    simd_lsh->insert(base, base_number);
    build_end = std::chrono::high_resolution_clock::now();
    auto simd_lsh_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);
    std::cout << "SIMD并行LSH构建时间: " << simd_lsh_build_time.count() << " ms" << std::endl;
    
    // 构建集成并行KDTree索引 (12棵树的森林)
    build_start = std::chrono::high_resolution_clock::now();
    EnsembleParallelKDTree* ensemble_kdtree = new EnsembleParallelKDTree(base, base_number, vecdim, 10000, 12);
    build_end = std::chrono::high_resolution_clock::now();
    auto ensemble_kdtree_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);
    std::cout << "集成并行KDTree构建时间: " << ensemble_kdtree_build_time.count() << " ms" << std::endl;
    
    // 构建智能混合优化KDTree索引 (10棵优化树，目标95%召回率)
    build_start = std::chrono::high_resolution_clock::now();
    HybridOptimizedKDTree* hybrid_kdtree = new HybridOptimizedKDTree(base, base_number, vecdim, 8000, 10, 0.95);
    build_end = std::chrono::high_resolution_clock::now();
    auto hybrid_kdtree_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);
    std::cout << "智能混合优化KDTree构建时间: " << hybrid_kdtree_build_time.count() << " ms" << std::endl;

    std::cout << "\n开始纯查询性能测试..." << std::endl;
    
    // 1. 原始Flat Search (基准算法) - 无需构建索引
    test_algorithm("Flat Search", 
                  [base, base_number, vecdim](float* query, size_t k) {
                      return flat_search(base, query, base_number, vecdim, k);
                  },
                  base, test_query, test_gt, base_number, vecdim, test_number, k, test_gt_d, all_results);

    // 2. KDTree CPU版本 - 使用预构建的索引
    test_algorithm("KDTree CPU", 
                  [kdtree](float* query, size_t k) {
                      return kdtree->search(query, k);
                  },
                  base, test_query, test_gt, base_number, vecdim, test_number, k, test_gt_d, all_results);

    // 3. LSH CPU版本 - 使用预构建的索引
    test_algorithm("LSH CPU", 
                  [lsh, base](float* query, size_t k) {
                      return lsh->search(base, query, k);
                  },
                  base, test_query, test_gt, base_number, vecdim, test_number, k, test_gt_d, all_results);
                  
    // 4. 近似KDTree版本(高召回率) - 使用预构建的索引
    test_algorithm("KDTree Approx 90%", 
                  [approx_kdtree](float* query, size_t k) {
                      return approx_kdtree->search(query, k);
                  },
                  base, test_query, test_gt, base_number, vecdim, test_number, k, test_gt_d, all_results);
                  
    // 5. 改进LSH版本(高召回率) - 使用预构建的索引
    test_algorithm("LSH Improved 90%", 
                  [improved_lsh, base](float* query, size_t k) {
                      return improved_lsh->search(base, query, k);
                  },
                  base, test_query, test_gt, base_number, vecdim, test_number, k, test_gt_d, all_results);
                  
    // 6. SIMD并行KDTree版本 - 使用预构建的索引
    test_algorithm("KDTree SIMD Parallel", 
                  [simd_kdtree](float* query, size_t k) {
                      return simd_kdtree->search(query, k);
                  },
                  base, test_query, test_gt, base_number, vecdim, test_number, k, test_gt_d, all_results);
                  
    // 7. SIMD并行LSH版本 - 使用预构建的索引
    test_algorithm("LSH SIMD Parallel", 
                  [simd_lsh, base](float* query, size_t k) {
                      return simd_lsh->search(base, query, k);
                  },
                  base, test_query, test_gt, base_number, vecdim, test_number, k, test_gt_d, all_results);
                  
    // 8. 集成并行KDTree版本 (12棵树森林) - 使用预构建的索引
    test_algorithm("KDTree Ensemble Parallel", 
                  [ensemble_kdtree](float* query, size_t k) {
                      return ensemble_kdtree->search(query, k);
                  },
                  base, test_query, test_gt, base_number, vecdim, test_number, k, test_gt_d, all_results);
                  
    // 9. SIMD并行Flat Search版本 (作为优化基准)
    test_algorithm("Flat Search SIMD Parallel", 
                  [base, base_number, vecdim](float* query, size_t k) {
                      return flat_search_simd_parallel(base, query, base_number, vecdim, k);
                  },
                  base, test_query, test_gt, base_number, vecdim, test_number, k, test_gt_d, all_results);
                  
    // 10. 智能混合优化KDTree版本 (终极优化)
    test_algorithm("KDTree Hybrid Optimized", 
                  [hybrid_kdtree](float* query, size_t k) {
                      return hybrid_kdtree->search(query, k);
                  },
                  base, test_query, test_gt, base_number, vecdim, test_number, k, test_gt_d, all_results);

    // 打印总结
    print_summary(all_results, test_number);

    // 打印构建时间对比
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "索引构建时间对比:" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Flat Search:         0 ms (无需构建)" << std::endl;
    std::cout << "KDTree CPU:          " << kdtree_build_time.count() << " ms" << std::endl;
    std::cout << "LSH CPU:             " << lsh_build_time.count() << " ms" << std::endl;
    std::cout << "KDTree Approx 90%:   " << approx_kdtree_build_time.count() << " ms" << std::endl;
    std::cout << "LSH Improved 90%:       " << improved_lsh_build_time.count() << " ms" << std::endl;
    std::cout << "KDTree SIMD Parallel:   " << simd_kdtree_build_time.count() << " ms" << std::endl;
    std::cout << "LSH SIMD Parallel:      " << simd_lsh_build_time.count() << " ms" << std::endl;
    std::cout << "KDTree Ensemble Parallel:" << ensemble_kdtree_build_time.count() << " ms" << std::endl;
    std::cout << "Flat Search SIMD Parallel: 0 ms (无需构建)" << std::endl;
    std::cout << "KDTree Hybrid Optimized:   " << hybrid_kdtree_build_time.count() << " ms" << std::endl;

    // 清理内存
    delete kdtree;
    delete lsh;
    delete approx_kdtree;
    delete improved_lsh;
    delete simd_kdtree;
    delete simd_lsh;
    delete ensemble_kdtree;
    delete hybrid_kdtree;
    delete[] test_query;
    delete[] test_gt;  
    delete[] base;

    std::cout << "\n纯查询时间测试完成!" << std::endl;
    std::cout << "注意: 此版本已分离索引构建时间和查询时间" << std::endl;
    std::cout << "新增: SIMD和OpenMP并行优化版本算法对比" << std::endl;
    std::cout << "新增: 集成并行KDTree (12棵树森林) 和 SIMD并行Flat Search" << std::endl;
    std::cout << "使用线程数: " << omp_get_max_threads() << std::endl;
    return 0;
} 