#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <unordered_map>

// Windows平台时间处理
#ifdef _WIN32
#include <windows.h>
// Windows平台上使用QueryPerformanceCounter替代gettimeofday
struct Performance_Counter {
    LARGE_INTEGER frequency;        // 计时器频率
    LARGE_INTEGER start;            // 开始时间点
    LARGE_INTEGER end;              // 结束时间点

    Performance_Counter() {
        QueryPerformanceFrequency(&frequency);
    }

    void start_counter() {
        QueryPerformanceCounter(&start);
    }

    void end_counter() {
        QueryPerformanceCounter(&end);
    }

    // 返回微秒数
    int64_t microseconds() const {
        LARGE_INTEGER elapsed;
        elapsed.QuadPart = end.QuadPart - start.QuadPart;
        
        // 转换为微秒: (elapsed * 1,000,000) / frequency
        elapsed.QuadPart *= 1000000;
        elapsed.QuadPart /= frequency.QuadPart;
        
        return elapsed.QuadPart;
    }
};
#else
#include <sys/time.h>
#endif

#include <omp.h>
#include <thread>
#include <algorithm>
#include <filesystem>
#include <regex>

// 引入相关头文件
#include "flat_ivf.h"        // 串行IVF
#include "omp_multi_ivf.h"   // OpenMP多查询并行
#include "ptd_multi_ivf.h"   // Pthread多查询并行

// 方法枚举
enum class MethodType {
    IVF_SERIAL,      // 串行IVF (每个查询串行处理)
    IVF_OMP_MULTI,   // OpenMP多查询并行
    IVF_PTD_MULTI    // Pthread多查询并行
};

// 方法名称映射
const std::unordered_map<MethodType, std::string> method_names = {
    {MethodType::IVF_SERIAL, "IVF-Serial(逐个处理查询)"},
    {MethodType::IVF_OMP_MULTI, "IVF-OMP-Multi(OpenMP多查询并行)"},
    {MethodType::IVF_PTD_MULTI, "IVF-PTD-Multi(Pthread多查询并行)"}
};

// 加载数据
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

// 搜索结果
struct SearchResult {
    float avg_recall;
    int64_t total_latency; // 单位us
    float throughput;     // 每秒查询数 (QPS)
};

// 测试多查询并行方法
SearchResult test_multi_query_method(
    MethodType method,
    size_t nlist,
    size_t nprobe,
    float* base,
    float* queries,
    int* groundtruth,
    size_t base_number,
    size_t query_number,
    size_t vecdim,
    size_t gt_k,
    size_t k,
    int num_threads) 
{
    // 加载IVF数据
    std::string centroids_path = "files/ivf_flat_centroids_" + std::to_string(nlist) + ".fbin";
    std::string invlists_path = "files/ivf_flat_invlists_" + std::to_string(nlist) + ".bin";
    
    std::vector<float> centroids;
    std::vector<std::vector<uint32_t>> invlists;
    
    try {
        centroids = load_ivf_centroids(centroids_path, nlist, vecdim);
        invlists = load_ivf_invlists(invlists_path, nlist);
    } catch (const std::exception& e) {
        std::cerr << "加载IVF数据失败: " << e.what() << std::endl;
        return {0.0f, 0, 0.0f};
    }
    
    int64_t total_latency = 0;
    float avg_recall = 0.0f;
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results;
    
    #ifdef _WIN32
    Performance_Counter timer;
    timer.start_counter();
    #else
    struct timeval start_time, end_time;
    gettimeofday(&start_time, nullptr);
    #endif
    
    // 根据方法选择搜索函数
    switch (method) {
        case MethodType::IVF_SERIAL: {
            // 串行处理所有查询
            results.resize(query_number);
            for (size_t q = 0; q < query_number; ++q) {
                float* query = queries + q * vecdim;
                results[q] = flat_ivf_search(base, query, base_number, vecdim, k,
                                           centroids.data(), nlist, invlists, nprobe);
            }
            break;
        }
        case MethodType::IVF_OMP_MULTI: {
            // OpenMP多查询并行
            results = multi_ivf_search_omp(base, queries, query_number, base_number, vecdim, k,
                                        centroids.data(), nlist, invlists, nprobe, num_threads);
            break;
        }
        case MethodType::IVF_PTD_MULTI: {
            // Pthread多查询并行
            results = multi_ivf_search_ptd(base, queries, query_number, base_number, vecdim, k,
                                       centroids.data(), nlist, invlists, nprobe, num_threads);
            break;
        }
        default:
            std::cerr << "未知的方法类型" << std::endl;
            return {0.0f, 0, 0.0f};
    }
    
    #ifdef _WIN32
    timer.end_counter();
    total_latency = timer.microseconds();
    #else
    gettimeofday(&end_time, nullptr);
    total_latency = (end_time.tv_sec - start_time.tv_sec) * 1000000 + 
                  (end_time.tv_usec - start_time.tv_usec);
    #endif
    
    // 计算平均召回率
    for (size_t q = 0; q < query_number; ++q) {
        std::set<uint32_t> gt_set;
        for (size_t j = 0; j < k && j < gt_k; ++j) {
            gt_set.insert(groundtruth[q * gt_k + j]);
        }
        
        size_t hits = 0;
        auto& res = results[q];
        std::vector<uint32_t> result_ids;
        
        while (!res.empty()) {
            uint32_t id = res.top().second;
            result_ids.push_back(id);
            res.pop();
        }
        
        // 按照正确的顺序检查（从最近到最远）
        std::reverse(result_ids.begin(), result_ids.end());
        for (uint32_t id : result_ids) {
            if (gt_set.find(id) != gt_set.end()) {
                ++hits;
            }
        }
        
        float recall = static_cast<float>(hits) / std::min<size_t>(k, gt_k);
        avg_recall += recall;
    }
    
    avg_recall /= query_number;
    float throughput = static_cast<float>(query_number) / (static_cast<float>(total_latency) / 1000000.0f);
    
    return {avg_recall, total_latency, throughput};
}

// 运行测试
void run_multi_query_tests(
    float* base,
    float* queries,
    int* groundtruth,
    size_t base_number,
    size_t query_number,
    size_t vecdim,
    size_t gt_k,
    size_t k)
{
    std::cout << "Method,nlist,nprobe,threads,avg_recall,avg_latency(us),throughput(QPS)" << std::endl;
    
    // 测试参数
    const std::vector<size_t> nlist_values = {128, 256};
    const std::vector<size_t> nprobe_values = {8, 16, 32};
    const std::vector<int> thread_counts = {1, 2, 4, 8, 16}; // 线程数
    
    // 测试所有方法
    for (MethodType method : {MethodType::IVF_SERIAL, MethodType::IVF_OMP_MULTI, MethodType::IVF_PTD_MULTI}) {
        for (size_t nlist : nlist_values) {
            for (size_t nprobe : nprobe_values) {
                // 对于串行方法，不需要测试多个线程数
                std::vector<int> threads_to_test = (method == MethodType::IVF_SERIAL) 
                                                ? std::vector<int>{1} 
                                                : thread_counts;
                
                for (int num_threads : threads_to_test) {
                    auto result = test_multi_query_method(
                        method, nlist, nprobe, base, queries, groundtruth,
                        base_number, query_number, vecdim, gt_k, k, num_threads
                    );
                    
                    std::cout << method_names.at(method) << ","
                              << nlist << ","
                              << nprobe << ","
                              << num_threads << ","
                              << result.avg_recall << ","
                              << result.total_latency/2000 << ","
                              << result.throughput << std::endl;
                }
            }
        }
    }
}

// 主函数
int main(int argc, char* argv[]) {
    // 加载数据
    std::string data_path = "anndata/";
    size_t query_number = 0, vecdim = 0;
    size_t gt_k = 0;
    size_t base_number = 0;
    
    auto queries = LoadData<float>(data_path + "DEEP100K.query.fbin", query_number, vecdim);
    auto groundtruth = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", query_number, gt_k);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 限制查询数量，避免测试时间过长
    query_number = std::min<size_t>(query_number, 2000);
    
    // 搜索参数
    const size_t k = 10;  // 返回前k个结果
    
    run_multi_query_tests(base, queries, groundtruth, base_number, query_number, vecdim, gt_k, k);
    
    // 释放内存
    delete[] queries;
    delete[] groundtruth;
    delete[] base;
    
    return 0;
}