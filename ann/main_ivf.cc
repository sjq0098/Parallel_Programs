#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
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
#include <unordered_map>
#include <algorithm>
#include <filesystem>
#include <regex>

// 引入相关头文件
#include "flat_scan.h"       // 添加flat_search暴力搜索
#include "flat_scan_simd.h"  // 添加SIMD优化的暴力搜索
#include "flat_ivf.h"
#include "omp_ivf.h"
#include "ptd_ivf.h"
#include "pq_ivf.h"
#include "pq_omp_ivf.h"
#include "pq_ptd_ivf.h"

// 方法枚举
enum class MethodType {
    FLAT_SEARCH,      // 原始暴力搜索
    FLAT_SEARCH_SIMD, // SIMD优化的暴力搜索
    IVF_FLAT,         // 串行IVF
    IVF_OMP,          // OpenMP并行IVF
    IVF_PTD,          // pthread并行IVF
    IVFPQ,            // 串行IVF+PQ
    IVFPQ_OMP,        // OpenMP并行IVF+PQ
    IVFPQ_PTD         // pthread并行IVF+PQ
};

// 方法名称映射
const std::unordered_map<MethodType, std::string> method_names = {
    {MethodType::FLAT_SEARCH, "Flat-Search(暴力搜索)"},
    {MethodType::FLAT_SEARCH_SIMD, "Flat-Search-SIMD(SIMD优化暴力搜索)"},
    {MethodType::IVF_FLAT, "IVF-Flat(Serial)"},
    {MethodType::IVF_OMP, "IVF-Flat(OpenMP)"},
    {MethodType::IVF_PTD, "IVF-Flat(pthread)"},
    {MethodType::IVFPQ, "IVF-PQ(Serial)"},
    {MethodType::IVFPQ_OMP, "IVF-PQ(OpenMP)"},
    {MethodType::IVFPQ_PTD, "IVF-PQ(pthread)"}
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
    float recall;
    int64_t latency; // 单位us
};

// 测试暴力搜索方法
std::vector<SearchResult> test_flat_search_method(
    MethodType method,
    float* base,
    float* queries,
    int* groundtruth,
    size_t base_number,
    size_t query_number,
    size_t vecdim,
    size_t gt_k,
    size_t k)
{
    std::vector<SearchResult> results(query_number);
    
    for (size_t q = 0; q < query_number; ++q) {
        int64_t latency = 0;
        
        #ifdef _WIN32
        Performance_Counter timer;
        timer.start_counter();
        #else
        struct timeval start_time, end_time;
        gettimeofday(&start_time, nullptr);
        #endif
        
        std::priority_queue<std::pair<float, uint32_t>> res;
        float* query = queries + q * vecdim;
        
        // 根据方法选择搜索函数
        switch (method) {
            case MethodType::FLAT_SEARCH:
                res = flat_search(base, query, base_number, vecdim, k);
                break;
            case MethodType::FLAT_SEARCH_SIMD:
                res = flat_search_sse(base, query, base_number, vecdim, k);
                break;
            default:
                std::cerr << "未知的暴力搜索方法类型" << std::endl;
                return {};
        }
        
        #ifdef _WIN32
        timer.end_counter();
        latency = timer.microseconds();
        #else
        gettimeofday(&end_time, nullptr);
        latency = (end_time.tv_sec - start_time.tv_sec) * 1000000 + 
                  (end_time.tv_usec - start_time.tv_usec);
        #endif
        
        // 计算召回率
        std::set<uint32_t> gt_set;
        for (size_t j = 0; j < k && j < gt_k; ++j) {
            gt_set.insert(groundtruth[q * gt_k + j]);
        }
        
        size_t hits = 0;
        while (!res.empty()) {
            uint32_t id = res.top().second;
            if (gt_set.find(id) != gt_set.end()) {
                ++hits;
            }
            res.pop();
        }
        
        float recall = static_cast<float>(hits) / std::min<size_t>(k, gt_k);
        results[q] = {recall, latency};
    }
    
    return results;
}

// 测试IVF方法
std::vector<SearchResult> test_ivf_method(
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
    size_t k) 
{
    std::vector<SearchResult> results(query_number);
    
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
        return {};
    }
    
    // 为每个查询执行搜索
    for (size_t q = 0; q < query_number; ++q) {
        int64_t latency = 0;
        #ifdef _WIN32
        Performance_Counter timer;
        timer.start_counter();
        #else
        struct timeval start_time, end_time;
        gettimeofday(&start_time, nullptr);
        #endif

        std::priority_queue<std::pair<float, uint32_t>> res;
        float* query = queries + q * vecdim;
        
        // 根据方法选择搜索函数
        switch (method) {
            case MethodType::IVF_FLAT:
                res = flat_ivf_search(base, query, base_number, vecdim, k,
                                      centroids.data(), nlist, invlists, nprobe);
                break;
            case MethodType::IVF_OMP:
                res = ivf_search_omp(base, query, base_number, vecdim, k,
                                     centroids.data(), nlist, invlists, nprobe);
                break;
            case MethodType::IVF_PTD: {
                int num_threads = std::thread::hardware_concurrency();
                res = ivf_search_ptd(base, query, base_number, vecdim, k,
                                   centroids.data(), nlist, invlists, nprobe, num_threads);
                break;
            }
            default:
                std::cerr << "未知的IVF方法类型" << std::endl;
                return {};
        }
        
        #ifdef _WIN32
        timer.end_counter();
        latency = timer.microseconds();
        #else
        gettimeofday(&end_time, nullptr);
        latency = (end_time.tv_sec - start_time.tv_sec) * 1000000 + 
                  (end_time.tv_usec - start_time.tv_usec);
        #endif
        
        // 计算召回率
        std::set<uint32_t> gt_set;
        for (size_t j = 0; j < k && j < gt_k; ++j) {
            gt_set.insert(groundtruth[q * gt_k + j]);
        }
        
        size_t hits = 0;
        while (!res.empty()) {
            uint32_t id = res.top().second;
            if (gt_set.find(id) != gt_set.end()) {
                ++hits;
            }
            res.pop();
        }
        
        float recall = static_cast<float>(hits) / std::min<size_t>(k, gt_k);
        results[q] = {recall, latency};
    }
    
    return results;
}

// 测试IVFPQ方法
std::vector<SearchResult> test_ivfpq_method(
    MethodType method,
    size_t nlist,
    size_t m,
    size_t nprobe,
    float* queries,
    int* groundtruth,
    size_t query_number,
    size_t gt_k,
    size_t k,
    bool do_rerank = false,
    size_t rerank_candidates = 100) 
{
    std::vector<SearchResult> results(query_number);
    
    // 加载IVFPQ索引
    std::string index_path = "files/ivf_pq_nlist" + std::to_string(nlist) + 
                             "_m" + std::to_string(m) + "_b8.bin";
    
    std::unique_ptr<IVFPQIndex> index;
    try {
        index = load_ivfpq_index(index_path);
        
        // 如果需要重排序，加载原始数据
        if (do_rerank) {
            std::string raw_data_path = "anndata/DEEP100K.base.100k.fbin";
            if (!load_raw_data_for_rerank(index.get(), raw_data_path)) {
                std::cerr << "无法加载重排序所需的原始数据，将不进行重排序" << std::endl;
                do_rerank = false;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "加载IVFPQ索引失败: " << e.what() << std::endl;
        return {};
    }
    
    // 为每个查询执行搜索
    for (size_t q = 0; q < query_number; ++q) {
        int64_t latency = 0;
        #ifdef _WIN32
        Performance_Counter timer;
        timer.start_counter();
        #else
        struct timeval start_time, end_time;
        gettimeofday(&start_time, nullptr);
        #endif

        std::priority_queue<std::pair<float, uint32_t>> res;
        float* query = queries + q * index->d;
        
        // 根据方法选择搜索函数
        switch (method) {
            case MethodType::IVFPQ:
                res = ivfpq_search(index.get(), query, k, nprobe, do_rerank, rerank_candidates);
                break;
            case MethodType::IVFPQ_OMP: {
                int num_threads = omp_get_max_threads();
                res = ivfpq_search_omp(index.get(), query, k, nprobe, num_threads, do_rerank, rerank_candidates);
                break;
            }
            case MethodType::IVFPQ_PTD: {
                int num_threads = std::thread::hardware_concurrency();
                res = ivfpq_search_ptd(index.get(), query, k, nprobe, num_threads, do_rerank, rerank_candidates);
                break;
            }
            default:
                std::cerr << "未知的IVFPQ方法类型" << std::endl;
                return {};
        }
        
        #ifdef _WIN32
        timer.end_counter();
        latency = timer.microseconds();
        #else
        gettimeofday(&end_time, nullptr);
        latency = (end_time.tv_sec - start_time.tv_sec) * 1000000 + 
                  (end_time.tv_usec - start_time.tv_usec);
        #endif
        
        // 计算召回率
        std::set<uint32_t> gt_set;
        for (size_t j = 0; j < k && j < gt_k; ++j) {
            gt_set.insert(groundtruth[q * gt_k + j]);
        }
        
        size_t hits = 0;
        while (!res.empty()) {
            uint32_t id = res.top().second;
            if (gt_set.find(id) != gt_set.end()) {
                ++hits;
            }
            res.pop();
        }
        
        float recall = static_cast<float>(hits) / std::min<size_t>(k, gt_k);
        results[q] = {recall, latency};
    }
    
    return results;
}

// 运行暴力搜索测试
void run_flat_search_tests(
    float* base,
    float* queries,
    int* groundtruth,
    size_t base_number,
    size_t query_number,
    size_t vecdim,
    size_t gt_k,
    size_t k)
{
    std::cout << "Method,Recall,Latency(us)" << std::endl;
    
    // 测试方法
    std::vector<MethodType> methods = {
        MethodType::FLAT_SEARCH,
        MethodType::FLAT_SEARCH_SIMD
    };
    
    for (auto method : methods) {
        auto results = test_flat_search_method(
            method, base, queries, groundtruth,
            base_number, query_number, vecdim, gt_k, k
        );
        
        if (results.empty()) continue;
        
        // 计算平均结果
        float avg_recall = 0;
        int64_t avg_latency = 0;
        
        for (const auto& res : results) {
            avg_recall += res.recall;
            avg_latency += res.latency;
        }
        
        avg_recall /= results.size();
        avg_latency /= results.size();
        
        std::cout << method_names.at(method) << "," 
                  << avg_recall << "," 
                  << avg_latency << std::endl;
    }
}

// 运行所有IVF测试
void run_all_ivf_tests(
    float* base,
    float* queries,
    int* groundtruth,
    size_t base_number,
    size_t query_number,
    size_t vecdim,
    size_t gt_k,
    size_t k)
{
    std::cout << "Method,Configuration,Recall,Latency(us)" << std::endl;
    
    // 测试参数
    const std::vector<size_t> nlist_values = {64, 128, 256, 512};
    const std::vector<size_t> nprobe_values = {4, 8, 12, 16, 20, 24, 32};
    
    // 测试IVF-Flat方法
    for (MethodType method : {MethodType::IVF_FLAT, MethodType::IVF_OMP, MethodType::IVF_PTD}) {
        for (size_t nlist : nlist_values) {
            for (size_t nprobe : nprobe_values) {
                // 确保nprobe <= nlist
                if (nprobe > nlist) continue;
                
                std::string config = "nlist=" + std::to_string(nlist) + 
                                     ",nprobe=" + std::to_string(nprobe);
                
                auto results = test_ivf_method(
                    method, nlist, nprobe, base, queries, groundtruth,
                    base_number, query_number, vecdim, gt_k, k
                );
                
                if (results.empty()) continue;
                
                // 计算平均结果
                float avg_recall = 0;
                int64_t avg_latency = 0;
                
                for (const auto& res : results) {
                    avg_recall += res.recall;
                    avg_latency += res.latency;
                }
                
                avg_recall /= results.size();
                avg_latency /= results.size();
                
                std::cout << method_names.at(method) << "," 
                          << config << "," 
                          << avg_recall << "," 
                          << avg_latency << std::endl;
            }
        }
    }
}

// 运行所有IVFPQ测试
void run_all_ivfpq_tests(
    float* queries,
    int* groundtruth,
    size_t query_number,
    size_t gt_k,
    size_t k)
{
    // 创建临时文件存储结果
    std::stringstream results_buffer;
    results_buffer << "Method,Configuration,Recall,Latency(us)" << std::endl;
    
    // 测试参数
    const std::vector<size_t> nlist_values = {64, 128, 256, 512};
    const std::vector<size_t> m_values = {8, 16, 32};
    // 减少测试的nprobe值，主要关注4,12,24几个有代表性的值
    const std::vector<size_t> nprobe_values = {4, 12, 24};
    // 添加重排序配置
    const std::vector<bool> rerank_options = {false, true};
    // 不再需要指定rerank_candidates，使用默认的计算方式
    
    // 获取可用的索引文件
    std::vector<std::tuple<size_t, size_t>> available_indices;
    
    std::regex index_pattern("ivf_pq_nlist(\\d+)_m(\\d+)_b8\\.bin");
    
    for (const auto& entry : std::filesystem::directory_iterator("files")) {
        if (!entry.is_regular_file()) continue;
        
        std::string filename = entry.path().filename().string();
        std::smatch match;
        
        if (std::regex_search(filename, match, index_pattern)) {
            size_t nlist = std::stoul(match[1].str());
            size_t m = std::stoul(match[2].str());
            available_indices.emplace_back(nlist, m);
        }
    }
    
    // 测试IVFPQ方法
    for (MethodType method : {MethodType::IVFPQ, MethodType::IVFPQ_OMP, MethodType::IVFPQ_PTD}) {
        for (const auto& [nlist, m] : available_indices) {
            for (size_t nprobe : nprobe_values) {
                // 确保nprobe <= nlist
                if (nprobe > nlist) continue;
                
                // 测试不同的重排序选项
                for (bool do_rerank : rerank_options) {
                    std::string rerank_str = do_rerank ? ",rerank=true" : "";
                    // 自动计算的候选集大小
                    size_t actual_rerank_candidates = do_rerank ? k * nprobe * 8 : 0;
                    
                    std::string config = "nlist=" + std::to_string(nlist) + 
                                        ",m=" + std::to_string(m) +
                                        ",nprobe=" + std::to_string(nprobe) +
                                        rerank_str;
                    
                    // 如果是重排序模式，添加候选集大小信息
                    if (do_rerank) {
                        config += ",candidates=" + std::to_string(actual_rerank_candidates);
                    }
                    
                    auto results = test_ivfpq_method(
                        method, nlist, m, nprobe, queries, groundtruth,
                        query_number, gt_k, k, do_rerank
                    );
                    
                    if (results.empty()) continue;
                    
                    // 计算平均结果
                    float avg_recall = 0;
                    int64_t avg_latency = 0;
                    
                    for (const auto& res : results) {
                        avg_recall += res.recall;
                        avg_latency += res.latency;
                    }
                    
                    avg_recall /= results.size();
                    avg_latency /= results.size();
                    
                    results_buffer << method_names.at(method) << "," 
                            << config << "," 
                            << std::fixed << std::setprecision(5) << avg_recall << "," 
                            << avg_latency << std::endl;
                }
            }
        }
    }
    
    // 输出最终结果到标准输出（跳过含有"加载IVFPQ索引"的行）
    std::string line;
    std::stringstream ss(results_buffer.str());
    
    while (std::getline(ss, line)) {
        if (line.find("加载IVFPQ索引") == std::string::npos) {
            std::cout << line << std::endl;
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
    query_number = std::min<size_t>(query_number, 200);
    
    // 搜索参数
    const size_t k = 10;  // 返回前k个结果
    
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " <method>" << std::endl;
        std::cout << "  method: flat - 测试暴力搜索方法" << std::endl;
        std::cout << "          ivf - 测试IVF相关方法" << std::endl;
        std::cout << "          ivfpq - 测试IVFPQ相关方法" << std::endl;
        std::cout << "          all - 测试所有方法" << std::endl;
        return 1;
    }
    
    std::string test_method = argv[1];
    
    if (test_method == "flat" || test_method == "all") {
        run_flat_search_tests(base, queries, groundtruth, base_number, query_number, vecdim, gt_k, k);
    }
    
    if (test_method == "ivf" || test_method == "all") {
        run_all_ivf_tests(base, queries, groundtruth, base_number, query_number, vecdim, gt_k, k);
    }
    
    if (test_method == "ivfpq" || test_method == "all") {
        run_all_ivfpq_tests(queries, groundtruth, query_number, gt_k, k);
    }
    
    // 释放内存
    delete[] queries;
    delete[] groundtruth;
    delete[] base;
    
    return 0;
} 