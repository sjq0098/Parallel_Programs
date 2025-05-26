#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <regex>
#ifdef _WIN32
#include <windows.h>
struct Performance_Counter {
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;

    Performance_Counter() { QueryPerformanceFrequency(&frequency); }
    void start_counter() { QueryPerformanceCounter(&start); }
    void end_counter() { QueryPerformanceCounter(&end); }
    int64_t microseconds() const {
        LARGE_INTEGER elapsed;
        elapsed.QuadPart = end.QuadPart - start.QuadPart;
        elapsed.QuadPart *= 1000000;
        elapsed.QuadPart /= frequency.QuadPart;
        return elapsed.QuadPart;
    }
};
#else
#include <sys/time.h>
#endif

#include "flat_scan.h"       // 暴力搜索
#include "flat_scan_simd.h"  // SIMD优化暴力搜索
#include "flat_hnsw.h"       // 串行HNSW
#include "omp_hnsw.h"        // 并行HNSW

// 方法枚举
enum class MethodType {
    FLAT_SEARCH,
    FLAT_SEARCH_SIMD,
    HNSW_SERIAL,
    HNSW_OMP
};

const std::unordered_map<MethodType, std::string> method_names = {
    {MethodType::FLAT_SEARCH, "Flat-Search(暴力搜索)"},
    {MethodType::FLAT_SEARCH_SIMD, "Flat-Search-SIMD(SIMD优化暴力搜索)"},
    {MethodType::HNSW_SERIAL, "HNSW(Serial)"},
    {MethodType::HNSW_OMP, "HNSW(OpenMP)"}
};

struct SearchResult {
    float recall;
    int64_t latency; // us
};

template<typename T>
T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    for (size_t i = 0; i < n; ++i) {
        fin.read(reinterpret_cast<char*>(data + i * d), d * sizeof(T));
    }
    fin.close();
    std::cerr << "load data " << data_path
              << "  #points=" << n << "  dim=" << d
              << "  sizeof(T)=" << sizeof(T) << "\n";
    return data;
}

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
        auto* query = queries + q * vecdim;
        std::priority_queue<std::pair<float, uint32_t>> res;
        if (method == MethodType::FLAT_SEARCH) {
            res = flat_search(base, query, base_number, vecdim, k);
        } else {
            res = flat_search_sse(base, query, base_number, vecdim, k);
        }
#ifdef _WIN32
        timer.end_counter(); latency = timer.microseconds();
#else
        gettimeofday(&end_time, nullptr);
        latency = (end_time.tv_sec - start_time.tv_sec) * 1000000 +
                  (end_time.tv_usec - start_time.tv_usec);
#endif
        std::set<uint32_t> gt_set;
        for (size_t j = 0; j < k && j < gt_k; ++j)
            gt_set.insert(groundtruth[q * gt_k + j]);
        size_t hits = 0;
        while (!res.empty()) {
            if (gt_set.count(res.top().second)) ++hits;
            res.pop();
        }
        results[q] = {static_cast<float>(hits) / std::min(k, gt_k), latency};
    }
    return results;
}

std::vector<SearchResult> test_hnsw_method(
    MethodType method,
    size_t M,
    size_t efConstruction,
    size_t efSearch,
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
    std::string index_path = "files/hnsw_M" + std::to_string(M)
                            + "_efC" + std::to_string(efConstruction) + ".index";

    if (method == MethodType::HNSW_SERIAL) {
        HNSWIndex index(vecdim, index_path, base_number);
        index.setEf(efSearch);
        for (size_t q = 0; q < query_number; ++q) {
            int64_t latency = 0;
#ifdef _WIN32
            Performance_Counter timer; timer.start_counter();
#else
            struct timeval start_time, end_time;
            gettimeofday(&start_time, nullptr);
#endif
            auto* query = queries + q * vecdim;
            auto res = index.flat_hnsw_search(query, k);
#ifdef _WIN32
            timer.end_counter(); latency = timer.microseconds();
#else
            gettimeofday(&end_time, nullptr);
            latency = (end_time.tv_sec - start_time.tv_sec) * 1000000 +
                      (end_time.tv_usec - start_time.tv_usec);
#endif
            std::set<uint32_t> gt_set;
            for (size_t j = 0; j < k && j < gt_k; ++j)
                gt_set.insert(groundtruth[q * gt_k + j]);
            size_t hits = 0;
            while (!res.empty()) {
                if (gt_set.count(res.top().second)) ++hits;
                res.pop();
            }
            results[q] = {static_cast<float>(hits) / std::min(k, gt_k), latency};
        }
    } else {
        HNSWIndex_omp index(vecdim, index_path, base_number);
        index.setEf(efSearch);
        for (size_t q = 0; q < query_number; ++q) {
            int64_t latency = 0;
#ifdef _WIN32
            Performance_Counter timer; timer.start_counter();
#else
            struct timeval start_time, end_time;
            gettimeofday(&start_time, nullptr);
#endif
            auto* query = queries + q * vecdim;
            auto res = index.flat_hnsw_search(query, k);
#ifdef _WIN32
            timer.end_counter(); latency = timer.microseconds();
#else
            gettimeofday(&end_time, nullptr);
            latency = (end_time.tv_sec - start_time.tv_sec) * 1000000 +
                      (end_time.tv_usec - start_time.tv_usec);
#endif
            std::set<uint32_t> gt_set;
            for (size_t j = 0; j < k && j < gt_k; ++j)
                gt_set.insert(groundtruth[q * gt_k + j]);
            size_t hits = 0;
            while (!res.empty()) {
                if (gt_set.count(res.top().second)) ++hits;
                res.pop();
            }
            results[q] = {static_cast<float>(hits) / std::min(k, gt_k), latency};
        }
    }

    return results;
}

void run_all_tests(
    float* base,
    float* queries,
    int* groundtruth,
    size_t base_number,
    size_t query_number,
    size_t vecdim,
    size_t gt_k,
    size_t k)
{
    std::cout << "Method,Configuration,Recall,Latency(us)\n";
    // 暴力搜索
    for (auto method : {MethodType::FLAT_SEARCH, MethodType::FLAT_SEARCH_SIMD}) {
        auto results = test_flat_search_method(method, base, queries, groundtruth,
                                               base_number, query_number, vecdim, gt_k, k);
        if (results.empty()) continue;
        float sum_r=0; int64_t sum_l=0;
        for (auto& r : results) { sum_r+=r.recall; sum_l+=r.latency; }
        std::cout << method_names.at(method) << ",k="<<k<<","<< (sum_r/results.size())
                  <<","<<(sum_l/results.size())<<"\n";
    }
    // HNSW
    std::vector<std::pair<size_t,size_t>> available;
    std::regex pat("hnsw_M(\\d+)_efC(\\d+)\\.index");
    for (auto& e : std::filesystem::directory_iterator("files")) {
        if (!e.is_regular_file()) continue;
        std::smatch m;
        auto fn = e.path().filename().string();
        if (std::regex_search(fn, m, pat))
            available.emplace_back(std::stoul(m[1]), std::stoul(m[2]));
    }
    std::vector<size_t> efS_vals={50,100,200,300,400,500};
    for (auto method : {MethodType::HNSW_SERIAL, MethodType::HNSW_OMP}) {
        for (auto [M,efC]: available) {
            for (auto efS: efS_vals) {
                auto results = test_hnsw_method(method, M, efC, efS,
                                               base, queries, groundtruth,
                                               base_number, query_number, vecdim, gt_k, k);
                if (results.empty()) continue;
                float sum_r=0; int64_t sum_l=0;
                for (auto& r : results) { sum_r+=r.recall; sum_l+=r.latency; }
                std::cout << method_names.at(method)
                          << ",M="<<M<<",efC="<<efC<<",efS="<<efS
                          <<","<<(sum_r/results.size())
                          <<","<<(sum_l/results.size())<<"\n";
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // 加载数据
    std::string data_path = "anndata/";
    size_t query_number = 0, vecdim = 0;
    size_t gt_k = 0;
    size_t base_number = 0;
    
    auto queries = LoadData<float>(data_path + "DEEP100K.query.fbin", query_number, vecdim);
    auto groundtruth = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", query_number, gt_k);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 限制查询数量
    query_number = std::min<size_t>(query_number, 200);
    
    // 搜索参数
    const size_t k = 10;  // 返回前k个结果
    
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " <method>" << std::endl;
        std::cout << "  method: flat - 测试暴力搜索方法" << std::endl;
        std::cout << "          hnsw - 测试HNSW相关方法" << std::endl;
        std::cout << "          all - 测试所有方法" << std::endl;
        return 1;
    }
    
    std::string test_method = argv[1];
    
    if (test_method == "flat" || test_method == "all") {
        run_all_tests(base, queries, groundtruth, base_number, query_number, vecdim, gt_k, k);
    }
    
    if (test_method == "hnsw" || test_method == "all") {
        run_all_tests(base, queries, groundtruth, base_number, query_number, vecdim, gt_k, k);
    }
    
    // 释放内存
    delete[] queries;
    delete[] groundtruth;
    delete[] base;
    
    return 0;
}