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
#include "hnswlib/hnswlib/hnswlib.h"
#include "pq.h"
#include "flat_scan_simd.h"  // 改为平台无关名称
#include "flat_scan_sq.h"
#include <unordered_map>
#include "flat_ivf.h" // 添加 IVF 头文件
#include "omp_ivf.h" // 添加 OMP IVF 头文件
#include "ptd_ivf.h" // 添加 pthread IVF 头文件
// 可以自行添加需要的头文件

using namespace hnswlib;

// 搜索方法枚举
enum SearchMethod {
    FLAT_SEARCH,       // 原始暴力搜索
    FLAT_SEARCH_SQ,    // 标量量化暴力搜索
    PQ4_SEARCH,        // PQ4量化搜索
    PQ8_SEARCH,        // PQ8量化搜索
    PQ16_SEARCH,       // PQ16量化搜索
    PQ32_SEARCH,       // PQ32量化搜索
    PQ4_RERANK_SEARCH, // PQ4量化+重排序搜索
    PQ8_RERANK_SEARCH, // PQ8量化+重排序搜索
    PQ16_RERANK_SEARCH,// PQ16量化+重排序搜索
    PQ32_RERANK_SEARCH, // PQ32量化+重排序搜索
    IVF_FLAT_N64_P8_SEARCH,
    IVF_FLAT_N128_P12_SEARCH,
    IVF_FLAT_N256_P16_SEARCH,
    IVF_FLAT_N512_P20_SEARCH,
    // IVF configurations
    IVF_FLAT_N64_P4_SEARCH,
    IVF_FLAT_N64_P12_SEARCH,
    IVF_FLAT_N128_P8_SEARCH,
    IVF_FLAT_N128_P16_SEARCH,
    IVF_FLAT_N256_P12_SEARCH,
    IVF_FLAT_N256_P24_SEARCH,
    IVF_FLAT_N512_P16_SEARCH,
    IVF_FLAT_N512_P32_SEARCH,
    // OMP IVF configurations
    IVF_OMP_N64_P4_SEARCH,
    IVF_OMP_N64_P8_SEARCH,
    IVF_OMP_N64_P12_SEARCH,
    IVF_OMP_N128_P8_SEARCH,
    IVF_OMP_N128_P12_SEARCH,
    IVF_OMP_N128_P16_SEARCH,
    IVF_OMP_N256_P12_SEARCH,
    IVF_OMP_N256_P16_SEARCH,
    IVF_OMP_N256_P24_SEARCH,
    IVF_OMP_N512_P16_SEARCH,
    IVF_OMP_N512_P20_SEARCH,
    IVF_OMP_N512_P32_SEARCH,
    // pthread IVF configurations
    IVF_PTD_N64_P4_SEARCH,
    IVF_PTD_N64_P8_SEARCH,
    IVF_PTD_N64_P12_SEARCH,
    IVF_PTD_N128_P8_SEARCH,
    IVF_PTD_N128_P12_SEARCH,
    IVF_PTD_N128_P16_SEARCH,
    IVF_PTD_N256_P12_SEARCH,
    IVF_PTD_N256_P16_SEARCH,
    IVF_PTD_N256_P24_SEARCH,
    IVF_PTD_N512_P16_SEARCH,
    IVF_PTD_N512_P20_SEARCH,
    IVF_PTD_N512_P32_SEARCH
};

// 方法名称映射
const std::unordered_map<SearchMethod, std::string> method_names = {
    {FLAT_SEARCH, "Flat Search"},
    {FLAT_SEARCH_SQ, "SQ Flat Search"},
    {PQ4_SEARCH, "PQ4 Search"},
    {PQ8_SEARCH, "PQ8 Search"},
    {PQ16_SEARCH, "PQ16 Search"},
    {PQ32_SEARCH, "PQ32 Search"},
    {PQ4_RERANK_SEARCH, "PQ4 Rerank Search"},
    {PQ8_RERANK_SEARCH, "PQ8 Rerank Search"},
    {PQ16_RERANK_SEARCH, "PQ16 Rerank Search"},
    {PQ32_RERANK_SEARCH, "PQ32 Rerank Search"},
    {IVF_FLAT_N64_P8_SEARCH, "IVF Flat (N=64, P=8)"},
    {IVF_FLAT_N128_P12_SEARCH, "IVF Flat (N=128, P=12)"},
    {IVF_FLAT_N256_P16_SEARCH, "IVF Flat (N=256, P=16)"},
    {IVF_FLAT_N512_P20_SEARCH, "IVF Flat (N=512, P=20)"},
    // IVF names
    {IVF_FLAT_N64_P4_SEARCH, "IVF Flat (N=64, P=4)"},
    {IVF_FLAT_N64_P12_SEARCH, "IVF Flat (N=64, P=12)"},
    {IVF_FLAT_N128_P8_SEARCH, "IVF Flat (N=128, P=8)"},
    {IVF_FLAT_N128_P16_SEARCH, "IVF Flat (N=128, P=16)"},
    {IVF_FLAT_N256_P12_SEARCH, "IVF Flat (N=256, P=12)"},
    {IVF_FLAT_N256_P24_SEARCH, "IVF Flat (N=256, P=24)"},
    {IVF_FLAT_N512_P16_SEARCH, "IVF Flat (N=512, P=16)"},
    {IVF_FLAT_N512_P32_SEARCH, "IVF Flat (N=512, P=32)"},
    // OMP IVF names
    {IVF_OMP_N64_P4_SEARCH, "IVF OMP (N=64, P=4)"},
    {IVF_OMP_N64_P8_SEARCH, "IVF OMP (N=64, P=8)"},
    {IVF_OMP_N64_P12_SEARCH, "IVF OMP (N=64, P=12)"},
    {IVF_OMP_N128_P8_SEARCH, "IVF OMP (N=128, P=8)"},
    {IVF_OMP_N128_P12_SEARCH, "IVF OMP (N=128, P=12)"},
    {IVF_OMP_N128_P16_SEARCH, "IVF OMP (N=128, P=16)"},
    {IVF_OMP_N256_P12_SEARCH, "IVF OMP (N=256, P=12)"},
    {IVF_OMP_N256_P16_SEARCH, "IVF OMP (N=256, P=16)"},
    {IVF_OMP_N256_P24_SEARCH, "IVF OMP (N=256, P=24)"},
    {IVF_OMP_N512_P16_SEARCH, "IVF OMP (N=512, P=16)"},
    {IVF_OMP_N512_P20_SEARCH, "IVF OMP (N=512, P=20)"},
    {IVF_OMP_N512_P32_SEARCH, "IVF OMP (N=512, P=32)"},
    // pthread IVF names
    {IVF_PTD_N64_P4_SEARCH, "IVF PTD (N=64, P=4)"},
    {IVF_PTD_N64_P8_SEARCH, "IVF PTD (N=64, P=8)"},
    {IVF_PTD_N64_P12_SEARCH, "IVF PTD (N=64, P=12)"},
    {IVF_PTD_N128_P8_SEARCH, "IVF PTD (N=128, P=8)"},
    {IVF_PTD_N128_P12_SEARCH, "IVF PTD (N=128, P=12)"},
    {IVF_PTD_N128_P16_SEARCH, "IVF PTD (N=128, P=16)"},
    {IVF_PTD_N256_P12_SEARCH, "IVF PTD (N=256, P=12)"},
    {IVF_PTD_N256_P16_SEARCH, "IVF PTD (N=256, P=16)"},
    {IVF_PTD_N256_P24_SEARCH, "IVF PTD (N=256, P=24)"},
    {IVF_PTD_N512_P16_SEARCH, "IVF PTD (N=512, P=16)"},
    {IVF_PTD_N512_P20_SEARCH, "IVF PTD (N=512, P=20)"},
    {IVF_PTD_N512_P32_SEARCH, "IVF PTD (N=512, P=32)"}
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

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

// 执行搜索并测量性能
std::vector<SearchResult> run_search_test(
    SearchMethod method,
    float* base,
    float* test_query,
    int* test_gt,
    size_t base_number,
    size_t test_number,
    size_t vecdim,
    size_t test_gt_d,
    size_t k)
{
    std::vector<SearchResult> results(test_number);
    
    // SQ量化预处理（如果需要）
    SQData* sq_data = nullptr;
    if (method == FLAT_SEARCH_SQ) {
        sq_data = new SQData(base_number, vecdim);
        *sq_data = quantize_base_sse(base, base_number, vecdim);
    }

    // IVF Flat 预处理（如果需要）
    std::vector<float> ivf_centroids_data;
    std::vector<std::vector<uint32_t>> ivf_invlists_data;
    size_t current_nlist = 0;
    size_t current_nprobe = 0;
    bool is_ivf = false;
    bool is_ivf_omp = false; // OpenMP版IVF标记
    bool is_ivf_ptd = false; // pthread版IVF标记

    switch (method) {
        case IVF_FLAT_N64_P8_SEARCH:
            current_nlist = 64; current_nprobe = 8; is_ivf = true;
            break;
        case IVF_FLAT_N128_P12_SEARCH:
            current_nlist = 128; current_nprobe = 12; is_ivf = true;
            break;
        case IVF_FLAT_N256_P16_SEARCH:
            current_nlist = 256; current_nprobe = 16; is_ivf = true;
            break;
        case IVF_FLAT_N512_P20_SEARCH:
            current_nlist = 512; current_nprobe = 20; is_ivf = true;
            break;
        // Add cases for new IVF configurations
        case IVF_FLAT_N64_P4_SEARCH:
            current_nlist = 64; current_nprobe = 4; is_ivf = true;
            break;
        case IVF_FLAT_N64_P12_SEARCH:
            current_nlist = 64; current_nprobe = 12; is_ivf = true;
            break;
        case IVF_FLAT_N128_P8_SEARCH:
            current_nlist = 128; current_nprobe = 8; is_ivf = true;
            break;
        case IVF_FLAT_N128_P16_SEARCH:
            current_nlist = 128; current_nprobe = 16; is_ivf = true;
            break;
        case IVF_FLAT_N256_P12_SEARCH:
            current_nlist = 256; current_nprobe = 12; is_ivf = true;
            break;
        case IVF_FLAT_N256_P24_SEARCH:
            current_nlist = 256; current_nprobe = 24; is_ivf = true;
            break;
        case IVF_FLAT_N512_P16_SEARCH:
            current_nlist = 512; current_nprobe = 16; is_ivf = true;
            break;
        case IVF_FLAT_N512_P32_SEARCH:
            current_nlist = 512; current_nprobe = 32; is_ivf = true;
            break;
        // Add cases for OMP IVF configurations
        case IVF_OMP_N64_P4_SEARCH:
            current_nlist = 64; current_nprobe = 4; is_ivf = true; is_ivf_omp = true;
            break;
        case IVF_OMP_N64_P8_SEARCH:
            current_nlist = 64; current_nprobe = 8; is_ivf = true; is_ivf_omp = true;
            break;
        case IVF_OMP_N64_P12_SEARCH:
            current_nlist = 64; current_nprobe = 12; is_ivf = true; is_ivf_omp = true;
            break;
        case IVF_OMP_N128_P8_SEARCH:
            current_nlist = 128; current_nprobe = 8; is_ivf = true; is_ivf_omp = true;
            break;
        case IVF_OMP_N128_P12_SEARCH:
            current_nlist = 128; current_nprobe = 12; is_ivf = true; is_ivf_omp = true;
            break;
        case IVF_OMP_N128_P16_SEARCH:
            current_nlist = 128; current_nprobe = 16; is_ivf = true; is_ivf_omp = true;
            break;
        case IVF_OMP_N256_P12_SEARCH:
            current_nlist = 256; current_nprobe = 12; is_ivf = true; is_ivf_omp = true;
            break;
        case IVF_OMP_N256_P16_SEARCH:
            current_nlist = 256; current_nprobe = 16; is_ivf = true; is_ivf_omp = true;
            break;
        case IVF_OMP_N256_P24_SEARCH:
            current_nlist = 256; current_nprobe = 24; is_ivf = true; is_ivf_omp = true;
            break;
        case IVF_OMP_N512_P16_SEARCH:
            current_nlist = 512; current_nprobe = 16; is_ivf = true; is_ivf_omp = true;
            break;
        case IVF_OMP_N512_P20_SEARCH:
            current_nlist = 512; current_nprobe = 20; is_ivf = true; is_ivf_omp = true;
            break;
        case IVF_OMP_N512_P32_SEARCH:
            current_nlist = 512; current_nprobe = 32; is_ivf = true; is_ivf_omp = true;
            break;
        // Add cases for pthread IVF configurations
        case IVF_PTD_N64_P4_SEARCH:
            current_nlist = 64; current_nprobe = 4; is_ivf = true; is_ivf_ptd = true;
            break;
        case IVF_PTD_N64_P8_SEARCH:
            current_nlist = 64; current_nprobe = 8; is_ivf = true; is_ivf_ptd = true;
            break;
        case IVF_PTD_N64_P12_SEARCH:
            current_nlist = 64; current_nprobe = 12; is_ivf = true; is_ivf_ptd = true;
            break;
        case IVF_PTD_N128_P8_SEARCH:
            current_nlist = 128; current_nprobe = 8; is_ivf = true; is_ivf_ptd = true;
            break;
        case IVF_PTD_N128_P12_SEARCH:
            current_nlist = 128; current_nprobe = 12; is_ivf = true; is_ivf_ptd = true;
            break;
        case IVF_PTD_N128_P16_SEARCH:
            current_nlist = 128; current_nprobe = 16; is_ivf = true; is_ivf_ptd = true;
            break;
        case IVF_PTD_N256_P12_SEARCH:
            current_nlist = 256; current_nprobe = 12; is_ivf = true; is_ivf_ptd = true;
            break;
        case IVF_PTD_N256_P16_SEARCH:
            current_nlist = 256; current_nprobe = 16; is_ivf = true; is_ivf_ptd = true;
            break;
        case IVF_PTD_N256_P24_SEARCH:
            current_nlist = 256; current_nprobe = 24; is_ivf = true; is_ivf_ptd = true;
            break;
        case IVF_PTD_N512_P16_SEARCH:
            current_nlist = 512; current_nprobe = 16; is_ivf = true; is_ivf_ptd = true;
            break;
        case IVF_PTD_N512_P20_SEARCH:
            current_nlist = 512; current_nprobe = 20; is_ivf = true; is_ivf_ptd = true;
            break;
        case IVF_PTD_N512_P32_SEARCH:
            current_nlist = 512; current_nprobe = 32; is_ivf = true; is_ivf_ptd = true;
            break;
        default:
            break;
    }

    if (is_ivf) {
        std::string centroids_path = "files/ivf_flat_centroids_" + std::to_string(current_nlist) + ".fbin";
        std::string invlists_path = "files/ivf_flat_invlists_" + std::to_string(current_nlist) + ".bin";
        std::cerr << "Loading IVF data for N=" << current_nlist << ": " << centroids_path << " and " << invlists_path << std::endl;
        try {
            ivf_centroids_data = load_ivf_centroids(centroids_path, current_nlist, vecdim);
            ivf_invlists_data = load_ivf_invlists(invlists_path, current_nlist);
            std::cerr << "IVF data loaded successfully. Centroids: " << ivf_centroids_data.size() / vecdim << "x" << vecdim 
                      << ", Invlists: " << ivf_invlists_data.size() << " lists." << std::endl;
        } catch (const std::runtime_error& e) {
            std::cerr << "Error loading IVF data for method " << method_names.at(method) << ": " << e.what() << std::endl;
            results.clear(); // Indicate error by returning empty results
            return results;
        }
    }
    
    for(int i = 0; i < test_number; ++i) {
        // 使用高精度计时器
        #ifdef _WIN32
        Performance_Counter timer;
        timer.start_counter();
        #else
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);
        #endif
        
        // 根据方法选择不同的搜索函数
        std::priority_queue<std::pair<float, uint32_t>> res;
        
        switch (method) {
            case FLAT_SEARCH:
                res = flat_search_sse(base, test_query + i*vecdim, base_number, vecdim, k);
                break;
            case FLAT_SEARCH_SQ:
                res = flat_search_sq_sse(sq_data->codes.data(), sq_data->min_vals.data(), 
                                        sq_data->scales.data(), test_query + i*vecdim, 
                                        base_number, vecdim, k);
                break;
            case PQ4_SEARCH:
                res = pq4_search_sse(base, test_query + i*vecdim, base_number, vecdim, k);
                break;
            case PQ8_SEARCH:
                res = pq8_search_sse(base, test_query + i*vecdim, base_number, vecdim, k);
                break;
            case PQ16_SEARCH:
                res = pq16_search_sse(base, test_query + i*vecdim, base_number, vecdim, k);
                break;
            case PQ32_SEARCH:
                res = pq32_search_sse(base, test_query + i*vecdim, base_number, vecdim, k);
                break;
            case PQ4_RERANK_SEARCH:
                res = pq4_rerank_sse(base, test_query + i*vecdim, base_number, vecdim, k);
                break;
            case PQ8_RERANK_SEARCH:
                res = pq8_rerank_sse(base, test_query + i*vecdim, base_number, vecdim, k);
                break;
            case PQ16_RERANK_SEARCH:
                res = pq16_rerank_sse(base, test_query + i*vecdim, base_number, vecdim, k);
                break;
            case PQ32_RERANK_SEARCH:
                res = pq32_rerank_sse(base, test_query + i*vecdim, base_number, vecdim, k);
                break;
            case IVF_FLAT_N64_P8_SEARCH:
            case IVF_FLAT_N128_P12_SEARCH:
            case IVF_FLAT_N256_P16_SEARCH:
            case IVF_FLAT_N512_P20_SEARCH:
            case IVF_FLAT_N64_P4_SEARCH:
            case IVF_FLAT_N64_P12_SEARCH:
            case IVF_FLAT_N128_P8_SEARCH:
            case IVF_FLAT_N128_P16_SEARCH:
            case IVF_FLAT_N256_P12_SEARCH:
            case IVF_FLAT_N256_P24_SEARCH:
            case IVF_FLAT_N512_P16_SEARCH:
            case IVF_FLAT_N512_P32_SEARCH:
                if (!ivf_centroids_data.empty() && !ivf_invlists_data.empty()) {
                    res = flat_ivf_search(base, test_query + i*vecdim, base_number, vecdim, k,
                                          ivf_centroids_data.data(), current_nlist, ivf_invlists_data, current_nprobe);
                } else {
                    std::cerr << "IVF data not properly loaded for query " << i << " (method: " << method_names.at(method) << "), skipping search." << std::endl;
                    // Populate with error/default to avoid issues later
                    for(size_t j=0; j<k; ++j) res.push({-1.0f, 0}); 
                }
                break;
            // Add OpenMP IVF cases
            case IVF_OMP_N64_P4_SEARCH:
            case IVF_OMP_N64_P8_SEARCH:
            case IVF_OMP_N64_P12_SEARCH:
            case IVF_OMP_N128_P8_SEARCH:
            case IVF_OMP_N128_P12_SEARCH:
            case IVF_OMP_N128_P16_SEARCH:
            case IVF_OMP_N256_P12_SEARCH:
            case IVF_OMP_N256_P16_SEARCH:
            case IVF_OMP_N256_P24_SEARCH:
            case IVF_OMP_N512_P16_SEARCH:
            case IVF_OMP_N512_P20_SEARCH:
            case IVF_OMP_N512_P32_SEARCH:
                if (!ivf_centroids_data.empty() && !ivf_invlists_data.empty()) {
                    int num_threads = omp_get_max_threads(); // 默认使用所有可用线程
                    res = ivf_search_omp(base, test_query + i*vecdim, base_number, vecdim, k,
                                       ivf_centroids_data.data(), current_nlist, ivf_invlists_data, current_nprobe, num_threads);
                } else {
                    std::cerr << "IVF data not properly loaded for OMP query " << i << " (method: " << method_names.at(method) << "), skipping search." << std::endl;
                    // Populate with error/default to avoid issues later
                    for(size_t j=0; j<k; ++j) res.push({-1.0f, 0}); 
                }
                break;
            // Add pthread IVF cases
            case IVF_PTD_N64_P4_SEARCH:
            case IVF_PTD_N64_P8_SEARCH:
            case IVF_PTD_N64_P12_SEARCH:
            case IVF_PTD_N128_P8_SEARCH:
            case IVF_PTD_N128_P12_SEARCH:
            case IVF_PTD_N128_P16_SEARCH:
            case IVF_PTD_N256_P12_SEARCH:
            case IVF_PTD_N256_P16_SEARCH:
            case IVF_PTD_N256_P24_SEARCH:
            case IVF_PTD_N512_P16_SEARCH:
            case IVF_PTD_N512_P20_SEARCH:
            case IVF_PTD_N512_P32_SEARCH:
                if (!ivf_centroids_data.empty() && !ivf_invlists_data.empty()) {
                    int num_threads = std::thread::hardware_concurrency(); // 获取硬件支持的线程数
                    res = ivf_search_ptd(base, test_query + i*vecdim, base_number, vecdim, k,
                                       ivf_centroids_data.data(), current_nlist, ivf_invlists_data, current_nprobe, num_threads);
                } else {
                    std::cerr << "IVF data not properly loaded for PTD query " << i << " (method: " << method_names.at(method) << "), skipping search." << std::endl;
                    // Populate with error/default to avoid issues later
                    for(size_t j=0; j<k; ++j) res.push({-1.0f, 0}); 
                }
                break;
        }

        // 计算耗时
        #ifdef _WIN32
        timer.end_counter();
        int64_t diff = timer.microseconds();
        #else
        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
        #endif

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }
    
    // 清理资源
    if (sq_data) {
        delete sq_data;
    }
    
    return results;
}

// 测试所有方法并返回结果
void test_all_methods(
    float* base,
    float* test_query,
    int* test_gt,
    size_t base_number,
    size_t test_number,
    size_t vecdim,
    size_t test_gt_d,
    size_t k)
{
    const std::vector<SearchMethod> methods_to_test = {
        FLAT_SEARCH, FLAT_SEARCH_SQ,
        PQ4_SEARCH, PQ8_SEARCH, PQ16_SEARCH, PQ32_SEARCH,
        PQ4_RERANK_SEARCH, PQ8_RERANK_SEARCH, PQ16_RERANK_SEARCH, PQ32_RERANK_SEARCH,
        // IVF Flat methods (serial)
        IVF_FLAT_N64_P4_SEARCH, IVF_FLAT_N64_P8_SEARCH, IVF_FLAT_N64_P12_SEARCH,
        IVF_FLAT_N128_P8_SEARCH, IVF_FLAT_N128_P12_SEARCH, IVF_FLAT_N128_P16_SEARCH,
        IVF_FLAT_N256_P12_SEARCH, IVF_FLAT_N256_P16_SEARCH, IVF_FLAT_N256_P24_SEARCH,
        IVF_FLAT_N512_P16_SEARCH, IVF_FLAT_N512_P20_SEARCH, IVF_FLAT_N512_P32_SEARCH,
        // IVF OMP methods (OpenMP parallel)
        IVF_OMP_N64_P4_SEARCH, IVF_OMP_N64_P8_SEARCH, IVF_OMP_N64_P12_SEARCH,
        IVF_OMP_N128_P8_SEARCH, IVF_OMP_N128_P12_SEARCH, IVF_OMP_N128_P16_SEARCH,
        IVF_OMP_N256_P12_SEARCH, IVF_OMP_N256_P16_SEARCH, IVF_OMP_N256_P24_SEARCH,
        IVF_OMP_N512_P16_SEARCH, IVF_OMP_N512_P20_SEARCH, IVF_OMP_N512_P32_SEARCH,
        // IVF PTD methods (pthread parallel)
        IVF_PTD_N64_P4_SEARCH, IVF_PTD_N64_P8_SEARCH, IVF_PTD_N64_P12_SEARCH,
        IVF_PTD_N128_P8_SEARCH, IVF_PTD_N128_P12_SEARCH, IVF_PTD_N128_P16_SEARCH,
        IVF_PTD_N256_P12_SEARCH, IVF_PTD_N256_P16_SEARCH, IVF_PTD_N256_P24_SEARCH,
        IVF_PTD_N512_P16_SEARCH, IVF_PTD_N512_P20_SEARCH, IVF_PTD_N512_P32_SEARCH
    };
    
    std::cout << "Method,Average Recall,Average Latency (us)" << std::endl;
    
    for (const auto& method : methods_to_test) {
        std::vector<SearchResult> results = run_search_test(
            method, base, test_query, test_gt, 
            base_number, test_number, vecdim, test_gt_d, k);
        
        float avg_recall = 0, avg_latency = 0;
        for(int i = 0; i < test_number; ++i) {
            avg_recall += results[i].recall;
            avg_latency += results[i].latency;
        }
        
        avg_recall /= test_number;
        avg_latency /= test_number;
        
        std::cout << method_names.at(method) << "," 
                  << avg_recall << ","
                  << avg_latency << std::endl;
    }
}

int main(int argc, char *argv[])
{
    bool test_all = true;
    SearchMethod single_method = FLAT_SEARCH;
    
    if (argc > 1) {
        std::string method_arg = argv[1];
        test_all = false;
        
        if (method_arg == "flat") single_method = FLAT_SEARCH;
        else if (method_arg == "sq") single_method = FLAT_SEARCH_SQ;
        else if (method_arg == "pq4") single_method = PQ4_SEARCH;
        else if (method_arg == "pq8") single_method = PQ8_SEARCH;
        else if (method_arg == "pq16") single_method = PQ16_SEARCH;
        else if (method_arg == "pq32") single_method = PQ32_SEARCH;
        else if (method_arg == "pq4rerank") single_method = PQ4_RERANK_SEARCH;
        else if (method_arg == "pq8rerank") single_method = PQ8_RERANK_SEARCH;
        else if (method_arg == "pq16rerank") single_method = PQ16_RERANK_SEARCH;
        else if (method_arg == "pq32rerank") single_method = PQ32_RERANK_SEARCH;
        else if (method_arg == "ivf_n64_p8") single_method = IVF_FLAT_N64_P8_SEARCH;
        else if (method_arg == "ivf_n128_p12") single_method = IVF_FLAT_N128_P12_SEARCH;
        else if (method_arg == "ivf_n256_p16") single_method = IVF_FLAT_N256_P16_SEARCH;
        else if (method_arg == "ivf_n512_p20") single_method = IVF_FLAT_N512_P20_SEARCH;
        else if (method_arg == "ivf_n64_p4") single_method = IVF_FLAT_N64_P4_SEARCH;
        else if (method_arg == "ivf_n64_p12") single_method = IVF_FLAT_N64_P12_SEARCH;
        else if (method_arg == "ivf_n128_p8") single_method = IVF_FLAT_N128_P8_SEARCH;
        else if (method_arg == "ivf_n128_p16") single_method = IVF_FLAT_N128_P16_SEARCH;
        else if (method_arg == "ivf_n256_p12") single_method = IVF_FLAT_N256_P12_SEARCH;
        else if (method_arg == "ivf_n256_p24") single_method = IVF_FLAT_N256_P24_SEARCH;
        else if (method_arg == "ivf_n512_p16") single_method = IVF_FLAT_N512_P16_SEARCH;
        else if (method_arg == "ivf_n512_p32") single_method = IVF_FLAT_N512_P32_SEARCH;
        else if (method_arg == "ivf_omp_n64_p4") single_method = IVF_OMP_N64_P4_SEARCH;
        else if (method_arg == "ivf_omp_n64_p8") single_method = IVF_OMP_N64_P8_SEARCH;
        else if (method_arg == "ivf_omp_n64_p12") single_method = IVF_OMP_N64_P12_SEARCH;
        else if (method_arg == "ivf_omp_n128_p8") single_method = IVF_OMP_N128_P8_SEARCH;
        else if (method_arg == "ivf_omp_n128_p12") single_method = IVF_OMP_N128_P12_SEARCH;
        else if (method_arg == "ivf_omp_n128_p16") single_method = IVF_OMP_N128_P16_SEARCH;
        else if (method_arg == "ivf_omp_n256_p12") single_method = IVF_OMP_N256_P12_SEARCH;
        else if (method_arg == "ivf_omp_n256_p16") single_method = IVF_OMP_N256_P16_SEARCH;
        else if (method_arg == "ivf_omp_n256_p24") single_method = IVF_OMP_N256_P24_SEARCH;
        else if (method_arg == "ivf_omp_n512_p16") single_method = IVF_OMP_N512_P16_SEARCH;
        else if (method_arg == "ivf_omp_n512_p20") single_method = IVF_OMP_N512_P20_SEARCH;
        else if (method_arg == "ivf_omp_n512_p32") single_method = IVF_OMP_N512_P32_SEARCH;
        else if (method_arg == "ivf_ptd_n64_p4") single_method = IVF_PTD_N64_P4_SEARCH;
        else if (method_arg == "ivf_ptd_n64_p8") single_method = IVF_PTD_N64_P8_SEARCH;
        else if (method_arg == "ivf_ptd_n64_p12") single_method = IVF_PTD_N64_P12_SEARCH;
        else if (method_arg == "ivf_ptd_n128_p8") single_method = IVF_PTD_N128_P8_SEARCH;
        else if (method_arg == "ivf_ptd_n128_p12") single_method = IVF_PTD_N128_P12_SEARCH;
        else if (method_arg == "ivf_ptd_n128_p16") single_method = IVF_PTD_N128_P16_SEARCH;
        else if (method_arg == "ivf_ptd_n256_p12") single_method = IVF_PTD_N256_P12_SEARCH;
        else if (method_arg == "ivf_ptd_n256_p16") single_method = IVF_PTD_N256_P16_SEARCH;
        else if (method_arg == "ivf_ptd_n256_p24") single_method = IVF_PTD_N256_P24_SEARCH;
        else if (method_arg == "ivf_ptd_n512_p16") single_method = IVF_PTD_N512_P16_SEARCH;
        else if (method_arg == "ivf_ptd_n512_p20") single_method = IVF_PTD_N512_P20_SEARCH;
        else if (method_arg == "ivf_ptd_n512_p32") single_method = IVF_PTD_N512_P32_SEARCH;
        else if (method_arg == "all") test_all = true;
        else {
            std::cerr << "Unknown method: " << method_arg << std::endl;
            return 1;
        }
    }
    
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    if (test_all) {
        test_all_methods(base, test_query, test_gt, base_number, test_number, vecdim, test_gt_d, k);
    } else {
        std::vector<SearchResult> results = run_search_test(
            single_method, base, test_query, test_gt, 
            base_number, test_number, vecdim, test_gt_d, k);
            
        float avg_recall = 0, avg_latency = 0;
        for(int i = 0; i < test_number; ++i) {
            avg_recall += results[i].recall;
            avg_latency += results[i].latency;
        }
        
        std::cout << "Method: " << method_names.at(single_method) << std::endl;
        std::cout << "Average recall: " << avg_recall / test_number << std::endl;
        std::cout << "Average latency (us): " << avg_latency / test_number << std::endl;
    }

    // 释放内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}