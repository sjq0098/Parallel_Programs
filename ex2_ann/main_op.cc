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
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "simd_utils.h"
#include "sq_index.h"
#include "pq_index.h"

using namespace hnswlib;

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
};

// 标量版本的暴力搜索 - 基线方法
std::priority_queue<std::pair<float, uint32_t>> scalar_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    std::priority_queue<std::pair<float, uint32_t>> q;

    for(int i = 0; i < base_number; ++i) {
        float ip = 0;

        // 标量内积计算
        for(int d = 0; d < vecdim; ++d) {
            ip += base[d + i*vecdim] * query[d];
        }
        float dis = 1.0f - ip;

        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}

// SSE优化的暴力搜索
std::priority_queue<std::pair<float, uint32_t>> sse_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    std::priority_queue<std::pair<float, uint32_t>> q;

    for(int i = 0; i < base_number; ++i) {
        // 使用SSE计算内积
        float ip = dot_product_sse(base + i*vecdim, query, vecdim);
        float dis = 1.0f - ip;

        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}

// AVX优化的暴力搜索
std::priority_queue<std::pair<float, uint32_t>> avx_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    std::priority_queue<std::pair<float, uint32_t>> q;

    for(int i = 0; i < base_number; ++i) {
        // 使用AVX计算内积
        float ip = dot_product_avx(base + i*vecdim, query, vecdim);
        float dis = 1.0f - ip;

        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}

// 构建和保存SQ索引
void build_sq_index(float* base, size_t base_number, size_t vecdim) {
    std::cout << "构建标量量化(SQ)索引...\n";
    
    ScalarQuantizer sq;
    sq.build(base, base_number, vecdim);
    
    // 保存索引
    sq.save("files/sq_index.bin");
    
    std::cout << "SQ索引已保存到 files/sq_index.bin\n";
}

// 使用SQ索引搜索
std::priority_queue<std::pair<float, uint32_t>> sq_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    static ScalarQuantizer sq;
    static bool loaded = false;
    
    if (!loaded) {
        std::cout << "加载SQ索引...\n";
        if (!sq.load("files/sq_index.bin")) {
            std::cerr << "无法加载SQ索引，尝试构建新索引...\n";
            build_sq_index(base, base_number, vecdim);
            if (!sq.load("files/sq_index.bin")) {
                std::cerr << "构建索引失败！返回空结果\n";
                return std::priority_queue<std::pair<float, uint32_t>>();
            }
        }
        loaded = true;
    }
    
    return sq.search(query, k);
}

// 加载PQ或OPQ索引
ProductQuantizer load_pq_index(const std::string& prefix, size_t m) {
    ProductQuantizer pq;
    std::string codebook_file = "files/" + prefix + std::to_string(m) + "_codebook.bin";
    std::string codes_file = "files/" + prefix + std::to_string(m) + "_codes.bin";
    
    if (!pq.load_codebook(codebook_file)) {
        std::cerr << "无法加载码本: " << codebook_file << "\n";
        exit(1);
    }
    
    if (!pq.load_codes(codes_file)) {
        std::cerr << "无法加载编码: " << codes_file << "\n";
        exit(1);
    }
    
    // 如果是OPQ，还需要加载旋转矩阵
    if (prefix == "opq") {
        std::string rotation_file = "files/" + prefix + std::to_string(m) + "_rotation.bin";
        if (!pq.load_rotation(rotation_file)) {
            std::cerr << "警告: 无法加载旋转矩阵: " << rotation_file << "\n";
        }
    }
    
    return pq;
}

// 使用PQ索引搜索
std::priority_queue<std::pair<float, uint32_t>> pq_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t m) {
    
    static std::vector<ProductQuantizer> pq_indices(4);
    static bool loaded[4] = {false, false, false, false};
    
    int m_idx;
    switch(m) {
        case 4: m_idx = 0; break;
        case 8: m_idx = 1; break;
        case 16: m_idx = 2; break;
        case 32: m_idx = 3; break;
        default:
            std::cerr << "不支持的PQ子空间数量: " << m << "，使用默认值16\n";
            m = 16;
            m_idx = 2;
    }
    
    if (!loaded[m_idx]) {
        std::cout << "加载PQ" << m << "索引...\n";
        pq_indices[m_idx] = load_pq_index("pq", m);
        loaded[m_idx] = true;
    }
    
    return pq_indices[m_idx].search(query, k);
}

// 使用OPQ索引搜索
std::priority_queue<std::pair<float, uint32_t>> opq_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t m) {
    
    static std::vector<ProductQuantizer> opq_indices(4);
    static bool loaded[4] = {false, false, false, false};
    
    int m_idx;
    switch(m) {
        case 4: m_idx = 0; break;
        case 8: m_idx = 1; break;
        case 16: m_idx = 2; break;
        case 32: m_idx = 3; break;
        default:
            std::cerr << "不支持的OPQ子空间数量: " << m << "，使用默认值16\n";
            m = 16;
            m_idx = 2;
    }
    
    if (!loaded[m_idx]) {
        std::cout << "加载OPQ" << m << "索引...\n";
        opq_indices[m_idx] = load_pq_index("opq", m);
        loaded[m_idx] = true;
    }
    
    return opq_indices[m_idx].search(query, k);
}

// 使用PQ索引搜索，并进行精确重排序
std::priority_queue<std::pair<float, uint32_t>> pq_search_with_rerank(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t m, size_t rerank_k) {
    
    static std::vector<ProductQuantizer> pq_indices(4);
    static bool loaded[4] = {false, false, false, false};
    
    int m_idx;
    switch(m) {
        case 4: m_idx = 0; break;
        case 8: m_idx = 1; break;
        case 16: m_idx = 2; break;
        case 32: m_idx = 3; break;
        default:
            std::cerr << "不支持的PQ子空间数量: " << m << "，使用默认值16\n";
            m = 16;
            m_idx = 2;
    }
    
    if (!loaded[m_idx]) {
        std::cout << "加载PQ" << m << "索引...\n";
        pq_indices[m_idx] = load_pq_index("pq", m);
        loaded[m_idx] = true;
    }
    
    return pq_indices[m_idx].search_with_rerank(query, base, k, rerank_k);
}

// 使用OPQ索引搜索，并进行精确重排序
std::priority_queue<std::pair<float, uint32_t>> opq_search_with_rerank(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t m, size_t rerank_k) {
    
    static std::vector<ProductQuantizer> opq_indices(4);
    static bool loaded[4] = {false, false, false, false};
    
    int m_idx;
    switch(m) {
        case 4: m_idx = 0; break;
        case 8: m_idx = 1; break;
        case 16: m_idx = 2; break;
        case 32: m_idx = 3; break;
        default:
            std::cerr << "不支持的OPQ子空间数量: " << m << "，使用默认值16\n";
            m = 16;
            m_idx = 2;
    }
    
    if (!loaded[m_idx]) {
        std::cout << "加载OPQ" << m << "索引...\n";
        opq_indices[m_idx] = load_pq_index("opq", m);
        loaded[m_idx] = true;
    }
    
    return opq_indices[m_idx].search_with_rerank(query, base, k, rerank_k);
}

// 搜索方法枚举
enum SearchMethod {
    SCALAR = 0,     // 标量暴力搜索
    SSE = 1,        // SSE暴力搜索
    AVX = 2,        // AVX暴力搜索
    SQ = 3,         // 标量量化
    PQ4 = 4,        // PQ m=4
    PQ8 = 5,        // PQ m=8
    PQ16 = 6,       // PQ m=16
    PQ32 = 7,       // PQ m=32
    OPQ4 = 8,       // OPQ m=4
    OPQ8 = 9,       // OPQ m=8
    OPQ16 = 10,     // OPQ m=16
    OPQ32 = 11,     // OPQ m=32
    PQ16_RERANK = 12, // PQ16 + 重排序
    PQ32_RERANK = 13, // PQ32 + 重排序
    OPQ16_RERANK = 14, // OPQ16 + 重排序
    OPQ32_RERANK = 15  // OPQ32 + 重排序
};

// 根据指定方法搜索
std::priority_queue<std::pair<float, uint32_t>> search(
    SearchMethod method, float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    const size_t rerank_k = 100; // 重排序候选数量
    
    switch(method) {
        case SCALAR:
            return scalar_search(base, query, base_number, vecdim, k);
        case SSE:
            return sse_search(base, query, base_number, vecdim, k);
        case AVX:
            return avx_search(base, query, base_number, vecdim, k);
        case SQ:
            return sq_search(base, query, base_number, vecdim, k);
        case PQ4:
            return pq_search(base, query, base_number, vecdim, k, 4);
        case PQ8:
            return pq_search(base, query, base_number, vecdim, k, 8);
        case PQ16:
            return pq_search(base, query, base_number, vecdim, k, 16);
        case PQ32:
            return pq_search(base, query, base_number, vecdim, k, 32);
        case OPQ4:
            return opq_search(base, query, base_number, vecdim, k, 4);
        case OPQ8:
            return opq_search(base, query, base_number, vecdim, k, 8);
        case OPQ16:
            return opq_search(base, query, base_number, vecdim, k, 16);
        case OPQ32:
            return opq_search(base, query, base_number, vecdim, k, 32);
        case PQ16_RERANK:
            return pq_search_with_rerank(base, query, base_number, vecdim, k, 16, rerank_k);
        case PQ32_RERANK:
            return pq_search_with_rerank(base, query, base_number, vecdim, k, 32, rerank_k);
        case OPQ16_RERANK:
            return opq_search_with_rerank(base, query, base_number, vecdim, k, 16, rerank_k);
        case OPQ32_RERANK:
            return opq_search_with_rerank(base, query, base_number, vecdim, k, 32, rerank_k);
        default:
            std::cerr << "未知搜索方法: " << method << "\n";
            return scalar_search(base, query, base_number, vecdim, k);
    }
}

// 获取搜索方法名称
std::string get_method_name(SearchMethod method) {
    switch(method) {
        case SCALAR: return "标量暴力搜索";
        case SSE: return "SSE优化暴力搜索";
        case AVX: return "AVX优化暴力搜索";
        case SQ: return "标量量化(SQ)";
        case PQ4: return "乘积量化(PQ) M=4";
        case PQ8: return "乘积量化(PQ) M=8";
        case PQ16: return "乘积量化(PQ) M=16";
        case PQ32: return "乘积量化(PQ) M=32";
        case OPQ4: return "优化乘积量化(OPQ) M=4";
        case OPQ8: return "优化乘积量化(OPQ) M=8";
        case OPQ16: return "优化乘积量化(OPQ) M=16";
        case OPQ32: return "优化乘积量化(OPQ) M=32";
        case PQ16_RERANK: return "混合搜索(PQ16+精确重排序)";
        case PQ32_RERANK: return "混合搜索(PQ32+精确重排序)";
        case OPQ16_RERANK: return "混合搜索(OPQ16+精确重排序)";
        case OPQ32_RERANK: return "混合搜索(OPQ32+精确重排序)";
        default: return "未知方法";
    }
}

int main(int argc, char *argv[])
{
    SearchMethod method = AVX;  // 默认使用AVX搜索
    
    // 如果提供了命令行参数，解析搜索方法
    if (argc > 1) {
        method = static_cast<SearchMethod>(atoi(argv[1]));
    }
    
    // 测试单个方法
    bool test_single = true;
    if (argc > 2) {
        test_single = atoi(argv[2]) != 0;
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
    
    if (test_single) {
        // 测试单个方法
        std::vector<SearchResult> results;
        results.resize(test_number);
        
        std::cout << "测试方法: " << get_method_name(method) << "\n";
        
        // 查询测试代码
        for(int i = 0; i < test_number; ++i) {
            const unsigned long Converter = 1000 * 1000;
            struct timeval val;
            gettimeofday(&val, NULL);
            
            auto res = search(method, base, test_query + i*vecdim, base_number, vecdim, k);
            
            struct timeval newVal;
            gettimeofday(&newVal, NULL);
            int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
            
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
        
        float avg_recall = 0, avg_latency = 0;
        for(int i = 0; i < test_number; ++i) {
            avg_recall += results[i].recall;
            avg_latency += results[i].latency;
        }
        
        // 创建输出文件
        std::string output_file = "results/main_op_" + std::to_string(static_cast<int>(method)) + ".txt";
        std::ofstream output(output_file);
        
        output << "方法: " << get_method_name(method) << "\n";
        output << "平均召回率: " << avg_recall / test_number << "\n";
        output << "平均延迟(微秒): " << avg_latency / test_number << "\n";
        output.close();
        
        // 输出到控制台
        std::cout << "平均召回率: " << avg_recall / test_number << "\n";
        std::cout << "平均延迟(微秒): " << avg_latency / test_number << "\n";
        std::cout << "结果已保存到 " << output_file << "\n";
    } else {
        // 测试所有方法
        for (int m = 0; m <= 15; m++) {
            SearchMethod curr_method = static_cast<SearchMethod>(m);
            
            // 创建单独进程测试每个方法
            std::string cmd = "./main_op " + std::to_string(m) + " 1";
            std::cout << "执行: " << cmd << "\n";
            system(cmd.c_str());
        }
    }
    
    return 0;
} 