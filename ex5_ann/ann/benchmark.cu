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
#include <random>
#include <algorithm>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan_gpu.h"     // GPU+GPU TOP-K实现
#include "flat_scan_gpu_cpu.h" // GPU+CPU TOP-K实现
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace hnswlib;

// 加载数据函数
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

// 结果结构体
struct SearchResult
{
    float recall;
    int64_t latency_us; // 延迟（微秒）
    int64_t total_time_ms; // 总时间（毫秒）
};

// 记录实验结果
struct ExperimentResult {
    std::string method;
    size_t batch_size;
    size_t query_count;
    float avg_recall;
    int64_t avg_latency_us;
    int64_t total_time_ms;
};

// 生成随机查询索引，确保每次运行使用相同的查询顺序
std::vector<int> generateRandomIndices(int max_val, int count) {
    std::vector<int> indices(max_val);
    for (int i = 0; i < max_val; ++i) {
        indices[i] = i;
    }
    
    std::mt19937 g(42); // 固定种子为42，确保可重复性
    std::shuffle(indices.begin(), indices.end(), g);
    
    std::vector<int> result;
    for (int i = 0; i < count && i < max_val; ++i) {
        result.push_back(indices[i]);
    }
    return result;
}

// 运行单次实验
ExperimentResult runExperiment(
    const std::string& method,
    float* base, float* test_query, int* test_gt,
    size_t base_number, size_t vecdim, size_t test_gt_d,
    size_t query_count, size_t batch_size, size_t k,
    int warm_up_count, int repeat_count) {
    
    std::vector<SearchResult> results(query_count);
    int64_t total_time_ms = 0;
    
    // 确定实验要用的查询向量索引
    std::vector<int> queryIndices = generateRandomIndices(query_count, query_count);
    
    // 暖机运行
    std::cout << "暖机 " << warm_up_count << " 条查询..." << std::endl;
    if (method == "GPU+GPU") {
        auto warm_up_res = flat_search_gpu(base, test_query, base_number, vecdim, warm_up_count, k);
    } else {
        auto warm_up_res = flat_search_gpu_cpu(base, test_query, base_number, vecdim, warm_up_count, k);
    }
    
    // 多次重复实验
    for (int rep = 0; rep < repeat_count; ++rep) {
        std::cout << "运行实验 " << method << " (重复 " << rep+1 << "/" << repeat_count 
                  << ", 批大小=" << batch_size << ", 查询数=" << query_count << ")" << std::endl;
        
        // 对每个批次进行处理
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < query_count; i += batch_size) {
            size_t current_batch_size = std::min(batch_size, query_count - i);
            
            // 准备当前批次的查询向量
            float* current_queries = new float[current_batch_size * vecdim];
            for (size_t b = 0; b < current_batch_size; ++b) {
                int query_idx = queryIndices[i + b];
                memcpy(current_queries + b * vecdim, test_query + query_idx * vecdim, vecdim * sizeof(float));
            }
            
            // 计时开始
            struct timeval val;
            gettimeofday(&val, NULL);
            
            // 根据方法选择不同的实现
            std::vector<std::priority_queue<std::pair<float, uint32_t>>> res;
            if (method == "GPU+GPU") {
                res = flat_search_gpu(base, current_queries, base_number, vecdim, current_batch_size, k);
            } else {
                res = flat_search_gpu_cpu(base, current_queries, base_number, vecdim, current_batch_size, k);
            }
            
            // 计时结束
            struct timeval newVal;
            gettimeofday(&newVal, NULL);
            int64_t diff = (newVal.tv_sec * 1000000 + newVal.tv_usec) - (val.tv_sec * 1000000 + val.tv_usec);
            
            // 计算每个查询的召回率
            for (size_t b = 0; b < current_batch_size; ++b) {
                int query_idx = queryIndices[i + b];
                std::set<uint32_t> gtset;
                for (int j = 0; j < k; ++j) {
                    gtset.insert(test_gt[j + query_idx * test_gt_d]);
                }
                
                size_t acc = 0;
                auto pq = res[b];
                while (!pq.empty()) {
                    int x = pq.top().second;
                    if (gtset.find(x) != gtset.end()) {
                        ++acc;
                    }
                    pq.pop();
                }
                
                float recall = (float)acc / k;
                int64_t latency = diff / current_batch_size;
                
                // 累积结果
                results[i + b].recall += recall / repeat_count;
                results[i + b].latency_us += latency / repeat_count;
            }
            
            delete[] current_queries;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        total_time_ms += duration.count();
    }
    
    // 计算平均结果
    float avg_recall = 0;
    int64_t avg_latency = 0;
    for (size_t i = 0; i < query_count; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency_us;
    }
    avg_recall /= query_count;
    avg_latency /= query_count;
    total_time_ms /= repeat_count;
    
    return {method, batch_size, query_count, avg_recall, avg_latency, total_time_ms};
}

// 主函数
int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "用法: " << argv[0] << " <数据路径> <最大查询数> <重复次数> <暖机查询数>" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];
    size_t max_query_count = std::stoi(argv[2]);
    int repeat_count = std::stoi(argv[3]);
    int warm_up_count = std::stoi(argv[4]);

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    // 限制最大查询数
    size_t query_count = std::min(max_query_count, test_number);
    const size_t k = 10;
    
    // 设置不同的批量大小
    std::vector<size_t> batch_sizes = {32, 64, 128, 256, 512, 1024,2048,4096};
    std::vector<size_t> query_counts = {500, 1000, 2000,4000,8000,10000};
    
    if (query_count < 2000) {
        query_counts = {query_count / 4, query_count / 2, query_count};
    }
    
    // 存储所有实验结果
    std::vector<ExperimentResult> all_results;
    
    // 输出结果的CSV文件头
    std::ofstream out_file("benchmark_results.csv");
    out_file << "方法,批大小,查询数量,平均召回率,平均延迟(μs),总耗时(ms)" << std::endl;
    
    // 运行两种方法的对比实验
    for (const auto& method : {"GPU+CPU", "GPU+GPU"}) {
        // 对比不同的批量大小
        for (size_t batch_size : batch_sizes) {
            // 对比不同的查询数量
            for (size_t qcount : query_counts) {
                if (qcount > query_count) continue;
                
                auto result = runExperiment(
                    method, base, test_query, test_gt, 
                    base_number, vecdim, test_gt_d,
                    qcount, batch_size, k, 
                    warm_up_count, repeat_count
                );
                
                all_results.push_back(result);
                
                // 输出结果到控制台和文件
                std::cout << "结果: " << method 
                          << ", 批大小=" << batch_size 
                          << ", 查询数=" << qcount
                          << ", 召回率=" << result.avg_recall
                          << ", 延迟=" << result.avg_latency_us << "μs"
                          << ", 总时间=" << result.total_time_ms << "ms" << std::endl;
                
                out_file << method << ","
                         << batch_size << ","
                         << qcount << ","
                         << result.avg_recall << ","
                         << result.avg_latency_us << ","
                         << result.total_time_ms << std::endl;
            }
        }
    }
    
    out_file.close();
    
    std::cout << "所有实验完成，结果已保存到 benchmark_results.csv" << std::endl;
    return 0;
} 