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
#include <map>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"           // 基准CPU实现
#include "flat_scan_gpu.h"       // GPU+GPU TOP-K实现
#include "flat_scan_gpu_cpu.h"   // GPU+CPU TOP-K实现  
#include "flat_ivf.h"            // CPU IVF实现
#include "gpu_ivf.h"             // GPU IVF实现
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
struct SpeedupResult {
    std::string method;
    size_t batch_size;
    double baseline_time_ms;
    double method_time_ms;
    double speedup;
    float recall;
};

// 计算召回率
float calculateRecall(const std::vector<std::priority_queue<std::pair<float, uint32_t>>>& results,
                     int* ground_truth, size_t query_count, size_t k, size_t gt_dim) {
    float total_recall = 0.0f;
    
    for (size_t q = 0; q < query_count; ++q) {
        std::set<uint32_t> gtset;
        for (size_t j = 0; j < k; ++j) {
            gtset.insert(ground_truth[j + q * gt_dim]);
        }
        
        size_t acc = 0;
        auto pq = results[q];
        while (!pq.empty()) {
            uint32_t x = pq.top().second;
            if (gtset.find(x) != gtset.end()) {
                ++acc;
            }
            pq.pop();
        }
        total_recall += (float)acc / k;
    }
    
    return total_recall / query_count;
}

// 单查询召回率计算
float calculateSingleRecall(const std::priority_queue<std::pair<float, uint32_t>>& result,
                           int* ground_truth, size_t query_idx, size_t k, size_t gt_dim) {
    std::set<uint32_t> gtset;
    for (size_t j = 0; j < k; ++j) {
        gtset.insert(ground_truth[j + query_idx * gt_dim]);
    }
    
    size_t acc = 0;
    auto pq = result;
    while (!pq.empty()) {
        uint32_t x = pq.top().second;
        if (gtset.find(x) != gtset.end()) {
            ++acc;
        }
        pq.pop();
    }
    
    return (float)acc / k;
}

// 运行基准CPU实验
double runBaselineExperiment(float* base, float* test_query, size_t base_number, 
                           size_t vecdim, size_t query_count, size_t k, int repeat_count) {
    double total_time = 0.0;
    
    for (int rep = 0; rep < repeat_count; ++rep) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t q = 0; q < query_count; ++q) {
            auto result = flat_search(base, test_query + q * vecdim, base_number, vecdim, k);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        total_time += duration.count();
    }
    
    return total_time / repeat_count / query_count;  // 返回平均每个查询的时间(us)
}

// 运行CPU IVF实验
double runCPUIVFExperiment(float* base, float* test_query, size_t base_number, 
                          size_t vecdim, size_t query_count, size_t k,
                          float* centroids, size_t nlist,
                          const std::vector<std::vector<uint32_t>>& invlists,
                          size_t nprobe, int repeat_count) {
    double total_time = 0.0;
    
    for (int rep = 0; rep < repeat_count; ++rep) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t q = 0; q < query_count; ++q) {
            auto result = flat_ivf_search(base, test_query + q * vecdim, base_number, 
                                        vecdim, k, centroids, nlist, invlists, nprobe);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        total_time += duration.count();
    }
    
    return total_time / repeat_count / query_count;  // 返回平均每个查询的时间(us)
}

// 运行GPU实验（批处理）
double runGPUExperiment(const std::string& method, float* base, float* test_query, 
                       size_t base_number, size_t vecdim, size_t query_count, 
                       size_t batch_size, size_t k, int repeat_count,
                       float* centroids = nullptr, size_t nlist = 0,
                       const std::vector<std::vector<uint32_t>>* invlists = nullptr,
                       size_t nprobe = 0) {
    double total_time = 0.0;
    
    for (int rep = 0; rep < repeat_count; ++rep) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < query_count; i += batch_size) {
            size_t current_batch_size = std::min(batch_size, query_count - i);
            
            if (method == "GPU+GPU") {
                auto result = flat_search_gpu(base, test_query + i * vecdim, 
                                            base_number, vecdim, current_batch_size, k);
            } else if (method == "GPU+CPU") {
                auto result = flat_search_gpu_cpu(base, test_query + i * vecdim, 
                                                base_number, vecdim, current_batch_size, k);
            } else if (method == "GPU_IVF") {
                std::vector<float> centroids_vec(centroids, centroids + nlist * vecdim);
                auto result = ivf_search_gpu(base, test_query + i * vecdim, 
                                           base_number, vecdim, current_batch_size, k,
                                           centroids_vec, nlist, *invlists, nprobe);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        total_time += duration.count();
    }
    
    return total_time / repeat_count / query_count;  // 返回平均每个查询的时间(us)
}

// 主函数
int main(int argc, char *argv[]) {
    if (argc < 6) {
        std::cerr << "用法: " << argv[0] << " <数据路径> <查询数> <重复次数> <暖机次数> <nprobe>" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];
    size_t query_count = std::stoi(argv[2]);
    int repeat_count = std::stoi(argv[3]);
    int warm_up_count = std::stoi(argv[4]);
    size_t nprobe = std::stoi(argv[5]);

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    // 加载数据
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    // 加载IVF数据
    size_t nlist = 256;  // 使用256个簇
    std::vector<float> centroids_vec;
    std::vector<std::vector<uint32_t>> invlists;
    float* centroids = nullptr;
    
    try {
        std::string ivf_path = "file/";  // IVF文件路径
        centroids_vec = load_ivf_centroids(ivf_path + "ivf_flat_centroids_256.fbin", nlist, vecdim);
        invlists = load_ivf_invlists(ivf_path + "ivf_flat_invlists_256.bin", nlist);
        centroids = centroids_vec.data();
        std::cout << "成功加载IVF数据，nlist=" << nlist << ", nprobe=" << nprobe << std::endl;
    } catch (const std::exception& e) {
        std::cout << "无法加载IVF数据: " << e.what() << "，将跳过IVF测试" << std::endl;
    }

    // 限制查询数量
    query_count = std::min(query_count, test_number);
    const size_t k = 10;
    
    // 不同的批大小设置
    std::vector<size_t> batch_sizes = {32, 64, 128, 256, 512, 1024, 2048};
    
    // 存储所有实验结果
    std::vector<SpeedupResult> all_results;
    
    // 输出结果的CSV文件头
    std::ofstream out_file("speedup_results.csv");
    out_file << "方法,批大小,基准时间(us),方法时间(us),加速比,召回率" << std::endl;
    
    std::cout << "开始加速比测试实验..." << std::endl;
    std::cout << "查询数量: " << query_count << ", 重复次数: " << repeat_count << std::endl;
    
    // 暖机
    std::cout << "暖机 " << warm_up_count << " 条查询..." << std::endl;
    for (int i = 0; i < warm_up_count; ++i) {
        auto warm_result = flat_search(base, test_query + i * vecdim, base_number, vecdim, k);
    }
    
    // 1. 运行基准CPU实验
    std::cout << "运行基准CPU实验..." << std::endl;
    double baseline_time = runBaselineExperiment(base, test_query, base_number, 
                                                vecdim, query_count, k, repeat_count);
    std::cout << "基准CPU时间: " << baseline_time << " us" << std::endl;
    
    // 2. 运行CPU IVF实验（如果有数据）
    if (centroids != nullptr) {
        std::cout << "运行CPU IVF实验..." << std::endl;
        double cpu_ivf_time = runCPUIVFExperiment(base, test_query, base_number, vecdim, 
                                                 query_count, k, centroids, nlist, 
                                                 invlists, nprobe, repeat_count);
        double speedup = baseline_time / cpu_ivf_time;
        
        SpeedupResult result = {"CPU_IVF", 1, baseline_time, cpu_ivf_time, speedup, 0.0f};
        all_results.push_back(result);
        
        std::cout << "CPU IVF: 时间=" << cpu_ivf_time << "us, 加速比=" << speedup << std::endl;
        out_file << "CPU_IVF,1," << baseline_time << "," << cpu_ivf_time << "," << speedup << ",N/A" << std::endl;
    }
    
    // 3. 运行GPU实验（不同批大小）
    std::vector<std::string> gpu_methods = {"GPU+CPU", "GPU+GPU"};
    if (centroids != nullptr) {
        gpu_methods.push_back("GPU_IVF");
    }
    
    for (const auto& method : gpu_methods) {
        std::cout << "测试 " << method << " 方法..." << std::endl;
        
        for (size_t batch_size : batch_sizes) {
            std::cout << "  批大小: " << batch_size << std::endl;
            
            // GPU暖机
            if (method == "GPU+GPU") {
                auto warm_result = flat_search_gpu(base, test_query, base_number, vecdim, warm_up_count, k);
            } else if (method == "GPU+CPU") {
                auto warm_result = flat_search_gpu_cpu(base, test_query, base_number, vecdim, warm_up_count, k);
            } else if (method == "GPU_IVF") {
                auto warm_result = ivf_search_gpu(base, test_query, base_number, vecdim, warm_up_count, k,
                                                centroids_vec, nlist, invlists, nprobe);
            }
            
            double gpu_time = runGPUExperiment(method, base, test_query, base_number, vecdim,
                                             query_count, batch_size, k, repeat_count,
                                             centroids, nlist, &invlists, nprobe);
            
            double speedup = baseline_time / gpu_time;
            
            SpeedupResult result = {method, batch_size, baseline_time, gpu_time, speedup, 0.0f};
            all_results.push_back(result);
            
            std::cout << "    时间=" << gpu_time << "us, 加速比=" << speedup << std::endl;
            out_file << method << "," << batch_size << "," << baseline_time << "," 
                     << gpu_time << "," << speedup << ",N/A" << std::endl;
        }
    }
    
    out_file.close();
    
    // 输出结果总结
    std::cout << "\n=== 加速比测试结果总结 ===" << std::endl;
    std::cout << "基准CPU时间: " << baseline_time << " us" << std::endl;
    std::cout << "\n最佳加速比:" << std::endl;
    
    // 找到每种方法的最佳加速比
    std::map<std::string, SpeedupResult> best_results;
    for (const auto& result : all_results) {
        if (best_results.find(result.method) == best_results.end() || 
            result.speedup > best_results[result.method].speedup) {
            best_results[result.method] = result;
        }
    }
    
    for (const auto& pair : best_results) {
        const auto& result = pair.second;
        std::cout << result.method << ": " << result.speedup << "x (批大小=" 
                  << result.batch_size << ", 时间=" << result.method_time_ms << "us)" << std::endl;
    }
    
    std::cout << "\n所有结果已保存到 speedup_results.csv" << std::endl;
    return 0;
} 