#include <mpi.h>
#include <omp.h>
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

// 包含算法头文件
#include "mpi_ivf.h"
#include "mpi_ivf_hnsw.h"
#include "mpi_simd_hybrid.h"
#include "mpi_sharded_hnsw.h"

using namespace std;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d, int rank)
{
    T* data = nullptr;
    
    if (rank == 0) {
        // 只有主进程读取数据
        std::ifstream fin;
        fin.open(data_path, std::ios::in | std::ios::binary);
        if (!fin) {
            std::cerr << "无法打开文件: " << data_path << std::endl;
            n = 0; d = 0;
            return nullptr;
        }
        
        fin.read((char*)&n, 4);
        fin.read((char*)&d, 4);
        data = new T[n * d];
        int sz = sizeof(T);
        for(size_t i = 0; i < n; ++i){
            fin.read(((char*)data + i*d*sz), d*sz);
        }
        fin.close();

        std::cerr << "加载数据 " << data_path << std::endl;
        std::cerr << "维度: " << d << "  数量: " << n << "  每元素大小: " << sizeof(T) << std::endl;
    }
    
    // 广播数据维度信息
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        data = new T[n * d];
    }
    
    // 广播数据
    if (std::is_same<T, float>::value) {
        MPI_Bcast(data, n * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else if (std::is_same<T, int>::value) {
        MPI_Bcast(data, n * d, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

enum class AlgorithmType {
    IVF = 1,
    IVF_HNSW = 2,
    SIMD_HYBRID = 3,
    SHARDED_HNSW = 4
};

void print_usage() {
    std::cout << "用法: mpirun -np <进程数> ./main_mpi <算法类型> [参数]" << std::endl;
    std::cout << "算法类型:" << std::endl;
    std::cout << "  1 - 基础IVF" << std::endl;
    std::cout << "  2 - IVF+HNSW混合索引" << std::endl;
    std::cout << "  3 - SIMD混合并行" << std::endl;
    std::cout << "  4 - 分片HNSW" << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  mpirun -np 4 ./main_mpi 1  # 使用4个进程运行基础IVF" << std::endl;
    std::cout << "  mpirun -np 4 ./main_mpi 2  # 使用4个进程运行IVF+HNSW" << std::endl;
    std::cout << "  mpirun -np 8 ./main_mpi 3  # 使用8个进程运行SIMD混合并行" << std::endl;
    std::cout << "  mpirun -np 6 ./main_mpi 4  # 使用6个进程运行分片HNSW" << std::endl;
}

int main(int argc, char *argv[])
{
    // 初始化MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 2) {
        if (rank == 0) {
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }
    
    AlgorithmType alg_type = static_cast<AlgorithmType>(std::atoi(argv[1]));
    
    if (rank == 0) {
        std::cout << "=== MPI并行向量搜索测试 ===" << std::endl;
        std::cout << "MPI进程数: " << size << std::endl;
        std::cout << "OpenMP线程数: " << omp_get_max_threads() << std::endl;
        std::cout << "算法类型: ";
        switch(alg_type) {
            case AlgorithmType::IVF: std::cout << "基础IVF"; break;
            case AlgorithmType::IVF_HNSW: std::cout << "IVF+HNSW混合索引"; break;
            case AlgorithmType::SIMD_HYBRID: std::cout << "SIMD混合并行"; break;
            case AlgorithmType::SHARDED_HNSW: std::cout << "分片HNSW"; break;
        }
        std::cout << std::endl << std::endl;
    }
    
    // 加载数据
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim, rank);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d, rank);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim, rank);
    
    if (!test_query || !test_gt || !base) {
        if (rank == 0) {
            std::cerr << "数据加载失败!" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // 限制测试查询数量
    test_number = std::min(test_number, size_t(2000));
    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 构建索引和搜索的时间测量
    auto start_build = std::chrono::high_resolution_clock::now();
    
    // 根据算法类型进行测试
    switch(alg_type) {
        case AlgorithmType::IVF:
        {
            if (rank == 0) std::cout << "开始构建基础IVF索引..." << std::endl;
            
                         MPIIVFIndex index(vecdim, 128);  // 128个簇
            
            // 这里需要预先生成或加载聚类中心
            // 简化实现：使用随机选择的点作为聚类中心
            if (rank == 0) {
                std::vector<float> centroids(128 * vecdim);
                for (size_t i = 0; i < 128; ++i) {
                    size_t random_idx = i * (base_number / 128) % base_number;
                    std::copy(base + random_idx * vecdim, 
                             base + (random_idx + 1) * vecdim,
                             centroids.begin() + i * vecdim);
                }
                std::ofstream out("temp_centroids.bin", std::ios::binary);
                out.write(reinterpret_cast<const char*>(centroids.data()), 
                         centroids.size() * sizeof(float));
                out.close();
            }
            MPI_Barrier(MPI_COMM_WORLD);
            
            index.load_centroids("temp_centroids.bin");
            index.build_index(base, base_number);
            
            auto end_build = std::chrono::high_resolution_clock::now();
            auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build);
            
            if (rank == 0) {
                std::cout << "索引构建完成，耗时: " << build_time.count() << "ms" << std::endl;
                index.print_index_stats();
                std::cout << "开始搜索测试..." << std::endl;
            }
            
            // 搜索测试
            for(size_t i = 0; i < test_number; ++i) {
                struct timeval val, newVal;
                gettimeofday(&val, NULL);
                
                auto res = index.mpi_search(test_query + i * vecdim, k, 16);
                
                gettimeofday(&newVal, NULL);
                int64_t diff = (newVal.tv_sec * 1000000 + newVal.tv_usec) - 
                              (val.tv_sec * 1000000 + val.tv_usec);
                
                // 计算recall（只在rank 0上计算）
                float recall = 0.0f;
                if (rank == 0) {
                    std::set<uint32_t> gtset;
                    for(size_t j = 0; j < k; ++j){
                        int t = test_gt[j + i * test_gt_d];
                        gtset.insert(t);
                    }
                    
                    size_t acc = 0;
                    std::priority_queue<std::pair<float, uint32_t>> temp_res = res;
                    while (!temp_res.empty()) {   
                        uint32_t x = temp_res.top().second;
                        if(gtset.find(x) != gtset.end()){
                            ++acc;
                        }
                        temp_res.pop();
                    }
                    recall = static_cast<float>(acc) / k;
                }
                
                results[i] = {recall, diff};
            }
            break;
        }
        
        case AlgorithmType::IVF_HNSW:
        {
            if (rank == 0) std::cout << "开始构建IVF+HNSW索引..." << std::endl;
            
            MPIIVFHNSWIndex index(vecdim, 128, 16, 200);  // 128个簇，M=16，efC=200
            
            // 这里需要预先生成或加载聚类中心
            // 简化实现：使用随机选择的点作为聚类中心
            if (rank == 0) {
                std::vector<float> centroids(128 * vecdim);
                for (size_t i = 0; i < 128; ++i) {
                    size_t random_idx = i * (base_number / 128) % base_number;
                    std::copy(base + random_idx * vecdim, 
                             base + (random_idx + 1) * vecdim,
                             centroids.begin() + i * vecdim);
                }
                std::ofstream out("temp_centroids.bin", std::ios::binary);
                out.write(reinterpret_cast<const char*>(centroids.data()), 
                         centroids.size() * sizeof(float));
                out.close();
            }
            MPI_Barrier(MPI_COMM_WORLD);
            
            index.load_centroids("temp_centroids.bin");
            index.build_index(base, base_number);
            
            auto end_build = std::chrono::high_resolution_clock::now();
            auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build);
            
            if (rank == 0) {
                std::cout << "索引构建完成，耗时: " << build_time.count() << "ms" << std::endl;
                index.print_index_stats();
                std::cout << "开始搜索测试..." << std::endl;
            }
            
            // 搜索测试
            for(size_t i = 0; i < test_number; ++i) {
                struct timeval val, newVal;
                gettimeofday(&val, NULL);
                
                auto res = index.mpi_search(test_query + i * vecdim, k, 16);
                
                gettimeofday(&newVal, NULL);
                int64_t diff = (newVal.tv_sec * 1000000 + newVal.tv_usec) - 
                              (val.tv_sec * 1000000 + val.tv_usec);
                
                // 计算recall（只在rank 0上计算）
                float recall = 0.0f;
                if (rank == 0) {
                    std::set<uint32_t> gtset;
                    for(size_t j = 0; j < k; ++j){
                        int t = test_gt[j + i * test_gt_d];
                        gtset.insert(t);
                    }
                    
                    size_t acc = 0;
                    std::priority_queue<std::pair<float, uint32_t>> temp_res = res;
                    while (!temp_res.empty()) {   
                        uint32_t x = temp_res.top().second;
                        if(gtset.find(x) != gtset.end()){
                            ++acc;
                        }
                        temp_res.pop();
                    }
                    recall = static_cast<float>(acc) / k;
                }
                
                results[i] = {recall, diff};
            }
            break;
        }
        
        case AlgorithmType::SIMD_HYBRID:
        {
            if (rank == 0) std::cout << "开始构建SIMD混合并行索引..." << std::endl;
            
            MPISIMDHybridIndex index(vecdim, 64);  // 64个簇
            index.print_performance_info();
            
            // 生成聚类中心
            if (rank == 0) {
                std::vector<float> centroids(64 * vecdim);
                for (size_t i = 0; i < 64; ++i) {
                    size_t random_idx = i * (base_number / 64) % base_number;
                    std::copy(base + random_idx * vecdim, 
                             base + (random_idx + 1) * vecdim,
                             centroids.begin() + i * vecdim);
                }
                std::ofstream out("temp_simd_centroids.bin", std::ios::binary);
                out.write(reinterpret_cast<const char*>(centroids.data()), 
                         centroids.size() * sizeof(float));
                out.close();
            }
            MPI_Barrier(MPI_COMM_WORLD);
            
            index.load_centroids("temp_simd_centroids.bin");
            index.build_index(base, base_number);
            
            auto end_build = std::chrono::high_resolution_clock::now();
            auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build);
            
            if (rank == 0) {
                std::cout << "索引构建完成，耗时: " << build_time.count() << "ms" << std::endl;
                std::cout << "开始搜索测试..." << std::endl;
            }
            
            // 搜索测试
            for(size_t i = 0; i < test_number; ++i) {
                struct timeval val, newVal;
                gettimeofday(&val, NULL);
                
                auto res = index.mpi_simd_search(test_query + i * vecdim, k, 16);
                
                gettimeofday(&newVal, NULL);
                int64_t diff = (newVal.tv_sec * 1000000 + newVal.tv_usec) - 
                              (val.tv_sec * 1000000 + val.tv_usec);
                
                // 计算recall
                float recall = 0.0f;
                if (rank == 0) {
                    std::set<uint32_t> gtset;
                    for(size_t j = 0; j < k; ++j){
                        int t = test_gt[j + i * test_gt_d];
                        gtset.insert(t);
                    }
                    
                    size_t acc = 0;
                    std::priority_queue<std::pair<float, uint32_t>> temp_res = res;
                    while (!temp_res.empty()) {   
                        uint32_t x = temp_res.top().second;
                        if(gtset.find(x) != gtset.end()){
                            ++acc;
                        }
                        temp_res.pop();
                    }
                    recall = static_cast<float>(acc) / k;
                }
                
                results[i] = {recall, diff};
            }
            break;
        }
        
        case AlgorithmType::SHARDED_HNSW:
        {
            if (rank == 0) std::cout << "开始构建分片HNSW索引..." << std::endl;
            
            MPIShardedHNSWIndex index(vecdim, 16, 200, ShardingStrategy::ROUND_ROBIN);
            index.build_index(base, base_number);
            
            auto end_build = std::chrono::high_resolution_clock::now();
            auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build);
            
            if (rank == 0) {
                std::cout << "索引构建完成，耗时: " << build_time.count() << "ms" << std::endl;
                std::cout << "开始搜索测试..." << std::endl;
            }
            
            // 搜索测试
            for(size_t i = 0; i < test_number; ++i) {
                struct timeval val, newVal;
                gettimeofday(&val, NULL);
                
                auto res = index.mpi_search(test_query + i * vecdim, k);
                
                gettimeofday(&newVal, NULL);
                int64_t diff = (newVal.tv_sec * 1000000 + newVal.tv_usec) - 
                              (val.tv_sec * 1000000 + val.tv_usec);
                
                // 计算recall
                float recall = 0.0f;
                if (rank == 0) {
                    std::set<uint32_t> gtset;
                    for(size_t j = 0; j < k; ++j){
                        int t = test_gt[j + i * test_gt_d];
                        gtset.insert(t);
                    }
                    
                    size_t acc = 0;
                    std::priority_queue<std::pair<float, uint32_t>> temp_res = res;
                    while (!temp_res.empty()) {   
                        uint32_t x = temp_res.top().second;
                        if(gtset.find(x) != gtset.end()){
                            ++acc;
                        }
                        temp_res.pop();
                    }
                    recall = static_cast<float>(acc) / k;
                }
                
                results[i] = {recall, diff};
            }
            break;
        }
    }

    // 收集和计算结果统计（只在rank 0上进行）
    if (rank == 0) {
        float avg_recall = 0, avg_latency = 0;
        for(size_t i = 0; i < test_number; ++i) {
            avg_recall += results[i].recall;
            avg_latency += results[i].latency;
        }

        std::cout << std::endl << "=== 测试结果 ===" << std::endl;
        std::cout << "平均召回率: " << std::fixed << std::setprecision(4) 
                  << avg_recall / test_number << std::endl;
        std::cout << "平均延迟 (us): " << std::fixed << std::setprecision(2) 
                  << avg_latency / test_number << std::endl;
        std::cout << "测试查询数: " << test_number << std::endl;
        std::cout << "Top-K: " << k << std::endl;
        
        // 延迟分布统计
        std::vector<int64_t> latencies;
        for(const auto& result : results) {
            latencies.push_back(result.latency);
        }
        std::sort(latencies.begin(), latencies.end());
        
        std::cout << "延迟分布:" << std::endl;
        std::cout << "  P50: " << latencies[test_number * 50 / 100] << " us" << std::endl;
        std::cout << "  P90: " << latencies[test_number * 90 / 100] << " us" << std::endl;
        std::cout << "  P99: " << latencies[test_number * 99 / 100] << " us" << std::endl;
    }

    // 清理资源
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    // 删除临时文件
    if (rank == 0) {
        system("rm -f temp_centroids.bin temp_simd_centroids.bin");
    }
    
    MPI_Finalize();
    return 0;
} 