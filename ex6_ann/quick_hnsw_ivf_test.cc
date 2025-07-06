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
#include "mpi_hnsw_ivf.h"

using namespace std;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d, int rank)
{
    T* data = nullptr;
    
    if (rank == 0) {
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
    
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        data = new T[n * d];
    }
    
    if (std::is_same<T, float>::value) {
        MPI_Bcast(data, n * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else if (std::is_same<T, int>::value) {
        MPI_Bcast(data, n * d, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    return data;
}

// 计算召回率
float calculate_recall(const std::vector<uint32_t>& result, const int* gt, size_t k, size_t gt_d) {
    std::set<int> gt_set;
    for (size_t i = 0; i < gt_d && i < k; ++i) {
        gt_set.insert(gt[i]);
    }
    
    int correct = 0;
    for (size_t i = 0; i < std::min(k, result.size()); ++i) {
        if (gt_set.count(result[i])) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / std::min(k, gt_d);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== 快速测试 MPI HNSW+IVF 混合算法 ===" << std::endl;
        std::cout << "MPI进程数: " << size << std::endl;
        std::cout << "OpenMP线程数: " << omp_get_max_threads() << std::endl;
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
    
    test_number = std::min(test_number, size_t(200));  // 快速测试，只用200条查询
    const size_t k = 10;
    
    // 精选的参数配置，用于快速对比
    struct Config {
        size_t nlist, nprobe, M, efConstruction, efSearch, candidate_factor;
        string name;
    };
    
    std::vector<Config> configs = {
        {128, 8, 16, 100, 100, 20, "平衡配置"},
        {256, 16, 16, 200, 150, 30, "高精度配置"},
        {64, 4, 8, 100, 50, 15, "快速配置"}
    };
    
    if (rank == 0) {
        std::cout << "\n开始快速测试...\n" << std::endl;
        std::cout << "配置名称,nlist,nprobe,M,efC,efS,candidate_factor,recall,latency_us,build_time_ms" << std::endl;
    }
    
    for (const auto& config : configs) {
        // 构建索引
        auto build_start = std::chrono::high_resolution_clock::now();
        
        MPIHNSWIVFIndex index(vecdim, config.nlist, config.M, config.efConstruction);
        index.setEfSearch(config.efSearch);
        index.setHNSWCandidateFactor(config.candidate_factor);
        index.build_index(base, base_number);
        
        auto build_end = std::chrono::high_resolution_clock::now();
        int64_t build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            build_end - build_start).count();
        
        if (rank == 0) {
            std::cout << "\n测试配置: " << config.name << std::endl;
        }
        
        std::vector<float> query_recalls;
        std::vector<int64_t> query_latencies;
        
        // 暖机
        if (rank == 0) {
            for (size_t i = 0; i < std::min(size_t(20), test_number); ++i) {
                auto result = index.mpi_search(test_query + i * vecdim, base, k, config.nprobe);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        // 实际测试
        for (size_t i = 0; i < test_number; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = index.mpi_search(test_query + i * vecdim, base, k, config.nprobe);
            auto end = std::chrono::high_resolution_clock::now();
            
            int64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
            
            if (rank == 0) {
                // 转换结果格式
                std::vector<uint32_t> result_ids;
                auto temp_result = result;
                while (!temp_result.empty()) {
                    result_ids.push_back(temp_result.top().second);
                    temp_result.pop();
                }
                std::reverse(result_ids.begin(), result_ids.end());
                
                float recall = calculate_recall(result_ids, test_gt + i * test_gt_d, k, test_gt_d);
                query_recalls.push_back(recall);
                query_latencies.push_back(latency);
            }
        }
        
        if (rank == 0) {
            // 计算平均召回率和延迟
            float avg_recall = 0.0f;
            int64_t avg_latency = 0;
            for (size_t i = 0; i < query_recalls.size(); ++i) {
                avg_recall += query_recalls[i];
                avg_latency += query_latencies[i];
            }
            avg_recall /= query_recalls.size();
            avg_latency /= query_latencies.size();
            
            // 输出结果
            std::cout << config.name << "," << config.nlist << "," << config.nprobe << "," 
                      << config.M << "," << config.efConstruction << "," << config.efSearch << "," 
                      << config.candidate_factor << "," << std::fixed << std::setprecision(4) 
                      << avg_recall << "," << avg_latency << "," << build_time << std::endl;
        }
        
        // 输出索引统计信息
        if (rank == 0) {
            index.print_index_stats();
            std::cout << std::endl;
        }
    }
    
    if (rank == 0) {
        std::cout << "\n=== 快速测试完成 ===" << std::endl;
        std::cout << "算法特点: 先用HNSW进行全局定位，再用IVF精化搜索结果" << std::endl;
        std::cout << "优势: 结合了HNSW的高质量全局搜索和IVF的局部性约束" << std::endl;
    }
    
    // 清理内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    MPI_Finalize();
    return 0;
} 