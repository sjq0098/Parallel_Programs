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
#include <numeric>
#include "mpi_simd_hybrid.h"

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

struct TestResult {
    size_t nlist;
    size_t nprobe;
    float recall_mean;
    float recall_std;
    int64_t latency_us_mean;
    int64_t latency_us_std;
    int64_t build_time_ms;
    int mpi_processes;
    int omp_threads;
    string simd_type;
};

// 计算标准差的辅助函数
float calculate_std(const std::vector<float>& values, float mean) {
    if (values.size() <= 1) return 0.0f;
    float sum_sq_diff = 0.0f;
    for (float v : values) {
        float diff = v - mean;
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / (values.size() - 1));
}

float calculate_std_int64(const std::vector<int64_t>& values, int64_t mean) {
    if (values.size() <= 1) return 0.0f;
    float sum_sq_diff = 0.0f;
    for (int64_t v : values) {
        float diff = static_cast<float>(v - mean);
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / (values.size() - 1));
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== MPI SIMD混合算法参数测试 ===" << std::endl;
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
    
    test_number = std::min(test_number, size_t(2000));  // 统一为2000条测试
    const size_t k = 10;
    const size_t warmup_queries = 100;  // 暖机查询数
    const int repeat_runs = 5;  // 重复运行次数
    
    if (rank == 0) {
        std::cout << "测试配置: " << test_number << " 条查询, " 
                  << warmup_queries << " 条暖机, " << repeat_runs << " 次重复" << std::endl;
    }
    
    // 参数组合定义
    std::vector<size_t> nlist_values = {32, 64, 128, 256, 512};
    std::vector<size_t> nprobe_values = {1, 2, 4, 8, 16, 32, 64};
    
    std::vector<TestResult> results;
    
    // 检测SIMD支持类型
    string simd_type = "SCALAR";
#ifdef __AVX2__
    simd_type = "AVX2";
#elif defined(__SSE2__)
    simd_type = "SSE2";
#elif defined(__ARM_NEON)
    simd_type = "NEON";
#endif
    
    for (size_t nlist : nlist_values) {
        if (rank == 0) {
            std::cout << "\n测试 nlist=" << nlist << std::endl;
        }
        
        // 生成聚类中心
        string centroid_file = "temp_simd_centroids_" + to_string(nlist) + ".bin";
        if (rank == 0) {
            std::vector<float> centroids(nlist * vecdim);
            for (size_t i = 0; i < nlist; ++i) {
                size_t random_idx = i * (base_number / nlist) % base_number;
                std::copy(base + random_idx * vecdim, 
                         base + (random_idx + 1) * vecdim,
                         centroids.begin() + i * vecdim);
            }
            std::ofstream out(centroid_file, std::ios::binary);
            out.write(reinterpret_cast<const char*>(centroids.data()), 
                     centroids.size() * sizeof(float));
            out.close();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        // 构建索引
        auto start_build = std::chrono::high_resolution_clock::now();
        MPISIMDHybridIndex index(vecdim, nlist);
        index.load_centroids(centroid_file);
        index.build_index(base, base_number);
        auto end_build = std::chrono::high_resolution_clock::now();
        
        int64_t build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count();
        
        for (size_t nprobe : nprobe_values) {
            if (nprobe > nlist) continue;  // nprobe不能大于nlist
            
            if (rank == 0) {
                std::cout << "  测试 nprobe=" << nprobe << std::endl;
            }
            
            // 暖机阶段
            if (rank == 0) {
                std::cout << "    执行暖机 (" << warmup_queries << " 条查询)..." << std::flush;
            }
            
            for (size_t i = 0; i < std::min(warmup_queries, test_number); ++i) {
                auto res = index.mpi_simd_search(test_query + i * vecdim, k, nprobe);
            }
            MPI_Barrier(MPI_COMM_WORLD);  // 确保所有进程完成暖机
            
            if (rank == 0) {
                std::cout << " 完成" << std::endl;
            }
            
            // 重复运行测试
            std::vector<float> recall_runs;
            std::vector<int64_t> latency_runs;
            
            for (int run = 0; run < repeat_runs; ++run) {
                if (rank == 0) {
                    std::cout << "    第 " << (run + 1) << "/" << repeat_runs << " 次运行..." << std::flush;
            }
            
            float total_recall = 0.0f;
            int64_t total_latency = 0;
            
            for(size_t i = 0; i < test_number; ++i) {
                struct timeval val, newVal;
                gettimeofday(&val, NULL);
                
                auto res = index.mpi_simd_search(test_query + i * vecdim, k, nprobe);
                
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
                
                total_recall += recall;
                total_latency += diff;
            }
            
                if (rank == 0) {
                    float avg_recall = total_recall / test_number;
                    int64_t avg_latency = total_latency / test_number;
                    
                    recall_runs.push_back(avg_recall);
                    latency_runs.push_back(avg_latency);
                    
                    std::cout << " R=" << std::fixed << std::setprecision(4) << avg_recall 
                             << ", L=" << avg_latency << "us" << std::endl;
                }
            }
            
            // 计算统计结果
            if (rank == 0) {
                TestResult result;
                result.nlist = nlist;
                result.nprobe = nprobe;
                
                // 计算平均值
                result.recall_mean = std::accumulate(recall_runs.begin(), recall_runs.end(), 0.0f) / repeat_runs;
                result.latency_us_mean = std::accumulate(latency_runs.begin(), latency_runs.end(), 0LL) / repeat_runs;
                
                // 计算标准差
                result.recall_std = calculate_std(recall_runs, result.recall_mean);
                result.latency_us_std = static_cast<int64_t>(calculate_std_int64(latency_runs, result.latency_us_mean));
                
                result.build_time_ms = build_time;
                result.mpi_processes = size;
                result.omp_threads = omp_get_max_threads();
                result.simd_type = simd_type;
                
                results.push_back(result);
                
                std::cout << "    平均结果: Recall=" << std::fixed << std::setprecision(4) << result.recall_mean 
                         << "±" << result.recall_std << ", Latency=" << result.latency_us_mean 
                         << "±" << result.latency_us_std << "us" << std::endl;
            }
        }
        
        // 清理临时文件
        if (rank == 0) {
            remove(centroid_file.c_str());
        }
    }
    
    // 输出CSV结果
    if (rank == 0) {
        std::ofstream csv_file("results_mpi_simd.csv");
        csv_file << "nlist,nprobe,recall_mean,recall_std,latency_us_mean,latency_us_std,build_time_ms,mpi_processes,omp_threads,simd_type\n";
        
        for (const auto& result : results) {
            csv_file << result.nlist << "," << result.nprobe << "," 
                    << std::fixed << std::setprecision(6) << result.recall_mean << ","
                    << std::fixed << std::setprecision(6) << result.recall_std << ","
                    << result.latency_us_mean << "," << result.latency_us_std << ","
                    << result.build_time_ms << "," << result.mpi_processes << "," 
                    << result.omp_threads << "," << result.simd_type << "\n";
        }
        
        csv_file.close();
        std::cout << "\n结果已保存到 results_mpi_simd.csv" << std::endl;
    }
    
    // 清理资源
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    MPI_Finalize();
    return 0;
}