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
#include <limits>
#include <random>
#include <cmath>
#include <numeric>
#include "mpi_pq_ivf.h"

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
    size_t m;
    size_t ksub;
    size_t rerank_factor;
    float recall;
    int64_t latency_us;
    int64_t build_time_ms;
    int mpi_processes;
    int omp_threads;
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

// 生成简单的PQ码本
void generate_pq_codebook(size_t m, size_t ksub, size_t dsub, const string& filename, int rank) {
    if (rank == 0) {
        std::vector<float> codebook(m * ksub * dsub);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0, 1.0);
        
        for (size_t i = 0; i < codebook.size(); ++i) {
            codebook[i] = dis(gen);
        }
        
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(codebook.data()), 
                 codebook.size() * sizeof(float));
        out.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// 生成简单的PQ聚类中心
void generate_pq_centroids(size_t nlist, size_t m, const string& filename, int rank) {
    if (rank == 0) {
        std::vector<float> centroids(nlist * m);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 255.0);
        
        for (size_t i = 0; i < centroids.size(); ++i) {
            centroids[i] = dis(gen);
        }
        
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(centroids.data()), 
                 centroids.size() * sizeof(float));
        out.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== MPI PQ-IVF算法参数测试 ===" << std::endl;
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
    
    test_number = std::min(test_number, size_t(1000));  // 减少到1000条测试
    const size_t k = 10;
    const size_t warmup_queries = 50;  // 减少暖机查询数
    
    if (rank == 0) {
        std::cout << "测试配置: " << test_number << " 条查询, " 
                  << warmup_queries << " 条暖机, 单次运行" << std::endl;
    }
    
    // 参数组合定义
    std::vector<size_t> nlist_values = {64, 128, 256};
    std::vector<size_t> nprobe_values = {4, 8, 16, 32};
    std::vector<size_t> m_values = {8, 16, 24, 32};
    std::vector<size_t> ksub_values = {256, 512};
    std::vector<size_t> rerank_values = {2, 4};  // 重排序因子
    
    std::vector<TestResult> results;
    
    for (size_t nlist : nlist_values) {
        for (size_t m : m_values) {
            if (vecdim % m != 0) continue;  // 确保维度可以被m整除
            
            size_t dsub = vecdim / m;
            
            for (size_t ksub : ksub_values) {
                if (rank == 0) {
                    std::cout << "\n测试 nlist=" << nlist << ", m=" << m << ", ksub=" << ksub << std::endl;
                }
                
                // 生成PQ码本和聚类中心
                generate_pq_codebook(m, ksub, dsub, "temp_pq_codebook.bin", rank);
                generate_pq_centroids(nlist, m, "temp_pq_centroids.bin", rank);
                
                // 构建索引
                auto start_build = std::chrono::high_resolution_clock::now();
                MPIPQIVFIndex index(vecdim, nlist, m, ksub, true);  // 使用内积距离
                
                if (!index.load_pq_codebook("temp_pq_codebook.bin") ||
                    !index.load_pq_centroids("temp_pq_centroids.bin")) {
                    if (rank == 0) {
                        std::cerr << "加载PQ码本或聚类中心失败!" << std::endl;
                    }
                    continue;
                }
                
                index.build_index(base, base_number);
                auto end_build = std::chrono::high_resolution_clock::now();
                
                int64_t build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count();
                
                for (size_t nprobe : nprobe_values) {
                    if (nprobe > nlist) continue;
                    
                    for (size_t rerank_factor : rerank_values) {
                    if (rank == 0) {
                            std::cout << "  测试 nprobe=" << nprobe << ", rerank_factor=" << rerank_factor << "..." << std::flush;
                    }
                    
                        // 暖机阶段
                        for (size_t i = 0; i < std::min(warmup_queries, test_number); ++i) {
                            auto res = index.mpi_search(test_query + i * vecdim, k, nprobe, 200);
                        }
                        MPI_Barrier(MPI_COMM_WORLD);  // 确保所有进程完成暖机
                        
                        // 单次测试运行
                    float total_recall = 0.0f;
                    int64_t total_latency = 0;
                    
                    for(size_t i = 0; i < test_number; ++i) {
                        struct timeval val, newVal;
                        gettimeofday(&val, NULL);
                        
                            auto res = index.mpi_search(test_query + i * vecdim, k, nprobe, 200);
                        
                        gettimeofday(&newVal, NULL);
                        int64_t diff = (newVal.tv_sec * 1000000 + newVal.tv_usec) - 
                                      (val.tv_sec * 1000000 + val.tv_usec);
                        
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
                        TestResult result;
                        result.nlist = nlist;
                        result.nprobe = nprobe;
                        result.m = m;
                        result.ksub = ksub;
                            result.rerank_factor = rerank_factor;
                        result.recall = total_recall / test_number;
                        result.latency_us = total_latency / test_number;
                        result.build_time_ms = build_time;
                        result.mpi_processes = size;
                        result.omp_threads = omp_get_max_threads();
                        
                        results.push_back(result);
                        
                        std::cout << " Recall=" << std::fixed << std::setprecision(4) << result.recall 
                                 << ", Latency=" << result.latency_us << "us" << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    // 输出CSV结果
    if (rank == 0) {
        std::ofstream csv_file("results_mpi_pq_ivf.csv");
        csv_file << "nlist,nprobe,m,ksub,rerank_factor,recall,latency_us,build_time_ms,mpi_processes,omp_threads\n";
        
        for (const auto& result : results) {
            csv_file << result.nlist << "," << result.nprobe << "," << result.m << ","
                    << result.ksub << "," << result.rerank_factor << ","
                    << std::fixed << std::setprecision(6) << result.recall << ","
                    << result.latency_us << "," << result.build_time_ms << ","
                    << result.mpi_processes << "," << result.omp_threads << "\n";
        }
        
        csv_file.close();
        std::cout << "\n结果已保存到 results_mpi_pq_ivf.csv" << std::endl;
    }
    
    // 清理临时文件
    if (rank == 0) {
        std::remove("temp_pq_codebook.bin");
        std::remove("temp_pq_centroids.bin");
    }
    
    // 清理资源
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    MPI_Finalize();
    return 0;
} 