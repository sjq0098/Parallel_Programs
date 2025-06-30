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
#include "mpi_ivf_hnsw.h"

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
    size_t M;
    size_t efConstruction;
    size_t efSearch;
    float recall;
    int64_t latency_us;
    int64_t build_time_ms;
    int mpi_processes;
    int omp_threads;
};

void generate_centroids(const float* base_data, size_t base_number, size_t vecdim, 
                       size_t nlist, const string& filename, int rank) {
    if (rank == 0) {
        std::vector<float> centroids(nlist * vecdim);
        for (size_t i = 0; i < nlist; ++i) {
            size_t random_idx = i * (base_number / nlist) % base_number;
            std::copy(base_data + random_idx * vecdim, 
                     base_data + (random_idx + 1) * vecdim,
                     centroids.begin() + i * vecdim);
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
        std::cout << "=== MPI IVF+HNSW算法参数测试 ===" << std::endl;
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
    
    test_number = std::min(test_number, size_t(500));  // 限制测试数量
    const size_t k = 10;
    
    // 参数组合定义
    std::vector<size_t> nlist_values = {64, 128, 256};
    std::vector<size_t> nprobe_values = {2, 4, 8, 16};
    std::vector<size_t> M_values = {8, 16, 24};
    std::vector<size_t> efConstruction_values = {100, 200};
    std::vector<size_t> efSearch_values = {32, 64, 128};
    
    std::vector<TestResult> results;
    
    for (size_t nlist : nlist_values) {
        if (rank == 0) {
            std::cout << "\n测试 nlist=" << nlist << std::endl;
        }
        
        // 生成聚类中心
        string centroid_file = "temp_ivf_hnsw_centroids_" + to_string(nlist) + ".bin";
        generate_centroids(base, base_number, vecdim, nlist, centroid_file, rank);
        
        for (size_t M : M_values) {
            for (size_t efC : efConstruction_values) {
                if (rank == 0) {
                    std::cout << "  构建索引 M=" << M << ", efC=" << efC << "..." << std::endl;
                }
                
                // 构建索引
                auto start_build = std::chrono::high_resolution_clock::now();
                MPIIVFHNSWIndex index(vecdim, nlist, M, efC);
                index.load_centroids(centroid_file);
                index.build_index(base, base_number);
                auto end_build = std::chrono::high_resolution_clock::now();
                
                int64_t build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count();
                
                for (size_t efS : efSearch_values) {
                    index.setEfSearch(efS);
                    
                    for (size_t nprobe : nprobe_values) {
                        if (nprobe > nlist) continue;
                        
                        if (rank == 0) {
                            std::cout << "    测试 efS=" << efS << ", nprobe=" << nprobe << "..." << std::flush;
                        }
                        
                        float total_recall = 0.0f;
                        int64_t total_latency = 0;
                        
                        for(size_t i = 0; i < test_number; ++i) {
                            struct timeval val, newVal;
                            gettimeofday(&val, NULL);
                            
                            auto res = index.mpi_search(test_query + i * vecdim, k, nprobe);
                            
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
                            result.M = M;
                            result.efConstruction = efC;
                            result.efSearch = efS;
                            result.recall = total_recall / test_number;
                            result.latency_us = total_latency / test_number;
                            result.build_time_ms = build_time;
                            result.mpi_processes = size;
                            result.omp_threads = omp_get_max_threads();
                            
                            results.push_back(result);
                            
                            std::cout << " R=" << std::fixed << std::setprecision(4) << result.recall 
                                     << ", L=" << result.latency_us << "us" << std::endl;
                        }
                    }
                }
            }
        }
        
        // 清理临时文件
        if (rank == 0) {
            remove(centroid_file.c_str());
        }
    }
    
    // 输出CSV结果
    if (rank == 0) {
        std::ofstream csv_file("results_mpi_ivf_hnsw.csv");
        csv_file << "nlist,nprobe,M,efConstruction,efSearch,recall,latency_us,build_time_ms,mpi_processes,omp_threads\n";
        
        for (const auto& result : results) {
            csv_file << result.nlist << "," << result.nprobe << "," << result.M << ","
                    << result.efConstruction << "," << result.efSearch << ","
                    << std::fixed << std::setprecision(6) << result.recall << ","
                    << result.latency_us << "," << result.build_time_ms << ","
                    << result.mpi_processes << "," << result.omp_threads << "\n";
        }
        
        csv_file.close();
        std::cout << "\n结果已保存到 results_mpi_ivf_hnsw.csv" << std::endl;
    }
    
    // 清理资源
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    MPI_Finalize();
    return 0;
}