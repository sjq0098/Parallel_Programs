#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include "mpi_ivf_hnsw.h"

// 数据加载函数
bool LoadData(const std::string& file, std::vector<float>& data, size_t& num, size_t& dim) {
    std::ifstream in(file, std::ios::binary);
    if (!in) return false;
    
    in.read(reinterpret_cast<char*>(&num), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    
    data.resize(num * dim);
    in.read(reinterpret_cast<char*>(data.data()), num * dim * sizeof(float));
    return true;
}

bool LoadGroundTruth(const std::string& file, std::vector<uint32_t>& gt, size_t& query_num, size_t& k) {
    std::ifstream in(file, std::ios::binary);
    if (!in) return false;
    
    in.read(reinterpret_cast<char*>(&query_num), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&k), sizeof(uint32_t));
    
    gt.resize(query_num * k);
    in.read(reinterpret_cast<char*>(gt.data()), query_num * k * sizeof(uint32_t));
    return true;
}

double CalculateRecall(const std::vector<uint32_t>& result, const uint32_t* gt, size_t k) {
    size_t intersection = 0;
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < k; ++j) {
            if (result[i] == gt[j]) {
                intersection++;
                break;
            }
        }
    }
    return static_cast<double>(intersection) / k;
}

// 测试配置结构
struct TestConfig {
    std::string name;
    std::string codebook_file;
    size_t nlist;
    size_t nprobe;
    size_t M;
    size_t efConstruction;
    size_t efSearch;
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "====================================================================\n";
        std::cout << "              原版IVF+HNSW多次运行性能测试\n";
        std::cout << "====================================================================\n";
        std::cout << "MPI进程数: " << size << "\n";
        std::cout << "OpenMP线程数: " << omp_get_max_threads() << "\n";
    }
    
    // 加载数据
    std::vector<float> base_data, query_data;
    std::vector<uint32_t> gt_data;
    size_t base_num, query_num, dim, gt_k;
    
    if (rank == 0) {
        std::cout << "\n加载数据...\n";
        
        if (!LoadData("anndata/DEEP100K.base.100k.fbin", base_data, base_num, dim)) {
            std::cerr << "错误: 无法加载基础数据\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (!LoadData("anndata/DEEP100K.query.fbin", query_data, query_num, dim)) {
            std::cerr << "错误: 无法加载查询数据\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (!LoadGroundTruth("anndata/DEEP100K.gt.query.100k.top100.bin", gt_data, query_num, gt_k)) {
            std::cerr << "错误: 无法加载Ground Truth\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        std::cout << "✓ 数据加载完成\n";
        std::cout << "  基础向量: " << base_num << " x " << dim << "\n";
        std::cout << "  查询向量: " << query_num << " x " << dim << "\n";
        std::cout << "  Ground Truth: top-" << gt_k << "\n";
    }
    
    // 广播数据维度
    MPI_Bcast(&base_num, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&query_num, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gt_k, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    
    // 广播数据
    if (rank != 0) {
        base_data.resize(base_num * dim);
        query_data.resize(query_num * dim);
        gt_data.resize(query_num * gt_k);
    }
    
    MPI_Bcast(base_data.data(), base_num * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(query_data.data(), query_num * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(gt_data.data(), query_num * gt_k, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // 测试配置
    std::vector<TestConfig> test_configs = {
        {"PQ4-快速", "files/pq4_codebook.bin", 256, 8, 8, 100, 50},
        {"PQ4-平衡", "files/pq4_codebook.bin", 256, 16, 12, 150, 80},
        {"PQ4-高精度", "files/pq4_codebook.bin", 256, 32, 16, 200, 100},
        
        {"PQ8-快速", "files/pq8_codebook.bin", 256, 8, 8, 100, 50},
        {"PQ8-平衡", "files/pq8_codebook.bin", 256, 16, 12, 150, 80},
        {"PQ8-高精度", "files/pq8_codebook.bin", 256, 32, 16, 200, 100},
        
        {"PQ16-快速", "files/pq16_codebook.bin", 256, 8, 8, 100, 50},
        {"PQ16-平衡", "files/pq16_codebook.bin", 256, 16, 12, 150, 80},
        {"PQ16-高精度", "files/pq16_codebook.bin", 256, 32, 16, 200, 100}
    };
    
    const size_t k = 100;
    const size_t warmup_queries = 10;
    const size_t test_queries = 200;
    const size_t repeat_count = 5;
    
    if (rank == 0) {
        std::cout << "\n测试参数:\n";
        std::cout << "  检索top-k: " << k << "\n";
        std::cout << "  暖机查询: " << warmup_queries << "\n";
        std::cout << "  测试查询: " << test_queries << "\n";
        std::cout << "  重复次数: " << repeat_count << "\n";
        
        // 输出CSV头
        std::cout << "\n算法,配置,nlist,nprobe,M,efC,efS,召回率,召回率std,延迟μs,延迟std,构建时间ms,重复次数\n";
    }
    
    for (const auto& config : test_configs) {
        if (rank == 0) {
            std::cout << "\n===== 测试配置: " << config.name << " =====\n";
        }
        
        std::vector<double> recall_results;
        std::vector<double> latency_results;
        double build_time_ms = 0;
        
        for (size_t repeat = 0; repeat < repeat_count; ++repeat) {
            if (rank == 0) {
                std::cout << "第 " << (repeat + 1) << "/" << repeat_count << " 轮测试...\n";
            }
            
            // 创建索引
            MPIIVFHNSWIndex index(dim, config.nlist, config.M, config.efConstruction);
            
            // 加载码本
            if (!index.load_centroids(config.codebook_file)) {
                if (rank == 0) {
                    std::cerr << "错误: 无法加载码本文件 " << config.codebook_file << "\n";
                }
                continue;
            }
            
            // 构建索引
            auto build_start = std::chrono::high_resolution_clock::now();
            index.build_index(base_data.data(), base_num);
            auto build_end = std::chrono::high_resolution_clock::now();
            
            if (repeat == 0) {
                build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
            }
            
            // 设置搜索参数
            index.setEfSearch(config.efSearch);
            
            // 暖机
            if (rank == 0) {
                std::cout << "  暖机中...\n";
            }
            for (size_t i = 0; i < warmup_queries; ++i) {
                auto result = index.mpi_search(query_data.data() + i * dim, k, config.nprobe);
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            
            // 性能测试
            if (rank == 0) {
                std::cout << "  性能测试中...\n";
            }
            
            std::vector<double> query_times;
            std::vector<double> query_recalls;
            
            auto test_start = std::chrono::high_resolution_clock::now();
            
            for (size_t i = 0; i < test_queries; ++i) {
                auto query_start = std::chrono::high_resolution_clock::now();
                auto result = index.mpi_search(query_data.data() + i * dim, k, config.nprobe);
                auto query_end = std::chrono::high_resolution_clock::now();
                
                if (rank == 0) {
                    double latency = std::chrono::duration<double, std::micro>(query_end - query_start).count();
                    query_times.push_back(latency);
                    
                    // 计算召回率
                    std::vector<uint32_t> result_ids;
                    auto temp_result = result;
                    while (!temp_result.empty() && result_ids.size() < k) {
                        result_ids.push_back(temp_result.top().second);
                        temp_result.pop();
                    }
                    
                    double recall = CalculateRecall(result_ids, gt_data.data() + i * gt_k, k);
                    query_recalls.push_back(recall);
                }
            }
            
            if (rank == 0) {
                // 计算统计量
                double avg_latency = 0, avg_recall = 0;
                for (double t : query_times) avg_latency += t;
                for (double r : query_recalls) avg_recall += r;
                avg_latency /= query_times.size();
                avg_recall /= query_recalls.size();
                
                recall_results.push_back(avg_recall);
                latency_results.push_back(avg_latency);
                
                std::cout << "    召回率: " << std::fixed << std::setprecision(4) << avg_recall 
                          << ", 延迟: " << std::setprecision(1) << avg_latency << "μs\n";
            }
        }
        
        if (rank == 0) {
            // 计算最终统计
            double recall_mean = 0, recall_std = 0;
            double latency_mean = 0, latency_std = 0;
            
            for (double r : recall_results) recall_mean += r;
            for (double l : latency_results) latency_mean += l;
            recall_mean /= recall_results.size();
            latency_mean /= latency_results.size();
            
            for (double r : recall_results) recall_std += (r - recall_mean) * (r - recall_mean);
            for (double l : latency_results) latency_std += (l - latency_mean) * (l - latency_mean);
            recall_std = std::sqrt(recall_std / recall_results.size());
            latency_std = std::sqrt(latency_std / latency_results.size());
            
            // 输出CSV格式结果
            std::cout << "原版IVF+HNSW," << config.name << "," << config.nlist << "," 
                      << config.nprobe << "," << config.M << "," << config.efConstruction << "," 
                      << config.efSearch << "," << std::fixed << std::setprecision(4) << recall_mean 
                      << "," << recall_std << "," << std::setprecision(1) << latency_mean 
                      << "," << latency_std << "," << std::setprecision(0) << build_time_ms 
                      << "," << repeat_count << "\n";
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        std::cout << "\n🎉 所有测试完成！\n";
    }
    
    MPI_Finalize();
    return 0;
} 