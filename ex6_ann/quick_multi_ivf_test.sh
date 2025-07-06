#!/bin/bash

# ====================================================================
# 原版IVF+HNSW快速验证测试脚本
# 使用预训练码本进行快速多配置验证
# ====================================================================

set -e

# 实验配置
MPI_PROCESSES=4
OMP_THREADS=4
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="quick_multi_ivf_results_${TIMESTAMP}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}======================================================================"
echo "              原版IVF+HNSW快速验证测试"
echo "======================================================================${NC}"
echo "MPI进程数: $MPI_PROCESSES"
echo "OpenMP线程数: $OMP_THREADS"
echo "结果目录: $RESULT_DIR"

# 设置环境
export OMP_NUM_THREADS=$OMP_THREADS

# 创建结果目录
mkdir -p "$RESULT_DIR"

# 创建快速测试版本
echo -e "\n${BLUE}创建快速测试程序...${NC}"
cat > quick_multi_test.cc << 'EOF'
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

// 快速测试配置
struct QuickTestConfig {
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
        std::cout << "              原版IVF+HNSW快速验证测试\n";
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
    
    // 快速测试配置（每种码本只测试一个配置）
    std::vector<QuickTestConfig> test_configs = {
        {"PQ4-平衡", "files/pq4_codebook.bin", 256, 16, 12, 150, 80},
        {"PQ8-平衡", "files/pq8_codebook.bin", 256, 16, 12, 150, 80},
        {"PQ16-平衡", "files/pq16_codebook.bin", 256, 16, 12, 150, 80}
    };
    
    const size_t k = 100;
    const size_t warmup_queries = 5;
    const size_t test_queries = 50;
    const size_t repeat_count = 2;  // 快速测试只重复2次
    
    if (rank == 0) {
        std::cout << "\n快速测试参数:\n";
        std::cout << "  检索top-k: " << k << "\n";
        std::cout << "  暖机查询: " << warmup_queries << "\n";
        std::cout << "  测试查询: " << test_queries << "\n";
        std::cout << "  重复次数: " << repeat_count << "\n";
        
        // 输出CSV头
        std::cout << "\n算法,配置,nlist,nprobe,M,efC,efS,召回率,延迟μs,构建时间ms\n";
    }
    
    for (const auto& config : test_configs) {
        if (rank == 0) {
            std::cout << "\n===== 快速测试: " << config.name << " =====\n";
        }
        
        double total_recall = 0, total_latency = 0, build_time_ms = 0;
        
        for (size_t repeat = 0; repeat < repeat_count; ++repeat) {
            if (rank == 0) {
                std::cout << "第 " << (repeat + 1) << "/" << repeat_count << " 轮...\n";
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
            for (size_t i = 0; i < warmup_queries; ++i) {
                auto result = index.mpi_search(query_data.data() + i * dim, k, config.nprobe);
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            
            // 快速性能测试
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
                // 计算平均值
                double avg_latency = 0, avg_recall = 0;
                for (double t : query_times) avg_latency += t;
                for (double r : query_recalls) avg_recall += r;
                avg_latency /= query_times.size();
                avg_recall /= query_recalls.size();
                
                total_recall += avg_recall;
                total_latency += avg_latency;
                
                std::cout << "    召回率: " << std::fixed << std::setprecision(4) << avg_recall 
                          << ", 延迟: " << std::setprecision(1) << avg_latency << "μs\n";
            }
        }
        
        if (rank == 0) {
            // 输出平均结果
            double final_recall = total_recall / repeat_count;
            double final_latency = total_latency / repeat_count;
            
            std::cout << "原版IVF+HNSW," << config.name << "," << config.nlist << "," 
                      << config.nprobe << "," << config.M << "," << config.efConstruction << "," 
                      << config.efSearch << "," << std::fixed << std::setprecision(4) << final_recall 
                      << "," << std::setprecision(1) << final_latency 
                      << "," << std::setprecision(0) << build_time_ms << "\n";
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        std::cout << "\n🎉 快速验证测试完成！\n";
    }
    
    MPI_Finalize();
    return 0;
}
EOF

echo -e "${GREEN}✓ 快速测试程序创建完成${NC}"

# 编译快速测试程序
echo -e "\n${BLUE}编译快速测试程序...${NC}"
mpic++ -O3 -std=c++11 -fopenmp -DWITH_OPENMP \
    -I. -Ihnswlib \
    quick_multi_test.cc \
    -o quick_multi_test \
    2> "$RESULT_DIR/compile.log"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 编译成功${NC}"
else
    echo -e "${RED}✗ 编译失败，查看日志: $RESULT_DIR/compile.log${NC}"
    exit 1
fi

# 运行快速测试
echo -e "\n${BLUE}开始快速验证测试...${NC}"
echo "测试配置: 3种PQ码本各一个平衡配置"
echo "测试参数: 暖机5次、重复2轮、每轮50次查询"
echo "预计时间: 3-8分钟"

start_time=$(date +%s)

# 运行测试 (超时10分钟)
timeout 600 mpirun -np $MPI_PROCESSES ./quick_multi_test \
    > "$RESULT_DIR/test_output.log" 2>&1

test_result=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

if [ $test_result -eq 0 ]; then
    echo -e "${GREEN}✓ 快速验证测试完成${NC}"
    echo "测试耗时: ${duration}秒"
    
    # 提取结果
    echo "算法,配置,nlist,nprobe,M,efC,efS,召回率,延迟μs,构建时间ms" > "$RESULT_DIR/results.csv"
    grep "原版IVF+HNSW," "$RESULT_DIR/test_output.log" >> "$RESULT_DIR/results.csv" 2>/dev/null || true
    
    # 显示结果
    if [ -f "$RESULT_DIR/results.csv" ] && [ -s "$RESULT_DIR/results.csv" ]; then
        echo -e "\n${YELLOW}=== 快速验证结果 ===${NC}"
        echo ""
        awk -F',' 'NR>1 {printf "%-12s | 召回率: %.4f | 延迟: %6.1fμs | 构建: %6.0fms\n", $2, $8, $9, $10}' "$RESULT_DIR/results.csv"
        
        # 检查结果合理性
        echo -e "\n${YELLOW}=== 结果验证 ===${NC}"
        
        avg_recall=$(awk -F',' 'NR>1 {sum+=$8; count++} END {print sum/count}' "$RESULT_DIR/results.csv")
        avg_latency=$(awk -F',' 'NR>1 {sum+=$9; count++} END {print sum/count}' "$RESULT_DIR/results.csv")
        
        echo "平均召回率: $(printf "%.4f" $avg_recall)"
        echo "平均延迟: $(printf "%.1f" $avg_latency)μs"
        
        # 简单的健全性检查
        recall_ok=$(awk -v r="$avg_recall" 'BEGIN {print (r > 0.1 && r < 1.0) ? "true" : "false"}')
        latency_ok=$(awk -v l="$avg_latency" 'BEGIN {print (l > 10 && l < 100000) ? "true" : "false"}')
        
        if [ "$recall_ok" = "true" ] && [ "$latency_ok" = "true" ]; then
            echo -e "${GREEN}✓ 结果看起来合理${NC}"
        else
            echo -e "${YELLOW}⚠️  结果可能异常，请检查详细日志${NC}"
        fi
        
    else
        echo -e "${RED}⚠️  未找到有效的CSV结果${NC}"
    fi
    
elif [ $test_result -eq 124 ]; then
    echo -e "${RED}✗ 测试超时(10分钟限制)${NC}"
else
    echo -e "${RED}✗ 测试失败，退出码: $test_result${NC}"
fi

# 清理临时文件
rm -f quick_multi_test.cc quick_multi_test

echo -e "\n${GREEN}=== 快速验证测试完成 ===${NC}"
echo -e "结果目录: ${BLUE}$RESULT_DIR${NC}"
echo -e "主要文件:"
echo -e "  📊 ${YELLOW}results.csv${NC} - 快速测试数据"
echo -e "  📋 ${YELLOW}test_output.log${NC} - 完整日志"

if [ $test_result -eq 0 ]; then
    echo -e "\n💡 如需完整测试，请运行: ${YELLOW}./run_multi_ivf_hnsw_test.sh${NC}"
fi 