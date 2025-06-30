#!/bin/bash

# 专注于高召回率参数的PQ-IVF vs IVF-PQ对比测试
# 用于论文中\subsection{PQ与IVF混合的两种方法对比分析}

set -e

echo "=== PQ-IVF vs IVF-PQ 高性能参数对比测试 ==="
echo "用途: 论文对比分析支撑数据"
echo "开始时间: $(date)"
echo ""

# 编译参数
MPI_CC="mpicxx"
CXX_FLAGS="-std=c++17 -O3 -march=native -fopenmp -g"
INCLUDE_FLAGS="-I. -I./hnswlib"
LINK_FLAGS="-lm -fopenmp -lstdc++ -lmpi_cxx"

# 固定使用4进程
MPI_PROCESSES=4
OMP_THREADS=4

echo "测试配置:"
echo "  MPI进程数: $MPI_PROCESSES"
echo "  OpenMP线程数: $OMP_THREADS" 
echo "  测试规模: 500查询，30暖机，单次运行"
echo "  专注参数: 高召回率组合"
echo ""

# 设置OpenMP线程数
export OMP_NUM_THREADS=$OMP_THREADS

# 创建结果目录
RESULTS_DIR="focused_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR
echo "结果将保存到目录: $RESULTS_DIR"
echo ""

# 首先停止当前运行的测试
pkill -f main_mpi || true

# 使用现有的简化程序但修改参数组合
echo "修改测试程序为精简高性能参数..."

# 创建精简的PQ-IVF测试
cat > test_pq_ivf_focused.cc << 'EOF'
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <random>
#include "mpi_pq_ivf.h"

using namespace std;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d, int rank) {
    T* data = nullptr;
    if (rank == 0) {
        std::ifstream fin(data_path, std::ios::in | std::ios::binary);
        fin.read((char*)&n, 4);
        fin.read((char*)&d, 4);
        data = new T[n * d];
        for(size_t i = 0; i < n; ++i){
            fin.read(((char*)data + i*d*sizeof(T)), d*sizeof(T));
        }
        fin.close();
    }
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) data = new T[n * d];
    if (std::is_same<T, float>::value) {
        MPI_Bcast(data, n * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else if (std::is_same<T, int>::value) {
        MPI_Bcast(data, n * d, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return data;
}

void generate_pq_codebook(size_t m, size_t ksub, size_t dsub, const string& filename, int rank) {
    if (rank == 0) {
        std::vector<float> codebook(m * ksub * dsub);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0, 1.0);
        for (size_t i = 0; i < codebook.size(); ++i) codebook[i] = dis(gen);
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(codebook.data()), codebook.size() * sizeof(float));
        out.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void generate_pq_centroids(size_t nlist, size_t m, const string& filename, int rank) {
    if (rank == 0) {
        std::vector<float> centroids(nlist * m);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 255.0);
        for (size_t i = 0; i < centroids.size(); ++i) centroids[i] = dis(gen);
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(centroids.data()), centroids.size() * sizeof(float));
        out.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    std::string data_path = "anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim, rank);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d, rank);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim, rank);

    if (!test_query || !test_gt || !base) { MPI_Finalize(); return 1; }

    test_number = std::min(test_number, size_t(500));
    const size_t k = 10, warmup = 30;

    // 精选高性能参数
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> configs = {
        {128, 16, 16, 256}, {128, 32, 16, 256}, {256, 16, 24, 512}, {256, 32, 24, 512}
    };

    if (rank == 0) {
        std::cout << "nlist,nprobe,m,ksub,recall,latency_us,build_time_ms,algorithm\n";
    }

    for (auto& config : configs) {
        size_t nlist = std::get<0>(config);
        size_t nprobe = std::get<1>(config);
        size_t m = std::get<2>(config);
        size_t ksub = std::get<3>(config);
        
        if (vecdim % m != 0) continue;
        size_t dsub = vecdim / m;

        generate_pq_codebook(m, ksub, dsub, "temp_pq_codebook.bin", rank);
        generate_pq_centroids(nlist, m, "temp_pq_centroids.bin", rank);

        auto start_build = std::chrono::high_resolution_clock::now();
        MPIPQIVFIndex index(vecdim, nlist, m, ksub, true);
        
        if (!index.load_pq_codebook("temp_pq_codebook.bin") || !index.load_pq_centroids("temp_pq_centroids.bin")) continue;
        
        index.build_index(base, base_number);
        auto end_build = std::chrono::high_resolution_clock::now();
        int64_t build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count();

        // 暖机
        for (size_t i = 0; i < warmup; ++i) {
            auto res = index.mpi_search(test_query + i * vecdim, k, nprobe, 200);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // 测试
        float total_recall = 0.0f;
        int64_t total_latency = 0;

        for(size_t i = 0; i < test_number; ++i) {
            struct timeval val, newVal;
            gettimeofday(&val, NULL);
            auto res = index.mpi_search(test_query + i * vecdim, k, nprobe, 200);
            gettimeofday(&newVal, NULL);
            int64_t diff = (newVal.tv_sec * 1000000 + newVal.tv_usec) - (val.tv_sec * 1000000 + val.tv_usec);

            float recall = 0.0f;
            if (rank == 0) {
                std::set<uint32_t> gtset;
                for(size_t j = 0; j < k; ++j) gtset.insert(test_gt[j + i * test_gt_d]);
                
                size_t acc = 0;
                std::priority_queue<std::pair<float, uint32_t>> temp_res = res;
                while (!temp_res.empty()) {   
                    if(gtset.find(temp_res.top().second) != gtset.end()) ++acc;
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
            std::cout << nlist << "," << nprobe << "," << m << "," << ksub << "," 
                     << std::fixed << std::setprecision(4) << avg_recall << "," << avg_latency 
                     << "," << build_time << ",PQ-IVF\n";
        }
    }

    if (rank == 0) {
        std::remove("temp_pq_codebook.bin");
        std::remove("temp_pq_centroids.bin");
    }
    delete[] test_query; delete[] test_gt; delete[] base;
    MPI_Finalize();
    return 0;
}
EOF

# 创建精简的IVF-PQ测试  
cat > test_ivf_pq_focused.cc << 'EOF'
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <random>
#include "mpi_ivf_pq.h"

using namespace std;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d, int rank) {
    T* data = nullptr;
    if (rank == 0) {
        std::ifstream fin(data_path, std::ios::in | std::ios::binary);
        fin.read((char*)&n, 4);
        fin.read((char*)&d, 4);
        data = new T[n * d];
        for(size_t i = 0; i < n; ++i){
            fin.read(((char*)data + i*d*sizeof(T)), d*sizeof(T));
        }
        fin.close();
    }
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) data = new T[n * d];
    if (std::is_same<T, float>::value) {
        MPI_Bcast(data, n * d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else if (std::is_same<T, int>::value) {
        MPI_Bcast(data, n * d, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return data;
}

void generate_centroids(const float* base_data, size_t base_number, size_t vecdim, size_t nlist, const string& filename, int rank) {
    if (rank == 0) {
        std::vector<float> centroids(nlist * vecdim);
        for (size_t i = 0; i < nlist; ++i) {
            size_t idx = i * (base_number / nlist) % base_number;
            std::copy(base_data + idx * vecdim, base_data + (idx + 1) * vecdim, centroids.begin() + i * vecdim);
        }
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(centroids.data()), centroids.size() * sizeof(float));
        out.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void generate_pq_codebook(size_t m, size_t ksub, size_t dsub, const string& filename, int rank) {
    if (rank == 0) {
        std::vector<float> codebook(m * ksub * dsub);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0, 1.0);
        for (size_t i = 0; i < codebook.size(); ++i) codebook[i] = dis(gen);
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(codebook.data()), codebook.size() * sizeof(float));
        out.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    std::string data_path = "anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim, rank);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d, rank);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim, rank);

    if (!test_query || !test_gt || !base) { MPI_Finalize(); return 1; }

    test_number = std::min(test_number, size_t(500));
    const size_t k = 10, warmup = 30;

    // 与PQ-IVF相同的参数配置
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> configs = {
        {128, 16, 16, 256}, {128, 32, 16, 256}, {256, 16, 24, 512}, {256, 32, 24, 512}
    };

    if (rank == 0) {
        std::cout << "nlist,nprobe,m,ksub,recall,latency_us,build_time_ms,algorithm\n";
    }

    for (auto& config : configs) {
        size_t nlist = std::get<0>(config);
        size_t nprobe = std::get<1>(config);
        size_t m = std::get<2>(config);
        size_t ksub = std::get<3>(config);
        
        if (vecdim % m != 0) continue;
        size_t dsub = vecdim / m;

        generate_centroids(base, base_number, vecdim, nlist, "temp_centroids.bin", rank);
        generate_pq_codebook(m, ksub, dsub, "temp_codebook.bin", rank);

        auto start_build = std::chrono::high_resolution_clock::now();
        MPIIVFPQIndex index(vecdim, nlist, m, ksub, true);
        
        if (!index.load_centroids("temp_centroids.bin") || !index.load_codebook("temp_codebook.bin")) continue;
        
        index.build_index(base, base_number);
        auto end_build = std::chrono::high_resolution_clock::now();
        int64_t build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count();

        // 暖机
        for (size_t i = 0; i < warmup; ++i) {
            auto res = index.mpi_search(test_query + i * vecdim, k, nprobe, 200);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // 测试
        float total_recall = 0.0f;
        int64_t total_latency = 0;

        for(size_t i = 0; i < test_number; ++i) {
            struct timeval val, newVal;
            gettimeofday(&val, NULL);
            auto res = index.mpi_search(test_query + i * vecdim, k, nprobe, 200);
            gettimeofday(&newVal, NULL);
            int64_t diff = (newVal.tv_sec * 1000000 + newVal.tv_usec) - (val.tv_sec * 1000000 + val.tv_usec);

            float recall = 0.0f;
            if (rank == 0) {
                std::set<uint32_t> gtset;
                for(size_t j = 0; j < k; ++j) gtset.insert(test_gt[j + i * test_gt_d]);
                
                size_t acc = 0;
                std::priority_queue<std::pair<float, uint32_t>> temp_res = res;
                while (!temp_res.empty()) {   
                    if(gtset.find(temp_res.top().second) != gtset.end()) ++acc;
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
            std::cout << nlist << "," << nprobe << "," << m << "," << ksub << "," 
                     << std::fixed << std::setprecision(4) << avg_recall << "," << avg_latency 
                     << "," << build_time << ",IVF-PQ\n";
        }
    }

    if (rank == 0) {
        std::remove("temp_centroids.bin");
        std::remove("temp_codebook.bin");
    }
    delete[] test_query; delete[] test_gt; delete[] base;
    MPI_Finalize();
    return 0;
}
EOF

echo "编译精简测试程序..."
$MPI_CC $CXX_FLAGS $INCLUDE_FLAGS -o test_pq_ivf_focused test_pq_ivf_focused.cc $LINK_FLAGS
$MPI_CC $CXX_FLAGS $INCLUDE_FLAGS -o test_ivf_pq_focused test_ivf_pq_focused.cc $LINK_FLAGS

echo "运行PQ-IVF专注测试..."
mpirun -np $MPI_PROCESSES ./test_pq_ivf_focused > "${RESULTS_DIR}/pq_ivf_focused.csv"

echo "运行IVF-PQ专注测试..."  
mpirun -np $MPI_PROCESSES ./test_ivf_pq_focused > "${RESULTS_DIR}/ivf_pq_focused.csv"

# 合并结果
cd $RESULTS_DIR
cat pq_ivf_focused.csv > combined_results.csv
tail -n +2 ivf_pq_focused.csv >> combined_results.csv

echo ""
echo "=== 论文对比分析结果 ==="

# 简单的分析脚本
cat > analyze.py << 'EOF'
import csv

print("PQ→IVF vs IVF→PQ 高性能参数对比分析")
print("=" * 50)

pq_results = []
ivf_results = []

with open('combined_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['algorithm'] == 'PQ-IVF':
            pq_results.append(row)
        else:
            ivf_results.append(row)

print(f"测试配置数: {len(pq_results)} 个参数组合")
print()

# 对比分析
print("详细对比结果:")
print("配置 | PQ→IVF召回率 | IVF→PQ召回率 | 召回率提升 | PQ→IVF延迟 | IVF→PQ延迟")
print("-" * 80)

total_pq_recall = 0
total_ivf_recall = 0  
total_pq_latency = 0
total_ivf_latency = 0

for i, pq in enumerate(pq_results):
    if i < len(ivf_results):
        ivf = ivf_results[i]
        pq_recall = float(pq['recall'])
        ivf_recall = float(ivf['recall'])
        improvement = ((ivf_recall - pq_recall) / pq_recall) * 100
        
        config = f"({pq['nlist']},{pq['nprobe']},{pq['m']},{pq['ksub']})"
        print(f"{config:<15} | {pq_recall:.4f}      | {ivf_recall:.4f}      | +{improvement:5.1f}%    | {pq['latency_us']:>6}μs | {ivf['latency_us']:>6}μs")
        
        total_pq_recall += pq_recall
        total_ivf_recall += ivf_recall
        total_pq_latency += int(pq['latency_us'])
        total_ivf_latency += int(ivf['latency_us'])

print()
print("总体性能对比:")
avg_pq_recall = total_pq_recall / len(pq_results)
avg_ivf_recall = total_ivf_recall / len(ivf_results)  
avg_pq_latency = total_pq_latency / len(pq_results)
avg_ivf_latency = total_ivf_latency / len(ivf_results)

overall_improvement = ((avg_ivf_recall - avg_pq_recall) / avg_pq_recall) * 100

print(f"PQ→IVF: 平均召回率 {avg_pq_recall:.4f}, 平均延迟 {avg_pq_latency:.0f}μs")
print(f"IVF→PQ: 平均召回率 {avg_ivf_recall:.4f}, 平均延迟 {avg_ivf_latency:.0f}μs")
print(f"整体召回率提升: +{overall_improvement:.1f}%")
print(f"延迟开销: +{avg_ivf_latency - avg_pq_latency:.0f}μs")

# 找出最佳配置
best_ivf_idx = max(range(len(ivf_results)), key=lambda i: float(ivf_results[i]['recall']))
best_config = ivf_results[best_ivf_idx]
corresponding_pq = pq_results[best_ivf_idx]

print()
print("最佳召回率配置:")
print(f"参数: nlist={best_config['nlist']}, nprobe={best_config['nprobe']}, m={best_config['m']}, ksub={best_config['ksub']}")
print(f"IVF→PQ召回率: {best_config['recall']}")
print(f"相比PQ→IVF提升: {((float(best_config['recall']) - float(corresponding_pq['recall'])) / float(corresponding_pq['recall']) * 100):.1f}%")
EOF

if command -v python3 &> /dev/null; then
    python3 analyze.py
else
    echo "请安装Python3查看详细分析"
    echo "结果文件: combined_results.csv"
fi

cd ..

echo ""
echo "专注对比测试完成!"
echo "结果保存在: $RESULTS_DIR"
echo "完成时间: $(date)"

# 清理
rm -f test_pq_ivf_focused.cc test_ivf_pq_focused.cc
rm -f test_pq_ivf_focused test_ivf_pq_focused

