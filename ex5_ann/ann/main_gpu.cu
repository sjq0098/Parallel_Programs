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
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan_gpu.h" // 修改后的头文件
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace hnswlib;

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

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

int main(int argc, char *argv[]) {
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    test_number = 2000; // 只测试前2000条查询
    const size_t k = 10;
    const size_t batch_size = 1024; // 设置批次大小，可调整

    std::vector<SearchResult> results(test_number);

    // 批处理查询
    for (size_t i = 0; i < test_number; i += batch_size) {
        size_t current_batch_size = std::min(batch_size, test_number - i);
        
        struct timeval val;
        gettimeofday(&val, NULL);

        // 调用GPU版本的flat_search
        auto res = flat_search_gpu(base, test_query + i * vecdim, base_number, vecdim, current_batch_size, k);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * 1000000 + newVal.tv_usec) - (val.tv_sec * 1000000 + val.tv_usec);

        // 计算recall
        for (size_t b = 0; b < current_batch_size; ++b) {
            std::set<uint32_t> gtset;
            for (int j = 0; j < k; ++j) {
                gtset.insert(test_gt[j + (i + b) * test_gt_d]);
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
            int64_t avg_us = static_cast<int64_t>(diff)/ static_cast<int64_t>(current_batch_size);
            results[i + b] = { recall, avg_us };

        }
    }

    // 计算平均recall和latency
    float avg_recall = 0, avg_latency = 0;
    for (int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    std::cout << "average recall: " << avg_recall / test_number << "\n";
    std::cout << "average latency (us): " << avg_latency / test_number << "\n";
    return 0;
}