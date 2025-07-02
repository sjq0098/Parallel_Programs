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
#include "gpu_ivf.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

int main(int argc, char *argv[]) {
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    test_number = 10000; // 只测试前2000条查询
    const size_t k = 10;
    const size_t batch_size = 2000; // 设置批次大小，为IVF优化调整
    const size_t nprobe = 8; // IVF参数：探测的簇数量
    const size_t nlist = 256; // IVF参数：簇的总数，对应256个簇的码本

    // 加载IVF码本
    std::string ivf_path = "file/";
    std::vector<float> centroids;
    std::vector<std::vector<uint32_t>> invlists;
    
    try {
        // 加载簇中心，使用256个簇的码本
        centroids = load_ivf_centroids(ivf_path + "ivf_flat_centroids_256.fbin", nlist, vecdim);
        std::cerr << "Successfully loaded " << nlist << " centroids with dimension " << vecdim << "\n";
        
        // 加载倒排列表
        invlists = load_ivf_invlists(ivf_path + "ivf_flat_invlists_256.bin", nlist);
        std::cerr << "Successfully loaded " << nlist << " inverted lists\n";
        
        // 打印一些统计信息
        size_t total_vectors = 0;
        size_t max_cluster_size = 0;
        size_t min_cluster_size = SIZE_MAX;
        for (const auto& invlist : invlists) {
            total_vectors += invlist.size();
            max_cluster_size = std::max(max_cluster_size, invlist.size());
            min_cluster_size = std::min(min_cluster_size, invlist.size());
        }
        std::cerr << "Total vectors in invlists: " << total_vectors << "\n";
        std::cerr << "Max cluster size: " << max_cluster_size << "\n";
        std::cerr << "Min cluster size: " << min_cluster_size << "\n";
        std::cerr << "Average cluster size: " << (float)total_vectors / nlist << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading IVF codebook: " << e.what() << "\n";
        return -1;
    }

    std::vector<SearchResult> results(test_number);

    // 批处理查询
    for (size_t i = 0; i < test_number; i += batch_size) {
        size_t current_batch_size = std::min(batch_size, test_number - i);
        
        struct timeval val;
        gettimeofday(&val, NULL);

        // 调用GPU版本的IVF搜索
        auto res = ivf_search_gpu(base, test_query + i * vecdim, base_number, vecdim, 
                                  current_batch_size, k, centroids, nlist, invlists, nprobe);

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
            int64_t avg_us = static_cast<int64_t>(diff) / static_cast<int64_t>(current_batch_size);
            results[i + b] = { recall, avg_us };
        }
        
        // 每处理100个batch打印一次进度
        if ((i / batch_size) % 100 == 0) {
            std::cerr << "Processed " << i + current_batch_size << "/" << test_number << " queries\n";
        }
    }

    // 计算平均recall和latency
    float avg_recall = 0, avg_latency = 0;
    for (int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    std::cout << "\n=== IVF-GPU Search Results ===\n";
    std::cout << "Dataset: DEEP100K\n";
    std::cout << "Number of clusters (nlist): " << nlist << "\n";
    std::cout << "Number of probes (nprobe): " << nprobe << "\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "Top-k: " << k << "\n";
    std::cout << "Test queries: " << test_number << "\n";
    std::cout << "Average recall: " << std::fixed << std::setprecision(4) 
              << avg_recall / test_number << "\n";
    std::cout << "Average latency (us): " << std::fixed << std::setprecision(2) 
              << avg_latency / test_number << "\n";
    
    // 清理内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    return 0;
} 