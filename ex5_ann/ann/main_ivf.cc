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
#include "flat_ivf.h"
// 可以自行添加需要的头文件

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


int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 10000;

    const size_t k = 10;
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

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

    
    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        auto res = flat_ivf_search(base, test_query + i*vecdim, base_number, vecdim, k, 
                                   centroids.data(), nlist, invlists, nprobe);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "\n=== IVF-CPU Search Results ===\n";
    std::cout << "Dataset: DEEP100K\n";
    std::cout << "Number of clusters (nlist): " << nlist << "\n";
    std::cout << "Number of probes (nprobe): " << nprobe << "\n";
    std::cout << "Top-k: " << k << "\n";
    std::cout << "Test queries: " << test_number << "\n";
    std::cout << "Average recall: " << std::fixed << std::setprecision(4) 
              << avg_recall / test_number << "\n";
    std::cout << "Average latency (us): " << std::fixed << std::setprecision(2) 
              << avg_latency / test_number << "\n";
    return 0;
}
