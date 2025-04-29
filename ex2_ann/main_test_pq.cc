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
#include "simd_utils.h"
#include "pq_index.h"
#include "pq_simd_ops.h"

using namespace std;

// 用于测试的计时器
class Timer {
public:
    Timer() { start(); }
    
    void start() {
        begin = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - begin).count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point begin;
};

// 加载数据
template<typename T>
T* LoadData(std::string data_path, size_t& n, size_t& d) {
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr << "加载数据 " << data_path << "\n";
    std::cerr << "维度: " << d << "  数量:" << n << "  每元素大小:" << sizeof(T) << "\n";

    return data;
}

// 测试不同M值的PQ搜索性能
void test_pq_performance(float* base, float* query, int* gt, 
                        size_t base_number, size_t query_number, size_t vecdim, 
                        size_t gt_dim, const vector<int>& m_values) {
    const size_t k = 10;
    
    cout << "\n======= PQ性能测试 =======\n";
    cout << "测试数据: " << base_number << " 基础向量, " << query_number << " 查询向量, 维度 " << vecdim << "\n";
    cout << "\n子空间数(M)\t平均召回率\t平均延迟(ms)\t相对加速比\n";
    cout << "--------------------------------------------------------------\n";
    
    // 基准测试：使用暴力搜索
    double base_time = 0;
    {
        Timer timer;
        
        for (size_t q = 0; q < query_number; q++) {
            std::priority_queue<std::pair<float, uint32_t>> result;
            
            for (size_t i = 0; i < base_number; i++) {
                float ip = dot_product_avx(base + i*vecdim, query + q*vecdim, vecdim);
                float dis = 1.0f - ip;
                
                if (result.size() < k) {
                    result.push({dis, i});
                } else if (dis < result.top().first) {
                    result.pop();
                    result.push({dis, i});
                }
            }
        }
        
        base_time = timer.elapsed() / query_number * 1000; // 转换为毫秒
    }
    
    cout << "暴力搜索\t1.0000\t\t" << fixed << setprecision(4) << base_time << "\t1.00x\n";
    
    // 测试不同M值的PQ
    for (int m : m_values) {
        if (vecdim % m != 0) {
            cout << m << "\t\t跳过 (维度不能被M整除)\n";
            continue;
        }
        
        // 加载PQ索引
        ProductQuantizer pq;
        std::string codebook_file = "files/pq" + std::to_string(m) + "_codebook.bin";
        std::string codes_file = "files/pq" + std::to_string(m) + "_codes.bin";
        
        if (!pq.load_codebook(codebook_file)) {
            cerr << "无法加载码本: " << codebook_file << "\n";
            continue;
        }
        
        if (!pq.load_codes(codes_file)) {
            cerr << "无法加载编码: " << codes_file << "\n";
            continue;
        }
        
        // 测试PQ搜索性能和召回率
        double total_time = 0;
        float total_recall = 0;
        
        for (size_t q = 0; q < query_number; q++) {
            Timer timer;
            
            // 执行PQ搜索
            auto results = pq.search(query + q*vecdim, k);
            
            total_time += timer.elapsed();
            
            // 计算召回率
            std::set<uint32_t> gtset;
            for (int j = 0; j < k; ++j) {
                int t = gt[j + q*gt_dim];
                gtset.insert(t);
            }
            
            size_t acc = 0;
            std::vector<uint32_t> result_ids;
            while (!results.empty()) {
                int x = results.top().second;
                if (gtset.find(x) != gtset.end()) {
                    ++acc;
                }
                results.pop();
            }
            
            float recall = (float)acc/k;
            total_recall += recall;
        }
        
        double avg_time = total_time / query_number * 1000; // 转换为毫秒
        float avg_recall = total_recall / query_number;
        double speedup = base_time / avg_time;
        
        cout << m << "\t\t" << fixed << setprecision(4) 
             << avg_recall << "\t\t" << avg_time 
             << "\t" << speedup << "x\n";
    }
    
    // 测试OPQ
    cout << "\n======= OPQ性能测试 =======\n";
    cout << "子空间数(M)\t平均召回率\t平均延迟(ms)\t相对加速比\n";
    cout << "--------------------------------------------------------------\n";
    
    for (int m : m_values) {
        if (vecdim % m != 0) {
            cout << m << "\t\t跳过 (维度不能被M整除)\n";
            continue;
        }
        
        // 加载OPQ索引
        ProductQuantizer opq;
        std::string codebook_file = "files/opq" + std::to_string(m) + "_codebook.bin";
        std::string codes_file = "files/opq" + std::to_string(m) + "_codes.bin";
        std::string rotation_file = "files/opq" + std::to_string(m) + "_rotation.bin";
        
        if (!opq.load_codebook(codebook_file)) {
            cerr << "无法加载码本: " << codebook_file << "\n";
            continue;
        }
        
        if (!opq.load_codes(codes_file)) {
            cerr << "无法加载编码: " << codes_file << "\n";
            continue;
        }
        
        opq.load_rotation(rotation_file); // 可选，如果加载失败会继续执行
        
        // 测试OPQ搜索性能和召回率
        double total_time = 0;
        float total_recall = 0;
        
        for (size_t q = 0; q < query_number; q++) {
            Timer timer;
            
            // 执行OPQ搜索
            auto results = opq.search(query + q*vecdim, k);
            
            total_time += timer.elapsed();
            
            // 计算召回率
            std::set<uint32_t> gtset;
            for (int j = 0; j < k; ++j) {
                int t = gt[j + q*gt_dim];
                gtset.insert(t);
            }
            
            size_t acc = 0;
            while (!results.empty()) {
                int x = results.top().second;
                if (gtset.find(x) != gtset.end()) {
                    ++acc;
                }
                results.pop();
            }
            
            float recall = (float)acc/k;
            total_recall += recall;
        }
        
        double avg_time = total_time / query_number * 1000; // 转换为毫秒
        float avg_recall = total_recall / query_number;
        double speedup = base_time / avg_time;
        
        cout << m << " (OPQ)\t\t" << fixed << setprecision(4) 
             << avg_recall << "\t\t" << avg_time 
             << "\t" << speedup << "x\n";
    }
    
    // 测试PQ+精确重排序的性能
    cout << "\n======= PQ+重排序性能测试 =======\n";
    cout << "配置\t\t平均召回率\t平均延迟(ms)\t相对加速比\n";
    cout << "--------------------------------------------------------------\n";
    
    // 测试不同重排序候选数量
    vector<int> rerank_ks = {50, 100, 200, 500};
    
    for (int rerank_k : rerank_ks) {
        // 使用PQ16
        ProductQuantizer pq;
        std::string codebook_file = "files/pq16_codebook.bin";
        std::string codes_file = "files/pq16_codes.bin";
        
        if (!pq.load_codebook(codebook_file) || !pq.load_codes(codes_file)) {
            cerr << "无法加载PQ16索引\n";
            continue;
        }
        
        // 测试性能和召回率
        double total_time = 0;
        float total_recall = 0;
        
        for (size_t q = 0; q < query_number; q++) {
            Timer timer;
            
            // 执行PQ搜索+重排序
            auto results = pq.search_with_rerank(query + q*vecdim, base, k, rerank_k);
            
            total_time += timer.elapsed();
            
            // 计算召回率
            std::set<uint32_t> gtset;
            for (int j = 0; j < k; ++j) {
                int t = gt[j + q*gt_dim];
                gtset.insert(t);
            }
            
            size_t acc = 0;
            while (!results.empty()) {
                int x = results.top().second;
                if (gtset.find(x) != gtset.end()) {
                    ++acc;
                }
                results.pop();
            }
            
            float recall = (float)acc/k;
            total_recall += recall;
        }
        
        double avg_time = total_time / query_number * 1000; // 转换为毫秒
        float avg_recall = total_recall / query_number;
        double speedup = base_time / avg_time;
        
        cout << "PQ16+R" << rerank_k << "\t" << fixed << setprecision(4) 
             << avg_recall << "\t\t" << avg_time 
             << "\t" << speedup << "x\n";
    }
}

int main(int argc, char* argv[]) {
    // 设置线程数
    int num_threads = omp_get_max_threads();
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);
    
    cout << "使用线程数: " << num_threads << endl;
    
    // 加载数据
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    
    std::string data_path = "anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 限制测试数量，加速测试过程
    size_t query_limit = 2000;
    if (test_number > query_limit) {
        test_number = query_limit;
        cout << "限制测试查询数量为: " << test_number << endl;
    }
    
    // 测试不同M值的PQ
    vector<int> m_values = {4, 8, 16, 32};
    test_pq_performance(base, test_query, test_gt, base_number, test_number, vecdim, test_gt_d, m_values);
    
    // 释放内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    return 0;
} 