#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h" 

// 从 main.cc 复制的 LoadData 函数
template<typename T>
T *LoadData(const std::string& data_path, size_t& n, size_t& d)
{
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open data file " << data_path << std::endl;
        exit(1);
    }
    fin.read(reinterpret_cast<char*>(&n), 4);
    fin.read(reinterpret_cast<char*>(&d), 4);
    T* data = new T[n * d];
    size_t sz = sizeof(T);
    for (size_t i = 0; i < n; ++i) {
        fin.read(reinterpret_cast<char*>(data + i * d), d * sz);
    }
    fin.close();

    std::cout << "Loaded data from " << data_path << "\n"
              << "Dimension: " << d
              << ", Number: " << n
              << ", Size per element: " << sizeof(T) << "\n";
    return data;
}

int main(int argc, char *argv[])
{
    size_t base_number = 0;
    size_t vecdim      = 0;

    // ----------- 配置参数 -----------
    std::string data_root_path   = "anndata/";            
    std::string base_data_file   = "DEEP100K.base.100k.fbin";
    std::string output_index_dir = "files/";              
    // 要测试的 HNSW 参数组合
    std::vector<int> M_values            = {128,256};
    std::vector<int> efConstruction_vals = {100, 150, 200};
    // -------------------------------

    std::string base_data_path = data_root_path + base_data_file;
    float* base_vectors = LoadData<float>(base_data_path, base_number, vecdim);

    if (!base_vectors || base_number == 0 || vecdim == 0) {
        std::cerr << "Failed to load base data. Exiting." << std::endl;
        return 1;
    }

    // 使用 InnerProductSpace，与示例一致
    hnswlib::InnerProductSpace space(vecdim);

    for (int M : M_values) {
        for (int efC : efConstruction_vals) {
            std::cout << "\nBuilding HNSW index with M=" << M
                      << ", efConstruction=" << efC << std::endl;

            hnswlib::HierarchicalNSW<float>* appr_alg = nullptr;
            try {
                appr_alg = new hnswlib::HierarchicalNSW<float>(
                    &space, base_number, M, efC
                );
            } catch (const std::exception& e) {
                std::cerr << "Error initializing HNSW: "
                          << e.what() << std::endl;
                continue;
            }

            std::cout << "Adding points to HNSW graph..." << std::endl;
            auto t1 = std::chrono::high_resolution_clock::now();

            // 添加第 0 个点
            if (base_number > 0) {
                appr_alg->addPoint(base_vectors, 0);
            }
            // 并行添加其余点
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 1; i < base_number; ++i) {
                try {
                    appr_alg->addPoint(base_vectors + i * vecdim, i);
                } catch (const std::exception& e) {
                    #pragma omp critical
                    std::cerr << "Error adding point " << i << ": "
                              << e.what() << std::endl;
                }
            }

            auto t2 = std::chrono::high_resolution_clock::now();
            double build_time = std::chrono::duration<double>(t2 - t1).count();
            std::cout << "Finished adding points. Build time: "
                      << std::fixed << std::setprecision(3)
                      << build_time << " seconds.\n";

            // 构造索引文件名并保存
            std::ostringstream idx_ss;
            idx_ss << output_index_dir
                   << "hnsw_M" << M << "_efC" << efC << ".index";
            std::string index_save_path = idx_ss.str();

            std::cout << "Saving index to " << index_save_path << " ... ";
            try {
                appr_alg->saveIndex(index_save_path);
                std::cout << "done.\n";
            } catch (const std::exception& e) {
                std::cerr << "\nError saving index: "
                          << e.what() << std::endl;
            }
            delete appr_alg;
        }
    }

    delete[] base_vectors;
    std::cout << "\nAll HNSW index building tasks completed." << std::endl;
    return 0;
}
