#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <unordered_map>
#include <set>
#include <cstring>

// 读取 IVF centroids (GPU版本)
std::vector<float> gpu_load_ivf_centroids(const std::string& filename, size_t expected_nlist, size_t vecdim) {
    std::vector<float> centroids(expected_nlist * vecdim);
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) throw std::runtime_error("无法打开 centroids 文件: " + filename);

    int read_nlist, read_dim;
    fin.read((char*)&read_nlist, 4);
    fin.read((char*)&read_dim, 4);

    if (read_nlist != expected_nlist || read_dim != vecdim)
        throw std::runtime_error("centroids 文件维度或簇数与期望不符");

    fin.read((char*)centroids.data(), sizeof(float) * expected_nlist * vecdim);
    if (!fin) throw std::runtime_error("读取 centroids 数据失败");

    return centroids;
}

// 读取 IVF 倒排列表 (GPU版本)
std::vector<std::vector<uint32_t>> gpu_load_ivf_invlists(const std::string& filename, size_t expected_nlist) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) throw std::runtime_error("无法打开 invlists 文件: " + filename);

    int read_nlist;
    fin.read((char*)&read_nlist, 4);
    if (read_nlist != expected_nlist)
        throw std::runtime_error("invlists 的 nlist 与期望值不一致");

    std::vector<std::vector<uint32_t>> invlists(expected_nlist);
    for (int i = 0; i < read_nlist; ++i) {
        int L;
        fin.read((char*)&L, 4);
        invlists[i].resize(L);
        fin.read((char*)invlists[i].data(), sizeof(uint32_t) * L);
        if (!fin) throw std::runtime_error("读取 invlist[" + std::to_string(i) + "] 失败");
    }

    return invlists;
}

// CUDA kernel: 计算查询向量到簇中心的距离
__global__ void compute_query_centroid_distances(
    float* queries, float* centroids, float* distances,
    size_t batch_size, size_t nlist, size_t vecdim) {
    
    int query_idx = blockIdx.x;
    int centroid_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (query_idx >= batch_size || centroid_idx >= nlist) return;
    
    float dot = 0.0f;
    for (int d = 0; d < vecdim; ++d) {
        dot += queries[query_idx * vecdim + d] * centroids[centroid_idx * vecdim + d];
    }
    distances[query_idx * nlist + centroid_idx] = 1.0f - dot; // IP距离
}

// CUDA kernel: 找到每个查询的top-nprobe个簇
__global__ void find_top_clusters_kernel(
    float* distances, uint32_t* cluster_ids, 
    size_t batch_size, size_t nlist, size_t nprobe) {
    
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= batch_size) return;
    
    // 使用简单的选择排序找到top-nprobe
    float* query_distances = distances + query_idx * nlist;
    uint32_t* query_clusters = cluster_ids + query_idx * nprobe;
    
    // 创建簇ID数组并初始化
    uint32_t cluster_indices[512]; // 假设最大nlist不超过512
    for (int i = 0; i < nlist; ++i) {
        cluster_indices[i] = i;
    }
    
    // 选择排序，同时跟踪簇ID
    for (int i = 0; i < nprobe && i < nlist; ++i) {
        int min_idx = i;
        for (int j = i + 1; j < nlist; ++j) {
            if (query_distances[j] < query_distances[min_idx]) {
                min_idx = j;
            }
        }
        // 交换距离和簇ID
        if (min_idx != i) {
            float temp_dist = query_distances[i];
            query_distances[i] = query_distances[min_idx];
            query_distances[min_idx] = temp_dist;
            
            uint32_t temp_id = cluster_indices[i];
            cluster_indices[i] = cluster_indices[min_idx];
            cluster_indices[min_idx] = temp_id;
        }
        query_clusters[i] = cluster_indices[i];
    }
}

// GPU Top-K 选择kernel
__global__ void gpu_topk_for_cluster(float* distances, uint32_t* vector_ids, 
                                     uint32_t* results_indices, float* results_distances, 
                                     size_t num_queries, size_t num_vectors, size_t k) {
    int query_idx = blockIdx.x;
    if (query_idx >= num_queries) return;
    
    // 每个查询的距离起始位置
    float* query_distances = distances + query_idx * num_vectors;
    float* query_results_dist = results_distances + query_idx * k;
    uint32_t* query_results_idx = results_indices + query_idx * k;
    
    // 只让thread 0处理，避免竞争条件
    if (threadIdx.x == 0) {
        // 初始化结果
        for (int i = 0; i < k && i < num_vectors; i++) {
            query_results_dist[i] = query_distances[i];
            query_results_idx[i] = vector_ids[i];
        }
        
        // 使用选择排序找前k小的元素
        for (int i = 0; i < k && i < num_vectors; i++) {
            int min_idx = i;
            for (int j = i + 1; j < num_vectors; j++) {
                if (query_distances[j] < query_distances[min_idx]) {
                    min_idx = j;
                }
            }
            
            if (i < k && min_idx < num_vectors) {
                query_results_dist[i] = query_distances[min_idx];
                query_results_idx[i] = vector_ids[min_idx];
                
                // 标记已选择的元素
                query_distances[min_idx] = 1e10f;
            }
        }
    }
}

// 将内积转换为距离的kernel
__global__ void convert_and_copy(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f - input[idx];
    }
}

struct BatchClusterInfo {
    uint32_t cluster_id;
    std::vector<uint32_t> query_indices; // 该簇对应的查询索引
};

struct ClusterMatrix {
    float* d_vectors; // GPU上该簇的向量矩阵
    uint32_t* h_vector_ids; // CPU上该簇向量在base中的ID
    size_t num_vectors;
    size_t dim;
};

class GPUIVFIndex {
private:
    float* d_base;
    float* d_centroids;
    size_t base_number;
    size_t vecdim;
    size_t nlist;
    std::vector<std::vector<uint32_t>> h_invlists;
    std::vector<ClusterMatrix> cluster_matrices;
    cublasHandle_t cublas_handle;
    
public:
    GPUIVFIndex(float* base, size_t base_num, size_t vec_dim, 
                const std::vector<float>& centroids, size_t num_list,
                const std::vector<std::vector<uint32_t>>& invlists)
        : base_number(base_num), vecdim(vec_dim), nlist(num_list), h_invlists(invlists) {
        
        // 创建cuBLAS句柄
        cublasCreate(&cublas_handle);
        
        // 分配GPU内存
        cudaMalloc(&d_base, base_number * vecdim * sizeof(float));
        cudaMalloc(&d_centroids, nlist * vecdim * sizeof(float));
        
        // 拷贝数据到GPU
        cudaMemcpy(d_base, base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_centroids, centroids.data(), nlist * vecdim * sizeof(float), cudaMemcpyHostToDevice);
        
        // 组织簇矩阵
        organize_cluster_matrices(base);
    }
    
    ~GPUIVFIndex() {
        // 释放GPU内存
        cudaFree(d_base);
        cudaFree(d_centroids);
        
        for (auto& cluster : cluster_matrices) {
            cudaFree(cluster.d_vectors);
            delete[] cluster.h_vector_ids;
        }
        
        cublasDestroy(cublas_handle);
    }
    
    void organize_cluster_matrices(float* base) {
        cluster_matrices.resize(nlist);
        
        for (size_t cluster_id = 0; cluster_id < nlist; ++cluster_id) {
            const auto& invlist = h_invlists[cluster_id];
            size_t num_vectors = invlist.size();
            
            if (num_vectors == 0) {
                cluster_matrices[cluster_id].d_vectors = nullptr;
                cluster_matrices[cluster_id].h_vector_ids = nullptr;
                cluster_matrices[cluster_id].num_vectors = 0;
                continue;
            }
            
            // 分配GPU内存存储该簇的向量矩阵
            cudaMalloc(&cluster_matrices[cluster_id].d_vectors, 
                       num_vectors * vecdim * sizeof(float));
            
            // 分配CPU内存存储向量ID
            cluster_matrices[cluster_id].h_vector_ids = new uint32_t[num_vectors];
            
            // 拷贝该簇的向量到GPU
            std::vector<float> cluster_data(num_vectors * vecdim);
            for (size_t i = 0; i < num_vectors; ++i) {
                uint32_t vec_id = invlist[i];
                cluster_matrices[cluster_id].h_vector_ids[i] = vec_id;
                memcpy(cluster_data.data() + i * vecdim, 
                       base + vec_id * vecdim, 
                       vecdim * sizeof(float));
            }
            
            cudaMemcpy(cluster_matrices[cluster_id].d_vectors, 
                       cluster_data.data(), 
                       num_vectors * vecdim * sizeof(float), 
                       cudaMemcpyHostToDevice);
            
            cluster_matrices[cluster_id].num_vectors = num_vectors;
            cluster_matrices[cluster_id].dim = vecdim;
        }
    }
    
    std::vector<BatchClusterInfo> group_queries_by_clusters(
        float* d_queries, size_t batch_size, size_t nprobe) {
        
        // 分配临时GPU内存
        float* d_distances;
        uint32_t* d_cluster_ids;
        cudaMalloc(&d_distances, batch_size * nlist * sizeof(float));
        cudaMalloc(&d_cluster_ids, batch_size * nprobe * sizeof(uint32_t));
        
        // 计算查询向量到簇中心的距离
        dim3 grid(batch_size, (nlist + 255) / 256);
        dim3 block(256);
        compute_query_centroid_distances<<<grid, block>>>(
            d_queries, d_centroids, d_distances, batch_size, nlist, vecdim);
        
        // 找到每个查询的top-nprobe个簇
        dim3 grid2((batch_size + 255) / 256);
        dim3 block2(256);
        find_top_clusters_kernel<<<grid2, block2>>>(
            d_distances, d_cluster_ids, batch_size, nlist, nprobe);
        
        // 拷贝结果到CPU进行分组
        std::vector<uint32_t> h_cluster_ids(batch_size * nprobe);
        cudaMemcpy(h_cluster_ids.data(), d_cluster_ids, 
                   batch_size * nprobe * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        // 按簇ID分组查询
        std::unordered_map<uint32_t, std::vector<uint32_t>> cluster_to_queries;
        for (size_t q = 0; q < batch_size; ++q) {
            for (size_t p = 0; p < nprobe; ++p) {
                uint32_t cluster_id = h_cluster_ids[q * nprobe + p];
                cluster_to_queries[cluster_id].push_back(q);
            }
        }
        
        // 转换为返回格式
        std::vector<BatchClusterInfo> groups;
        for (const auto& pair : cluster_to_queries) {
            BatchClusterInfo info;
            info.cluster_id = pair.first;
            info.query_indices = pair.second;
            groups.push_back(info);
        }
        
        cudaFree(d_distances);
        cudaFree(d_cluster_ids);
        
        return groups;
    }
    
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> 
    search_batch(float* queries, size_t batch_size, size_t k, size_t nprobe) {
        
        // 分配GPU内存存储查询向量
        float* d_queries;
        cudaMalloc(&d_queries, batch_size * vecdim * sizeof(float));
        cudaMemcpy(d_queries, queries, batch_size * vecdim * sizeof(float), cudaMemcpyHostToDevice);
        
        // 对查询进行分组
        auto groups = group_queries_by_clusters(d_queries, batch_size, nprobe);
        
        // 初始化结果
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(batch_size);
        
        // 对每个分组进行矩阵乘法计算
        for (const auto& group : groups) {
            uint32_t cluster_id = group.cluster_id;
            const auto& query_indices = group.query_indices;
            
            if (cluster_matrices[cluster_id].num_vectors == 0) continue;
            
            size_t num_queries = query_indices.size();
            size_t num_vectors = cluster_matrices[cluster_id].num_vectors;
            
            // 分配GPU内存存储查询子集
            float* d_query_subset;
            cudaMalloc(&d_query_subset, num_queries * vecdim * sizeof(float));
            
            // 拷贝对应的查询向量
            std::vector<float> h_query_subset(num_queries * vecdim);
            for (size_t i = 0; i < num_queries; ++i) {
                uint32_t query_idx = query_indices[i];
                memcpy(h_query_subset.data() + i * vecdim, 
                       queries + query_idx * vecdim, 
                       vecdim * sizeof(float));
            }
            cudaMemcpy(d_query_subset, h_query_subset.data(), 
                       num_queries * vecdim * sizeof(float), cudaMemcpyHostToDevice);
            
            // 使用cuBLAS进行矩阵乘法
            const float alpha = 1.0f, beta = 0.0f;
            float* d_result;
            cudaMalloc(&d_result, num_queries * num_vectors * sizeof(float));
            
            // 矩阵乘法: result = query_subset * cluster_vectors^T
            cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        num_vectors, num_queries, vecdim,
                        &alpha,
                        cluster_matrices[cluster_id].d_vectors, vecdim,
                        d_query_subset, vecdim,
                        &beta,
                        d_result, num_vectors);
            
            // 转换为距离并进行GPU top-k选择
            size_t total_elements = num_queries * num_vectors;
            int block_size = 256;
            int grid_size = (total_elements + block_size - 1) / block_size;
            
            // 直接在GPU上进行top-k选择
            uint32_t* d_results_indices;
            float* d_results_distances;
            cudaMalloc(&d_results_indices, num_queries * k * sizeof(uint32_t));
            cudaMalloc(&d_results_distances, num_queries * k * sizeof(float));
            
            // 分配向量ID数组
            uint32_t* d_vector_ids;
            cudaMalloc(&d_vector_ids, num_vectors * sizeof(uint32_t));
            cudaMemcpy(d_vector_ids, cluster_matrices[cluster_id].h_vector_ids, 
                       num_vectors * sizeof(uint32_t), cudaMemcpyHostToDevice);
            
                         // 将内积转换为距离
             float* d_distances_copy;
             cudaMalloc(&d_distances_copy, num_queries * num_vectors * sizeof(float));
             convert_and_copy<<<grid_size, block_size>>>(d_result, d_distances_copy, total_elements);
            
            // GPU top-k选择
            gpu_topk_for_cluster<<<num_queries, 1>>>(
                d_distances_copy, d_vector_ids, d_results_indices, d_results_distances,
                num_queries, num_vectors, k);
            
            // 拷贝结果到CPU
            std::vector<float> h_distances(num_queries * k);
            std::vector<uint32_t> h_indices(num_queries * k);
            cudaMemcpy(h_distances.data(), d_results_distances, 
                       num_queries * k * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_indices.data(), d_results_indices, 
                       num_queries * k * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            // 更新结果
            for (size_t q = 0; q < num_queries; ++q) {
                uint32_t query_idx = query_indices[q];
                auto& pq = results[query_idx];
                
                for (size_t i = 0; i < k; ++i) {
                    float dist = h_distances[q * k + i];
                    uint32_t vec_id = h_indices[q * k + i];
                    
                    if (pq.size() < k) {
                        pq.push({dist, vec_id});
                    } else if (dist < pq.top().first) {
                        pq.push({dist, vec_id});
                        pq.pop();
                    }
                }
            }
            
            cudaFree(d_query_subset);
            cudaFree(d_result);
            cudaFree(d_distances_copy);
            cudaFree(d_results_indices);
            cudaFree(d_results_distances);
            cudaFree(d_vector_ids);
        }
        
        cudaFree(d_queries);
        return results;
    }
};

// 主要接口函数
std::vector<std::priority_queue<std::pair<float, uint32_t>>>
ivf_search_gpu(float* base,
               float* queries,
               size_t base_number,
               size_t vecdim,
               size_t batch_size,
               size_t k,
               const std::vector<float>& centroids,
               size_t nlist,
               const std::vector<std::vector<uint32_t>>& invlists,
               size_t nprobe) {
    
    GPUIVFIndex index(base, base_number, vecdim, centroids, nlist, invlists);
    return index.search_batch(queries, batch_size, k, nprobe);
} 