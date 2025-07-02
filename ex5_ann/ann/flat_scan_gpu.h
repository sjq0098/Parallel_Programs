#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <queue>
#include <vector>
#include <algorithm>
#include <cfloat>
#include <climits>

// CUDA kernel：将内积转换为距离（1 - 内积）
__global__ void convert_dot_to_distance(float* distances, size_t total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        distances[idx] = 1.0f - distances[idx];
    }
}

// 简化但高效的GPU Top-K搜索（每个线程块处理一个查询）
__global__ void gpu_topk_simple(float* distances, uint32_t* results_indices, 
                                float* results_distances, size_t base_number, 
                                size_t batch_size, size_t k) {
    int query_idx = blockIdx.x;
    if (query_idx >= batch_size) return;
    
    // 每个查询的距离起始位置
    float* query_distances = distances + query_idx * base_number;
    float* query_results_dist = results_distances + query_idx * k;
    uint32_t* query_results_idx = results_indices + query_idx * k;
    
    // 只让thread 0处理，避免竞争条件（简单但有效）
    if (threadIdx.x == 0) {
        // 初始化结果为前k个元素
        for (int i = 0; i < k && i < base_number; i++) {
            query_results_dist[i] = query_distances[i];
            query_results_idx[i] = i;
        }
        
        // 使用选择排序找前k小的元素
        for (int i = 0; i < k && i < base_number; i++) {
            int min_idx = i;
            for (int j = i + 1; j < base_number; j++) {
                if (query_distances[j] < query_distances[min_idx]) {
                    min_idx = j;
                }
            }
            
            // 只在前k个位置记录结果
            if (i < k) {
                query_results_dist[i] = query_distances[min_idx];
                query_results_idx[i] = min_idx;
                
                // 标记已选择的元素（设为很大的值）
                query_distances[min_idx] = FLT_MAX;
            }
        }
    }
}

// 并行版本的GPU Top-K（使用线程协作）
__global__ void gpu_topk_parallel(float* distances, uint32_t* results_indices, 
                                  float* results_distances, size_t base_number, 
                                  size_t batch_size, size_t k) {
    int query_idx = blockIdx.x;
    if (query_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // 每个查询的距离起始位置
    float* query_distances = distances + query_idx * base_number;
    float* query_results_dist = results_distances + query_idx * k;
    uint32_t* query_results_idx = results_indices + query_idx * k;
    
    // 使用归约方式找最小值
    for (int round = 0; round < k; round++) {
        float min_val = FLT_MAX;
        int min_idx = -1;
        
        // 每个线程找自己负责范围内的最小值
        for (int i = tid; i < base_number; i += stride) {
            if (query_distances[i] < min_val) {
                min_val = query_distances[i];
                min_idx = i;
            }
        }
        
        // 使用共享内存进行归约
        __shared__ float shared_min_vals[256];
        __shared__ int shared_min_indices[256];
        
        shared_min_vals[tid] = min_val;
        shared_min_indices[tid] = min_idx;
        __syncthreads();
        
        // 归约找全局最小值
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockDim.x) {
                if (shared_min_vals[tid + s] < shared_min_vals[tid]) {
                    shared_min_vals[tid] = shared_min_vals[tid + s];
                    shared_min_indices[tid] = shared_min_indices[tid + s];
                }
            }
            __syncthreads();
        }
        
        // thread 0写入结果并标记已选择的元素
        if (tid == 0 && round < k) {
            query_results_dist[round] = shared_min_vals[0];
            query_results_idx[round] = shared_min_indices[0];
            if (shared_min_indices[0] >= 0) {
                query_distances[shared_min_indices[0]] = FLT_MAX;
            }
        }
        __syncthreads();
    }
}

// GPU版本的flat_search，处理一个查询批次
std::vector<std::priority_queue<std::pair<float, uint32_t>>> flat_search_gpu(
    float* base, float* query, size_t base_number, size_t vecdim, size_t batch_size, size_t k) {
    
    // 初始化结果
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(batch_size);

    // GPU内存分配
    float *d_base, *d_query, *d_distances, *d_distances_copy;
    uint32_t *d_results_indices;
    float *d_results_distances;
    
    cudaMalloc(&d_base, base_number * vecdim * sizeof(float));
    cudaMalloc(&d_query, vecdim * batch_size * sizeof(float));
    cudaMalloc(&d_distances, base_number * batch_size * sizeof(float));
    cudaMalloc(&d_distances_copy, base_number * batch_size * sizeof(float));
    cudaMalloc(&d_results_indices, batch_size * k * sizeof(uint32_t));
    cudaMalloc(&d_results_distances, batch_size * k * sizeof(float));

    // 创建CUDA流以实现异步执行
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 将数据异步拷贝到GPU
    cudaMemcpyAsync(d_base, base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_query, query, vecdim * batch_size * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 初始化cuBLAS并设置流
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    // 设置矩阵乘法参数
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                base_number, batch_size, vecdim,
                &alpha,
                d_base, vecdim,      // base: [vecdim, base_number]，转置后
                d_query, vecdim,     // query: [vecdim, batch_size]
                &beta,
                d_distances, base_number);

    // GPU上执行距离转换
    size_t total_elements = base_number * batch_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    convert_dot_to_distance<<<grid_size, block_size, 0, stream>>>(d_distances, total_elements);

    // 复制距离数组（因为Top-K搜索会修改原数组）
    cudaMemcpyAsync(d_distances_copy, d_distances, base_number * batch_size * sizeof(float), 
                   cudaMemcpyDeviceToDevice, stream);

    // GPU上执行Top-K搜索
    // 根据数据集大小选择不同的kernel
    if (base_number < 100000) {
        // 小数据集使用并行版本
        gpu_topk_parallel<<<batch_size, 256, 0, stream>>>(
            d_distances_copy, d_results_indices, d_results_distances, 
            base_number, batch_size, k);
    } else {
        // 大数据集使用简化版本（避免共享内存限制）
        gpu_topk_simple<<<batch_size, 1, 0, stream>>>(
            d_distances_copy, d_results_indices, d_results_distances, 
            base_number, batch_size, k);
    }

    // 将结果异步拷贝回CPU
    float* h_results_distances = new float[batch_size * k];
    uint32_t* h_results_indices = new uint32_t[batch_size * k];
    
    cudaMemcpyAsync(h_results_distances, d_results_distances, batch_size * k * sizeof(float), 
                   cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_results_indices, d_results_indices, batch_size * k * sizeof(uint32_t), 
                   cudaMemcpyDeviceToHost, stream);

    // 等待所有GPU操作完成
    cudaStreamSynchronize(stream);

    // 构建结果
    for (size_t q = 0; q < batch_size; ++q) {
        std::priority_queue<std::pair<float, uint32_t>> pq;
        for (size_t i = 0; i < k; ++i) {
            float dis = h_results_distances[q * k + i];
            uint32_t idx = h_results_indices[q * k + i];
            pq.push({dis, idx});
        }
        results[q] = pq;
    }

    // 释放资源
    delete[] h_results_distances;
    delete[] h_results_indices;
    cudaFree(d_base);
    cudaFree(d_query);
    cudaFree(d_distances);
    cudaFree(d_distances_copy);
    cudaFree(d_results_indices);
    cudaFree(d_results_distances);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    return results;
}