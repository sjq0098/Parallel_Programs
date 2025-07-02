#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <queue>
#include <vector>
#include <algorithm>
#include <cfloat>
#include <climits>

// CUDA kernel：将内积转换为距离（1 - 内积）- CPU版本专用
__global__ void convert_dot_to_distance_cpu_version(float* distances, size_t total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        distances[idx] = 1.0f - distances[idx];
    }
}

// GPU矩阵计算 + CPU TOP-K选择
std::vector<std::priority_queue<std::pair<float, uint32_t>>> flat_search_gpu_cpu(
    float* base, float* query, size_t base_number, size_t vecdim, size_t batch_size, size_t k) {
    
    // 初始化结果
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(batch_size);

    // GPU内存分配
    float *d_base, *d_query, *d_distances;
    cudaMalloc(&d_base, base_number * vecdim * sizeof(float));
    cudaMalloc(&d_query, vecdim * batch_size * sizeof(float));
    cudaMalloc(&d_distances, base_number * batch_size * sizeof(float));

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
    convert_dot_to_distance_cpu_version<<<grid_size, block_size, 0, stream>>>(d_distances, total_elements);

    // 将距离拷贝回CPU
    float* h_distances = new float[base_number * batch_size];
    cudaMemcpyAsync(h_distances, d_distances, base_number * batch_size * sizeof(float), 
                   cudaMemcpyDeviceToHost, stream);

    // 等待GPU操作完成
    cudaStreamSynchronize(stream);

    // 在CPU上进行TOP-K选择
    #pragma omp parallel for
    for (size_t q = 0; q < batch_size; ++q) {
        std::priority_queue<std::pair<float, uint32_t>> pq;
        float* query_distances = h_distances + q * base_number;
        
        for (size_t i = 0; i < base_number; ++i) {
            float dis = query_distances[i];
            
            if (pq.size() < k) {
                pq.push({dis, i});
            } else if (dis < pq.top().first) {
                pq.push({dis, i});
                pq.pop();
            }
        }
        
        results[q] = pq;
    }

    // 释放资源
    delete[] h_distances;
    cudaFree(d_base);
    cudaFree(d_query);
    cudaFree(d_distances);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    return results;
} 