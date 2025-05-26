#!/bin/bash

# 创建结果目录
mkdir -p results 2>/dev/null || mkdir results

# 测试所有方法并保存结果
echo "开始运行所有测试..."
echo "结果将保存在 results 目录中"

# 运行所有测试并保存结果到CSV文件
./build/ann_benchmark.exe all > results/all_methods.csv

# 单独运行每个方法并保存详细结果
METHODS=("flat" "sq" "pq4" "pq8" "pq16" "pq32" "pq4rerank" "pq8rerank" "pq16rerank" "pq32rerank" \
         "ivf_n64_p4" "ivf_n64_p8" "ivf_n64_p12" \
         "ivf_n128_p8" "ivf_n128_p12" "ivf_n128_p16" \
         "ivf_n256_p12" "ivf_n256_p16" "ivf_n256_p24" \
         "ivf_n512_p16" "ivf_n512_p20" "ivf_n512_p32" \
         "ivf_omp_n64_p4" "ivf_omp_n64_p8" "ivf_omp_n64_p12" \
         "ivf_omp_n128_p8" "ivf_omp_n128_p12" "ivf_omp_n128_p16" \
         "ivf_omp_n256_p12" "ivf_omp_n256_p16" "ivf_omp_n256_p24" \
         "ivf_omp_n512_p16" "ivf_omp_n512_p20" "ivf_omp_n512_p32" \
         "ivf_ptd_n64_p4" "ivf_ptd_n64_p8" "ivf_ptd_n64_p12" \
         "ivf_ptd_n128_p8" "ivf_ptd_n128_p12" "ivf_ptd_n128_p16" \
         "ivf_ptd_n256_p12" "ivf_ptd_n256_p16" "ivf_ptd_n256_p24" \
         "ivf_ptd_n512_p16" "ivf_ptd_n512_p20" "ivf_ptd_n512_p32")
METHOD_NAMES=("FLAT_SEARCH" "FLAT_SEARCH_SQ" "PQ4" "PQ8" "PQ16" "PQ32" \
              "PQ4_RERANK" "PQ8_RERANK" "PQ16_RERANK" "PQ32_RERANK" \
              "IVF_N64_P4" "IVF_N64_P8" "IVF_N64_P12" \
              "IVF_N128_P8" "IVF_N128_P12" "IVF_N128_P16" \
              "IVF_N256_P12" "IVF_N256_P16" "IVF_N256_P24" \
              "IVF_N512_P16" "IVF_N512_P20" "IVF_N512_P32" \
              "IVF_OMP_N64_P4" "IVF_OMP_N64_P8" "IVF_OMP_N64_P12" \
              "IVF_OMP_N128_P8" "IVF_OMP_N128_P12" "IVF_OMP_N128_P16" \
              "IVF_OMP_N256_P12" "IVF_OMP_N256_P16" "IVF_OMP_N256_P24" \
              "IVF_OMP_N512_P16" "IVF_OMP_N512_P20" "IVF_OMP_N512_P32" \
              "IVF_PTD_N64_P4" "IVF_PTD_N64_P8" "IVF_PTD_N64_P12" \
              "IVF_PTD_N128_P8" "IVF_PTD_N128_P12" "IVF_PTD_N128_P16" \
              "IVF_PTD_N256_P12" "IVF_PTD_N256_P16" "IVF_PTD_N256_P24" \
              "IVF_PTD_N512_P16" "IVF_PTD_N512_P20" "IVF_PTD_N512_P32")

for i in "${!METHODS[@]}"; do
    method=${METHODS[$i]}
    name=${METHOD_NAMES[$i]}
    echo "运行 $name 测试..."
    ./build/ann_benchmark.exe "$method" > "results/${method}_results.txt"
done

# 生成汇总报告
echo "生成汇总报告..."
echo "方法,召回率,延迟(微秒)" > results/summary.csv
for i in "${!METHODS[@]}"; do
    method=${METHODS[$i]}
    name=${METHOD_NAMES[$i]}
    
    # 从文件中提取召回率和延迟
    recall=$(grep "Average recall" "results/${method}_results.txt" | awk '{print $3}')
    latency=$(grep "Average latency" "results/${method}_results.txt" | awk '{print $4}')
    
    echo "$name,$recall,$latency" >> results/summary.csv
done


echo "所有测试完成，汇总结果保存在 results/summary.csv" 