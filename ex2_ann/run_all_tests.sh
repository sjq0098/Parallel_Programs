#!/bin/bash

# 设置中文输出编码
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8

# 创建结果文件夹
mkdir -p results

echo "======= 开始测试所有算法 ======="
echo "测试时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee results/summary.txt
echo "=============================\n" | tee -a results/summary.txt

# 编译PQ优化测试程序
echo "编译PQ优化测试程序..." | tee -a results/summary.txt
g++ -std=c++14 -O3 -mavx2 -mfma -fopenmp -o pq_test main_test_pq.cc
if [ $? -eq 0 ]; then
    echo "PQ优化测试程序编译成功" | tee -a results/summary.txt
    
    # 运行PQ基准测试
    echo "运行PQ基准测试..." | tee -a results/summary.txt
    ./pq_test | tee results/pq_test.txt
    echo -e "\n" | tee -a results/summary.txt
else
    echo "PQ优化测试程序编译失败，跳过PQ测试" | tee -a results/summary.txt
fi

# 测试原始版本
echo "测试原始版本..." | tee -a results/summary.txt
./main | tee results/main_original.txt
echo -e "\n" | tee -a results/summary.txt

# 测试所有优化版本
for i in {0..15}; do
    echo "测试方法 $i..."
    ./main_op $i 1 | tee results/main_op_$i.txt
    echo -e "\n" | tee -a results/summary.txt
done

# 提取各方法的结果并添加到摘要
# 定义方法名称数组
method_names=(
    "标量暴力搜索"
    "SSE优化暴力搜索"
    "AVX优化暴力搜索"
    "标量量化(SQ)"
    "乘积量化(PQ) M=4"
    "乘积量化(PQ) M=8"
    "乘积量化(PQ) M=16"
    "乘积量化(PQ) M=32"
    "优化乘积量化(OPQ) M=4"
    "优化乘积量化(OPQ) M=8"
    "优化乘积量化(OPQ) M=16"
    "优化乘积量化(OPQ) M=32"
    "混合搜索(PQ16+精确重排序)"
    "混合搜索(PQ32+精确重排序)"
    "混合搜索(OPQ16+精确重排序)"
    "混合搜索(OPQ32+精确重排序)"
)

# 创建结果表格
echo "\n\n测试结果汇总表：" | tee -a results/summary.txt
echo "算法名称,平均召回率,平均延迟(微秒)" | tee results/results.csv
echo "原始版本,$(grep "average recall" results/main_original.txt | awk '{print $3}'),$(grep "average latency" results/main_original.txt | awk '{print $4}')" | tee -a results/results.csv

for i in {0..15}; do
    echo "提取方法 $i (${method_names[$i]}) 的结果..."
    recall=$(grep "平均召回率" results/main_op_$i.txt | awk '{print $2}')
    latency=$(grep "平均延迟" results/main_op_$i.txt | awk '{print $4}')
    
    echo "  结果: 平均召回率=$recall, 平均延迟=$latency 微秒" | tee -a results/summary.txt
    echo "${method_names[$i]},$recall,$latency" | tee -a results/results.csv
done

# 如果有PQ测试结果，也添加到表格中
if [ -f "results/pq_test.txt" ]; then
    echo "添加PQ优化测试结果..." | tee -a results/summary.txt
    pq_recall=$(grep "平均召回率" results/pq_test.txt | awk '{print $2}')
    pq_latency=$(grep "平均延迟" results/pq_test.txt | awk '{print $4}')
    
    if [ ! -z "$pq_recall" ] && [ ! -z "$pq_latency" ]; then
        echo "  结果: 平均召回率=$pq_recall, 平均延迟=$pq_latency 微秒" | tee -a results/summary.txt
        echo "SIMD优化乘积量化,$pq_recall,$pq_latency" | tee -a results/results.csv
    fi
fi

# 找出召回率≥0.9且延迟最低的算法
echo "召回率≥0.9的算法性能排名：" | tee -a results/summary.txt
grep -v "原始版本" results/results.csv | awk -F, '$2 >= 0.9 {print $0}' | sort -t, -k3,3n | tee results/high_recall_ranking.csv

echo "======= 所有测试完成 =======" | tee -a results/summary.txt
echo "详细结果已保存到 results/ 目录" | tee -a results/summary.txt
echo "召回率≥0.9的最佳算法已保存到 results/high_recall_ranking.csv" | tee -a results/summary.txt 