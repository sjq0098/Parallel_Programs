#!/bin/bash

# MPI并行向量搜索算法参数测试脚本
# 编译和运行所有测试脚本，输出结果到CSV文件

# set -e  # 注释掉，允许单个测试失败

echo "=== MPI并行向量搜索算法参数测试套件 ==="
echo "开始时间: $(date)"
echo ""

# 检查MPI环境
if ! command -v mpicc &> /dev/null; then
    echo "错误: 未找到MPI编译器 (mpicc)"
    exit 1
fi

if ! command -v mpirun &> /dev/null; then
    echo "错误: 未找到MPI运行环境 (mpirun)"
    exit 1
fi

# 编译参数
MPI_CC="mpicxx"
CXX_FLAGS="-std=c++17 -O3 -march=native -fopenmp -g"
INCLUDE_FLAGS="-I. -I./hnswlib"
LINK_FLAGS="-lm -fopenmp -lstdc++ -lmpi_cxx"

# 进程数配置 - 测试多种进程数
MPI_PROCESSES_LIST=(1 2 4)
OMP_THREADS=4

echo "编译配置:"
echo "  编译器: $MPI_CC"
echo "  编译选项: $CXX_FLAGS"
echo "  测试进程数: ${MPI_PROCESSES_LIST[*]}"
echo "  OpenMP线程数: $OMP_THREADS"
echo ""

# 设置OpenMP线程数
export OMP_NUM_THREADS=$OMP_THREADS

# 创建结果目录
RESULTS_DIR="parameter_test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR
echo "结果将保存到目录: $RESULTS_DIR"
echo ""

# 定义测试列表
declare -a tests=(
    "main_mpi_ivf:基础IVF算法"
    "main_mpi_simd:SIMD混合算法"
)

# 如果存在其他测试文件，添加到列表中
if [ -f "main_mpi_ivf_hnsw.cc" ]; then
    tests+=("main_mpi_ivf_hnsw:IVF+HNSW算法")
fi

if [ -f "main_mpi_sharded_hnsw.cc" ]; then
    tests+=("main_mpi_sharded_hnsw:分片HNSW算法")
fi

if [ -f "main_mpi_pq_ivf.cc" ]; then
    tests+=("main_mpi_pq_ivf:PQ-IVF算法(带重排序)")
fi

if [ -f "main_mpi_ivf_pq.cc" ]; then
    tests+=("main_mpi_ivf_pq:IVF-PQ算法(带重排序)")
fi

# 函数：编译测试程序
compile_test() {
    local test_name=$1
    local description=$2
    
    echo "编译 $description ($test_name)..."
    
    if [ ! -f "${test_name}.cc" ]; then
        echo "  警告: 源文件 ${test_name}.cc 不存在，跳过"
        return 1
    fi
    
    $MPI_CC $CXX_FLAGS $INCLUDE_FLAGS -o $test_name ${test_name}.cc $LINK_FLAGS
    
    if [ $? -eq 0 ]; then
        echo "  编译成功"
        return 0
    else
        echo "  编译失败"
        return 1
    fi
}

# 函数：运行测试程序（支持多进程数）
run_test() {
    local test_name=$1
    local description=$2
    
    echo ""
    echo "========================================"
    echo "运行 $description 参数测试"
    echo "========================================"
    
    local test_success=0
    local test_failure=0
    
    # 对每种进程数分别测试
    for processes in "${MPI_PROCESSES_LIST[@]}"; do
        echo ""
        echo "测试 $description 使用 $processes 进程..."
        
        # 记录测试开始时间
        local start_time=$(date +%s)
        
        # 运行测试（添加超时机制，20分钟超时）
        timeout 1200 mpirun -np $processes ./$test_name
        
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        if [ $exit_code -eq 0 ]; then
            echo "$description ($processes 进程) 测试完成，耗时: ${duration}秒"
            ((test_success++))
            
            # 移动结果文件到结果目录，添加进程数标识
            for csv_file in results_*.csv; do
                if [ -f "$csv_file" ]; then
                    # 在文件名中添加进程数标识
                    new_name="${csv_file%.csv}_${processes}proc.csv"
                    mv "$csv_file" "$RESULTS_DIR/$new_name"
                    echo "结果文件已保存: $RESULTS_DIR/$new_name"
                fi
            done
        else
            echo "错误: $description ($processes 进程) 测试失败 (退出码: $exit_code)"
            ((test_failure++))
        fi
    done
    
    echo ""
    echo "$description 总结: 成功 $test_success 次, 失败 $test_failure 次"
    
    if [ $test_success -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

# 主测试流程
echo "开始编译所有测试程序..."
echo ""

compiled_tests=()
failed_compilations=()

# 编译所有测试
for test_info in "${tests[@]}"; do
    IFS=':' read -r test_name description <<< "$test_info"
    
    if compile_test "$test_name" "$description"; then
        compiled_tests+=("$test_info")
    else
        failed_compilations+=("$test_info")
    fi
done

echo ""
echo "编译完成:"
echo "  成功: ${#compiled_tests[@]} 个测试"
echo "  失败: ${#failed_compilations[@]} 个测试"

if [ ${#failed_compilations[@]} -gt 0 ]; then
    echo ""
    echo "编译失败的测试:"
    for test_info in "${failed_compilations[@]}"; do
        IFS=':' read -r test_name description <<< "$test_info"
        echo "  - $description ($test_name)"
    done
fi

if [ ${#compiled_tests[@]} -eq 0 ]; then
    echo ""
    echo "错误: 没有成功编译的测试程序"
    exit 1
fi

echo ""
echo "开始运行参数测试..."

# 运行所有编译成功的测试
successful_tests=0
failed_tests=0

for test_info in "${compiled_tests[@]}"; do
    IFS=':' read -r test_name description <<< "$test_info"
    
    if run_test "$test_name" "$description"; then
        ((successful_tests++))
    else
        ((failed_tests++))
    fi
    
    # 清理可执行文件
    rm -f $test_name
done

# 生成测试总结报告
echo ""
echo "========================================"
echo "测试总结报告"
echo "========================================"
echo "开始时间: $(date)"
echo "测试进程数: ${MPI_PROCESSES_LIST[*]}"
echo "OpenMP线程数: $OMP_THREADS"
echo ""
echo "测试结果:"
echo "  成功: $successful_tests 个测试"
echo "  失败: $failed_tests 个测试"
echo ""

# 列出生成的结果文件
if [ -d "$RESULTS_DIR" ] && [ "$(ls -A $RESULTS_DIR)" ]; then
    echo "生成的结果文件:"
    ls -la $RESULTS_DIR/
    echo ""
    echo "结果文件位置: $(pwd)/$RESULTS_DIR/"
else
    echo "警告: 没有生成结果文件"
fi

# 生成增强的分析脚本
cat > $RESULTS_DIR/analyze_results.py << 'EOF'
#!/usr/bin/env python3
"""
增强的结果分析脚本
分析CSV文件中的参数对性能的影响，包括多进程数和统计指标
"""

import pandas as pd
import glob
import os
import numpy as np

def analyze_csv_files():
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("没有找到CSV结果文件")
        return
    
    # 按算法类型分组分析
    algorithms = {}
    for csv_file in csv_files:
        # 提取算法名称和进程数
        if '_' in csv_file:
            parts = csv_file.replace('.csv', '').split('_')
            if 'proc' in parts[-1]:
                proc_num = parts[-1].replace('proc', '')
                algo_name = '_'.join(parts[1:-1])  # 去掉results_前缀和进程数后缀
            else:
                algo_name = '_'.join(parts[1:])
                proc_num = 'unknown'
        else:
            algo_name = csv_file.replace('.csv', '')
            proc_num = 'unknown'
        
        if algo_name not in algorithms:
            algorithms[algo_name] = {}
        algorithms[algo_name][proc_num] = csv_file
    
    for algo_name, proc_files in algorithms.items():
        print(f"\n=== 分析算法: {algo_name} ===")
        
        # 合并所有进程数的数据
        all_data = []
        for proc_num, csv_file in proc_files.items():
            try:
                df = pd.read_csv(csv_file)
                df['proc_num'] = proc_num
                all_data.append(df)
                
                print(f"\n进程数 {proc_num}: {len(df)} 个测试点")
                
                if 'recall_mean' in df.columns:
                    print(f"  召回率: {df['recall_mean'].min():.4f} - {df['recall_mean'].max():.4f} (平均: {df['recall_mean'].mean():.4f})")
                    if 'recall_std' in df.columns:
                        print(f"  召回率标准差: 平均 {df['recall_std'].mean():.4f}")
                
                if 'latency_us_mean' in df.columns:
                    print(f"  延迟: {df['latency_us_mean'].min():.0f} - {df['latency_us_mean'].max():.0f} μs (平均: {df['latency_us_mean'].mean():.0f})")
                    if 'latency_us_std' in df.columns:
                        print(f"  延迟标准差: 平均 {df['latency_us_std'].mean():.0f} μs")
                
            except Exception as e:
                print(f"  分析 {csv_file} 时出错: {e}")
        
        # 合并分析
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            print(f"\n{algo_name} 综合分析:")
            print(f"  总测试点数: {len(combined_df)}")
            
            # 按进程数分析性能
            if 'proc_num' in combined_df.columns and len(proc_files) > 1:
                print("\n进程数对性能的影响:")
                proc_analysis = combined_df.groupby('proc_num').agg({
                    'recall_mean': ['mean', 'std'] if 'recall_mean' in combined_df.columns else 'count',
                    'latency_us_mean': ['mean', 'std'] if 'latency_us_mean' in combined_df.columns else 'count'
                })
                print(proc_analysis)
            
            # 找出最佳配置
            if 'recall_mean' in combined_df.columns and 'latency_us_mean' in combined_df.columns:
                # 效率得分 = recall / (latency_ms) 
                combined_df['efficiency'] = combined_df['recall_mean'] / (combined_df['latency_us_mean'] / 1000)
                best_idx = combined_df['efficiency'].idxmax()
                
                print(f"\n{algo_name} 最高效率配置:")
                for col in combined_df.columns:
                    if col != 'efficiency':
                        print(f"  {col}: {combined_df.loc[best_idx, col]}")

def compare_algorithms():
    """跨算法性能对比"""
    print("\n=== 跨算法性能对比 ===")
    
    csv_files = glob.glob("*.csv")
    if len(csv_files) < 2:
        print("需要至少2个算法的结果文件进行对比")
        return
    
    algo_summary = {}
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # 提取算法名称
            if '_' in csv_file:
                parts = csv_file.replace('.csv', '').split('_')
                if 'proc' in parts[-1]:
                    algo_name = '_'.join(parts[1:-1])
                else:
                    algo_name = '_'.join(parts[1:])
            else:
                algo_name = csv_file.replace('.csv', '')
            
            if 'recall_mean' in df.columns and 'latency_us_mean' in df.columns:
                # 计算该算法的最佳性能
                best_recall = df['recall_mean'].max()
                min_latency_for_best_recall = df[df['recall_mean'] >= best_recall * 0.95]['latency_us_mean'].min()
                
                algo_summary[algo_name] = {
                    'best_recall': best_recall,
                    'best_latency': df['latency_us_mean'].min(),
                    'balanced_latency': min_latency_for_best_recall,
                    'test_points': len(df)
                }
                
        except Exception as e:
            print(f"处理 {csv_file} 时出错: {e}")
    
    if algo_summary:
        print("\n算法性能总结:")
        print("算法名称 | 最佳召回率 | 最低延迟(μs) | 平衡延迟(μs) | 测试点数")
        print("-" * 70)
        for algo, stats in algo_summary.items():
            print(f"{algo:15} | {stats['best_recall']:8.4f} | {stats['best_latency']:10.0f} | {stats['balanced_latency']:11.0f} | {stats['test_points']:8d}")

if __name__ == "__main__":
    analyze_csv_files()
    compare_algorithms()
EOF

chmod +x $RESULTS_DIR/analyze_results.py

echo "已生成增强的结果分析脚本: $RESULTS_DIR/analyze_results.py"
echo "使用方法: cd $RESULTS_DIR && python3 analyze_results.py"
echo ""

# 清理临时文件
echo "清理临时文件..."
rm -f temp_*.bin

echo "所有测试完成!"
echo "结束时间: $(date)"

if [ $failed_tests -gt 0 ]; then
    exit 1
else
    exit 0
fi