import hnswlib
import numpy as np
import os
import time
import struct # 用于解析 .fbin 文件

def load_fbin_data(filename):
    """
    从 .fbin 文件加载数据。
    文件格式假定为：[num_vectors (int32), dim (int32), data (float32 * num_vectors * dim)]
    """
    print(f"Loading data from {filename}...")
    with open(filename, "rb") as f:
        num_vectors, dim = struct.unpack('ii', f.read(8))
        print(f"Number of vectors: {num_vectors}, Dimension: {dim}")
        data = np.fromfile(f, dtype=np.float32, count=num_vectors * dim)
        data = data.reshape((num_vectors, dim))
    print("Data loaded successfully.")
    return data, dim, num_vectors

def ensure_dir_exists(directory_path):
    """确保目录存在，如果不存在则创建它。"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def main():
    # ----------- 配置参数 -----------
    data_root_path = "/anndata/"  # 数据集根目录
    base_data_file = "DEEP100K.base.100k.fbin"  # 基础数据集文件名
    output_index_dir = "files_py/"  # Python 构建的索引保存目录

    # 要测试的 HNSW 参数组合
    M_values = [8, 16, 24]
    ef_construction_values = [100, 150, 200]
    # -------------------------------

    ensure_dir_exists(output_index_dir)

    base_data_path = os.path.join(data_root_path, base_data_file)
    
    try:
        base_vectors, dim, num_elements = load_fbin_data(base_data_path)
    except FileNotFoundError:
        print(f"Error: Base data file not found at {base_data_path}")
        return
    except Exception as e:
        print(f"Error loading base data: {e}")
        return

    if num_elements == 0 or dim == 0:
        print("No data loaded. Exiting.")
        return

    # HNSW 空间类型: 'l2' (欧氏距离平方) 或 'ip' (内积).
    # 对于 DEEP100K，通常使用 L2 空间。
    # 如果您的 C++ 版本使用的是 InnerProductSpace，并且您的数据已经归一化，
    # 那么使用 'ip' 并将距离解释为 1 - cos(theta) 是可以的。
    # 否则，'l2' 更常见。
    space_name = 'l2' # 或者 'ip'

    for M_val in M_values:
        for ef_construction_val in ef_construction_values:
            print(f"\nBuilding HNSW index with M = {M_val}, ef_construction = {ef_construction_val}")

            # 声明索引
            # hnswlib.Index(space, dim)
            index = hnswlib.Index(space=space_name, dim=dim)

            # 初始化索引
            # init_index(max_elements, M, ef_construction, random_seed, allow_replace_deleted)
            index.init_index(max_elements=num_elements, M=M_val, ef_construction=ef_construction_val, random_seed=100)

            # 设置并行添加数据点的线程数 (可选, 默认使用所有可用核心)
            # num_threads 参数在 add_items 中控制，而不是全局设置
            # index.set_num_threads(4) # 例如使用4个线程

            print("Adding items to HNSW graph...")
            start_time = time.time()

            # add_items(data, ids=None, num_threads=-1, replace_deleted=False)
            # ids 如果为 None, 将使用 0 到 num_elements-1
            # num_threads = -1 表示使用所有可用的 OpenMP 线程
            index.add_items(base_vectors, num_threads=-1)
            
            end_time = time.time()
            build_time_s = end_time - start_time
            print(f"Finished adding items. Build time: {build_time_s:.3f} seconds.")

            index_filename = f"hnsw_py_M{M_val}_efC{ef_construction_val}.bin" # Python通常保存为 .bin
            index_save_path = os.path.join(output_index_dir, index_filename)

            print(f"Saving index to {index_save_path} ...")
            try:
                index.save_index(index_save_path)
                print("Index saved successfully.")
            except Exception as e:
                print(f"Error saving index: {e}")
            
            # 在 Python 中，索引对象在作用域结束时会自动清理，无需显式 delete

    print("\nAll HNSW index building tasks (Python) completed.")

if __name__ == "__main__":
    main() 