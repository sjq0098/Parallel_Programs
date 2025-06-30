#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <string>
#include <stdexcept>

// 读取 IVF centroids
std::vector<float> load_ivf_centroids(const std::string& filename, size_t expected_nlist, size_t vecdim) {
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

// 读取 IVF 倒排列表
std::vector<std::vector<uint32_t>> load_ivf_invlists(const std::string& filename, size_t expected_nlist) {
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


// ivf_search 接口：
//   base:      所有原始向量的起始地址，行优先，每个向量长度 vecdim
//   query:     单个查询向量，长度 vecdim
//   k:         top-k
//   centroids: IVF 簇中心数组，大小 nlist * vecdim
//   invlists:  每个簇对应的倒排列表（存储该簇中向量在 base 中的索引）
//   nprobe:    查询时要扫描的簇个数
std::priority_queue<std::pair<float, uint32_t>>
flat_ivf_search(float* base,
           float* query,
           size_t base_number,
           size_t vecdim,
           size_t k,
           float* centroids,
           size_t nlist,
           const std::vector<std::vector<uint32_t>>& invlists,
           size_t nprobe)
{
    // 1. 先对 nlist 个簇中心，计算与 query 的“粗”距离，选出 top-nprobe 个簇
    //    这里使用 IP 距离转换： dis = 1 - dot(c, q)
    struct CentDist { float dist; uint32_t idx; };
    std::vector<CentDist> cd(nlist);
    for (size_t i = 0; i < nlist; ++i) {
        float dot = 0;
        float* cptr = centroids + i * vecdim;
        for (size_t d = 0; d < vecdim; ++d) {
            dot += cptr[d] * query[d];
        }
        cd[i].dist = 1.0f - dot;
        cd[i].idx  = (uint32_t)i;
    }
    // 部分排序取最小的 nprobe 簇
    if (nprobe < nlist) {
        std::nth_element(cd.begin(), cd.begin() + nprobe, cd.end(),
                         [](auto &a, auto &b){ return a.dist < b.dist; });
    }
    // 前 nprobe 就是我们的倒排簇索引列表
    std::vector<uint32_t> probe_list;
    probe_list.reserve(nprobe);
    for (size_t i = 0; i < std::min(nprobe, nlist); ++i) {
        probe_list.push_back(cd[i].idx);
    }

    // 2. 在选中的倒排列表里，逐个对 base 向量计算精确距离，并维护 top-k
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (uint32_t list_id : probe_list) {
        const auto& vec_ids = invlists[list_id];
        for (uint32_t vid : vec_ids) {
            // 计算 base[vid] 与 query 的 IP 距离
            float dot = 0;
            float* vptr = base + vid * vecdim;
            for (size_t d = 0; d < vecdim; ++d) {
                dot += vptr[d] * query[d];
            }
            float dist = 1.0f - dot;

            if (q.size() < k) {
                q.push({dist, vid});
            } else if (dist < q.top().first) {
                q.push({dist, vid});
                q.pop();
            }
        }
    }

    return q;
}
