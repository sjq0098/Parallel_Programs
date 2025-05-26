#pragma once
#include <thread>
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <mutex>
#include <condition_variable>
#include "pq_ivf.h" // 引入基本的IVFPQ索引定义

// 简易线程池实现（用于 IVFPQ 模块）
class ThreadPool_pq {
public:
    ThreadPool_pq(size_t n) : stop_flag(false) {
        for (size_t i = 0; i < n; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lk(this->mtx);
                        this->cv.wait(lk, [this]{ return stop_flag || !tasks.empty(); });
                        if (stop_flag && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    void enqueue(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            tasks.emplace(std::move(f));
        }
        cv.notify_one();
    }

    ~ThreadPool_pq() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop_flag = true;
        }
        cv.notify_all();
        for (auto &t : workers) t.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop_flag;
};

// 用于存储 IVFPQ 候选的参数结构
struct IVFPQThreadArgs {
    std::vector<std::pair<float,uint32_t>> candidates;
};

std::priority_queue<std::pair<float, uint32_t>> ivfpq_search_ptd(
    const IVFPQIndex* index,
    float* query,
    size_t k,
    size_t nprobe,
    int num_threads)
{
    // 1. 串行选簇（L2 距离平方）
    struct CentDist { float dist2; uint32_t idx; };
    std::vector<CentDist> cd(index->nlist);
    for (size_t i = 0; i < index->nlist; ++i) {
        const float* cptr = index->centroids.data() + i * index->d;
        float d2 = 0;
        for (size_t dd = 0; dd < index->d; ++dd) {
            float diff = query[dd] - cptr[dd];
            d2 += diff * diff;
        }
        cd[i] = {d2, (uint32_t)i};
    }
    if (nprobe < index->nlist) {
        std::nth_element(cd.begin(), cd.begin() + nprobe, cd.end(),
            [](auto &a, auto &b){ return a.dist2 < b.dist2; });
    }
    std::vector<uint32_t> probe_list(nprobe);
    for (size_t i = 0; i < nprobe && i < index->nlist; ++i)
        probe_list[i] = cd[i].idx;

    // 2. 串行构建距离表
    std::vector<float> pq_dist_table = compute_pq_distance_table(index, query);

    // 3. 并行生成候选（作用域管理线程池_pq）
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }
    std::vector<IVFPQThreadArgs> args(nprobe);
    {
        ThreadPool_pq pool(num_threads);
        for (size_t i = 0; i < nprobe; ++i) {
            pool.enqueue([&, i] {
                auto &out = args[i].candidates;
                out.reserve(k * 10);
                uint32_t lid = probe_list[i];
                const auto &ids   = index->invlists[lid];
                const auto &codes = index->codes[lid];
                for (size_t j = 0; j < ids.size(); ++j) {
                    float dist2 = 0;
                    for (size_t m = 0; m < index->m; ++m) {
                        uint8_t code = codes[j*index->m + m];
                        dist2 += pq_dist_table[m*index->ksub + code];
                    }
                    out.emplace_back(dist2, ids[j]);
                }
            });
        }
    }

    // 4. 聚合并选取 Top-k
    std::vector<std::pair<float,uint32_t>> all;
    size_t total = 0;
    for (auto &a : args) total += a.candidates.size();
    all.reserve(total);
    for (auto &a : args) {
        all.insert(all.end(), a.candidates.begin(), a.candidates.end());
    }
    if (all.size() > k) {
        std::nth_element(all.begin(), all.begin() + k, all.end(),
            [](auto &a, auto &b){ return a.first < b.first; });
        all.resize(k);
    }
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (auto &p : all) result.push(p);
    return result;
}

// 带重排序的并行 IVFPQ 搜索
std::vector<std::pair<float, uint32_t>> rerank_with_ptd(
    const float* base,
    const float* query,
    const std::vector<std::pair<float, uint32_t>>& candidates,
    size_t d,
    size_t k,
    int num_threads)
{
    std::vector<std::pair<float, uint32_t>> result(candidates);
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }
    size_t chunk = (candidates.size() + num_threads - 1) / num_threads;
    {
        ThreadPool_pq pool(num_threads);
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, candidates.size());
            pool.enqueue([&, start, end] {
                for (size_t i = start; i < end; ++i) {
                    uint32_t id = candidates[i].second;
                    const float* vec = base + id * d;
                    float d2 = 0;
                    for (size_t j = 0; j < d; ++j) {
                        float diff = query[j] - vec[j];
                        d2 += diff * diff;
                    }
                    result[i].first = d2;
                }
            });
        }
    }
    if (k < result.size()) {
        std::partial_sort(result.begin(), result.begin() + k, result.end(),
            [](auto &a, auto &b){ return a.first < b.first; });
        result.resize(k);
    } else {
        std::sort(result.begin(), result.end(),
            [](auto &a, auto &b){ return a.first < b.first; });
    }
    return result;
}

std::priority_queue<std::pair<float, uint32_t>> ivfpq_search_ptd_rerank(
    const IVFPQIndex* index,
    const float* base,
    float* query,
    size_t k,
    size_t nprobe,
    size_t L,
    int num_threads)
{
    // 1. 串行选簇
    struct CentDist { float dist2; uint32_t idx; };
    std::vector<CentDist> cd(index->nlist);
    for (size_t i = 0; i < index->nlist; ++i) {
        const float* cptr = index->centroids.data() + i * index->d;
        float d2 = 0;
        for (size_t dd = 0; dd < index->d; ++dd) {
            float diff = query[dd] - cptr[dd];
            d2 += diff * diff;
        }
        cd[i] = {d2, (uint32_t)i};
    }
    if (nprobe < index->nlist) {
        std::nth_element(cd.begin(), cd.begin() + nprobe, cd.end(),
            [](auto &a, auto &b){ return a.dist2 < b.dist2; });
    }
    std::vector<uint32_t> probe_list(nprobe);
    for (size_t i = 0; i < nprobe && i < index->nlist; ++i)
        probe_list[i] = cd[i].idx;

    // 2. 串行构建距离表
    std::vector<float> pq_dist_table = compute_pq_distance_table(index, query);

    // 3. 并行生成候选
    std::vector<IVFPQThreadArgs> args(nprobe);
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }
    {
        ThreadPool_pq pool(num_threads);
        for (size_t i = 0; i < nprobe; ++i) {
            pool.enqueue([&, i] {
                auto &out = args[i].candidates;
                out.reserve(L * 10);
                uint32_t lid = probe_list[i];
                const auto &ids   = index->invlists[lid];
                const auto &codes = index->codes[lid];
                for (size_t j = 0; j < ids.size(); ++j) {
                    float dist2 = 0;
                    for (size_t m = 0; m < index->m; ++m) {
                        uint8_t code = codes[j*index->m + m];
                        dist2 += pq_dist_table[m*index->ksub + code];
                    }
                    out.emplace_back(dist2, ids[j]);
                }
            });
        }
    }

    // 4. 聚合 top-L
    std::vector<std::pair<float,uint32_t>> all;
    size_t total = 0;
    for (auto &a : args) total += a.candidates.size();
    all.reserve(total);
    for (auto &a : args) all.insert(all.end(), a.candidates.begin(), a.candidates.end());
    if (all.size() > L) {
        std::nth_element(all.begin(), all.begin() + L, all.end(),
            [](auto &a, auto &b){ return a.first < b.first; });
        all.resize(L);
    }

    // 5. 精排
    auto reranked = rerank_with_ptd(base, query, all, index->d, k, num_threads);

    // 6. 返回优先队列
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (auto &p : reranked) result.push(p);
    return result;
}