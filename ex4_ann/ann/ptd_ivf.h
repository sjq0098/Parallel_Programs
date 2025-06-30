#include <thread>
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <mutex>
#include <condition_variable>

// 简易线程池实现
class ThreadPool {
public:
    ThreadPool(size_t n) : stop_flag(false) {
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

    // 提交任务
    void enqueue(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            tasks.emplace(std::move(f));
        }
        cv.notify_one();
    }

    ~ThreadPool() {
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

// 原有 ThreadArgs，用于存储每个 probe 的候选
struct ThreadArgs {
    std::vector<std::pair<float, uint32_t>> candidates;
};

std::priority_queue<std::pair<float, uint32_t>>
ivf_search_ptd(float* base,
               float* query,
               size_t base_number,
               size_t vecdim,
               size_t k,
               float* centroids,
               size_t nlist,
               const std::vector<std::vector<uint32_t>>& invlists,
               size_t nprobe,
               int num_threads)
{
    // —— 串行粗筛（计算质心距离并选出 probe_list） ——
    struct CentDist { float dist; uint32_t idx; };
    std::vector<CentDist> cd(nlist);
    for (size_t i = 0; i < nlist; ++i) {
        float dot = 0, *cptr = centroids + i * vecdim;
        for (size_t d = 0; d < vecdim; ++d) dot += cptr[d] * query[d];
        cd[i].dist = 1.f - dot;
        cd[i].idx  = uint32_t(i);
    }
    if (nprobe < nlist) {
        std::nth_element(cd.begin(), cd.begin() + nprobe, cd.end(),
                         [](auto &a, auto &b){ return a.dist < b.dist; });
    }
    std::vector<uint32_t> probe_list(nprobe);
    for (size_t i = 0; i < nprobe; ++i) probe_list[i] = cd[i].idx;

    // 确定线程数
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }

    // 为每个 probe 准备结果存储
    std::vector<ThreadArgs> targs(nprobe);

    // —— 并行精排（线程池作用域） ——
    {
        ThreadPool pool(num_threads);
        for (size_t i = 0; i < nprobe; ++i) {
            pool.enqueue([&, i] {
                auto &out = targs[i].candidates;
                out.reserve(k * 10);
                auto &vec_ids = invlists[probe_list[i]];
                for (uint32_t vid : vec_ids) {
                    float dot = 0;
                    float* vptr = base + vid * vecdim;
                    for (size_t d = 0; d < vecdim; ++d) dot += vptr[d] * query[d];
                    float dist = 1.f - dot;
                    out.emplace_back(dist, vid);
                }
            });
        }
        // 离开作用域时，pool 析构并等待所有任务完成
    }

    // —— 合并所有线程结果并选 Top-k ——
    size_t total = 0;
    for (auto &A : targs) total += A.candidates.size();
    std::vector<std::pair<float, uint32_t>> all;
    all.reserve(total);
    for (auto &A : targs) {
        all.insert(all.end(), A.candidates.begin(), A.candidates.end());
    }

    if (all.size() > k) {
        std::nth_element(all.begin(), all.begin() + k, all.end(),
                         [](auto &a, auto &b){ return a.first < b.first; });
        all.resize(k);
    }

    std::priority_queue<std::pair<float, uint32_t>> global_q;
    for (auto &p : all) global_q.push(p);
    return global_q;
}
