#include <thread>
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>

// 每个线程的输入／输出结构
struct ThreadArgs {
    // 输入
    float* base;               // 所有向量基址
    float* query;              // 查询向量
    size_t vecdim;
    size_t k;
    const std::vector<uint32_t>* probe_list;      // 指向主线程的 probe_list
    const std::vector<std::vector<uint32_t>>* invlists; 
    size_t start_idx, end_idx;  // 本线程处理 probe_list[start_idx .. end_idx)
    // 输出
    std::vector<std::pair<float, uint32_t>> candidates; // 替换为vector而非priority_queue
};

// 线程函数：只处理 probe_list 中的一个区间
void* ivf_thread_fn(void* _args) {
    ThreadArgs* args = (ThreadArgs*)_args;
    auto &probe = *args->probe_list;
    auto &invl  = *args->invlists;

    // 预先分配一定容量以避免频繁的重分配
    args->candidates.reserve(args->k * 10); // 设置一个适当的初始容量

    for (size_t pi = args->start_idx; pi < args->end_idx; ++pi) {
        uint32_t list_id = probe[pi];
        for (uint32_t vid : invl[list_id]) {
            // 计算 IP 距离
            float dot = 0;
            float* vptr = args->base + vid * args->vecdim;
            
            // 向量点积计算
            for (size_t d = 0; d < args->vecdim; ++d) {
                dot += vptr[d] * args->query[d];
            }
            float dist = 1.0f - dot;

            // 直接添加到vector中，无需维护堆结构
            args->candidates.push_back({dist, vid});
        }
    }
    return nullptr;
}

// 并行版 ivf_search
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
                   int num_threads)    // 新增参数：线程数
{
    // —— 第 1 部分：串行粗筛（和原来一样） ——
    struct CentDist { float dist; uint32_t idx; };
    std::vector<CentDist> cd(nlist);
    for (size_t i = 0; i < nlist; ++i) {
        float dot = 0, *cptr = centroids + i*vecdim;
        for (size_t d = 0; d < vecdim; ++d) dot += cptr[d] * query[d];
        cd[i].dist = 1.f - dot; cd[i].idx = uint32_t(i);
    }
    if (nprobe < nlist) {
        std::nth_element(cd.begin(), cd.begin()+nprobe, cd.end(),
                         [](const CentDist &a, const CentDist &b){
                             return a.dist < b.dist;
                         });
    }
    std::vector<uint32_t> probe_list;
    probe_list.reserve(nprobe);
    for (size_t i = 0; i < nprobe; ++i) probe_list.push_back(cd[i].idx);

    // —— 第 2 部分：并行精排，使用线程私有vector ——
    std::vector<ThreadArgs> targs(num_threads);
    std::vector<std::thread> tids(num_threads);

    size_t per = (nprobe + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        auto &A = targs[t];
        A.base       = base;
        A.query      = query;
        A.vecdim     = vecdim;
        A.k          = k;
        A.probe_list = &probe_list;
        A.invlists   = &invlists;
        A.start_idx  = std::min<size_t>(t*per, nprobe);
        A.end_idx    = std::min<size_t>((t+1)*per, nprobe);
        // 启动线程
        tids[t] = std::thread(ivf_thread_fn, &A);
    }
    
    // 等待所有线程结束
    for (int t = 0; t < num_threads; ++t) {
        tids[t].join();
    }

    // —— 第 3 部分：高效归并线程结果 ——
    // 1. 计算总候选项数量
    size_t total_candidates = 0;
    for (const auto &A : targs) {
        total_candidates += A.candidates.size();
    }
    
    // 2. 一次性分配存储空间，合并所有线程的候选项
    std::vector<std::pair<float, uint32_t>> all_candidates;
    all_candidates.reserve(total_candidates);
    
    for (auto &A : targs) {
        all_candidates.insert(all_candidates.end(), A.candidates.begin(), A.candidates.end());
        // 清空线程的candidates节省内存
        std::vector<std::pair<float, uint32_t>>().swap(A.candidates);
    }
    
    // 3. 如果候选数量多于k，使用nth_element找出前k个
    if (all_candidates.size() > k) {
        std::nth_element(
            all_candidates.begin(), 
            all_candidates.begin() + k, 
            all_candidates.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; }
        );
        // 截断至k个元素
        all_candidates.resize(k);
    }
    
    // 4. 构建返回的优先队列
    std::priority_queue<std::pair<float, uint32_t>> global_q;
    for (const auto &cand : all_candidates) {
        global_q.push(cand);
    }
    
    return global_q;
}
