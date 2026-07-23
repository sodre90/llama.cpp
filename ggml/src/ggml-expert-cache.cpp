// Expert-residency cache for CPU-resident MoE expert weights.
//
// When a model's routed experts exceed host RAM they demand-page from disk
// through the model mmap. The OS page cache neither protects hot experts from
// eviction by unrelated IO nor overlaps expert slice reads with compute. This
// module tracks per-(tensor, expert) usage from mul_mat_id, pins the hottest
// expert slices in RAM with mlock() up to a byte budget, and issues
// MADV_WILLNEED readahead for the slices the current and upcoming layers are
// about to touch, so slice reads proceed in parallel with the matmuls.
//
// Opt-in via environment:
//   GGML_EXPERT_CACHE=1           master switch
//   GGML_EXPERT_CACHE_MB=4096     mlock budget for hot expert slices
//   GGML_EXPERT_CACHE_PREDICT=1   layers ahead to prefetch using last-used ids
//   GGML_EXPERT_CACHE_STATS=path  periodic CSV dump of usage counters
//
// Locking: `mtx` guards the registry and queue and is the only lock compute
// threads take. `op_mtx` serializes mlock/madvise phases against invalidate()
// so registry entries cannot be erased while the worker holds pointers to
// them across a syscall phase. Order: op_mtx before mtx.

#include "ggml-impl.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

struct expert_tensor_state {
    std::string           name;
    int                   layer      = -1;
    const char *          data       = nullptr;
    size_t                slice_size = 0; // bytes per expert (nb[2])
    int                   n_expert   = 0;
    bool                  file_backed_checked = false;
    bool                  file_backed         = false;
    std::vector<uint32_t> counts;
    std::vector<uint8_t>  pinned;
    std::vector<int32_t>  last_ids;
};

struct prefetch_request {
    int                  layer;
    std::vector<int32_t> ids;
};

// POD copy handed to the syscall phase so no registry pointers are needed there
struct willneed_target {
    const char *         data;
    size_t               slice_size;
    int                  n_expert;
    std::vector<int32_t> ids;
};

struct expert_cache {
    bool        enabled       = false;
    size_t      budget_bytes  = 0;
    int         predict_ahead = 1;
    std::string stats_path;

    std::mutex              op_mtx;
    std::mutex              mtx;
    std::condition_variable cv;

    std::unordered_map<const void *, expert_tensor_state> tensors;
    std::map<int, std::vector<expert_tensor_state *>>     by_layer;
    std::deque<prefetch_request>                          queue;
    uint64_t track_calls  = 0;
    size_t   pinned_bytes = 0;
    bool     mlock_failed = false;

    // last ids issued per layer: gate/down hooks repeat the up hook's routing
    std::map<int, std::vector<int32_t>> last_issued;

    static expert_cache & instance() {
        static expert_cache cache;
        return cache;
    }

    expert_cache() {
        const char * env = getenv("GGML_EXPERT_CACHE");
        enabled = env && atoi(env) != 0;
        if (!enabled) {
            return;
        }
        const char * mb = getenv("GGML_EXPERT_CACHE_MB");
        budget_bytes = (mb ? (size_t) atoll(mb) : 4096) * 1024u * 1024u;
        const char * pred = getenv("GGML_EXPERT_CACHE_PREDICT");
        predict_ahead = pred ? atoi(pred) : 1;
        const char * stats = getenv("GGML_EXPERT_CACHE_STATS");
        stats_path = stats ? stats : "";

        GGML_LOG_INFO("expert-cache: enabled, pin budget %zu MiB, predict %d layer(s) ahead%s%s\n",
                budget_bytes/(1024u*1024u), predict_ahead,
                stats_path.empty() ? "" : ", stats -> ", stats_path.c_str());

        std::thread(&expert_cache::worker, this).detach();
    }

    static int parse_layer(const char * name) {
        // tensor names look like "blk.12.ffn_up_exps.weight"
        if (strncmp(name, "blk.", 4) != 0) {
            return -1;
        }
        return atoi(name + 4);
    }

    // compute-thread fast path: registry/queue mutation only, no syscalls
    void track(const struct ggml_tensor * weights, const int64_t * row_counts, int n_as) {
        std::unique_lock<std::mutex> lock(mtx);
        track_calls++;

        expert_tensor_state & st = tensors[weights->data];
        if (st.data == nullptr) {
            st.name       = weights->name;
            st.layer      = parse_layer(weights->name);
            st.data       = (const char *) weights->data;
            st.slice_size = weights->nb[2];
            st.n_expert   = n_as;
            st.counts.assign(n_as, 0);
            st.pinned.assign(n_as, 0);
            st.file_backed = check_file_backed(st);
            if (st.layer >= 0) {
                by_layer[st.layer].push_back(&st);
            }
        }

        st.last_ids.clear();
        for (int e = 0; e < n_as; e++) {
            if (row_counts[e] > 0) {
                st.counts[e]++;
                st.last_ids.push_back(e);
            }
        }

        // near-dense evals (large-batch prefill) gain nothing from readahead
        if (st.layer >= 0 && (int) st.last_ids.size() <= n_as/4 && queue.size() < 64) {
            queue.push_back({st.layer, st.last_ids});
            lock.unlock();
            cv.notify_one();
        }
    }

    void invalidate(const void * base, size_t size) {
        std::lock_guard<std::mutex> op_lock(op_mtx);
        std::lock_guard<std::mutex> lock(mtx);
        const char * lo = (const char *) base;
        const char * hi = lo + size;
        for (auto it = tensors.begin(); it != tensors.end(); ) {
            expert_tensor_state & st = it->second;
            if (st.data >= lo && st.data < hi) {
                for (int e = 0; e < st.n_expert; e++) {
                    if (st.pinned[e]) {
                        pinned_bytes -= pin_range_size(st, e);
                    }
                }
                if (st.layer >= 0) {
                    auto & vec = by_layer[st.layer];
                    vec.erase(std::remove(vec.begin(), vec.end(), &st), vec.end());
                }
                it = tensors.erase(it);
            } else {
                ++it;
            }
        }
    }

#if defined(__linux__)
    static size_t page_size() {
        static const size_t ps = (size_t) sysconf(_SC_PAGESIZE);
        return ps;
    }

    // page-align inward so we never touch pages shared with neighboring slices
    static bool pin_range(const char * data, size_t slice_size, int e, uintptr_t & start, uintptr_t & end) {
        const size_t ps = page_size();
        start = ((uintptr_t) data + e*slice_size + ps - 1) & ~(ps - 1);
        end   = ((uintptr_t) data + (e + 1)*slice_size) & ~(ps - 1);
        return end > start;
    }

    static size_t pin_range_size(const expert_tensor_state & st, int e) {
        uintptr_t start, end;
        return pin_range(st.data, st.slice_size, e, start, end) ? end - start : 0;
    }

    static bool check_file_backed(const expert_tensor_state & st) {
        FILE * f = fopen("/proc/self/maps", "r");
        if (!f) {
            return false;
        }
        char line[1024];
        bool backed = false;
        while (fgets(line, sizeof(line), f)) {
            unsigned long lo = 0, hi = 0;
            int path_off = 0;
            if (sscanf(line, "%lx-%lx %*s %*s %*s %*s %n", &lo, &hi, &path_off) < 2) {
                continue;
            }
            if ((uintptr_t) st.data >= lo && (uintptr_t) st.data < hi) {
                backed = path_off > 0 && line[path_off] == '/';
                break;
            }
        }
        fclose(f);
        if (!backed) {
            GGML_LOG_INFO("expert-cache: %s is not file-backed, pin/prefetch disabled for it\n", st.name.c_str());
        }
        return backed;
    }

    static void willneed(const willneed_target & t) {
        for (int32_t e : t.ids) {
            if (e < 0 || e >= t.n_expert) {
                continue;
            }
            uintptr_t start, end;
            if (pin_range(t.data, t.slice_size, e, start, end)) {
                posix_madvise((void *) start, end - start, POSIX_MADV_WILLNEED);
            }
        }
    }

    // caller holds op_mtx (so entries cannot disappear) but NOT mtx
    void apply_pin_set() {
        struct candidate {
            expert_tensor_state * st;
            int      expert;
            uint32_t count;
        };
        std::vector<candidate> cands;
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (auto & kv : tensors) {
                expert_tensor_state & st = kv.second;
                if (!st.file_backed) {
                    continue;
                }
                for (int e = 0; e < st.n_expert; e++) {
                    if (st.counts[e] > 0 || st.pinned[e]) {
                        // hysteresis: currently-pinned experts get a small boost
                        cands.push_back({&st, e, st.counts[e] + (st.pinned[e] ? st.counts[e]/8 : 0)});
                    }
                }
            }
        }
        std::sort(cands.begin(), cands.end(), [](const candidate & a, const candidate & b) {
            return a.count > b.count;
        });

        size_t want_bytes = 0;
        std::vector<std::pair<expert_tensor_state *, int>> to_pin;
        std::vector<std::pair<expert_tensor_state *, int>> to_unpin;
        for (auto & c : cands) {
            const size_t sz = pin_range_size(*c.st, c.expert);
            // after an mlock failure, hold the current pin set instead of growing it
            const bool eligible = mlock_failed ? c.st->pinned[c.expert] != 0 : true;
            if (eligible && want_bytes + sz <= budget_bytes) {
                want_bytes += sz;
                if (!c.st->pinned[c.expert]) {
                    to_pin.push_back({c.st, c.expert});
                }
            } else if (c.st->pinned[c.expert]) {
                to_unpin.push_back({c.st, c.expert});
            }
        }

        // mlock faults pages in from disk and can take a while; only op_mtx held
        std::vector<std::pair<expert_tensor_state *, int>> unpinned_ok;
        std::vector<std::pair<expert_tensor_state *, int>> pinned_ok;
        for (auto & p : to_unpin) {
            uintptr_t start, end;
            if (pin_range(p.first->data, p.first->slice_size, p.second, start, end)) {
                munlock((void *) start, end - start);
            }
            unpinned_ok.push_back(p);
        }
        for (auto & p : to_pin) {
            uintptr_t start, end;
            if (!pin_range(p.first->data, p.first->slice_size, p.second, start, end)) {
                continue;
            }
            if (mlock((void *) start, end - start) != 0) {
                GGML_LOG_WARN("expert-cache: mlock failed (%s), pinning capped at current set\n", strerror(errno));
                mlock_failed = true;
                break;
            }
            pinned_ok.push_back(p);
        }

        size_t pinned_now = 0;
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (auto & p : unpinned_ok) {
                if (p.first->pinned[p.second]) {
                    p.first->pinned[p.second] = 0;
                    pinned_bytes -= pin_range_size(*p.first, p.second);
                }
            }
            for (auto & p : pinned_ok) {
                if (!p.first->pinned[p.second]) {
                    p.first->pinned[p.second] = 1;
                    pinned_bytes += pin_range_size(*p.first, p.second);
                }
            }
            pinned_now = pinned_bytes;
        }
        if (!pinned_ok.empty() || !unpinned_ok.empty()) {
            GGML_LOG_INFO("expert-cache: pinned %zu MiB of hot experts (+%zu/-%zu slices this pass)\n",
                    pinned_now/(1024u*1024u), pinned_ok.size(), unpinned_ok.size());
        }
    }
#else
    static size_t pin_range_size(const expert_tensor_state &, int) { return 0; }
    static bool check_file_backed(const expert_tensor_state &) { return false; }
    static void willneed(const willneed_target &) {}
    void apply_pin_set() {}
#endif

    void dump_stats() {
        std::string tmp = stats_path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "w");
        if (!f) {
            return;
        }
        fprintf(f, "tensor,layer,expert,count,pinned\n");
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (auto & kv : tensors) {
                expert_tensor_state & st = kv.second;
                for (int e = 0; e < st.n_expert; e++) {
                    if (st.counts[e] > 0 || st.pinned[e]) {
                        fprintf(f, "%s,%d,%d,%u,%d\n", st.name.c_str(), st.layer, e, st.counts[e], st.pinned[e] ? 1 : 0);
                    }
                }
            }
            fprintf(f, "#pinned_bytes,%zu\n#track_calls,%llu\n", pinned_bytes, (unsigned long long) track_calls);
        }
        fclose(f);
        rename(tmp.c_str(), stats_path.c_str());
    }

    void worker() {
        using clock = std::chrono::steady_clock;
        auto next_pin   = clock::now() + std::chrono::seconds(20);
        auto next_decay = clock::now() + std::chrono::seconds(300);
        auto next_stats = clock::now() + std::chrono::seconds(60);

        for (;;) {
            std::deque<prefetch_request> local;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait_for(lock, std::chrono::seconds(1), [this] { return !queue.empty(); });
                local.swap(queue);
            }

            {
                std::lock_guard<std::mutex> op_lock(op_mtx);
                process_prefetches(local);

                const auto now = clock::now();
                if (now >= next_pin && budget_bytes > 0) {
                    bool warmed_up;
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        warmed_up = track_calls >= 100;
                    }
                    if (warmed_up) {
                        apply_pin_set();
                    }
                    next_pin = now + std::chrono::seconds(20);
                }
            }

            const auto now = clock::now();
            if (now >= next_decay) {
                std::lock_guard<std::mutex> lock(mtx);
                for (auto & kv : tensors) {
                    for (auto & c : kv.second.counts) {
                        c >>= 1;
                    }
                }
                next_decay = now + std::chrono::seconds(300);
            }
            if (!stats_path.empty() && now >= next_stats) {
                dump_stats();
                next_stats = now + std::chrono::seconds(60);
            }
        }
    }

    // caller holds op_mtx but NOT mtx
    void process_prefetches(std::deque<prefetch_request> & local) {
        for (auto & req : local) {
            std::vector<willneed_target> targets;
            {
                std::lock_guard<std::mutex> lock(mtx);

                auto & issued = last_issued[req.layer];
                const bool duplicate = issued == req.ids;
                issued = req.ids;

                if (!duplicate) {
                    auto it = by_layer.find(req.layer);
                    if (it != by_layer.end()) {
                        for (expert_tensor_state * st : it->second) {
                            if (st->file_backed) {
                                targets.push_back({st->data, st->slice_size, st->n_expert, req.ids});
                            }
                        }
                    }
                }
                auto ahead = by_layer.upper_bound(req.layer);
                for (int i = 0; i < predict_ahead && ahead != by_layer.end(); i++, ++ahead) {
                    for (expert_tensor_state * st : ahead->second) {
                        if (st->file_backed && !st->last_ids.empty()) {
                            targets.push_back({st->data, st->slice_size, st->n_expert, st->last_ids});
                        }
                    }
                }
            }
            for (auto & t : targets) {
                willneed(t);
            }
        }
    }
};

} // namespace

extern "C" void ggml_expert_cache_track(const struct ggml_tensor * weights, const int64_t * row_counts, int n_as) {
    expert_cache & cache = expert_cache::instance();
    if (!cache.enabled) {
        return;
    }
    cache.track(weights, row_counts, n_as);
}

extern "C" void ggml_expert_cache_invalidate(const void * base, size_t size) {
    expert_cache & cache = expert_cache::instance();
    if (!cache.enabled) {
        return;
    }
    cache.invalidate(base, size);
}
