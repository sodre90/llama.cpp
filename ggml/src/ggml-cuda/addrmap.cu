#include "addrmap.cuh"

#include "ggml-impl.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace {

constexpr size_t LOG_THRESHOLD_BYTES = 1024 * 1024;
constexpr size_t FREED_RING_SIZE     = 64;

struct addrmap_entry {
    uintptr_t   base;
    size_t      size;
    const char *kind;
    double      t_alloc;
    double      t_free;
};

struct addrmap_state {
    std::mutex                mtx;
    std::vector<addrmap_entry> live;
    std::deque<addrmap_entry>  freed;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    double now() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

addrmap_state & state() {
    static addrmap_state s;
    return s;
}

double mib(size_t bytes) {
    return bytes / 1024.0 / 1024.0;
}

} // namespace

void ggml_cuda_addrmap_alloc(const char * kind, const void * ptr, size_t size) {
    if (ptr == nullptr) {
        return;
    }
    addrmap_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    const double t = s.now();
    s.live.push_back({(uintptr_t) ptr, size, kind, t, 0.0});
    if (size >= LOG_THRESHOLD_BYTES) {
        GGML_LOG_WARN("cuda-addrmap: alloc %-12s 0x%012" PRIxPTR " - 0x%012" PRIxPTR " %9.1f MiB at t=%.3f s (%zu live)\n",
                kind, (uintptr_t) ptr, (uintptr_t) ptr + size, mib(size), t, s.live.size());
    }
}

void ggml_cuda_addrmap_free(const void * ptr) {
    if (ptr == nullptr) {
        return;
    }
    addrmap_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    auto it = std::find_if(s.live.begin(), s.live.end(), [ptr](const addrmap_entry & e) { return e.base == (uintptr_t) ptr; });
    if (it == s.live.end()) {
        GGML_LOG_WARN("cuda-addrmap: free of unknown 0x%012" PRIxPTR "\n", (uintptr_t) ptr);
        return;
    }
    addrmap_entry e = *it;
    e.t_free = s.now();
    s.live.erase(it);
    s.freed.push_back(e);
    if (s.freed.size() > FREED_RING_SIZE) {
        s.freed.pop_front();
    }
    if (e.size >= LOG_THRESHOLD_BYTES) {
        GGML_LOG_WARN("cuda-addrmap: free  %-12s 0x%012" PRIxPTR " - 0x%012" PRIxPTR " %9.1f MiB at t=%.3f s, lived %.1f s (%zu live)\n",
                e.kind, e.base, e.base + e.size, mib(e.size), e.t_free, e.t_free - e.t_alloc, s.live.size());
    }
}

void ggml_cuda_addrmap_dump(void) {
    addrmap_state & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    const double t = s.now();
    std::vector<addrmap_entry> live = s.live;
    std::sort(live.begin(), live.end(), [](const addrmap_entry & a, const addrmap_entry & b) { return a.base < b.base; });

    GGML_LOG_ERROR("cuda-addrmap: dump at t=%.3f s: %zu live allocations, %zu recent frees (match the Xid 31 fault address from dmesg)\n",
            t, live.size(), s.freed.size());
    for (const addrmap_entry & e : live) {
        GGML_LOG_ERROR("cuda-addrmap:   live  %-12s 0x%012" PRIxPTR " - 0x%012" PRIxPTR " %9.1f MiB, allocated %.1f s ago\n",
                e.kind, e.base, e.base + e.size, mib(e.size), t - e.t_alloc);
    }
    for (auto it = s.freed.rbegin(); it != s.freed.rend(); ++it) {
        const addrmap_entry & e = *it;
        GGML_LOG_ERROR("cuda-addrmap:   freed %-12s 0x%012" PRIxPTR " - 0x%012" PRIxPTR " %9.1f MiB, freed %.1f s ago after %.1f s\n",
                e.kind, e.base, e.base + e.size, mib(e.size), t - e.t_free, e.t_free - e.t_alloc);
    }
}
