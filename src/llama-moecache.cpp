#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <unistd.h>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

namespace {

// a host expert matrix in its model file: the descriptor the reader opened and the offset of expert 0
struct host_file {
    int    fd     = -1;
    size_t offset = 0;
};

struct layer_state {
    llama_moe_cache_layer pub;

    // LRU and SLRU bookkeeping (host side; tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;    // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;    // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use;  // slot -> lamport clock of last hit
    std::vector<bool>     slot_protected; // SLRU: true if in protected segment

    // [TAG_MOE_CACHE_DEMAND_RANK] uncached ids observed this step and the step before, with their
    // (token, routed-id) multiplicity: an expert wanted by several sequences ranks first, and a
    // miss survives one step so a budget of 1-2 inserts is spent on the most wanted experts
    // instead of whichever miss was observed last. Ordering only - every step still spends its
    // full insert budget (LFU admission gating is refuted).
    std::vector<int32_t>  pending;        // ids with demand_cur > 0 or demand_prev > 0 (dedup)
    std::vector<uint16_t> demand_cur;     // expert id -> misses observed this step
    std::vector<uint16_t> demand_prev;    // expert id -> misses observed the previous step

    std::vector<bool>     slot_in_flight; // slot has an upload pending

    // [TAG_MOE_CACHE_WARM_FILL] experts a prefill ubatch already streamed to the device and that
    // are being copied into free or probation slots device-to-device; published at the next step
    // once all three of up/gate/down have been copied (each sets its bit in `copied`)
    struct warm_entry {
        int32_t expert;
        int32_t slot;
        int     copied;
    };
    std::vector<warm_entry> warm_plan;   // the ubatch being staged now
    std::vector<warm_entry> warm_done;   // fully copied by an earlier ubatch of this step, awaiting publish
    int                     warm_seen = 0; // up/gate/down bits already staged for warm_plan
    std::vector<uint16_t>   warm_demand;   // scratch: expert id -> tokens routed to it

    int32_t n_slots       = 0; // this layer's slot count (mirrors pub.n_slots)
    int32_t max_protected = 0; // SLRU protected cap for this layer

    uint64_t n_hit    = 0;
    uint64_t n_miss   = 0;
    uint64_t n_insert = 0;
    uint64_t n_evict  = 0;
    uint64_t n_warm   = 0; // slots filled from a prefill's staged copy

    uint64_t last_log_hits   = 0;
    uint64_t last_log_misses = 0;

    // host tier (pub.n_host_slots > 0), changed only by host_map_op: which pool slot holds which expert, and the
    // decayed use count (same policy as the device cache) that picks the slot to reuse
    std::vector<int32_t>  host_slot;        // expert id -> pool slot, -1 when absent
    std::vector<int32_t>  host_slot_expert; // pool slot -> expert id, -1 when empty
    std::vector<float>    host_score;       // expert id -> use count as of host_last
    std::vector<uint32_t> host_last;        // expert id -> host_clock when it was last routed
    std::vector<uint8_t>  host_routed;      // expert id -> routed by the running ubatch
    uint32_t              host_clock = 0;

    uint64_t n_host_hit      = 0; // routed experts already in the pool
    uint64_t n_host_miss     = 0; // routed experts copied into the pool
    uint64_t host_bytes_read = 0; // bytes copied into the pool
    uint64_t host_read_us    = 0; // time spent copying

    std::array<host_file, 3> host_files; // up, gate, down; set when the tier has a reader
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
};

// opens path for O_DIRECT reads and raises alignment to the block size of its file; -1 with errno set when it cannot
int open_direct(const char * path, size_t & alignment) {
#if !defined(_WIN32) && defined(O_DIRECT)
    const int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        return -1;
    }
    struct stat file_stat = {};
    if (fstat(fd, &file_stat) != 0) {
        close(fd);
        return -1;
    }
    alignment = std::max<size_t>(alignment, file_stat.st_blksize);
    return fd;
#else
    GGML_UNUSED(path);
    GGML_UNUSED(alignment);
    errno = ENOTSUP;
    return -1;
#endif
}

// pread of at most n bytes; -1 with errno set on error
int64_t read_at(int fd, void * buf, size_t n, size_t offset) {
#ifndef _WIN32
    return pread(fd, buf, n, (off_t) offset);
#else
    GGML_UNUSED(fd);
    GGML_UNUSED(buf);
    GGML_UNUSED(n);
    GGML_UNUSED(offset);
    errno = ENOTSUP;
    return -1;
#endif
}

// parallel O_DIRECT reads of byte ranges of the model files, so that nothing goes through the page cache
struct expert_reader {
    struct job {
        int          fd;
        size_t       offset; // in the file
        void *       dst;
        size_t       len;
        const void * mapped; // the same bytes in the mapping, copied when the read fails
        int32_t *    done_flag = nullptr; // when set, stored 0 once every byte of the job is in dst
    };

    // demand reads are taken before speculative ones
    enum class priority { high, low };

    // the chunks of one submit that are not done yet
    struct batch {
        size_t  pending     = 0;
        int64_t t_submit_us = 0;
        int64_t t_done_us   = 0; // when pending reached 0
    };
    using ticket = std::shared_ptr<batch>;

    struct queued_chunk {
        job    part;
        ticket owner;
        std::shared_ptr<std::atomic<size_t>> job_chunks_left; // the chunks of the job of part that are not done, null without done_flag
    };

    std::vector<int>         fds;
    size_t                   alignment   = 4096;
    size_t                   chunk_bytes = 512*1024;
    std::vector<void *>      bounces;
    std::vector<std::thread> threads;

    std::mutex               mtx; // guards the members below and batch::pending
    std::condition_variable  work_cv;
    std::condition_variable  done_cv;
    std::deque<queued_chunk> high;
    std::deque<queued_chunk> low;
    bool                     stop = false;

    std::atomic<bool> warned{false};
    std::atomic<int>  failed_errno{0}; // the first read error a worker saw

    ~expert_reader() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        work_cv.notify_all();
        for (auto & thread : threads) {
            thread.join();
        }
        for (const std::deque<queued_chunk> * queue : {&high, &low}) {
            std::for_each(queue->begin(), queue->end(), finish_job_chunk);
        }
        for (void * bounce : bounces) {
            free(bounce);
        }
        for (const int fd : fds) {
            close(fd);
        }
    }

    // false when a read buffer cannot be allocated
    bool start(int n_threads) {
        for (int i = 0; i < n_threads; ++i) {
            void * bounce = nullptr;
            if (posix_memalign(&bounce, alignment, chunk_bytes + 2*alignment) != 0) {
                return false;
            }
            bounces.push_back(bounce);
        }
        for (void * bounce : bounces) {
            threads.emplace_back([this, bounce]() { work((char *) bounce); });
        }
        return true;
    }

    // queues the jobs and returns at once; wait(ticket) returns when every byte of every job is in its dst
    ticket submit(const std::vector<job> & jobs, priority prio) {
        report_failure();
        auto t = std::make_shared<batch>();
        t->t_submit_us = t->t_done_us = ggml_time_us();
        std::lock_guard<std::mutex> lock(mtx);
        std::deque<queued_chunk> & queue = prio == priority::high ? high : low;
        for (const job & j : jobs) {
            std::shared_ptr<std::atomic<size_t>> chunks_left;
            if (j.done_flag && j.len == 0) {
                __atomic_store_n(j.done_flag, 0, __ATOMIC_RELEASE);
            } else if (j.done_flag) {
                chunks_left = std::make_shared<std::atomic<size_t>>((j.len + chunk_bytes - 1)/chunk_bytes);
            }
            for (size_t done = 0; done < j.len; done += chunk_bytes) {
                queue.push_back({{j.fd, j.offset + done, (char *) j.dst + done, std::min(chunk_bytes, j.len - done), (const char *) j.mapped + done, j.done_flag}, t, chunks_left});
                t->pending++;
            }
        }
        work_cv.notify_all();
        return t;
    }

    void wait(const ticket & t) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            done_cv.wait(lock, [&t]() { return t->pending == 0; });
        }
        report_failure();
    }

    // the chunks of t that no worker has started are taken before every speculative one
    void promote(const ticket & t) {
        std::lock_guard<std::mutex> lock(mtx);
        if (t->pending == 0) {
            return;
        }
        const auto first_of_t = std::stable_partition(low.begin(), low.end(), [&t](const queued_chunk & c) { return c.owner != t; });
        std::move(first_of_t, low.end(), std::back_inserter(high));
        low.erase(first_of_t, low.end());
    }

    // returns when every byte of every job is in its dst
    void read(const std::vector<job> & jobs) {
        wait(submit(jobs, priority::high));
    }

    void work(char * bounce) {
        std::unique_lock<std::mutex> lock(mtx);
        for (;;) {
            work_cv.wait(lock, [this]() { return stop || !high.empty() || !low.empty(); });
            if (stop) {
                return;
            }
            std::deque<queued_chunk> & queue = high.empty() ? low : high;
            queued_chunk next = std::move(queue.front());
            queue.pop_front();
            lock.unlock();
            if (const int err = read_chunk(next.part, bounce); err != 0) {
                copy_from_mapping(next.part, err);
            }
            finish_job_chunk(next);
            lock.lock();
            if (--next.owner->pending == 0) {
                next.owner->t_done_us = ggml_time_us();
                done_cv.notify_all();
            }
        }
    }

    // The thread that ends the last chunk of a job clears its flag, which a device kernel may wait on. It takes no lock and calls no backend:
    // the main thread can hold any lock while it waits for that kernel.
    static void finish_job_chunk(const queued_chunk & chunk) {
        if (!chunk.job_chunks_left) {
            return;
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (chunk.job_chunks_left->fetch_sub(1, std::memory_order_acq_rel) == 1) {
            __atomic_store_n(chunk.part.done_flag, 0, __ATOMIC_RELEASE);
        }
    }

    // 0, or the errno of the failed read; a read that stops at the end of the file is fine when it covers the chunk
    int read_chunk(const job & chunk, char * bounce) const {
        const size_t begin = chunk.offset/alignment*alignment;
        const size_t end   = (chunk.offset + chunk.len + alignment - 1)/alignment*alignment;
        const size_t skip  = chunk.offset - begin;
        size_t got = 0;
        while (got < skip + chunk.len) {
            const int64_t n = read_at(chunk.fd, bounce + got, end - begin - got, begin + got);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n < 0) {
                return errno;
            }
            if (n == 0) {
                return EIO;
            }
            got += (size_t) n;
        }
        memcpy(chunk.dst, bounce + skip, chunk.len);
        return 0;
    }

    // no logging here, see finish_job_chunk
    void copy_from_mapping(const job & chunk, int err) {
        int no_error = 0;
        failed_errno.compare_exchange_strong(no_error, err);
        memcpy(chunk.dst, chunk.mapped, chunk.len);
    }

    void report_failure() {
        if (const int err = failed_errno.load(); err != 0 && !warned.exchange(true)) {
            LLAMA_LOG_WARN("moe-cache: direct read of the model file failed (%s), copying from the mapping\n", strerror(err));
        }
    }
};

// the experts of one layer that a prefill reads from the files ahead of its upload, in ascending id order:
// [up experts][gate experts][down experts]
struct lookahead_buffer {
    ggml_backend_buffer_t buf = nullptr;
    int64_t               layer = -1;  // index in moe_cache::layers of the layer the buffer holds, -1: none
    int32_t               count = 0;   // experts per matrix in the buffer
    std::vector<int32_t>  position;    // expert id -> index among the buffered experts, -1 when not buffered
    expert_reader::ticket reads;       // the reads that fill the buffer
};

struct guess_stats {
    uint64_t read    = 0; // experts read ahead because a guess named them
    uint64_t used    = 0; // of those, experts the real routing then used
    uint64_t wanted  = 0; // experts the real routing used that the pool did not hold before the guess
    uint64_t wait_us = 0; // time spent waiting for the reads at the start of the layer's step

    guess_stats & operator+=(const guess_stats & other) {
        read    += other.read;
        used    += other.used;
        wanted  += other.wanted;
        wait_us += other.wait_us;
        return *this;
    }

    guess_stats operator-(const guess_stats & other) const {
        return {read - other.read, used - other.used, wanted - other.wanted, wait_us - other.wait_us};
    }
};

// what the guess made while the previous layer ran left in a layer's pool, until the layer's real routing arrives
struct host_guess {
    expert_reader::ticket reads;        // the reads into the pool, nullptr: none
    bool                  made = false; // a guess was made for the layer's next step
    std::vector<int32_t>  admitted;     // experts the guess put into the pool
    std::vector<int32_t>  evicted;      // experts the guess pushed out of the pool
    guess_stats           stats;
};

struct moe_cache {
    int32_t n_slots       = 0;  // default slots per layer; LLAMA_MOE_CACHE_LAYER_SLOTS overrides per layer
    int32_t max_inserts   = 2;
    int32_t protected_pct = 75; // SLRU protected segment share, LLAMA_MOE_CACHE_PROTECTED_PCT
    float   decay         = 0.05f; // device policy: an expert's score halves every 1/decay steps, LLAMA_MOE_CACHE_DECAY
    bool    rank_by_demand = true; // LLAMA_MOE_CACHE_INSERT_ORDER=recency restores last-observed-first
    int32_t warm_max      = 0;  // LLAMA_MOE_CACHE_WARM_MAX: slots per layer a prefill may fill from its staged copy, 0 = off
    int32_t max_in_flight = 4;  // LLAMA_MOE_CACHE_MAX_IN_FLIGHT: pending uploads per layer, 0 = unbounded

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    uint64_t last_log_hits   = 0;
    uint64_t last_log_misses = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec) and the host tier counters

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;

    // any of the three host weights -> its layer and the cache tensor holding its rows
    struct cached_rows {
        size_t layer_idx;
        ggml_tensor * llama_moe_cache_layer::* rows;
    };
    std::map<const ggml_tensor *, cached_rows> by_src;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;

    // host tier: slots per layer (0 = off) and the pools, one buffer per layer
    int32_t                             host_slots = 0;
    std::vector<ggml_context *>         host_ctxs;
    std::vector<ggml_backend_buffer_t>  host_bufs;
    std::unique_ptr<expert_reader>      host_reader; // nullptr: the tier copies from the mapping
    ggml_backend_buffer_type_t          host_buft = nullptr; // where the pools and lookahead buffers live

    // host_map_op queues the reads of a ubatch's missing experts and returns; the device kernels wait for them (LLAMA_MOE_HOST_OVERLAP)
    bool                                host_overlap = false;
    std::string                         host_overlap_state = "off";
    ggml_backend_buffer_t               host_pending_buf = nullptr; // the read flags of every layer, see llama_moe_cache_layer::host_pending
    std::vector<expert_reader::ticket>  demand_reads;      // per layer, parallel to layers: the reads host_map_op left running, nullptr: none

    // a prefill ubatch reads the next layer's experts ahead into one of two buffers, by layer parity (LLAMA_MOE_HOST_LOOKAHEAD)
    bool                                lookahead = false;
    float                               lookahead_min_used = 0.9f;
    int64_t                             lookahead_layer = -1; // the layer of the last host weight the scheduler asked about
    std::array<lookahead_buffer, 2>     lookahead_bufs;

    // a decode ubatch guesses the next layer's routing and reads the guessed experts into its pool (LLAMA_MOE_HOST_PREDICT)
    bool                                predict     = false;
    size_t                              predict_max = 8; // LLAMA_MOE_HOST_PREDICT_MAX: experts read ahead per layer and step
    std::vector<host_guess>             host_guesses;    // per layer, parallel to layers
    guess_stats                         last_log_guess;

    uint64_t last_log_host_hits   = 0;
    uint64_t last_log_host_misses = 0;
    uint64_t last_log_host_bytes  = 0;
    uint64_t last_log_host_us     = 0;

    // every layer's host table is a row of host_tables; each device group's tables are views of one
    // block whose rows are a contiguous run of host_tables, so a step publishes them in one copy each
    struct table_block {
        ggml_tensor * dev;
        size_t        host_offset;
    };
    ggml_tensor *            host_tables = nullptr;
    std::vector<table_block> table_blocks;
    bool                     tables_dirty = false;

    // see llama_moe_cache_layer::routing; nullptr unless the layers read host experts
    ggml_tensor *        routing = nullptr;
    std::vector<int32_t> routing_host;

    // device policy (LLAMA_MOE_CACHE_DEVICE): one I32 row of GPU-owned state per layer, see alloc_device_state
    bool                  device_policy = false;
    ggml_tensor *         dev_state     = nullptr;
    std::vector<int32_t>  state_host;
    std::vector<uint32_t> prev_counters; // per layer: raw n_hit, n_miss, n_fill, n_evict of the last mirror
    bool                  state_dirty = false; // a ubatch ran the device policy after the last mirror

    // LLAMA_MOE_CACHE_AUDIT=N: slots whose bytes are compared with the host experts per step, 0 = off
    int32_t      audit = 0;
    std::mt19937 audit_rng;
    int64_t      audit_compared = 0;
    int64_t      audit_differ   = 0;
    int64_t      audit_table_errors = 0;
    int64_t      audit_reports  = 0;
    int64_t      audit_mid_mirrors = 0; // mirrors before a staging ubatch of the same decode
    int64_t      audit_mid_stale   = 0; // host table entries those mirrors corrected

    FILE * trace = nullptr; // LLAMA_MOE_CACHE_TRACE=<path>: routed ids, one line per layer and token

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;

    // the worker's own stream per cache device: it queues a whole batch of copies and waits once,
    // instead of one synchronous copy per slice
    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_backend_t>> upload_backends;
};

moe_cache * g_cache = nullptr;
std::mutex g_init_mtx;
bool g_init_done = false;

int parse_layer_from_name(const char * name) {
    // "blk.<il>.ffn_gate_exps.weight"
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    return atoi(name + 4);
}

void promote_to_protected(layer_state & ls, int32_t slot, int32_t n_slots, int32_t max_protected, uint64_t clock);
int32_t find_eviction_victim(const layer_state & ls, int32_t n_slots);

void observe_routed_id(moe_cache & mc, layer_state & ls, int32_t id) {
    if (id < 0 || id >= (int32_t) ls.expert_slot.size()) {
        return;
    }

    const int32_t slot = ls.expert_slot[id];
    if (slot >= 0) {
        ls.n_hit++;
        promote_to_protected(ls, slot, ls.n_slots, ls.max_protected, ++mc.clock);
    } else {
        ls.n_miss++;
        if (slot == -2) {
            return;
        }
        if (ls.demand_cur[id] == 0 && ls.demand_prev[id] == 0) {
            ls.pending.push_back(id);
        }
        if (ls.demand_cur[id] < UINT16_MAX) {
            ls.demand_cur[id]++;
        }
    }
}

void trace_routed_ids(moe_cache & mc, int il, const int32_t * ids, int64_t n_ids) {
    fprintf(mc.trace, "blk.%d.ffn_gate_exps.weight", il);
    std::for_each(ids, ids + n_ids, [&](int32_t id) { fprintf(mc.trace, " %d", id); });
    fputc('\n', mc.trace);
}

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens > LLAMA_MOE_CACHE_MAX_TOKENS) {
        return; // batch/prefill: the cache graph is not built there, don't pollute the LRU
    }

    const int il = parse_layer_from_name(name);
    if (il < 0) {
        return;
    }

    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (!ls) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t t = 0; t < n_tokens; ++t) {
        if (mc->trace) {
            std::vector<int32_t> token_ids(n_ids);
            for (int64_t i = 0; i < n_ids; ++i) {
                token_ids[i] = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            }
            trace_routed_ids(*mc, il, token_ids.data(), n_ids);
        }
        for (int64_t i = 0; i < n_ids; ++i) {
            observe_routed_id(*mc, *ls, *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]));
        }
    }
}

void trace_device_routing(moe_cache & mc, int64_t n_ids) {
    const int64_t n_expert_used = n_ids / LLAMA_MOE_CACHE_MAX_TOKENS;

    std::vector<const layer_state *> by_il;
    for (const auto & ls : mc.layers) {
        by_il.push_back(&ls);
    }
    std::sort(by_il.begin(), by_il.end(), [](const layer_state * a, const layer_state * b) { return a->pub.il < b->pub.il; });

    for (const layer_state * ls : by_il) {
        const int32_t * row = mc.routing_host.data() + ls->pub.routing_row*n_ids;
        for (int64_t t = 0; t < LLAMA_MOE_CACHE_MAX_TOKENS; ++t) {
            if (row[t*n_expert_used] >= 0) {
                trace_routed_ids(mc, ls->pub.il, row + t*n_expert_used, n_expert_used);
            }
        }
    }
    fflush(mc.trace);
}

// reads the routing back into mc.routing_host and resets it; call under mc.mtx with no graph in flight
int64_t read_device_routing(moe_cache & mc) {
    mc.routing_host.resize(ggml_nelements(mc.routing));
    ggml_backend_tensor_get(mc.routing, mc.routing_host.data(), 0, ggml_nbytes(mc.routing));
    ggml_backend_tensor_memset(mc.routing, 0xFF, 0, ggml_nbytes(mc.routing));
    return mc.routing->ne[0];
}

// the routing a reads_host_experts graph left on the device; call under mc.mtx with no graph in flight
void observe_device_routing(moe_cache & mc) {
    const int64_t n_ids = read_device_routing(mc);

    for (auto & ls : mc.layers) {
        const int32_t * row = mc.routing_host.data() + ls.pub.routing_row*n_ids;
        std::for_each(row, row + n_ids, [&](int32_t id) { observe_routed_id(mc, ls, id); });
    }

    if (mc.trace) {
        trace_device_routing(mc, n_ids);
    }
}

ggml_backend_t upload_backend_of(const moe_cache & mc, const ggml_tensor * dst_c) {
    const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(dst_c->buffer);
    for (const auto & ub : mc.upload_backends) {
        if (ub.first == buft) {
            return ub.second;
        }
    }
    return nullptr;
}

// queued on the cache device's upload stream when it has one; the worker waits for the whole batch
void upload_slice(const moe_cache & mc, ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    const void * data   = (const char *) src->data + (size_t) expert*sz;
    const size_t offset = (size_t) slot*dst_c->nb[2];

    if (ggml_backend_t backend = upload_backend_of(mc, dst_c)) {
        ggml_backend_tensor_set_async(backend, dst_c, data, offset, sz);
    } else {
        ggml_backend_tensor_set(dst_c, data, offset, sz);
    }
}

// the device tables follow at the next flush_tables()
void set_table_entry(moe_cache & mc, llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
    mc.tables_dirty = true;
}

void flush_tables(moe_cache & mc) {
    if (!mc.tables_dirty) {
        return;
    }
    for (const auto & b : mc.table_blocks) {
        ggml_backend_tensor_set(b.dev, (const char *) mc.host_tables->data + b.host_offset, 0, ggml_nbytes(b.dev));
    }
    mc.tables_dirty = false;
}

void set_host_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
}

// while graphs may be in flight: the device table changes in stream order, after the graph that
// may still gather through it and before the one built from the host table (one pageable H2D,
// which CUDA fences against the stream, per layer)
void mirror_table_in_stream(llama_moe_cache_layer & pub, ggml_backend_t backend) {
    ggml_backend_tensor_set_async(backend, pub.dev_table, pub.host_table->data, 0, ggml_nbytes(pub.host_table));
}

void promote_to_protected(layer_state & ls, int32_t slot, int32_t n_slots, int32_t max_protected, uint64_t clock) {
    ls.slot_last_use[slot] = clock;
    if (ls.slot_protected[slot]) {
        return;
    }

    ls.slot_protected[slot] = true;

    int n_protected = 0;
    for (int32_t s = 0; s < n_slots; ++s) {
        if (ls.slot_protected[s]) {
            n_protected++;
        }
    }

    if (n_protected <= max_protected) {
        return;
    }

    int32_t lru_slot = -1;
    uint64_t oldest_clock = UINT64_MAX;
    for (int32_t s = 0; s < n_slots; ++s) {
        if (ls.slot_protected[s] && s != slot && ls.slot_last_use[s] < oldest_clock) {
            oldest_clock = ls.slot_last_use[s];
            lru_slot = s;
        }
    }

    if (lru_slot >= 0) {
        ls.slot_protected[lru_slot] = false;
    }
}

int32_t find_eviction_victim(const layer_state & ls, int32_t n_slots) {
    for (int32_t s = 0; s < n_slots; ++s) {
        if (!ls.slot_in_flight[s] && ls.slot_expert[s] < 0) {
            return s;
        }
    }

    int32_t victim = -1;
    uint64_t oldest_probation = UINT64_MAX;
    for (int32_t s = 0; s < n_slots; ++s) {
        if (!ls.slot_in_flight[s] && !ls.slot_protected[s] && ls.slot_last_use[s] < oldest_probation) {
            oldest_probation = ls.slot_last_use[s];
            victim = s;
        }
    }

    if (victim >= 0) {
        return victim;
    }

    uint64_t oldest_protected = UINT64_MAX;
    for (int32_t s = 0; s < n_slots; ++s) {
        if (!ls.slot_in_flight[s] && ls.slot_last_use[s] < oldest_protected) {
            oldest_protected = ls.slot_last_use[s];
            victim = s;
        }
    }

    return victim;
}


// "il:slots,il:slots,..." - slots for the listed layers, the default for the rest
std::map<int, int32_t> parse_layer_slots(const char * spec) {
    std::map<int, int32_t> out;
    if (spec == nullptr) {
        return out;
    }
    const char * p = spec;
    while (*p) {
        char * end = nullptr;
        const long il = strtol(p, &end, 10);
        if (end == p || *end != ':') {
            LLAMA_LOG_WARN("moe-cache: LLAMA_MOE_CACHE_LAYER_SLOTS: cannot parse at '%s' - ignoring the rest\n", p);
            break;
        }
        p = end + 1;
        const long slots = strtol(p, &end, 10);
        if (end == p || slots < 1) {
            LLAMA_LOG_WARN("moe-cache: LLAMA_MOE_CACHE_LAYER_SLOTS: bad slot count at '%s' - ignoring the rest\n", p);
            break;
        }
        out[(int) il] = (int32_t) slots;
        p = *end == ',' ? end + 1 : end;
    }
    return out;
}

// most wanted first: this step's demand, then last step's; recency order keeps the previous
// behaviour (the last observed miss first) for an in-place A/B
void order_pending(layer_state & ls, bool by_demand) {
    if (by_demand) {
        std::stable_sort(ls.pending.begin(), ls.pending.end(), [&ls](int32_t a, int32_t b) {
            if (ls.demand_cur[a] != ls.demand_cur[b]) {
                return ls.demand_cur[a] > ls.demand_cur[b];
            }
            return ls.demand_prev[a] > ls.demand_prev[b];
        });
    } else {
        std::reverse(ls.pending.begin(), ls.pending.end());
    }
}

// this step's misses survive one more step (demand order only); older ones and scheduled
// inserts drop out
void age_demand(layer_state & ls, bool keep_one_step) {
    std::vector<int32_t> kept;
    kept.reserve(ls.pending.size());
    for (int32_t id : ls.pending) {
        const bool scheduled = ls.expert_slot[id] == -2 || ls.expert_slot[id] >= 0;
        ls.demand_prev[id] = scheduled || !keep_one_step ? 0 : ls.demand_cur[id];
        ls.demand_cur[id]  = 0;
        if (ls.demand_prev[id] > 0) {
            kept.push_back(id);
        }
    }
    ls.pending.swap(kept);
}

// the ubatch's staging is over: fully copied entries wait for the next step to be published,
// a partial one gives its slot back (its table entry is still the dummy)
void close_warm_plan(layer_state & ls) {
    for (const auto & w : ls.warm_plan) {
        if (w.copied == 7) {
            ls.warm_done.push_back(w);
        } else {
            ls.expert_slot[w.expert]  = -1;
            ls.slot_in_flight[w.slot] = false;
            if (w.copied != 0) {
                LLAMA_LOG_WARN("moe-cache: layer %d expert %d warm-fill incomplete (copied mask %d), slot %d released\n",
                        ls.pub.il, w.expert, w.copied, w.slot);
            }
        }
    }
    ls.warm_plan.clear();
    ls.warm_seen = 0;
}

// between graphs: the copies ran on the compute stream ahead of the next graph, publish them
void settle_warm_plans(moe_cache & mc) {
    for (auto & ls : mc.layers) {
        close_warm_plan(ls);
        for (const auto & w : ls.warm_done) {
            ls.slot_expert[w.slot]    = w.expert;
            ls.expert_slot[w.expert]  = w.slot;
            ls.slot_last_use[w.slot]  = ++mc.clock;
            ls.slot_in_flight[w.slot] = false;
            ls.n_warm++;
            set_table_entry(mc, ls.pub, w.expert, w.slot);
        }
        ls.warm_done.clear();
    }
}

// victims: empty slots, then probation slots holding an expert this ubatch does not route to,
// least recently used first; protected and in-flight slots are never taken
void plan_warm_fill(moe_cache & mc, layer_state & ls, const int32_t * ids, int64_t ne0, int64_t ne1, size_t s0, size_t s1, ggml_backend_t backend) {
    const int64_t n_expert = (int64_t) ls.warm_demand.size();
    std::fill(ls.warm_demand.begin(), ls.warm_demand.end(), 0);
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            const int32_t id = ids[i1*s1 + i0*s0];
            if (id >= 0 && id < n_expert && ls.warm_demand[id] < UINT16_MAX) {
                ls.warm_demand[id]++;
            }
        }
    }

    std::vector<int32_t> wanted;
    for (int32_t id = 0; id < n_expert; ++id) {
        if (ls.warm_demand[id] > 0 && ls.expert_slot[id] == -1) {
            wanted.push_back(id);
        }
    }
    std::stable_sort(wanted.begin(), wanted.end(), [&ls](int32_t a, int32_t b) { return ls.warm_demand[a] > ls.warm_demand[b]; });
    if ((int32_t) wanted.size() > mc.warm_max) {
        wanted.resize(mc.warm_max);
    }

    std::vector<int32_t> victims;
    for (int32_t s = 0; s < ls.n_slots && victims.size() < wanted.size(); ++s) {
        if (ls.slot_expert[s] < 0 && !ls.slot_in_flight[s]) {
            victims.push_back(s);
        }
    }
    if (victims.size() < wanted.size()) {
        std::vector<int32_t> probation;
        for (int32_t s = 0; s < ls.n_slots; ++s) {
            const int32_t e = ls.slot_expert[s];
            if (e >= 0 && !ls.slot_protected[s] && !ls.slot_in_flight[s] && ls.warm_demand[e] == 0) {
                probation.push_back(s);
            }
        }
        std::stable_sort(probation.begin(), probation.end(), [&ls](int32_t a, int32_t b) { return ls.slot_last_use[a] < ls.slot_last_use[b]; });
        for (int32_t s : probation) {
            if (victims.size() >= wanted.size()) {
                break;
            }
            victims.push_back(s);
        }
    }

    ls.warm_plan.clear();
    bool evicted = false;
    for (size_t i = 0; i < victims.size(); ++i) {
        const int32_t slot   = victims[i];
        const int32_t id     = wanted[i];
        const int32_t victim = ls.slot_expert[slot];
        if (victim >= 0) {
            ls.expert_slot[victim] = -1;
            ls.slot_expert[slot]   = -1;
            set_host_table_entry(ls.pub, victim, ls.n_slots);
            ls.n_evict++;
            evicted = true;
        }
        ls.slot_protected[slot] = false;
        ls.slot_in_flight[slot] = true;
        ls.expert_slot[id]      = -2;
        ls.warm_plan.push_back({id, slot, 0});
    }
    if (evicted) {
        mirror_table_in_stream(ls.pub, backend);
    }
}

// the pages of the range are read in the background, so that the copy that follows finds them
void advise_will_need(const void * data, size_t nbytes) {
#ifndef _WIN32
    static const uintptr_t page_size = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t begin = (uintptr_t) data & ~(page_size - 1);
    const uintptr_t end   = ((uintptr_t) data + nbytes + page_size - 1) & ~(page_size - 1);
    posix_madvise((void *) begin, end - begin, POSIX_MADV_WILLNEED);
#else
    GGML_UNUSED(data);
    GGML_UNUSED(nbytes);
#endif
}

struct host_matrix {
    const ggml_tensor * src;
    const ggml_tensor * pool;
};

std::array<host_matrix, 3> host_matrices_of(const llama_moe_cache_layer & pub) {
    return {{{pub.up_src, pub.up_h}, {pub.gate_src, pub.gate_h}, {pub.down_src, pub.down_h}}};
}

// the pool that holds the experts of one of the layer's host weights (up_src, gate_src or down_src)
const ggml_tensor * host_pool_of(const llama_moe_cache_layer & pub, const ggml_tensor * src) {
    GGML_ASSERT(src == pub.up_src || src == pub.gate_src || src == pub.down_src);
    return src == pub.up_src ? pub.up_h : src == pub.gate_src ? pub.gate_h : pub.down_h;
}

size_t host_matrix_index_of(const llama_moe_cache_layer & pub, const ggml_tensor * src) {
    GGML_ASSERT(src == pub.up_src || src == pub.gate_src || src == pub.down_src);
    return src == pub.up_src ? 0 : src == pub.gate_src ? 1 : 2;
}

// the read flags of one matrix: a word per pool slot, then the number of reads queued for the step
int32_t * host_pending_row(const llama_moe_cache_layer & pub, size_t matrix) {
    return pub.host_pending + matrix*(size_t) (pub.n_host_slots + 2);
}

// the read of n_experts experts from first_expert of the layer's matrix (0: up, 1: gate, 2: down) into dst
expert_reader::job host_read_job(const layer_state & ls, size_t matrix, int64_t first_expert, int64_t n_experts, void * dst) {
    const ggml_tensor * src = host_matrices_of(ls.pub)[matrix].src;
    const size_t offset = (size_t) first_expert*src->nb[2];
    return {ls.host_files[matrix].fd, ls.host_files[matrix].offset + offset, dst, (size_t) n_experts*src->nb[2], (const char *) src->data + offset};
}

float host_score_at(const moe_cache & mc, const layer_state & ls, uint32_t now, int32_t expert) {
    return ls.host_score[expert] * exp2f(-mc.decay * (float) (now - ls.host_last[expert]));
}

struct host_copy {
    int32_t expert;
    int32_t slot;
};

// the pool slots for the routed experts that are missing: empty slots first, then the lowest decayed score,
// never a slot whose expert the running ubatch routes to
std::vector<host_copy> assign_host_slots(const moe_cache & mc, layer_state & ls, const std::vector<int32_t> & missed, uint32_t now) {
    struct candidate {
        int     is_filled;
        float   score;
        int32_t slot;
    };
    std::vector<candidate> candidates;
    for (int32_t slot = 0; slot < ls.pub.n_host_slots; ++slot) {
        const int32_t expert = ls.host_slot_expert[slot];
        if (expert < 0) {
            candidates.push_back({0, 0.0f, slot});
        } else if (!ls.host_routed[expert]) {
            candidates.push_back({1, host_score_at(mc, ls, now, expert), slot});
        }
    }
    GGML_ASSERT(candidates.size() >= missed.size());
    std::partial_sort(candidates.begin(), candidates.begin() + missed.size(), candidates.end(), [](const candidate & a, const candidate & b) {
        return std::tie(a.is_filled, a.score, a.slot) < std::tie(b.is_filled, b.score, b.slot);
    });

    std::vector<host_copy> copies;
    for (size_t i = 0; i < missed.size(); ++i) {
        const int32_t slot = candidates[i].slot;
        if (const int32_t evicted = ls.host_slot_expert[slot]; evicted >= 0) {
            ls.host_slot[evicted] = -1;
        }
        ls.host_slot[missed[i]]   = slot;
        ls.host_slot_expert[slot] = missed[i];
        copies.push_back({missed[i], slot});
    }
    return copies;
}

// the reads that fill the pool slots of the copies; overlapped: up first, then gate, then down (the order the device uses them), each with its flag
std::vector<expert_reader::job> host_copy_jobs(const layer_state & ls, const std::vector<host_copy> & copies, bool overlapped = false) {
    const auto matrices = host_matrices_of(ls.pub);
    std::vector<expert_reader::job> jobs;
    const auto add = [&](const host_copy & c, size_t k) {
        jobs.push_back(host_read_job(ls, k, c.expert, 1, (char *) matrices[k].pool->data + (size_t) c.slot*matrices[k].pool->nb[2]));
        if (overlapped) {
            jobs.back().done_flag = host_pending_row(ls.pub, k) + c.slot;
        }
    };
    if (overlapped) {
        for (size_t k = 0; k < matrices.size(); ++k) {
            for (const host_copy & c : copies) {
                add(c, k);
            }
        }
    } else {
        for (const host_copy & c : copies) {
            for (size_t k = 0; k < matrices.size(); ++k) {
                add(c, k);
            }
        }
    }
    return jobs;
}

// flags first, so that a device wait never sees a slot as ready before its read is queued
expert_reader::ticket submit_overlapped_reads(moe_cache & mc, const layer_state & ls, const std::vector<host_copy> & copies) {
    for (size_t k = 0; k < 3; ++k) {
        int32_t * row = host_pending_row(ls.pub, k);
        for (const host_copy & c : copies) {
            __atomic_store_n(row + c.slot, 1, __ATOMIC_RELAXED);
        }
        __atomic_store_n(row + ls.pub.n_host_slots + 1, (int32_t) copies.size(), __ATOMIC_RELAXED);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (copies.empty()) {
        return nullptr;
    }
    return mc.host_reader->submit(host_copy_jobs(ls, copies, true), expert_reader::priority::high);
}

// with a reader the reads are only queued, and the caller waits for the returned ticket; without one the copies are done on return
expert_reader::ticket copy_into_host_pool(const moe_cache & mc, const layer_state & ls, const std::vector<host_copy> & copies) {
    if (mc.host_reader) {
        return mc.host_reader->submit(host_copy_jobs(ls, copies), expert_reader::priority::high);
    }
    const auto matrices = host_matrices_of(ls.pub);
    for (const host_copy & c : copies) {
        for (const host_matrix & m : matrices) {
            advise_will_need((const char *) m.src->data + (size_t) c.expert*m.src->nb[2], m.src->nb[2]);
        }
    }
    for (const host_copy & c : copies) {
        for (const host_matrix & m : matrices) {
            memcpy((char *) m.pool->data + (size_t) c.slot*m.pool->nb[2], (const char *) m.src->data + (size_t) c.expert*m.src->nb[2], m.src->nb[2]);
        }
    }
    return nullptr;
}

size_t host_expert_bytes(const llama_moe_cache_layer & pub) {
    return pub.up_src->nb[2] + pub.gate_src->nb[2] + pub.down_src->nb[2];
}

size_t layer_index(const moe_cache & mc, const layer_state & ls) {
    return &ls - mc.layers.data();
}

// the reads host_map_op left running must land before the CPU uses the layer's pool or gives its slots away again;
// the device kernels of the step wait for the ones they need, so this finds them done
void retire_demand_reads(moe_cache & mc, size_t layer_idx) {
    if (!mc.host_overlap || !mc.demand_reads[layer_idx]) {
        return;
    }
    const expert_reader::ticket reads = std::move(mc.demand_reads[layer_idx]);
    mc.host_reader->wait(reads);
    std::lock_guard<std::mutex> lock(mc.mtx);
    mc.layers[layer_idx].host_read_us += reads->t_done_us - reads->t_submit_us;
}

// the reads of an earlier guess must land before the layer's pool is used or its slots are given away again;
// returns the microseconds spent waiting
int64_t wait_guess_reads(moe_cache & mc, size_t layer_idx) {
    if (!mc.predict) {
        return 0;
    }
    expert_reader::ticket & reads = mc.host_guesses[layer_idx].reads;
    if (!reads) {
        return 0;
    }
    const int64_t t_start = ggml_time_us();
    mc.host_reader->promote(reads);
    mc.host_reader->wait(reads);
    reads.reset();
    return ggml_time_us() - t_start;
}

// how the real routing of a step compares with the guess made for it; the guess is spent afterwards
guess_stats settle_guess(host_guess & guess, const layer_state & ls, const std::vector<int32_t> & routed) {
    const auto contains = [](const std::vector<int32_t> & experts, int32_t expert) {
        return std::find(experts.begin(), experts.end(), expert) != experts.end();
    };
    guess_stats outcome;
    if (guess.made) {
        for (const int32_t e : routed) {
            const bool admitted = contains(guess.admitted, e);
            outcome.used   += admitted;
            outcome.wanted += admitted || (ls.host_slot[e] < 0 && !contains(guess.evicted, e));
        }
    }
    guess.made = false;
    guess.admitted.clear();
    guess.evicted.clear();
    return outcome;
}

std::vector<int32_t> experts_outside_pool_and_vram(const layer_state & ls);

// the experts the guess names that the layer holds in neither its pool nor its device cache: token 0 first, at most max_experts
std::vector<int32_t> guessed_misses(const layer_state & ls, const ggml_tensor * guess, size_t max_experts) {
    const std::vector<int32_t> outside = experts_outside_pool_and_vram(ls);
    std::vector<int32_t> experts;
    for (int64_t i1 = 0; i1 < guess->ne[1]; ++i1) {
        for (int64_t i0 = 0; i0 < guess->ne[0] && experts.size() < max_experts; ++i0) {
            const int32_t id = *(const int32_t *) ((const char *) guess->data + i1*guess->nb[1] + i0*guess->nb[0]);
            if (std::binary_search(outside.begin(), outside.end(), id) && std::find(experts.begin(), experts.end(), id) == experts.end()) {
                experts.push_back(id);
            }
        }
    }
    return experts;
}

// puts the experts the guess names into the layer's pool and queues their reads behind every demand read; returns how many
size_t admit_guess(moe_cache & mc, size_t layer_idx, const ggml_tensor * guessed_ids) {
    layer_state & ls   = mc.layers[layer_idx];
    host_guess & guess = mc.host_guesses[layer_idx];
    wait_guess_reads(mc, layer_idx);
    retire_demand_reads(mc, layer_idx);

    const std::vector<int32_t> missed = guessed_misses(ls, guessed_ids, std::min<size_t>(mc.predict_max, ls.pub.n_host_slots));
    const std::vector<int32_t> slot_expert_before = ls.host_slot_expert;
    const std::vector<host_copy> copies = assign_host_slots(mc, ls, missed, ls.host_clock + 1);

    guess.made     = true;
    guess.admitted = missed;
    guess.evicted.clear();
    for (const host_copy & c : copies) {
        if (slot_expert_before[c.slot] >= 0) {
            guess.evicted.push_back(slot_expert_before[c.slot]);
        }
    }
    if (!copies.empty()) {
        guess.reads = mc.host_reader->submit(host_copy_jobs(ls, copies), expert_reader::priority::low);
    }
    return copies.size();
}

// GGML_OP_CUSTOM on the CPU, once per layer per decode ubatch: after it every expert dst->src[0] routes to is in
// the pool. dst is I32 [n_expert]: the pool slot of each expert (-1 when absent). dst->src[1], when set, is the guess
// of the next layer's routing: its experts that the next layer's pool lacks are read ahead while this layer computes
void host_map_op(ggml_tensor * dst, int ith, int /*nth*/, void * userdata) {
    if (ith != 0) {
        return;
    }
    moe_cache & mc   = *g_cache;
    layer_state & ls = *(layer_state *) userdata;
    const ggml_tensor * ids = dst->src[0];
    const int32_t n_expert  = (int32_t) ls.host_slot.size();
    const size_t layer_idx  = layer_index(mc, ls);
    const int64_t guess_wait_us = wait_guess_reads(mc, layer_idx);
    retire_demand_reads(mc, layer_idx);
    const uint32_t now      = ++ls.host_clock;

    std::vector<int32_t> routed;
    for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
        for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + i1*ids->nb[1] + i0*ids->nb[0]);
            if (id >= 0 && id < n_expert && !ls.host_routed[id]) {
                ls.host_routed[id] = 1;
                routed.push_back(id);
            }
        }
    }
    for (const int32_t e : routed) {
        ls.host_score[e] = host_score_at(mc, ls, now, e) + 1.0f;
        ls.host_last[e]  = now;
    }

    std::vector<int32_t> missed;
    std::copy_if(routed.begin(), routed.end(), std::back_inserter(missed), [&ls](int32_t e) { return ls.host_slot[e] < 0; });
    const guess_stats outcome = mc.predict ? settle_guess(mc.host_guesses[layer_idx], ls, routed) : guess_stats();

    const int64_t t_start = ggml_time_us();
    const std::vector<host_copy> copies = assign_host_slots(mc, ls, missed, now);
    const expert_reader::ticket miss_reads = mc.host_overlap ? submit_overlapped_reads(mc, ls, copies) : copy_into_host_pool(mc, ls, copies);
    const size_t n_admitted = dst->src[1] ? admit_guess(mc, layer_idx + 1, dst->src[1]) : 0;
    if (mc.host_overlap) {
        mc.demand_reads[layer_idx] = miss_reads;
    } else if (miss_reads) {
        mc.host_reader->wait(miss_reads);
    }
    const int64_t t_us = mc.host_overlap ? 0 : ggml_time_us() - t_start;

    std::for_each(routed.begin(), routed.end(), [&ls](int32_t e) { ls.host_routed[e] = 0; });
    memcpy(dst->data, ls.host_slot.data(), n_expert*sizeof(int32_t));
    if (mc.host_overlap) {
        for (const host_copy & c : copies) {
            ((int32_t *) dst->data)[c.expert] |= GGML_MOE_CACHE_SLOT_PENDING;
        }
    }

    std::lock_guard<std::mutex> lock(mc.mtx);
    ls.n_host_hit      += routed.size() - missed.size();
    ls.n_host_miss     += missed.size();
    ls.host_bytes_read += missed.size()*host_expert_bytes(ls.pub);
    ls.host_read_us    += t_us;
    if (mc.predict) {
        guess_stats & stats = mc.host_guesses[layer_idx].stats;
        stats += outcome;
        stats.wait_us += guess_wait_us;
        if (dst->src[1]) {
            mc.host_guesses[layer_idx + 1].stats.read += n_admitted;
        }
    }
}

// the guess numbers since the previous log line, as the tail of the host tier line; "" when the guess is off
std::string guess_window_info(moe_cache & mc) {
    if (!mc.predict) {
        return "";
    }
    guess_stats total;
    for (const host_guess & guess : mc.host_guesses) {
        total += guess.stats;
    }
    const guess_stats win = total - mc.last_log_guess;
    mc.last_log_guess = total;
    const auto percent = [](uint64_t part, uint64_t whole) { return whole > 0 ? 100.0 * part / whole : 0.0; };
    return format(" win_guess_read=%" PRIu64 " win_guess_used=%" PRIu64 " win_recall=%.1f%% (%" PRIu64 "/%" PRIu64 ") win_guess_wait=%.1f ms"
            " total_guess_read=%" PRIu64 " total_guess_used=%" PRIu64 " total_recall=%.1f%% (%" PRIu64 "/%" PRIu64 ") total_guess_wait=%.2f s",
            win.read, win.used, percent(win.used, win.wanted), win.used, win.wanted, win.wait_us/1000.0,
            total.read, total.used, percent(total.used, total.wanted), total.used, total.wanted, total.wait_us/1e6);
}

// the host tier numbers since the previous log line
void log_host_tier_window(moe_cache & mc) {
    uint64_t hits   = 0;
    uint64_t misses = 0;
    uint64_t bytes  = 0;
    uint64_t us     = 0;
    for (const auto & ls : mc.layers) {
        hits   += ls.n_host_hit;
        misses += ls.n_host_miss;
        bytes  += ls.host_bytes_read;
        us     += ls.host_read_us;
    }
    const uint64_t win_hits   = hits   - mc.last_log_host_hits;
    const uint64_t win_misses = misses - mc.last_log_host_misses;
    const uint64_t win_bytes  = bytes  - mc.last_log_host_bytes;
    const uint64_t win_us     = us     - mc.last_log_host_us;
    mc.last_log_host_hits   = hits;
    mc.last_log_host_misses = misses;
    mc.last_log_host_bytes  = bytes;
    mc.last_log_host_us     = us;

    LLAMA_LOG_WARN("moe-cache: host tier slots=%d win_hit=%.1f%% (%" PRIu64 "/%" PRIu64 ") total_hit=%.1f%% (%" PRIu64 "/%" PRIu64 ") win_copied=%.1f MiB in %.1f ms total_copied=%.2f GiB in %.1f s%s\n",
            mc.host_slots, win_hits + win_misses > 0 ? 100.0 * win_hits / (win_hits + win_misses) : 0.0, win_hits, win_hits + win_misses,
            hits + misses > 0 ? 100.0 * hits / (hits + misses) : 0.0, hits, hits + misses,
            win_bytes/1024.0/1024.0, win_us/1000.0, bytes/1024.0/1024.0/1024.0, us/1e6, guess_window_info(mc).c_str());
}

struct layer_window_rate {
    int    il;
    double hit_pct;
};

// window = the steps since the previous log line; the per-layer spread tells whether
// a uniform slot count per layer is wasting slots on layers that cannot use them
void log_window_stats(moe_cache & mc) {
    uint64_t h   = 0;
    uint64_t m   = 0;
    uint64_t ins = 0;
    uint64_t ev  = 0;
    uint64_t wm  = 0;
    std::vector<layer_window_rate> rates;
    rates.reserve(mc.layers.size());
    for (auto & ls : mc.layers) {
        h   += ls.n_hit;
        m   += ls.n_miss;
        ins += ls.n_insert;
        ev  += ls.n_evict;
        wm  += ls.n_warm;

        const uint64_t lh = ls.n_hit  - ls.last_log_hits;
        const uint64_t lm = ls.n_miss - ls.last_log_misses;
        ls.last_log_hits   = ls.n_hit;
        ls.last_log_misses = ls.n_miss;
        if (lh + lm > 0) {
            rates.push_back({ls.pub.il, 100.0 * lh / (lh + lm)});
        }
    }
    const uint64_t dh = h - mc.last_log_hits;
    const uint64_t dm = m - mc.last_log_misses;
    mc.last_log_hits   = h;
    mc.last_log_misses = m;

    const double total_rate = (h + m > 0) ? (100.0 * h / (h + m)) : 0.0;
    const double win_rate   = (dh + dm > 0) ? (100.0 * dh / (dh + dm)) : 0.0;

    LLAMA_LOG_WARN("moe-cache: steps=%" PRIu64 " win_hit=%.1f%% (%" PRIu64 "/%" PRIu64 ") total_hit=%.1f%% (%" PRIu64 "/%" PRIu64 ") ins=%" PRIu64 " warm=%" PRIu64 " evict=%" PRIu64 " prot=%d%%\n",
            mc.n_steps, win_rate, dh, dh + dm, total_rate, h, h + m, ins, wm, ev, mc.protected_pct);
    if (mc.host_slots > 0) {
        log_host_tier_window(mc);
    }

    if (rates.empty()) {
        return;
    }
    std::sort(rates.begin(), rates.end(), [](const layer_window_rate & a, const layer_window_rate & b) {
        return a.hit_pct < b.hit_pct;
    });
    const layer_window_rate & mid = rates[rates.size() / 2];
    const size_t n_listed = std::min<size_t>(3, rates.size());
    auto append_rate = [](std::string & out, const layer_window_rate & r) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%s%d:%.0f", out.empty() ? "" : ",", r.il, r.hit_pct);
        out += buf;
    };
    std::string worst;
    std::string best;
    for (size_t k = 0; k < n_listed; ++k) {
        append_rate(worst, rates[k]);
        append_rate(best,  rates[rates.size() - 1 - k]);
    }
    LLAMA_LOG_WARN("moe-cache: layer win_hit median=%.1f%% (blk %d) worst=[%s] best=[%s] (blk:pct)\n",
            mid.hit_pct, mid.il, worst.c_str(), best.c_str());
}

bool backend_has_feature(ggml_backend_reg_t reg, const char * name) {
    auto * get_features = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (!get_features) {
        return false;
    }
    for (const ggml_backend_feature * f = get_features(reg); f->name; ++f) {
        if (strcmp(f->name, name) == 0) {
            return true;
        }
    }
    return false;
}

// the device can read the uncached experts in place when its backend supports that and every expert tensor sits
// in that device's pinned host buffer (load-mode none or mlock, not mmap) or the layer has a host pool, which does;
// LLAMA_MOE_CACHE_HOST_READS=0 opts out
bool can_read_host_experts(const moe_cache & mc, std::string & why_not) {
    if (const char * env = getenv("LLAMA_MOE_CACHE_HOST_READS"); env && atoi(env) == 0) {
        why_not = "LLAMA_MOE_CACHE_HOST_READS=0";
        return false;
    }
    const ggml_backend_buffer_type_t cache_buft = ggml_backend_buffer_get_type(mc.layers.front().pub.up_c->buffer);
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(cache_buft);
    if (!dev || !backend_has_feature(ggml_backend_dev_backend_reg(dev), "MMID_HOST_EXPERTS")) {
        why_not = std::string(ggml_backend_buft_name(cache_buft)) + " cannot read host experts";
        return false;
    }
    const ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    for (const auto & ls : mc.layers) {
        if (ggml_backend_buffer_get_type(ls.pub.up_c->buffer) != cache_buft) {
            why_not = "the cache spans devices";
            return false;
        }
        if (ls.pub.n_host_slots > 0) {
            continue;
        }
        for (const ggml_tensor * src : {ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src}) {
            if (ggml_backend_buffer_get_type(src->buffer) != host_buft) {
                why_not = std::string(src->name) + " is in " + ggml_backend_buffer_name(src->buffer) + ", not pinned host memory";
                return false;
            }
        }
    }
    return true;
}

// why the host tier cannot be set up, or "" when every layer can have it
std::string host_tier_blocker(const moe_cache & mc, const llama_model & model, int32_t n_host_slots, ggml_backend_buffer_type_t host_buft) {
    if (const char * env = getenv("LLAMA_MOE_CACHE_DEVICE"); !env || atoi(env) == 0) {
        return "the host tier needs LLAMA_MOE_CACHE_DEVICE=1";
    }
    if (host_buft == nullptr) {
        return "the cache device has no pinned host buffer type";
    }
    for (const auto & ls : mc.layers) {
        for (const ggml_tensor * src : {ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src}) {
            if (!src->buffer || !ggml_backend_buffer_is_host(src->buffer)) {
                return std::string(src->name) + " is not in host memory";
            }
            if (ggml_backend_buffer_get_type(src->buffer) == host_buft) {
                return "experts already in pinned host memory";
            }
            if (!ggml_is_contiguous(src)) {
                return std::string(src->name) + " is not contiguous";
            }
        }
    }
    const int64_t n_ids = model.hparams.n_expert_used_max()*LLAMA_MOE_CACHE_MAX_TOKENS;
    const int64_t slots = std::min<int64_t>(n_host_slots, mc.layers.front().pub.up_src->ne[2]);
    if (slots < n_ids) {
        return format("%" PRId64 " host slots cannot hold the %" PRId64 " expert ids of one step", slots, n_ids);
    }
    return "";
}

bool alloc_host_pools(moe_cache & mc, ggml_backend_buffer_type_t host_buft) {
    for (auto & ls : mc.layers) {
        ggml_init_params ip = {
            /*.mem_size  =*/ ggml_tensor_overhead()*3,
            /*.mem_buffer=*/ nullptr,
            /*.no_alloc  =*/ true,
        };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            return false;
        }
        mc.host_ctxs.push_back(ctx);

        const auto new_pool = [&](const ggml_tensor * src, const char * name) {
            ggml_tensor * pool = ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], ls.pub.n_host_slots + 1);
            ggml_format_name(pool, "moe_host_%s.%d", name, ls.pub.il);
            return pool;
        };
        ls.pub.up_h   = new_pool(ls.pub.up_src,   "up");
        ls.pub.gate_h = new_pool(ls.pub.gate_src, "gate");
        ls.pub.down_h = new_pool(ls.pub.down_src, "down");

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, host_buft);
        if (!buf) {
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        mc.host_bufs.push_back(buf);
    }
    return true;
}

void free_lookahead_buffers(moe_cache & mc) {
    for (auto & la : mc.lookahead_bufs) {
        if (la.buf) {
            ggml_backend_buffer_free(la.buf);
        }
        la = {};
    }
}

// the tier is off again: the pools are freed and the layers stop reading the experts in place
void drop_host_tier(moe_cache & mc) {
    mc.host_reader.reset();
    free_lookahead_buffers(mc);
    mc.lookahead = false;
    mc.host_buft = nullptr;
    for (auto * buf : mc.host_bufs) { ggml_backend_buffer_free(buf); }
    for (auto * ctx : mc.host_ctxs) { ggml_free(ctx); }
    mc.host_bufs.clear();
    mc.host_ctxs.clear();
    mc.host_slots = 0;
    if (mc.host_pending_buf) {
        ggml_backend_buffer_free(mc.host_pending_buf);
        mc.host_pending_buf = nullptr;
    }
    mc.host_overlap = false;
    mc.demand_reads.clear();
    for (auto & ls : mc.layers) {
        ls.pub.host_pending = nullptr;
        ls.pub.n_host_slots = 0;
        ls.pub.up_h   = nullptr;
        ls.pub.gate_h = nullptr;
        ls.pub.down_h = nullptr;
        ls.pub.reads_host_experts = false;
        ls.pub.routing            = nullptr;
    }
    mc.routing = nullptr;
}

// the reader of the model files that hold the host experts; nullptr, and why_not, when they cannot be read with O_DIRECT
std::unique_ptr<expert_reader> open_expert_reader(moe_cache & mc, const llama_model & model, std::string & why_not) {
    int n_threads = 16;
    if (const char * env = getenv("LLAMA_MOE_HOST_IO_THREADS")) {
        n_threads = atoi(env);
    }
    if (n_threads <= 0) {
        why_not = "LLAMA_MOE_HOST_IO_THREADS=0";
        return nullptr;
    }
    auto reader = std::make_unique<expert_reader>();
    if (const char * env = getenv("LLAMA_MOE_HOST_IO_CHUNK_KB")) {
        reader->chunk_bytes = (size_t) std::max(atoi(env), 1)*1024;
    }

    std::map<std::string, int> fd_of_path;
    for (auto & ls : mc.layers) {
        const auto matrices = host_matrices_of(ls.pub);
        for (size_t k = 0; k < matrices.size(); ++k) {
            std::string path;
            size_t offset = 0;
            if (!model.mapped_file_of(matrices[k].src->data, path, offset) || path.empty()) {
                why_not = std::string(matrices[k].src->name) + " is not in a mapped model file";
                return nullptr;
            }
            if (fd_of_path.count(path) == 0) {
                const int fd = open_direct(path.c_str(), reader->alignment);
                if (fd < 0) {
                    why_not = "cannot open " + path + " with O_DIRECT: " + strerror(errno);
                    return nullptr;
                }
                reader->fds.push_back(fd);
                fd_of_path[path] = fd;
            }
            ls.host_files[k] = {fd_of_path[path], offset};
        }
    }
    if (!reader->start(n_threads)) {
        why_not = "no memory for the read buffers";
        return nullptr;
    }
    return reader;
}

ggml_backend_buffer_type_t dev_mapped_buffer_type(ggml_backend_dev_t dev) {
    using mapped_buffer_type_fn = ggml_backend_buffer_type_t (*)(ggml_backend_dev_t);

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto * fn = reg ? (mapped_buffer_type_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_mapped_buffer_type") : nullptr;

    return fn ? fn(dev) : nullptr;
}

// LLAMA_MOE_HOST_OVERLAP (default 1): the read flags of all layers sit in one mapped host buffer, which is fine-grained, so
// a running kernel sees the writes of the reader threads; the pools keep their buffer. Sets host_overlap_state.
void setup_host_overlap(moe_cache & mc, ggml_backend_dev_t dev) {
    if (const char * env = getenv("LLAMA_MOE_HOST_OVERLAP"); env && atoi(env) == 0) {
        mc.host_overlap_state = "off: LLAMA_MOE_HOST_OVERLAP=0";
        return;
    }
    if (!mc.host_reader) {
        mc.host_overlap_state = "off: no file reader";
        return;
    }
    const ggml_backend_buffer_type_t mapped_buft = dev ? dev_mapped_buffer_type(dev) : nullptr;
    if (!mapped_buft) {
        mc.host_overlap_state = "off: the device has no mapped host buffer type";
        return;
    }
    const size_t words_per_layer = 3*(size_t) (mc.host_slots + 2);
    mc.host_pending_buf = ggml_backend_buft_alloc_buffer(mapped_buft, mc.layers.size()*words_per_layer*sizeof(int32_t));
    if (!mc.host_pending_buf) {
        mc.host_overlap_state = "off: no mapped host memory for the read flags";
        return;
    }
    ggml_backend_buffer_clear(mc.host_pending_buf, 0);
    int32_t * flags = (int32_t *) ggml_backend_buffer_get_base(mc.host_pending_buf);
    for (auto & ls : mc.layers) {
        ls.pub.host_pending = flags + layer_index(mc, ls)*words_per_layer;
    }
    mc.demand_reads.assign(mc.layers.size(), nullptr);
    mc.host_overlap       = true;
    mc.host_overlap_state = "on";
}

// "" when the tier is on: every layer has n_host_slots and its pools; else why it is off
std::string setup_host_tier(moe_cache & mc, const llama_model & model, int32_t n_host_slots) {
    const ggml_backend_buffer_type_t cache_buft = ggml_backend_buffer_get_type(mc.layers.front().pub.up_c->buffer);
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(cache_buft);
    const ggml_backend_buffer_type_t host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;

    if (std::string blocker = host_tier_blocker(mc, model, n_host_slots, host_buft); !blocker.empty()) {
        return blocker;
    }

    mc.host_slots = (int32_t) std::min<int64_t>(n_host_slots, mc.layers.front().pub.up_src->ne[2]);
    mc.host_buft  = host_buft;
    for (auto & ls : mc.layers) {
        ls.pub.n_host_slots = mc.host_slots;
    }
    std::string why_not;
    if (!can_read_host_experts(mc, why_not)) {
        drop_host_tier(mc);
        return why_not;
    }
    if (!alloc_host_pools(mc, host_buft)) {
        drop_host_tier(mc);
        return "no pinned host memory for the pools";
    }

    std::string reader_why_not;
    mc.host_reader = open_expert_reader(mc, model, reader_why_not);
    if (mc.host_reader) {
        mc.lookahead = mc.host_slots < mc.layers.front().pub.up_src->ne[2];
        if (const char * env = getenv("LLAMA_MOE_HOST_LOOKAHEAD")) {
            mc.lookahead = mc.lookahead && atoi(env) != 0;
        }
        if (const char * env = getenv("LLAMA_MOE_HOST_LOOKAHEAD_MIN_USED")) {
            mc.lookahead_min_used = std::clamp((float) atof(env), 0.0f, 1.0f);
        }
        LLAMA_LOG_WARN("moe-cache: host tier reads the model files with O_DIRECT: %zu files, %zu threads, %zu KiB chunks, %zu byte alignment, lookahead %s\n",
                mc.host_reader->fds.size(), mc.host_reader->threads.size(), mc.host_reader->chunk_bytes/1024, mc.host_reader->alignment, mc.lookahead ? "on" : "off");
    } else {
        LLAMA_LOG_WARN("moe-cache: host tier reads through the model mapping: %s\n", reader_why_not.c_str());
    }
    setup_host_overlap(mc, dev);
    return "";
}

// slot h holds expert h; the experts 0..H-1 are one contiguous range of each tensor
void populate_host_tier(moe_cache & mc) {
    const int64_t t_start = ggml_time_us();
    size_t pinned_bytes = 0;
    for (auto & ls : mc.layers) {
        const int64_t n_expert = ls.pub.up_src->ne[2];
        ls.host_slot.assign(n_expert, -1);
        std::iota(ls.host_slot.begin(), ls.host_slot.begin() + ls.pub.n_host_slots, 0);
        ls.host_slot_expert.resize(ls.pub.n_host_slots);
        std::iota(ls.host_slot_expert.begin(), ls.host_slot_expert.end(), 0);
        ls.host_score.assign(n_expert, 0.0f);
        ls.host_last.assign(n_expert, 0);
        ls.host_routed.assign(n_expert, 0);

        const auto matrices = host_matrices_of(ls.pub);
        if (mc.host_reader) {
            std::vector<expert_reader::job> jobs;
            for (size_t k = 0; k < matrices.size(); ++k) {
                jobs.push_back(host_read_job(ls, k, 0, ls.pub.n_host_slots, matrices[k].pool->data));
            }
            mc.host_reader->read(jobs);
        } else {
            for (const host_matrix & m : matrices) {
                advise_will_need(m.src->data, (size_t) ls.pub.n_host_slots*m.src->nb[2]);
            }
            for (const host_matrix & m : matrices) {
                memcpy(m.pool->data, m.src->data, (size_t) ls.pub.n_host_slots*m.src->nb[2]);
            }
        }
        for (const host_matrix & m : matrices) {
            pinned_bytes += ggml_nbytes(m.pool);
        }
    }
    LLAMA_LOG_INFO("moe-cache: host tier: %d of %" PRId64 " experts per layer are pinned, the others are read from the model file on demand\n",
            mc.host_slots, mc.layers.front().pub.up_src->ne[2]);
    LLAMA_LOG_WARN("moe-cache: host tier on: %zu layers, %d slots per layer, %.1f GiB pinned, populated in %.1f s, read overlap %s\n",
            mc.layers.size(), mc.host_slots, pinned_bytes/1024.0/1024.0/1024.0, (ggml_time_us() - t_start)/1e6, mc.host_overlap_state.c_str());
}

// LLAMA_MOE_HOST_PREDICT=1: every cache layer but the last guesses the routing of the next one, see host_map_op
void setup_host_predict(moe_cache & mc, const llama_model & model) {
    if (const char * env = getenv("LLAMA_MOE_HOST_PREDICT"); !env || atoi(env) == 0) {
        return;
    }
    if (!mc.host_reader) {
        LLAMA_LOG_WARN("moe-cache: LLAMA_MOE_HOST_PREDICT needs the file reader, guess off\n");
        return;
    }
    int32_t predict_k = 0;
    if (const char * env = getenv("LLAMA_MOE_HOST_PREDICT_K")) {
        predict_k = std::max(atoi(env), 0);
    }
    if (const char * env = getenv("LLAMA_MOE_HOST_PREDICT_MAX")) {
        mc.predict_max = (size_t) std::max(atoi(env), 0);
    }
    for (size_t i = 0; i + 1 < mc.layers.size(); ++i) {
        const llama_moe_cache_layer & next = mc.layers[i + 1].pub;
        const int32_t k = predict_k > 0 ? predict_k : (int32_t) model.hparams.n_expert_used(next.il);
        mc.layers[i].pub.predict_next = &next;
        mc.layers[i].pub.predict_k    = (int32_t) std::clamp<int64_t>(k, 1, next.gate_inp->ne[1]);
    }
    mc.host_guesses.resize(mc.layers.size());
    mc.predict = true;
    LLAMA_LOG_WARN("moe-cache: host tier guesses the next layer's routing: %d experts per token (0: n_expert_used), at most %zu read ahead per layer and step\n",
            predict_k, mc.predict_max);
}

// where a matrix's experts start in a lookahead buffer: the matrices before it come first
size_t lookahead_section_offset(const layer_state & ls, size_t matrix, int32_t count) {
    const auto matrices = host_matrices_of(ls.pub);
    const size_t expert_bytes = std::accumulate(matrices.begin(), matrices.begin() + matrix, (size_t) 0, [](size_t sum, const host_matrix & m) { return sum + m.src->nb[2]; });
    return (size_t) count*expert_bytes;
}

// the experts of the layer that are in neither the host pool nor the device cache; a prefill ubatch reads them from the files
std::vector<int32_t> experts_outside_pool_and_vram(const layer_state & ls) {
    const int32_t * vram_slot = ls.pub.host_table && ls.pub.host_table->data ? (const int32_t *) ls.pub.host_table->data : nullptr;
    std::vector<int32_t> experts;
    for (int32_t e = 0; e < (int32_t) ls.host_slot.size(); ++e) {
        const bool in_vram = vram_slot && vram_slot[e] >= 0 && vram_slot[e] < ls.pub.n_slots;
        if (ls.host_slot[e] < 0 && !in_vram) {
            experts.push_back(e);
        }
    }
    return experts;
}

// two pinned buffers, each for the most experts a layer can have outside the pool; false when there is no memory for them
bool alloc_lookahead_buffers(moe_cache & mc) {
    size_t bytes = 0;
    for (const auto & ls : mc.layers) {
        bytes = std::max(bytes, (size_t) (ls.pub.up_src->ne[2] - ls.pub.n_host_slots)*host_expert_bytes(ls.pub));
    }
    for (auto & la : mc.lookahead_bufs) {
        la.buf = ggml_backend_buft_alloc_buffer(mc.host_buft, bytes);
        if (!la.buf) {
            free_lookahead_buffers(mc);
            return false;
        }
    }
    LLAMA_LOG_WARN("moe-cache: host tier lookahead: 2 pinned buffers of %.2f GiB\n", bytes/1024.0/1024.0/1024.0);
    return true;
}

// the reads of the layer's experts outside the pool and the device cache, into the buffer of the layer's parity
void lookahead_read_layer(moe_cache & mc, size_t layer_idx) {
    const layer_state & ls = mc.layers[layer_idx];
    lookahead_buffer & la  = mc.lookahead_bufs[layer_idx % 2];
    if (la.reads) {
        mc.host_reader->wait(la.reads);
        la.reads.reset();
    }
    la.layer = -1;

    const std::vector<int32_t> experts = experts_outside_pool_and_vram(ls);
    if (experts.empty() || experts.size()*host_expert_bytes(ls.pub) > ggml_backend_buffer_get_size(la.buf)) {
        return;
    }
    la.count = (int32_t) experts.size();
    la.position.assign(ls.host_slot.size(), -1);
    for (size_t i = 0; i < experts.size(); ++i) {
        la.position[experts[i]] = (int32_t) i;
    }

    std::vector<expert_reader::job> jobs;
    const auto matrices = host_matrices_of(ls.pub);
    for (size_t k = 0; k < matrices.size(); ++k) {
        char * section = (char *) ggml_backend_buffer_get_base(la.buf) + lookahead_section_offset(ls, k, la.count);
        for (size_t run_begin = 0; run_begin < experts.size(); ) {
            size_t run_end = run_begin + 1;
            while (run_end < experts.size() && experts[run_end] == experts[run_end - 1] + 1) {
                ++run_end;
            }
            jobs.push_back(host_read_job(ls, k, experts[run_begin], run_end - run_begin, section + run_begin*matrices[k].src->nb[2]));
            run_begin = run_end;
        }
    }
    la.reads = mc.host_reader->submit(jobs, expert_reader::priority::low);
    la.layer = (int64_t) layer_idx;
}

int64_t count_used_experts(const uint32_t * used_ids, int64_t n_expert) {
    int64_t n_used = 0;
    for (int64_t id = 0; id < n_expert; ++id) {
        n_used += used_ids[id >> 5] >> (id & 31) & 1;
    }
    return n_used;
}

// the scheduler is about to upload a weight of the layer; the first time for a layer, its own reads are needed now
// and the next layer's reads start in the background
void lookahead_enter_layer(moe_cache & mc, size_t layer_idx, const uint32_t * used_ids) {
    if (!mc.lookahead || (int64_t) layer_idx == mc.lookahead_layer) {
        return;
    }
    mc.lookahead_layer = (int64_t) layer_idx;

    const lookahead_buffer & own = mc.lookahead_bufs[layer_idx % 2];
    if (own.layer == (int64_t) layer_idx && own.reads) {
        mc.host_reader->promote(own.reads);
    }

    const size_t next_idx = layer_idx + 1;
    const int64_t n_expert = mc.layers[layer_idx].pub.up_src->ne[2];
    if (next_idx >= mc.layers.size() || count_used_experts(used_ids, n_expert) < mc.lookahead_min_used*n_expert) {
        return;
    }
    if (!mc.lookahead_bufs.front().buf && !alloc_lookahead_buffers(mc)) {
        LLAMA_LOG_WARN("moe-cache: no pinned host memory for the lookahead buffers, lookahead off\n");
        mc.lookahead = false;
        return;
    }
    lookahead_read_layer(mc, next_idx);
}

bool alloc_routing(moe_cache & mc, int64_t n_expert_used) {
    ggml_init_params ip = {
        /*.mem_size  =*/ ggml_tensor_overhead(),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    mc.ctxs.push_back(ctx);

    ggml_tensor * routing = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used*LLAMA_MOE_CACHE_MAX_TOKENS, mc.layers.size());
    ggml_set_name(routing, "moe_cache_routing");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_buffer_get_type(mc.layers.front().pub.up_c->buffer));
    if (!buf) {
        return false;
    }
    ggml_backend_buffer_clear(buf, 0xFF);
    mc.bufs.push_back(buf);
    mc.routing = routing;

    for (size_t li = 0; li < mc.layers.size(); ++li) {
        auto & pub = mc.layers[li].pub;
        pub.reads_host_experts = true;
        pub.routing            = routing;
        pub.routing_row        = (int32_t) li;
    }
    return true;
}

constexpr int64_t DEV_STATE_HEADER = 8; // K, E, decay, clock, n_hit, n_miss, n_fill, n_evict

int64_t dev_state_len(int32_t n_slots, int64_t n_expert) {
    return DEV_STATE_HEADER + n_slots + 3*n_expert;
}

// per layer, one row of an I32 tensor: header, slot_expert[K], fill_slot[E], score[E] (float bits), last[E]
bool alloc_device_state(moe_cache & mc) {
    const int64_t n_expert = mc.layers.front().pub.up_src->ne[2];
    int32_t max_slots = 0;
    for (const auto & ls : mc.layers) {
        max_slots = std::max(max_slots, ls.n_slots);
    }
    const int64_t row_len = dev_state_len(max_slots, n_expert);

    ggml_init_params ip = {
        /*.mem_size  =*/ ggml_tensor_overhead()*(mc.layers.size()*2 + 1),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    mc.ctxs.push_back(ctx);

    ggml_tensor * state = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, row_len, mc.layers.size());
    ggml_set_name(state, "moe_cache_state");
    for (size_t li = 0; li < mc.layers.size(); ++li) {
        auto & pub = mc.layers[li].pub;
        pub.dev_state = ggml_view_1d(ctx, state, dev_state_len(pub.n_slots, n_expert), li*state->nb[1]);
        pub.fill_slot = ggml_view_1d(ctx, state, n_expert, li*state->nb[1] + (DEV_STATE_HEADER + pub.n_slots)*sizeof(int32_t));
        ggml_format_name(pub.dev_state, "moe_cache_state.%d", pub.il);
        ggml_format_name(pub.fill_slot, "moe_cache_fill.%d",  pub.il);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_buffer_get_type(mc.layers.front().pub.up_c->buffer));
    if (!buf) {
        for (auto & ls : mc.layers) {
            ls.pub.dev_state = nullptr;
            ls.pub.fill_slot = nullptr;
        }
        return false;
    }
    mc.bufs.push_back(buf);

    for (size_t li = 0; li < mc.layers.size(); ++li) {
        const auto & ls = mc.layers[li];
        std::vector<int32_t> row(row_len, 0);
        row[0] = ls.n_slots;
        row[1] = (int32_t) n_expert;
        std::memcpy(&row[2], &mc.decay, sizeof(int32_t));
        std::fill_n(row.begin() + DEV_STATE_HEADER, ls.n_slots, -1);
        std::fill_n(row.begin() + DEV_STATE_HEADER + ls.n_slots, n_expert, -1);
        ggml_backend_tensor_set(state, row.data(), li*state->nb[1], row_len*sizeof(int32_t));
    }

    mc.dev_state = state;
    mc.prev_counters.assign(mc.layers.size()*4, 0);
    for (auto & ls : mc.layers) {
        ls.pub.device_policy = true;
    }
    return true;
}

// the device decides and fills; the host copies its slot map after each step
void mirror_device_state(moe_cache & mc) {
    const int64_t row_len = mc.dev_state->ne[0];
    mc.state_host.resize(ggml_nelements(mc.dev_state));
    ggml_backend_tensor_get(mc.dev_state, mc.state_host.data(), 0, ggml_nbytes(mc.dev_state));

    for (size_t li = 0; li < mc.layers.size(); ++li) {
        auto & ls = mc.layers[li];
        const int32_t * row = mc.state_host.data() + li*row_len;

        uint32_t counters[4];
        for (int k = 0; k < 4; ++k) {
            counters[k] = (uint32_t) row[4 + k];
            counters[k] -= mc.prev_counters[li*4 + k];
            mc.prev_counters[li*4 + k] = (uint32_t) row[4 + k];
        }
        ls.n_hit    += counters[0];
        ls.n_miss   += counters[1];
        ls.n_insert += counters[2];
        ls.n_evict  += counters[3];

        std::copy_n(row + DEV_STATE_HEADER, ls.n_slots, ls.slot_expert.begin());
        std::fill(ls.expert_slot.begin(), ls.expert_slot.end(), -1);
        for (int32_t s = 0; s < ls.n_slots; ++s) {
            if (ls.slot_expert[s] >= 0) {
                ls.expert_slot[ls.slot_expert[s]] = s;
            }
        }

        std::vector<int32_t> table(ls.expert_slot.size());
        std::transform(ls.expert_slot.begin(), ls.expert_slot.end(), table.begin(), [&ls](int32_t s) { return s < 0 ? ls.n_slots : s; });
        ggml_backend_tensor_set(ls.pub.host_table, table.data(), 0, table.size()*sizeof(int32_t));
    }
}

constexpr int64_t AUDIT_MAX_REPORTS = 100;

bool audit_report_allowed(moe_cache & mc) {
    return mc.audit_reports++ < AUDIT_MAX_REPORTS;
}

// the device table must agree with the slot map and the pending fills, all read after the step
void audit_tables(moe_cache & mc) {
    for (size_t li = 0; li < mc.layers.size(); ++li) {
        const auto & ls = mc.layers[li];
        const int32_t K = ls.n_slots;
        const int64_t n_expert = (int64_t) ls.expert_slot.size();
        const int32_t * row       = mc.state_host.data() + li*mc.dev_state->ne[0];
        const int32_t * slot_expert = row + DEV_STATE_HEADER;
        const int32_t * fill_slot   = row + DEV_STATE_HEADER + K;

        std::vector<int32_t> table(n_expert);
        ggml_backend_tensor_get(ls.pub.dev_table, table.data(), 0, table.size()*sizeof(int32_t));
        for (int32_t & entry : table) {
            entry &= ~GGML_MOE_CACHE_SLOT_PENDING;
        }

        const auto table_error = [&](const char * what, int64_t slot, int64_t expert) {
            mc.audit_table_errors++;
            if (audit_report_allowed(mc)) {
                LLAMA_LOG_WARN("moe-cache audit: layer %d slot %" PRId64 " expert %" PRId64 " table error: %s, step %" PRIu64 "\n",
                        ls.pub.il, slot, expert, what, mc.n_steps);
            }
        };

        for (int32_t s = 0; s < K; ++s) {
            const int32_t e = slot_expert[s];
            if (e >= 0 && (e >= n_expert || table[e] != s)) {
                table_error("slot_expert without matching table entry", s, e);
            }
        }
        const int32_t max_table = K + std::max(ls.pub.n_host_slots - 1, 0);
        for (int64_t e = 0; e < n_expert; ++e) {
            if (table[e] < 0 || table[e] > max_table) {
                table_error("table entry out of range", -1, e);
            } else if (table[e] < K && slot_expert[table[e]] != e) {
                table_error("table entry without matching slot_expert", table[e], e);
            }
            const int32_t s = fill_slot[e];
            const bool has_slot = ls.pub.n_host_slots > 0 ? table[e] < K : table[e] != K;
            if (s >= 0 && (s >= K || has_slot || slot_expert[s] != -1)) {
                table_error("fill_slot on a cached expert or a used slot", s, e);
            }
        }
    }
}

// host table entries that disagree with the device table
int64_t count_stale_host_entries(moe_cache & mc) {
    int64_t n_stale = 0;
    for (const auto & ls : mc.layers) {
        std::vector<int32_t> table(ls.expert_slot.size());
        ggml_backend_tensor_get(ls.pub.dev_table, table.data(), 0, table.size()*sizeof(int32_t));
        const int32_t * host = (const int32_t *) ls.pub.host_table->data;
        for (size_t e = 0; e < table.size(); ++e) {
            n_stale += host[e] != std::min(table[e], ls.pub.n_slots);
        }
    }
    return n_stale;
}

// how a differing slot differs: byte count and span, a second read, and the host expert it equals, if any
void log_slot_difference(const layer_state & ls, const char * name, const ggml_tensor * cache, const ggml_tensor * src,
        int32_t slot, int32_t expert, const std::vector<char> & bytes, uint64_t step) {
    const char * want = (const char *) src->data + (size_t) expert*src->nb[2];
    size_t n_diff = 0;
    size_t first = bytes.size();
    size_t last  = 0;
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (bytes[i] != want[i]) {
            n_diff++;
            first = std::min(first, i);
            last  = i;
        }
    }
    std::vector<char> again(bytes.size());
    ggml_backend_tensor_get(cache, again.data(), (size_t) slot*cache->nb[2], again.size());
    const bool stable = again == bytes;
    int64_t equal_expert = -1;
    for (int64_t e = 0; e < src->ne[2] && equal_expert < 0; ++e) {
        if (memcmp(bytes.data(), (const char *) src->data + (size_t) e*src->nb[2], bytes.size()) == 0) {
            equal_expert = e;
        }
    }
    LLAMA_LOG_WARN("moe-cache audit: layer %d slot %d expert %d differs in %s, step %" PRIu64 ": %zu of %zu bytes in [%zu, %zu], second read %s, equals host expert %" PRId64 "\n",
            ls.pub.il, slot, expert, name, step, n_diff, bytes.size(), first, last, stable ? "same" : "changed", equal_expert);
}

// bytes of N random committed slots against the host experts
void audit_slot_bytes(moe_cache & mc) {
    for (int32_t n = 0; n < mc.audit; ++n) {
        const auto & ls = mc.layers[mc.audit_rng() % mc.layers.size()];
        std::vector<int32_t> committed;
        for (int32_t s = 0; s < ls.n_slots; ++s) {
            if (ls.slot_expert[s] >= 0) {
                committed.push_back(s);
            }
        }
        if (committed.empty()) {
            continue;
        }
        const int32_t slot   = committed[mc.audit_rng() % committed.size()];
        const int32_t expert = ls.slot_expert[slot];

        const struct { const char * name; const ggml_tensor * cache; const ggml_tensor * src; } matrices[] = {
            {"up",   ls.pub.up_c,   ls.pub.up_src},
            {"gate", ls.pub.gate_c, ls.pub.gate_src},
            {"down", ls.pub.down_c, ls.pub.down_src},
        };
        mc.audit_compared++;
        bool differs = false;
        for (const auto & m : matrices) {
            std::vector<char> bytes(m.cache->nb[2]);
            ggml_backend_tensor_get(m.cache, bytes.data(), (size_t) slot*m.cache->nb[2], bytes.size());
            if (memcmp(bytes.data(), (const char *) m.src->data + (size_t) expert*m.src->nb[2], bytes.size()) != 0) {
                differs = true;
                if (audit_report_allowed(mc)) {
                    log_slot_difference(ls, m.name, m.cache, m.src, slot, expert, bytes, mc.n_steps);
                }
            }
        }
        mc.audit_differ += differs;
    }
}

} // namespace

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts, int32_t n_host_slots) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return;
    }
    [&]() {
        const auto warn_host_tier_off = [&](const std::string & why) {
            if (n_host_slots > 0 && !why.empty()) {
                LLAMA_LOG_WARN("%s: %s, host tier off\n", __func__, why.c_str());
            }
        };

        if (n_slots <= 0) {
            warn_host_tier_off("the MoE expert cache is off");
            g_init_done = true;
            return;
        }

        auto * mc = new moe_cache();
        mc->n_slots = n_slots;
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }
        if (const char * env = getenv("LLAMA_MOE_CACHE_PROTECTED_PCT")) {
            mc->protected_pct = std::clamp(atoi(env), 0, 100);
        }
        if (const char * env = getenv("LLAMA_MOE_CACHE_DECAY")) {
            mc->decay = std::clamp((float) atof(env), 0.0f, 1.0f);
        }
        if (const char * env = getenv("LLAMA_MOE_CACHE_INSERT_ORDER")) {
            mc->rank_by_demand = strcmp(env, "recency") != 0;
        }
        if (const char * env = getenv("LLAMA_MOE_CACHE_WARM_MAX")) {
            mc->warm_max = std::max(0, atoi(env));
        }
        if (const char * env = getenv("LLAMA_MOE_CACHE_MAX_IN_FLIGHT")) {
            mc->max_in_flight = std::max(0, atoi(env));
        }
        const std::map<int, int32_t> layer_slots_override = parse_layer_slots(getenv("LLAMA_MOE_CACHE_LAYER_SLOTS"));

        // collect the host-resident expert layers, grouped by the device buffer
        // type of that layer's router (the cache lives next to the router)
        struct cand { int il; const llama_layer * l; };
        std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
                continue;
            }
            if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
                continue; // dry-run / memory-estimation model: weights not loaded, don't bind to it
            }
            if (!l.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
                continue; // experts already on a device: nothing to cache
            }
            if (!l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
                continue; // no device home for the cache
            }
            groups[ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)].push_back({(int) il, &l});
        }

        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\n", __func__, n_slots);
            warn_host_tier_off("no host-resident expert layers");
            delete mc;
            return;
        }

        // host buffer for the CPU-side tables
        std::vector<cand> all;
        for (auto & g : groups) {
            all.insert(all.end(), g.second.begin(), g.second.end());
        }

        const int64_t n_expert = all.front().l->ffn_up_exps->ne[2];
        size_t n_table_rows = 0;

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*5 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            ggml_tensor * tables = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert, cands.size());
            if (tables_only) {
                ggml_set_name(tables, "moe_cache_htbl");
                mc->host_tables = tables;
            } else {
                ggml_set_name(tables, "moe_cache_tbl");
                mc->table_blocks.push_back({tables, n_table_rows*tables->nb[1]});
                n_table_rows += cands.size();
            }

            for (const auto & c : cands) {
                GGML_ASSERT(c.l->ffn_up_exps->ne[2] == n_expert);
                ggml_tensor * table = ggml_view_2d(ctx, tables, 1, n_expert, sizeof(int32_t), (&c - cands.data())*tables->nb[1]);

                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.n_slots  = layer_slots_override.count(c.il) ? layer_slots_override.at(c.il) : n_slots;
                    ls->n_slots      = ls->pub.n_slots;
                    ls->max_protected = (ls->n_slots * mc->protected_pct) / 100;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                    ls->pub.gate_inp    = c.l->ffn_gate_inp;
                    ls->pub.gate_inp_b  = c.l->ffn_gate_inp_b;
                    ls->pub.exp_probs_b = c.l->ffn_exp_probs_b;
                }

                if (tables_only) {
                    ls->pub.host_table = table;
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], ls->n_slots + 1);
                    ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], ls->n_slots + 1);
                    ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], ls->n_slots + 1);
                    ls->pub.dev_table = table;
                    ggml_format_name(ls->pub.up_c,      "moe_cache_up.%d",   c.il);
                    ggml_format_name(ls->pub.gate_c,    "moe_cache_gate.%d", c.il);
                    ggml_format_name(ls->pub.down_c,    "moe_cache_down.%d", c.il);
                    ggml_format_name(ls->pub.dev_table, "moe_cache_tbl.%d",  c.il);
                }
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                        __func__, ggml_backend_buft_name(buft));
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            mc->bufs.push_back(buf);
            return true;
        };

        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            ok = alloc_group(g.first, g.second, /*tables_only=*/false);
        }

        if (!ok) {
            warn_host_tier_off("the MoE expert cache is off");
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true; // a real model was seen and allocation failed: stay disabled
            return;
        }

        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE")) {
            mc->trace = fopen(path, "a");
            if (mc->trace) {
                LLAMA_LOG_INFO("%s: routing trace on, appending to %s\n", __func__, path);
            } else {
                LLAMA_LOG_WARN("%s: cannot open LLAMA_MOE_CACHE_TRACE=%s\n", __func__, path);
            }
        }

        // init LRU state + tables (everything uncached -> dummy slot n_slots)
        size_t vram = 0;
        int64_t slots_total = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            ls.slot_expert.assign(ls.n_slots, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(ls.n_slots, 0);
            ls.slot_protected.assign(ls.n_slots, false);
            ls.slot_in_flight.assign(ls.n_slots, false);
            ls.demand_cur.assign(n_expert, 0);
            ls.demand_prev.assign(n_expert, 0);
            ls.warm_demand.assign(n_expert, 0);
            slots_total += ls.n_slots;

            std::vector<int32_t> dummy(n_expert, ls.n_slots);
            ggml_backend_tensor_set(ls.pub.dev_table,  dummy.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            mc->by_src[ls.pub.up_src]   = {(size_t) (&ls - mc->layers.data()), &llama_moe_cache_layer::up_c};
            mc->by_src[ls.pub.gate_src] = {(size_t) (&ls - mc->layers.data()), &llama_moe_cache_layer::gate_c};
            mc->by_src[ls.pub.down_src] = {(size_t) (&ls - mc->layers.data()), &llama_moe_cache_layer::down_c};
            vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
            LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert\n",
                    ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2]);
        }

        std::string host_tier_off;
        if (n_host_slots > 0) {
            host_tier_off = setup_host_tier(*mc, model, n_host_slots);
        }

        std::string host_reads = "off: ";
        if (std::string why_not; !can_read_host_experts(*mc, why_not)) {
            host_reads += why_not;
        } else if (!alloc_routing(*mc, model.hparams.n_expert_used_max())) {
            host_reads += "no device memory for the routing readback";
        } else {
            host_reads = mc->host_slots > 0 ? "on, uncached experts are read in place from the host pools" : "on, uncached experts are read in place from pinned host memory";
        }

        std::string policy = "host";
        if (const char * env = getenv("LLAMA_MOE_CACHE_DEVICE"); env && atoi(env) != 0) {
            if (!mc->routing) {
                LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_DEVICE needs device reads of host experts - staying on the host policy\n", __func__);
            } else if (!alloc_device_state(*mc)) {
                LLAMA_LOG_WARN("%s: no device memory for the device policy state - staying on the host policy\n", __func__);
            } else {
                mc->device_policy = true;
                mc->warm_max      = 0;
                policy = "device";
                if (const char * audit_env = getenv("LLAMA_MOE_CACHE_AUDIT")) {
                    mc->audit = std::max(0, atoi(audit_env));
                }
            }
        }

        if (mc->host_slots > 0 && !mc->device_policy) {
            drop_host_tier(*mc);
            host_tier_off = "no device memory for the routing readback or the device policy state";
            host_reads    = "off: " + host_tier_off;
        }
        warn_host_tier_off(host_tier_off);
        if (mc->host_slots > 0) {
            populate_host_tier(*mc);
            setup_host_predict(*mc, model);
        }

        for (ggml_backend_buffer_t buf : mc->bufs) {
            const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU || ggml_backend_dev_buffer_type(dev) != buft) {
                continue;
            }
            if (ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr)) {
                mc->upload_backends.push_back({buft, backend});
            }
        }

        if (!mc->device_policy) mc->worker = std::thread([mc]() {
            for (;;) {
                std::vector<upload_job> batch;
                {
                    std::unique_lock<std::mutex> lk(mc->wmtx);
                    mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                    if (mc->stop) {
                        return;
                    }
                    batch.assign(mc->todo.begin(), mc->todo.end());
                    mc->todo.clear();
                }
                for (const auto & j : batch) {
                    auto & ls = mc->layers[j.layer_idx];
                    upload_slice(*mc, ls.pub.up_c,   ls.pub.up_src,   j.expert, j.slot);
                    upload_slice(*mc, ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot);
                    upload_slice(*mc, ls.pub.down_c, ls.pub.down_src, j.expert, j.slot);
                }
                for (const auto & ub : mc->upload_backends) {
                    ggml_backend_synchronize(ub.second);
                }
                {
                    std::lock_guard<std::mutex> lk(mc->wmtx);
                    for (auto & j : batch) {
                        j.done = true;
                        mc->done.push_back(j);
                    }
                }
            }
        });

        ggml_set_moe_obs_callback(moe_obs_cb, mc);
        g_cache = mc;
        g_init_done = true;

        LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu layers, %" PRId64 " slots total (%d per layer%s), %d inserts/step ordered by %s, %d%% protected, %.1f MiB device memory\n",
                __func__, mc->layers.size(), slots_total, n_slots, layer_slots_override.empty() ? "" : ", LLAMA_MOE_CACHE_LAYER_SLOTS applied",
                mc->max_inserts, mc->rank_by_demand ? "demand" : "recency", mc->protected_pct, vram/1024.0/1024.0);
        LLAMA_LOG_INFO("%s: device reads of uncached experts: %s\n", __func__, host_reads.c_str());
        const std::string policy_info = mc->device_policy ? policy + format(", decayed frequency, decay %.3f", mc->decay) : policy;
        LLAMA_LOG_INFO("%s: eviction and fill policy: %s (audit %d)\n", __func__, policy_info.c_str(), mc->audit);
        LLAMA_LOG_WARN("%s: MoE cache policy: %s\n", __func__, policy_info.c_str());
        LLAMA_LOG_INFO("%s: prefill warm-fill: %s\n", __func__, mc->warm_max > 0 ? (std::to_string(mc->warm_max) + " slots/layer/ubatch").c_str() : "off");
        if (!layer_slots_override.empty()) {
            std::string per_layer;
            for (const auto & ls : mc->layers) {
                if (ls.n_slots != n_slots) {
                    per_layer += (per_layer.empty() ? "" : ",") + std::to_string(ls.pub.il) + ":" + std::to_string(ls.n_slots);
                }
            }
            LLAMA_LOG_INFO("%s: per-layer slot overrides: %s (%" PRId64 " vs %zu uniform)\n", __func__, per_layer.c_str(), slots_total, mc->layers.size()*(size_t) n_slots);
            for (const auto & kv : layer_slots_override) {
                const bool cached = std::any_of(mc->layers.begin(), mc->layers.end(), [&kv](const layer_state & ls) { return ls.pub.il == kv.first; });
                if (!cached) {
                    LLAMA_LOG_WARN("%s: LLAMA_MOE_CACHE_LAYER_SLOTS names layer %d, which has no cached experts - ignored\n", __func__, kv.first);
                }
            }
        }
    }();
}

bool llama_moe_cache_get_stats(llama_moe_cache_stats * out) {
    moe_cache * mc = g_cache;
    if (!mc || !out) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);

    *out = {};
    out->n_layers = (int32_t) mc->layers.size();
    out->n_slots  = mc->n_slots;
    out->n_host_slots = mc->host_slots;
    for (const auto & ls : mc->layers) {
        out->n_hit    += ls.n_hit;
        out->n_miss   += ls.n_miss;
        out->n_insert += ls.n_insert;
        out->n_evict  += ls.n_evict;
        out->n_host_hit      += ls.n_host_hit;
        out->n_host_miss     += ls.n_host_miss;
        out->host_bytes_read += ls.host_bytes_read;
        out->host_read_us    += ls.host_read_us;
    }

    return true;
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    if (!g_cache) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

ggml_tensor * llama_moe_cache_mul_mat_id(ggml_context * ctx, const llama_moe_cache_layer & layer,
        ggml_tensor * cache, const ggml_tensor * host_src, ggml_tensor * b, ggml_tensor * ids) {
    GGML_ASSERT(layer.reads_host_experts);
    ggml_tensor * cur = ggml_mul_mat_id(ctx, cache, b, ids);
    // the layout ggml_cuda_mmid_host_experts reads
    cur->src[3]       = layer.dev_table;
    if (layer.device_policy) {
        cur->src[4]   = layer.fill_slot;
    }
    static_assert(GGML_MOE_CACHE_OP_HOST_EXPERTS*sizeof(int32_t) + sizeof(void *) <= GGML_MAX_OP_PARAMS, "MoE cache op_params overflow");
    static_assert(GGML_MOE_CACHE_OP_HOST_SLOTS*sizeof(int32_t) + sizeof(int32_t) <= GGML_MAX_OP_PARAMS, "MoE cache op_params overflow");
    static_assert(GGML_MOE_CACHE_OP_HOST_EXPERTS + 2 <= GGML_MOE_CACHE_OP_HOST_SLOTS, "MoE cache op_params overlap");
    static_assert(GGML_MOE_CACHE_OP_HOST_PENDING*sizeof(int32_t) + sizeof(void *) <= GGML_MAX_OP_PARAMS, "MoE cache op_params overflow");
    static_assert(GGML_MOE_CACHE_OP_HOST_PENDING + 2 <= GGML_MOE_CACHE_OP_N_SLOTS, "MoE cache op_params overlap");
    static_assert(GGML_MOE_CACHE_OP_HOST_PENDING > 3, "MoE cache op_params overlap the ggml_prec slots");
    const ggml_tensor * host_experts = layer.n_host_slots > 0 ? host_pool_of(layer, host_src) : host_src;
    cur->op_params[GGML_MOE_CACHE_OP_N_SLOTS]    = layer.n_slots;
    cur->op_params[GGML_MOE_CACHE_OP_HOST_SLOTS] = layer.n_host_slots;
    memcpy(&cur->op_params[GGML_MOE_CACHE_OP_HOST_EXPERTS], &host_experts->data, sizeof(host_experts->data));
    if (layer.host_pending) {
        const int32_t * pending_row = host_pending_row(layer, host_matrix_index_of(layer, host_src));
        memcpy(&cur->op_params[GGML_MOE_CACHE_OP_HOST_PENDING], &pending_row, sizeof(pending_row));
    }
    return cur;
}

ggml_tensor * llama_moe_cache_host_map(ggml_context * ctx, const llama_moe_cache_layer & layer, ggml_tensor * ids, ggml_tensor * guess) {
    GGML_ASSERT(g_cache && layer.n_host_slots > 0);
    layer_state & ls = g_cache->layers[g_cache->by_up_src.at(layer.up_src)];
    ggml_tensor * args[] = {ids, guess};
    return ggml_custom_4d(ctx, GGML_TYPE_I32, (int64_t) ls.host_slot.size(), 1, 1, 1, args, guess ? 2 : 1, host_map_op, 1, &ls);
}

bool llama_moe_cache_host_tier() {
    return g_cache && g_cache->host_slots > 0;
}

bool llama_moe_cache_expert_host(const ggml_tensor * weight, const uint32_t * used_ids, const ggml_tensor ** pool, const int32_t ** host_slot, int32_t * n_slots, void * /*user_data*/) {
    moe_cache * mc = g_cache;
    if (!mc || mc->host_slots <= 0) {
        return false;
    }
    auto it = mc->by_src.find(weight);
    if (it == mc->by_src.end()) {
        return false;
    }
    const layer_state & ls = mc->layers[it->second.layer_idx];
    wait_guess_reads(*mc, it->second.layer_idx);
    retire_demand_reads(*mc, it->second.layer_idx);
    *pool      = host_pool_of(ls.pub, weight);
    *host_slot = ls.host_slot.data();
    *n_slots   = ls.pub.n_host_slots;

    lookahead_enter_layer(*mc, it->second.layer_idx, used_ids);
    if (!mc->host_reader) {
        for (int64_t id = 0; id < weight->ne[2]; ++id) {
            if ((used_ids[id >> 5] >> (id & 31) & 1) && ls.host_slot[id] < 0) {
                advise_will_need((const char *) weight->data + (size_t) id*weight->nb[2], weight->nb[2]);
            }
        }
    }
    return *pool != nullptr;
}

bool llama_moe_cache_direct_reads() {
    return g_cache && g_cache->host_reader;
}

bool llama_moe_cache_expert_read(const ggml_tensor * weight, int64_t first, int64_t n, void * dst, void * /*user_data*/) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->host_reader) {
        return false;
    }
    auto it = mc->by_src.find(weight);
    if (it == mc->by_src.end()) {
        return false;
    }
    const layer_state & ls = mc->layers[it->second.layer_idx];
    const auto matrices = host_matrices_of(ls.pub);
    const size_t matrix = std::find_if(matrices.begin(), matrices.end(), [weight](const host_matrix & m) { return m.src == weight; }) - matrices.begin();
    if (matrix == matrices.size()) {
        return false;
    }
    mc->host_reader->read({host_read_job(ls, matrix, first, n, dst)});
    return true;
}

const void * llama_moe_cache_expert_src(const ggml_tensor * weight, int64_t first, int64_t n, void * /*user_data*/) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->lookahead) {
        return nullptr;
    }
    auto it = mc->by_src.find(weight);
    if (it == mc->by_src.end()) {
        return nullptr;
    }
    const size_t layer_idx = it->second.layer_idx;
    const layer_state & ls = mc->layers[layer_idx];
    const lookahead_buffer & la = mc->lookahead_bufs[layer_idx % 2];
    if (la.layer != (int64_t) layer_idx || first < 0 || n <= 0 || first + n > (int64_t) la.position.size()) {
        return nullptr;
    }
    const auto run = la.position.begin() + first;
    const bool buffered = *run >= 0 && std::adjacent_find(run, run + n, [](int32_t a, int32_t b) { return b != a + 1; }) == run + n;
    const auto matrices = host_matrices_of(ls.pub);
    const size_t matrix = std::find_if(matrices.begin(), matrices.end(), [weight](const host_matrix & m) { return m.src == weight; }) - matrices.begin();
    if (!buffered || matrix == matrices.size()) {
        return nullptr;
    }
    mc->host_reader->promote(la.reads);
    mc->host_reader->wait(la.reads);
    return (const char *) ggml_backend_buffer_get_base(la.buf) + lookahead_section_offset(ls, matrix, la.count) + (size_t) *run*weight->nb[2];
}

bool llama_moe_cache_expert_rows(const ggml_tensor * weight, const ggml_tensor ** rows, const int32_t ** expert_slot, int32_t * n_slots, void * /*user_data*/) {
    moe_cache * mc = g_cache;
    if (!mc) {
        return false;
    }
    auto it = mc->by_src.find(weight);
    if (it == mc->by_src.end()) {
        return false;
    }
    const llama_moe_cache_layer & pub = mc->layers[it->second.layer_idx].pub;
    if (!pub.host_table || !pub.host_table->data) {
        return false;
    }
    *rows        = pub.*(it->second.rows);
    *expert_slot = (const int32_t *) pub.host_table->data;
    *n_slots     = pub.n_slots;
    return *rows != nullptr;
}

size_t llama_moe_cache_device_bytes() {
    const moe_cache * mc = g_cache;
    if (!mc) {
        return 0;
    }
    size_t bytes = 0;
    for (const auto & ls : mc->layers) {
        bytes += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
    }
    return bytes;
}

void llama_moe_cache_warm_from_staging(const ggml_tensor * weight, const ggml_tensor * staged,
        const int32_t * ids, int64_t ne0, int64_t ne1, size_t s0, size_t s1, ggml_backend_t backend, void * /*user_data*/) {
    moe_cache * mc = g_cache;
    if (!mc || mc->warm_max <= 0 || !staged || !staged->data) {
        return;
    }
    auto it = mc->by_src.find(weight);
    if (it == mc->by_src.end()) {
        return;
    }
    layer_state & ls = mc->layers[it->second.layer_idx];
    ggml_tensor * rows = ls.pub.*(it->second.rows);
    if (!rows || !rows->data || rows->type != staged->type || rows->nb[2] != staged->nb[2]) {
        return;
    }
    const size_t expert_size = staged->nb[2];

    const int bit = it->second.rows == &llama_moe_cache_layer::up_c ? 1 : it->second.rows == &llama_moe_cache_layer::gate_c ? 2 : 4;

    std::lock_guard<std::mutex> lock(mc->mtx);
    // a repeated member means the next ubatch of this step is being staged
    if (ls.warm_seen & bit) {
        close_warm_plan(ls);
    }
    if (ls.warm_seen == 0) {
        plan_warm_fill(*mc, ls, ids, ne0, ne1, s0, s1, backend);
    }
    ls.warm_seen |= bit;
    if (ls.warm_plan.empty()) {
        return;
    }

    // the plan may be older than this staged copy (a member skipped the staging path); a row is
    // only taken from a copy that this ubatch actually streamed it into
    std::vector<bool> staged_here(ls.warm_demand.size(), false);
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            const int32_t id = ids[i1*s1 + i0*s0];
            if (id >= 0 && (size_t) id < staged_here.size()) {
                staged_here[id] = true;
            }
        }
    }
    for (auto & w : ls.warm_plan) {
        if (!staged_here[w.expert]) {
            continue;
        }
        if (ggml_backend_tensor_copy_range_async(backend, staged, (size_t) w.expert*expert_size, rows, (size_t) w.slot*expert_size, expert_size)) {
            w.copied |= bit;
        }
    }
}

void llama_moe_cache_ubatch_begin(ggml_backend_sched_t sched, int64_t n_tokens) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->device_policy) {
        return;
    }
    if (n_tokens <= LLAMA_MOE_CACHE_MAX_TOKENS) {
        mc->state_dirty = true;
        return;
    }
    if (mc->state_dirty) {
        ggml_backend_sched_synchronize(sched);
        std::lock_guard<std::mutex> lk(mc->mtx);
        if (mc->audit > 0) {
            mc->audit_mid_mirrors++;
            mc->audit_mid_stale += count_stale_host_entries(*mc);
        }
        mirror_device_state(*mc);
        mc->state_dirty = false;
        if (mc->audit > 0) {
            audit_tables(*mc);
        }
    }
}

void llama_moe_cache_step(ggml_backend_sched_t sched) {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }

    if (mc->device_policy) {
        ggml_backend_sched_synchronize(sched);
        std::lock_guard<std::mutex> lk(mc->mtx);
        if (mc->trace) {
            const int64_t n_ids = read_device_routing(*mc);
            trace_device_routing(*mc, n_ids);
        }
        mirror_device_state(*mc);
        mc->state_dirty = false;
        if (mc->audit > 0) {
            audit_tables(*mc);
            audit_slot_bytes(*mc);
        }
        mc->n_steps++;
        if (mc->n_steps % 128 == 0) {
            log_window_stats(*mc);
            if (mc->audit > 0) {
                LLAMA_LOG_WARN("moe-cache audit: %" PRId64 " slots compared, %" PRId64 " differ, %" PRId64 " table errors, %" PRId64 " mid-decode mirrors fixed %" PRId64 " stale entries, step %" PRIu64 "\n",
                        mc->audit_compared, mc->audit_differ, mc->audit_table_errors, mc->audit_mid_mirrors, mc->audit_mid_stale, mc->n_steps);
            }
        }
        return;
    }

    if (mc->routing) {
        ggml_backend_sched_synchronize(sched);
        std::lock_guard<std::mutex> lk(mc->mtx);
        observe_device_routing(*mc);
    }

    // 1) publish completed uploads (sync point: no graph is executing)
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (const auto & j : mc->done) {
            auto & ls = mc->layers[j.layer_idx];
            ls.slot_expert[j.slot]     = j.expert;
            ls.expert_slot[j.expert]   = j.slot;
            ls.slot_last_use[j.slot]   = ++mc->clock;
            ls.slot_in_flight[j.slot]  = false;
            set_table_entry(*mc, ls.pub, j.expert, j.slot);
        }
        mc->done.clear();
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    settle_warm_plans(*mc);
    mc->n_steps++;

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    std::vector<upload_job> uploads;
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        auto & ls = mc->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        int n_empty     = 0;
        int n_in_flight = 0;
        for (int32_t s = 0; s < ls.n_slots; ++s) {
            if (ls.slot_in_flight[s]) {
                n_in_flight++;
            } else if (ls.slot_expert[s] < 0) {
                n_empty++;
            }
        }
        int budget = n_empty > 0 ? std::max(mc->max_inserts, std::min(4, n_empty)) : mc->max_inserts;
        // uploads slower than the step would otherwise evict the whole cache, hot experts included
        if (mc->max_in_flight > 0) {
            budget = std::min(budget, mc->max_in_flight - n_in_flight);
        }

        order_pending(ls, mc->rank_by_demand);
        for (int32_t id : ls.pending) {
            if (budget <= 0) {
                break;
            }
            if (ls.expert_slot[id] >= 0 || ls.expert_slot[id] == -2) {
                continue;
            }

            const int32_t slot = find_eviction_victim(ls, ls.n_slots);
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(*mc, ls.pub, victim, ls.n_slots);
                ls.n_evict++;
            }
            ls.slot_protected[slot] = false;
            ls.slot_in_flight[slot] = true;
            ls.expert_slot[id]      = -2;
            ls.n_insert++;
            budget--;

            uploads.push_back({li, id, slot});
        }
        age_demand(ls, mc->rank_by_demand);
    }

    // the evicted slots stop being read before the worker overwrites them
    flush_tables(*mc);
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        mc->todo.insert(mc->todo.end(), uploads.begin(), uploads.end());
    }
    mc->wcv.notify_one();

    if (mc->n_steps % 128 == 0) {
        log_window_stats(*mc);
    }
}
