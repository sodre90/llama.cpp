#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {

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
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
};

struct moe_cache {
    int32_t n_slots       = 0;  // default slots per layer; LLAMA_MOE_CACHE_LAYER_SLOTS overrides per layer
    int32_t max_inserts   = 2;
    int32_t protected_pct = 75; // SLRU protected segment share, LLAMA_MOE_CACHE_PROTECTED_PCT
    bool    rank_by_demand = true; // LLAMA_MOE_CACHE_INSERT_ORDER=recency restores last-observed-first
    int32_t warm_max      = 0;  // LLAMA_MOE_CACHE_WARM_MAX: slots per layer a prefill may fill from its staged copy, 0 = off

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    uint64_t last_log_hits   = 0;
    uint64_t last_log_misses = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

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

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;
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
        for (int64_t i = 0; i < n_ids; ++i) {
            observe_routed_id(*mc, *ls, *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]));
        }
    }
}

// the routing a reads_host_experts graph left on the device; call under mc.mtx with no graph in flight
void observe_device_routing(moe_cache & mc) {
    const int64_t n_ids = mc.routing->ne[0];
    mc.routing_host.resize(ggml_nelements(mc.routing));
    ggml_backend_tensor_get(mc.routing, mc.routing_host.data(), 0, ggml_nbytes(mc.routing));
    ggml_backend_tensor_memset(mc.routing, 0xFF, 0, ggml_nbytes(mc.routing));

    for (auto & ls : mc.layers) {
        const int32_t * row = mc.routing_host.data() + ls.pub.routing_row*n_ids;
        std::for_each(row, row + n_ids, [&](int32_t id) { observe_routed_id(mc, ls, id); });
    }
}

void upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
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
// in that device's pinned host buffer (load-mode none or mlock, not mmap); LLAMA_MOE_CACHE_HOST_READS=0 opts out
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
        for (const ggml_tensor * src : {ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src}) {
            if (ggml_backend_buffer_get_type(src->buffer) != host_buft) {
                why_not = std::string(src->name) + " is in " + ggml_backend_buffer_name(src->buffer) + ", not pinned host memory";
                return false;
            }
        }
    }
    return true;
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

} // namespace

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return;
    }
    [&]() {
        if (n_slots <= 0) {
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
        if (const char * env = getenv("LLAMA_MOE_CACHE_INSERT_ORDER")) {
            mc->rank_by_demand = strcmp(env, "recency") != 0;
        }
        if (const char * env = getenv("LLAMA_MOE_CACHE_WARM_MAX")) {
            mc->warm_max = std::max(0, atoi(env));
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
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true; // a real model was seen and allocation failed: stay disabled
            return;
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

        std::string host_reads = "off: ";
        if (std::string why_not; !can_read_host_experts(*mc, why_not)) {
            host_reads += why_not;
        } else if (!alloc_routing(*mc, model.hparams.n_expert_used_max())) {
            host_reads += "no device memory for the routing readback";
        } else {
            host_reads = "on, uncached experts are read in place from pinned host memory";
        }

        mc->worker = std::thread([mc]() {
            for (;;) {
                upload_job j;
                {
                    std::unique_lock<std::mutex> lk(mc->wmtx);
                    mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                    if (mc->stop) {
                        return;
                    }
                    j = mc->todo.front();
                    mc->todo.pop_front();
                }
                auto & ls = mc->layers[j.layer_idx];
                upload_slice(ls.pub.up_c,   ls.pub.up_src,   j.expert, j.slot);
                upload_slice(ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot);
                upload_slice(ls.pub.down_c, ls.pub.down_src, j.expert, j.slot);
                {
                    std::lock_guard<std::mutex> lk(mc->wmtx);
                    j.done = true;
                    mc->done.push_back(j);
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
    for (const auto & ls : mc->layers) {
        out->n_hit    += ls.n_hit;
        out->n_miss   += ls.n_miss;
        out->n_insert += ls.n_insert;
        out->n_evict  += ls.n_evict;
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
    cur->op_params[0] = layer.n_slots;
    memcpy(&cur->op_params[2], &host_src->data, sizeof(host_src->data));
    return cur;
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

void llama_moe_cache_step(ggml_backend_sched_t sched) {
    moe_cache * mc = g_cache;
    if (!mc) {
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

        int n_empty = 0;
        for (int32_t s = 0; s < ls.n_slots; ++s) {
            if (ls.slot_expert[s] < 0 && !ls.slot_in_flight[s]) {
                n_empty++;
            }
        }
        int budget = n_empty > 0 ? std::max(mc->max_inserts, std::min(4, n_empty)) : mc->max_inserts;

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
