#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
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
    std::vector<int32_t>  pending;        // uncached ids observed since last step (dedup, obs order)

    std::vector<bool>     slot_in_flight; // slot has an upload pending

    std::vector<uint64_t> expert_hits;    // expert id -> total hits observed

    uint64_t n_hit    = 0;
    uint64_t n_miss   = 0;
    uint64_t n_insert = 0;
    uint64_t n_evict  = 0;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
};

struct moe_cache {
    int32_t n_slots     = 0;
    int32_t max_inserts = 2;

    // where to persist the expert popularity histogram; empty disables persistence entirely
    std::string map_path;

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

void promote_to_protected(layer_state & ls, int32_t slot, int32_t n_slots, uint64_t clock);
int32_t find_eviction_victim(const layer_state & ls, int32_t n_slots);

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
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            ls->expert_hits[id]++;

            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit++;
                promote_to_protected(*ls, slot, mc->n_slots, ++mc->clock);
            } else {
                ls->n_miss++;
                if (slot == -2) {
                    continue;
                }
                bool dup = false;
                for (int32_t p : ls->pending) {
                    if (p == id) { dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                }
            }
        }
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

void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    ggml_backend_tensor_set(pub.dev_table,  &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
    ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
}

void promote_to_protected(layer_state & ls, int32_t slot, int32_t n_slots, uint64_t clock) {
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

    const int max_protected = (n_slots * 3) / 4;
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

// Persisting the expert popularity histogram only ever buys a warm start after a restart, so it is
// opt-in: --moe-expert-cache-map. It used to default to /models/moe-cache-map.csv whenever /models
// happened to exist, which made a file that survives restarts AND image changes invisible in both
// models.ini and the rendered command line - and it faked a +18% A/B result exactly once.
const char * get_moe_cache_map_path() {
    return g_cache && !g_cache->map_path.empty() ? g_cache->map_path.c_str() : nullptr;
}

void save_cache_map(const moe_cache & mc, const char * path) {
    if (!path || !path[0]) {
        return;
    }

    char tmp_path[512];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", path, (int) getpid());

    FILE * f = fopen(tmp_path, "w");
    if (!f) {
        return;
    }

    fprintf(f, "# layer,expert,hits\n");
    for (size_t li = 0; li < mc.layers.size(); ++li) {
        const auto & ls = mc.layers[li];
        std::vector<std::pair<uint64_t, int32_t>> ranked;
        ranked.reserve(ls.expert_hits.size());
        for (int32_t e = 0; e < (int32_t) ls.expert_hits.size(); ++e) {
            if (ls.expert_hits[e] > 0) {
                ranked.push_back({ls.expert_hits[e], e});
            }
        }
        std::sort(ranked.rbegin(), ranked.rend());
        for (const auto & p : ranked) {
            fprintf(f, "%d,%d,%" PRIu64 "\n", ls.pub.il, p.second, p.first);
        }
    }

    fclose(f);
    rename(tmp_path, path);
    LLAMA_LOG_INFO("moe-cache: saved expert cache map to %s\n", path);
}

void load_cache_map(moe_cache & mc, const char * path) {
    if (!path || !path[0]) {
        return;
    }

    FILE * f = fopen(path, "r");
    if (!f) {
        return;
    }

    std::map<int, std::vector<std::pair<uint64_t, int32_t>>> layer_experts;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        int il = -1;
        int expert = -1;
        uint64_t hits = 0;
        if (sscanf(line, "%d,%d,%" SCNu64, &il, &expert, &hits) >= 2) {
            layer_experts[il].push_back({hits, expert});
        }
    }
    fclose(f);

    int loaded_count = 0;
    for (auto & ls : mc.layers) {
        const auto it = layer_experts.find(ls.pub.il);
        if (it == layer_experts.end()) {
            continue;
        }
        int32_t slot = 0;
        for (const auto & h_exp : it->second) {
            const int32_t exp = h_exp.second;
            const uint64_t hits = h_exp.first;
            if (exp >= 0 && exp < (int32_t) ls.expert_hits.size()) {
                ls.expert_hits[exp] = hits;
            }
            if (slot >= mc.n_slots) {
                continue;
            }
            if (exp < 0 || exp >= (int32_t) ls.expert_slot.size() || ls.expert_slot[exp] >= 0) {
                continue;
            }

            ls.slot_expert[slot]    = exp;
            ls.expert_slot[exp]     = slot;
            ls.slot_protected[slot] = slot < (mc.n_slots * 3) / 4;
            ls.slot_last_use[slot]  = mc.clock + (mc.n_slots - slot);
            ls.slot_in_flight[slot] = false;

            upload_slice(ls.pub.up_c,   ls.pub.up_src,   exp, slot);
            upload_slice(ls.pub.gate_c, ls.pub.gate_src, exp, slot);
            upload_slice(ls.pub.down_c, ls.pub.down_src, exp, slot);
            set_table_entry(ls.pub, exp, slot);

            slot++;
            loaded_count++;
        }
        mc.clock += mc.n_slots;
    }

    LLAMA_LOG_INFO("%s: pre-warmed %d expert slots from %s\n", __func__, loaded_count, path);
}

void moe_cache_atexit() {
    if (g_cache) {
        const char * map_path = get_moe_cache_map_path();
        if (map_path) {
            std::lock_guard<std::mutex> lock(g_cache->mtx);
            save_cache_map(*g_cache, map_path);
        }
    }
}

} // namespace

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts, const char * map_path) {
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
        if (map_path && map_path[0]) {
            mc->map_path = map_path;
        }
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }

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

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*4 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.n_slots  = n_slots;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], n_slots + 1);
                    ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], n_slots + 1);
                    ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], n_slots + 1);
                    ls->pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
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
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            ls.slot_expert.assign(n_slots, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(n_slots, 0);
            ls.slot_protected.assign(n_slots, false);
            ls.slot_in_flight.assign(n_slots, false);
            ls.expert_hits.assign(n_expert, 0);

            std::vector<int32_t> dummy(n_expert, n_slots);
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

        // not get_moe_cache_map_path(): that reads g_cache, which is only published below
        if (!mc->map_path.empty()) {
            load_cache_map(*mc, mc->map_path.c_str());
        }
        std::atexit(moe_cache_atexit);

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

        LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu layers x %d slots, %d inserts/step, %.1f MiB device memory\n",
                __func__, mc->layers.size(), n_slots, mc->max_inserts, vram/1024.0/1024.0);
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

void llama_moe_cache_step() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
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
            set_table_entry(ls.pub, j.expert, j.slot);
        }
        mc->done.clear();
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        auto & ls = mc->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        int n_empty = 0;
        for (int32_t s = 0; s < mc->n_slots; ++s) {
            if (ls.slot_expert[s] < 0 && !ls.slot_in_flight[s]) {
                n_empty++;
            }
        }
        int budget = n_empty > 0 ? std::max(mc->max_inserts, std::min(4, n_empty)) : mc->max_inserts;
        for (auto it = ls.pending.rbegin(); it != ls.pending.rend() && budget > 0; ++it, --budget) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0 || ls.expert_slot[id] == -2) {
                continue;
            }

            const int32_t slot = find_eviction_victim(ls, mc->n_slots);
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(ls.pub, victim, mc->n_slots);
                ls.n_evict++;
            }
            ls.slot_protected[slot] = false;
            ls.slot_in_flight[slot] = true;
            ls.expert_slot[id]      = -2;
            ls.n_insert++;

            std::lock_guard<std::mutex> wlk(mc->wmtx);
            mc->todo.push_back({li, id, slot});
        }
        ls.pending.clear();
    }
    mc->wcv.notify_one();

    if (mc->n_steps % 128 == 0) {
        uint64_t h = 0;
        uint64_t m = 0;
        uint64_t ins = 0;
        uint64_t ev = 0;
        for (const auto & ls : mc->layers) {
            h   += ls.n_hit;
            m   += ls.n_miss;
            ins += ls.n_insert;
            ev  += ls.n_evict;
        }
        const uint64_t dh = h - mc->last_log_hits;
        const uint64_t dm = m - mc->last_log_misses;
        mc->last_log_hits   = h;
        mc->last_log_misses = m;

        const double total_rate = (h + m > 0) ? (100.0 * h / (h + m)) : 0.0;
        const double win_rate   = (dh + dm > 0) ? (100.0 * dh / (dh + dm)) : 0.0;

        LLAMA_LOG_WARN("moe-cache: steps=%" PRIu64 " win_hit=%.1f%% (%" PRIu64 "/%" PRIu64 ") total_hit=%.1f%% (%" PRIu64 "/%" PRIu64 ") ins=%" PRIu64 " evict=%" PRIu64 "\n",
                mc->n_steps, win_rate, dh, dh + dm, total_rate, h, h + m, ins, ev);
    }
}
