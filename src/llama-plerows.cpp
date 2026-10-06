#include "llama-plerows.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// 2-way set-associative host row cache for the lazy per-layer token embedding table.
// Sets are indexed by row % n_sets; each set holds 2 ways with 1-bit pseudo-LRU.
// Adjacent ways share the same cache line in the tag array.
struct ple_row_cache {
    const ggml_tensor * table    = nullptr;
    int64_t             row_size = 0;
    int32_t             n_slots  = 0;
    int32_t             n_sets   = 0;
    uint8_t *           rows     = nullptr;
    std::vector<int32_t> tag;                      // slot -> row, -1 when empty
    std::vector<uint8_t> lru;                      // set -> next victim way (0 or 1)
    bool                prefetch = false;          // model.can_prefetch holds the table

    uint64_t n_hit      = 0;
    uint64_t n_miss     = 0;
    uint64_t n_read     = 0;                       // bytes read from the mapping
    uint64_t log_hit    = 0;
    uint64_t log_miss   = 0;
    uint64_t log_read   = 0;
    int64_t  log_us     = 0;
};

static ple_row_cache * g_cache = nullptr;

void llama_ple_rows_init(const llama_model & model, int32_t n_rows) {
    if (g_cache != nullptr || n_rows <= 0) {
        return;
    }

    const ggml_tensor * table = model.per_layer_tok_embd;
    // gemma3n and gemma4 build the same table; only a host-resident one can be cached
    if (table == nullptr || table->data == nullptr || table->buffer == nullptr ||
            !ggml_backend_buffer_is_host(table->buffer)) {
        LLAMA_LOG_WARN("%s: no host-resident per-layer token embedding table, row cache off\n", __func__);
        return;
    }

    const int32_t slots_req = (int32_t) std::min<int64_t>(n_rows, table->ne[1]);
    const int32_t sets      = std::max<int32_t>(1, slots_req / 2);
    const int32_t slots     = sets * 2;
    const int64_t row_size  = ggml_row_size(table->type, table->ne[0]);

    g_cache = new ple_row_cache();
    g_cache->table    = table;
    g_cache->row_size = row_size;
    g_cache->n_slots  = slots;
    g_cache->n_sets   = sets;
    g_cache->rows     = (uint8_t *) malloc((size_t) slots * row_size);
    g_cache->tag.assign(slots, -1);
    g_cache->lru.assign(sets, 0);
    g_cache->prefetch = model.can_prefetch.count(table) > 0;
    g_cache->log_us   = ggml_time_us();

    if (g_cache->rows == nullptr) {
        LLAMA_LOG_WARN("%s: cannot allocate %.0f MiB for the row cache, off\n", __func__,
                (double) slots * row_size / 1024.0 / 1024.0);
        delete g_cache;
        g_cache = nullptr;
        return;
    }

    LLAMA_LOG_WARN("ple-rows: %d slots in %d sets (2-way, %.0f MiB), %s pf\n", slots, sets,
            (double) slots * row_size / 1024.0 / 1024.0, g_cache->prefetch ? "with" : "without");
}

bool llama_ple_rows_enabled() {
    return g_cache != nullptr;
}

void llama_ple_rows_gather(const ggml_tensor * table, const int32_t * idx, int64_t n_rows, float * dst) {
    ple_row_cache & c = *g_cache;

    GGML_ASSERT(table == c.table);

    const int64_t nc = table->ne[0];
    const auto to_float = ggml_get_type_traits(table->type)->to_float;

    // every row this ubatch reads that the cache does not hold yet
    std::vector<int32_t> miss;
    for (int64_t i = 0; i < n_rows; ++i) {
        const int32_t row   = idx[i];
        const int32_t set   = (uint32_t) row % c.n_sets;
        const int32_t slot0 = set * 2;
        const int32_t slot1 = set * 2 + 1;
        if (c.tag[slot0] == row) {
            ++c.n_hit;
            c.lru[set] = 1;
        } else if (c.tag[slot1] == row) {
            ++c.n_hit;
            c.lru[set] = 0;
        } else {
            miss.push_back(row);
        }
    }

    if (!miss.empty()) {
        // read the misses ahead, as the direct get_rows path does with the whole ubatch
        if (c.prefetch) {
            llama_prefetch_rows(table, miss.data(), miss.size());
        }

        const char * src0 = (const char *) table->data;
        for (int32_t row : miss) {
            const int32_t set   = (uint32_t) row % c.n_sets;
            const int32_t slot0 = set * 2;
            const int32_t slot1 = set * 2 + 1;
            if (c.tag[slot0] == row || c.tag[slot1] == row) {
                continue; // an earlier miss of this ubatch already filled the slot
            }
            int32_t slot;
            if (c.tag[slot0] < 0) {
                slot = slot0;
                c.lru[set] = 1;
            } else if (c.tag[slot1] < 0) {
                slot = slot1;
                c.lru[set] = 0;
            } else {
                const uint8_t victim = c.lru[set];
                slot = set * 2 + victim;
                c.lru[set] = 1 - victim;
            }
            memcpy(c.rows + (size_t) slot * c.row_size, src0 + (size_t) row * c.row_size, c.row_size);
            c.tag[slot] = row;
            ++c.n_miss;
            c.n_read += c.row_size;
        }
    }

    for (int64_t i = 0; i < n_rows; ++i) {
        const int32_t row   = idx[i];
        const int32_t set   = (uint32_t) row % c.n_sets;
        const int32_t slot0 = set * 2;
        const int32_t slot1 = set * 2 + 1;
        int32_t hit_slot = -1;
        if (c.tag[slot0] == row) {
            hit_slot = slot0;
        } else if (c.tag[slot1] == row) {
            hit_slot = slot1;
        }
        const char * src = hit_slot >= 0
                ? (const char *) c.rows + (size_t) hit_slot * c.row_size
                : (const char *) table->data + (size_t) row * c.row_size;
        to_float(src, dst + i * nc, nc);
    }

    const int64_t now = ggml_time_us();
    if (now - c.log_us >= 10 * 1000 * 1000) {
        const uint64_t win_hit  = c.n_hit  - c.log_hit;
        const uint64_t win_miss = c.n_miss - c.log_miss;
        const uint64_t win_read = c.n_read - c.log_read;
        const uint64_t win      = win_hit + win_miss;
        const uint64_t total    = c.n_hit + c.n_miss;
        LLAMA_LOG_WARN("ple-rows: slots=%d win_hit=%.1f%% (%" PRIu64 "/%" PRIu64 ") total_hit=%.1f%% (%" PRIu64 "/%" PRIu64 ") win_read=%.1f MiB total_read=%.2f GiB\n",
                c.n_slots, win > 0 ? 100.0 * win_hit / win : 0.0, win_hit, win,
                total > 0 ? 100.0 * c.n_hit / total : 0.0, c.n_hit, total,
                win_read / 1024.0 / 1024.0, c.n_read / 1024.0 / 1024.0 / 1024.0);
        c.log_hit  = c.n_hit;
        c.log_miss = c.n_miss;
        c.log_read = c.n_read;
        c.log_us   = now;
    }
}
