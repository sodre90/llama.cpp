#include "moe-cache.cuh"

#define MOE_CACHE_BLOCK_SIZE 256
#define MOE_CACHE_MAX_SLOTS  1024
#define MOE_CACHE_MAX_IDS    64
#define MOE_CACHE_HEADER     8

// state words: [0] K, [1] E, [2] decay (float), [3] clock, [4] n_hit, [5] n_miss, [6] n_fill, [7] n_evict,
// then slot_expert[K], fill_slot[E], score[E] (float), last[E]
// score[e] is a hit count that halves every 1/decay steps, last[e] is the step it was updated at
// table[e]: [0, K) VRAM slot, K without a slot; with a host tier (host_map != nullptr) a routed expert without a VRAM slot gets K + its host pool slot

// the score of expert e decayed to step now
static __device__ __forceinline__ float moe_cache_crf(
        const int32_t * g_score, const int32_t * g_last, const float decay, const uint32_t now, const int e) {
    return __int_as_float(g_score[e]) * exp2f(-decay * (float) (now - (uint32_t) g_last[e]));
}

// the number of slots in class >= 0 that sort before slot s by (class, key, slot)
static __device__ __forceinline__ int moe_cache_rank(
        const int8_t * slot_class, const float * slot_key, const int n_slots, const int s) {
    const int   c   = slot_class[s];
    const float key = slot_key[s];
    int rank = 0;
    for (int s2 = 0; s2 < n_slots; ++s2) {
        const int c2 = slot_class[s2];
        if (c2 < 0) {
            continue;
        }
        const float key2 = slot_key[s2];
        rank += c2 < c || (c2 == c && (key2 < key || (key2 == key && s2 < s)));
    }
    return rank;
}

static __global__ void __launch_bounds__(MOE_CACHE_BLOCK_SIZE) moe_cache_assign(
        const int32_t * ids, const int n_ids, int32_t * state, int32_t * table, const int32_t * host_map) {
    constexpr int slots_per_thread = MOE_CACHE_MAX_SLOTS/MOE_CACHE_BLOCK_SIZE;

    const int   n_slots   = state[0];
    const int   n_experts = state[1];
    const float decay     = __int_as_float(state[2]);
    if (n_slots > MOE_CACHE_MAX_SLOTS || n_experts > MOE_CACHE_MAX_SLOTS || n_slots < 0 || n_experts < 0) {
        return;
    }

    int32_t * g_slot_expert = state + MOE_CACHE_HEADER;
    int32_t * g_fill_slot   = g_slot_expert + n_slots;
    int32_t * g_score       = g_fill_slot + n_experts;
    int32_t * g_last        = g_score + n_experts;

    __shared__ int32_t  slot_expert[MOE_CACHE_MAX_SLOTS];
    __shared__ float    slot_key[MOE_CACHE_MAX_SLOTS];
    __shared__ int8_t   slot_class[MOE_CACHE_MAX_SLOTS];
    __shared__ uint8_t  routed[MOE_CACHE_MAX_SLOTS];
    __shared__ int32_t  routed_id[MOE_CACHE_MAX_IDS];
    __shared__ int32_t  is_first_miss[MOE_CACHE_MAX_IDS];
    __shared__ int32_t  missed_expert[MOE_CACHE_MAX_IDS];
    __shared__ uint32_t now;
    __shared__ uint32_t n_hit_uses;
    __shared__ uint32_t n_miss_uses;
    __shared__ uint32_t n_evicted;
    __shared__ uint32_t n_filled;
    __shared__ int32_t  n_distinct_misses;

    const int tid = threadIdx.x;

    if (tid == 0) {
        now               = (uint32_t) state[3] + 1;
        state[3]          = (int32_t) now;
        n_hit_uses        = 0;
        n_miss_uses       = 0;
        n_evicted         = 0;
        n_filled          = 0;
        n_distinct_misses = 0;
    }
    for (int s = tid; s < n_slots; s += MOE_CACHE_BLOCK_SIZE) {
        slot_expert[s] = g_slot_expert[s];
    }
    for (int e = tid; e < n_experts; e += MOE_CACHE_BLOCK_SIZE) {
        routed[e] = 0;
    }
    __syncthreads();

    for (int e = tid; e < n_experts; e += MOE_CACHE_BLOCK_SIZE) {
        const int s = g_fill_slot[e];
        if (s >= 0 && s < n_slots) {
            table[e]        = s;
            slot_expert[s]  = e;
            g_fill_slot[e]  = -1;
        }
    }
    __syncthreads();

    if (tid < n_ids) {
        const int e = ids[tid];
        routed_id[tid] = e;
        if (e >= 0 && e < n_experts) {
            routed[e] = 1;
            if (table[e] < n_slots) {
                atomicAdd(&n_hit_uses, 1u);
            } else {
                atomicAdd(&n_miss_uses, 1u);
            }
        }
    }
    __syncthreads();

    if (tid < n_ids) {
        const int e = routed_id[tid];
        bool first_miss = e >= 0 && e < n_experts && table[e] >= n_slots;
        for (int j = 0; j < tid && first_miss; ++j) {
            first_miss = routed_id[j] != e;
        }
        is_first_miss[tid] = first_miss;
    }
    __syncthreads();

    if (tid < n_ids) {
        int pos = 0;
        for (int j = 0; j < tid; ++j) {
            pos += is_first_miss[j];
        }
        if (is_first_miss[tid]) {
            missed_expert[pos] = routed_id[tid];
        }
        if (tid == n_ids - 1) {
            n_distinct_misses = pos + is_first_miss[tid];
        }
    }

    for (int e = tid; e < n_experts; e += MOE_CACHE_BLOCK_SIZE) {
        if (routed[e]) {
            g_score[e] = __float_as_int(moe_cache_crf(g_score, g_last, decay, now, e) + 1.0f);
            g_last[e]  = (int32_t) now;
        }
    }
    __syncthreads();

    if (n_distinct_misses > 0) {
        for (int s = tid; s < n_slots; s += MOE_CACHE_BLOCK_SIZE) {
            const int e = slot_expert[s];
            const bool in_range  = e >= 0 && e < n_experts;
            const bool candidate = !in_range || !routed[e];
            slot_class[s] = !candidate ? -1 : e < 0 ? 0 : 1;
            slot_key[s]   = in_range ? moe_cache_crf(g_score, g_last, decay, now, e) : 0.0f;
        }
        __syncthreads();

        int victim_of[slots_per_thread];
#pragma unroll
        for (int i = 0; i < slots_per_thread; ++i) {
            const int s = tid + i*MOE_CACHE_BLOCK_SIZE;
            victim_of[i] = -1;
            if (s < n_slots && slot_class[s] >= 0) {
                const int rank = moe_cache_rank(slot_class, slot_key, n_slots, s);
                if (rank < n_distinct_misses) {
                    victim_of[i] = missed_expert[rank];
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < slots_per_thread; ++i) {
            const int s = tid + i*MOE_CACHE_BLOCK_SIZE;
            const int e = victim_of[i];
            if (e < 0) {
                continue;
            }
            const int old = slot_expert[s];
            if (old >= 0 && old < n_experts) {
                table[old] = n_slots;
                atomicAdd(&n_evicted, 1u);
            }
            slot_expert[s] = -1;
            g_fill_slot[e] = s;
            atomicAdd(&n_filled, 1u);
        }
        __syncthreads();
    }

    if (host_map != nullptr && tid < n_ids) {
        const int e = routed_id[tid];
        if (e >= 0 && e < n_experts && table[e] >= n_slots) {
            table[e] = n_slots + host_map[e];
        }
    }

    for (int s = tid; s < n_slots; s += MOE_CACHE_BLOCK_SIZE) {
        g_slot_expert[s] = slot_expert[s];
    }
    if (tid == 0) {
        state[4] = (int32_t) ((uint32_t) state[4] + n_hit_uses);
        state[5] = (int32_t) ((uint32_t) state[5] + n_miss_uses);
        state[6] = (int32_t) ((uint32_t) state[6] + n_filled);
        state[7] = (int32_t) ((uint32_t) state[7] + n_evicted);
    }
}

void ggml_cuda_moe_cache_assign(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * state = dst->src[2];
    const ggml_tensor * table = dst->src[3];
    const ggml_tensor * host_map = dst->src[4];

    GGML_ASSERT(dst->type == GGML_TYPE_I32 && state->type == GGML_TYPE_I32 && table->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(state) && ggml_is_contiguous(table));
    GGML_ASSERT(host_map == nullptr || (host_map->type == GGML_TYPE_I32 && ggml_is_contiguous(host_map) &&
                ggml_nelements(host_map) == ggml_nelements(table)));
    GGML_ASSERT(ggml_nelements(dst) <= MOE_CACHE_MAX_IDS);
    GGML_ASSERT(ggml_nelements(table) <= MOE_CACHE_MAX_SLOTS);
    GGML_ASSERT(ggml_nelements(state) >= MOE_CACHE_HEADER);

    moe_cache_assign<<<1, MOE_CACHE_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const int32_t *) dst->data, (int) ggml_nelements(dst), (int32_t *) state->data, (int32_t *) table->data,
        host_map ? (const int32_t *) host_map->data : nullptr);
}
