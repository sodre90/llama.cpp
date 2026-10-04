#pragma once

// GPU-resident LRU cache for MoE expert weights that -ot pinned to host memory.
//
// Motivation (measured on Qwen3.8-Flash-Next, 512 experts / 10 routed): expert
// routing has strong temporal locality (LRU-64 hit rate ~67% over a mixed
// workload) even though the long-run distribution is near-uniform. Decode on a
// host-offloaded MoE layer is bound by host RAM bandwidth, so serving the hot
// experts from VRAM removes most of the per-token DIMM traffic.
//
// Mechanism (no custom kernels):
//  - per cached layer, companion tensors up_c/gate_c/down_c of shape
//    [ne0, ne1, n_slots+1] live in the device buffer of that layer's router;
//    slot n_slots is permanently zero (the "dummy" slot).
//  - an I32 table[512] maps expert id -> slot, or n_slots when uncached.
//    One copy on device (read by get_rows to remap ids for the cache-side
//    mul_mat_id chain) and one on host (read by the CPU mul_mat_id via
//    src[3] to SKIP cached ids, zeroing their dst rows).
//  - the two down-projection outputs are summed; uncached ids contribute 0
//    through the cache chain (zero slot) and cached ids contribute 0 through
//    the CPU chain (skip), so the result is exact.
//  - llama_moe_cache_step(), called at the end of llama_context::decode(),
//    performs throttled LRU updates: at most LLAMA_MOE_CACHE_INSERTS expert
//    uploads per layer per step via ggml_backend_tensor_set.
//
// Host reads (reads_host_experts): when the experts sit in the device's pinned
// host buffer and its backend supports it, there is no CPU chain. The cache
// chain takes the routed ids, and the device reads each uncached expert in
// place over the host link, so a decode step is one device graph instead of
// a device/CPU hand-off per layer. The graph copies its routing to the device,
// and llama_moe_cache_step() reads it back to drive the LRU.
// LLAMA_MOE_CACHE_DEVICE=1 moves the policy to the device: a kernel after the routing copy picks victims and
// the matvec kernels fill them while computing; llama_moe_cache_step() only mirrors the device state.
//
// Host tier (llama_context_params.n_moe_host_slots, CLI: --moe-expert-host-slots): the experts need not sit in pinned
// host memory. Per layer a pool of that many pinned host slots holds the experts that were routed lately, and the
// rest stay in the model file's mapping. A CPU op in front of each decode ubatch's cache chain copies the routed
// experts that are not in the pool from the mapping into it, and the device reads a VRAM miss from its pool slot.
// Needs the device policy (LLAMA_MOE_CACHE_DEVICE=1).
// The tier reads the experts from the model files with parallel O_DIRECT preads when it can, else from the mapping
// (LLAMA_MOE_HOST_IO_THREADS, default 16, 0 = mapping; LLAMA_MOE_HOST_IO_CHUNK_KB, default 512).
// With the reader, a prefill ubatch reads the next layer's experts that are in neither the pool nor the device cache
// into one of two pinned lookahead buffers while the current layer computes (LLAMA_MOE_HOST_LOOKAHEAD, default 1, 0 = off;
// LLAMA_MOE_HOST_LOOKAHEAD_MIN_USED, default 0.9: the fraction of a layer's experts the ubatch must use for that).
// LLAMA_MOE_HOST_PREDICT=1 (needs the reader) makes a decode ubatch guess the next cache layer's routing by running that
// layer's router on this layer's router input, and read the guessed experts that the next layer's pool lacks into the pool
// while this layer computes (LLAMA_MOE_HOST_PREDICT_K guessed experts per token, default n_expert_used;
// LLAMA_MOE_HOST_PREDICT_MAX experts read ahead per layer and step, default 8). The real routing is not changed.
//
// Enabled via llama_context_params.n_moe_cache_slots (CLI: --moe-expert-cache).

#include <cstddef>
#include <cstdint>

struct llama_model;
struct ggml_context;
struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;
typedef struct ggml_backend_sched * ggml_backend_sched_t;

// the cache chain is built (and routing observed) only for ubatches this small:
// decode of up to this many concurrent sequences. Larger ubatches are prefill,
// where streaming the experts once already amortises over the whole batch.
constexpr int64_t LLAMA_MOE_CACHE_MAX_TOKENS = 4;

struct llama_moe_cache_layer {
    int il = -1;

    int32_t n_slots = 0;

    // host-resident source weights (the authoritative experts)
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    // device-resident cache slots, ne[2] == n_slots + 1 (last slot all zeros)
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // expert id -> slot (or n_slots when uncached); I32 [1, n_expert]
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;

    // the device reads uncached experts in place from the pinned host tensors: the cache chain serves every
    // routed id (llama_moe_cache_mul_mat_id) and no CPU chain is built
    bool reads_host_experts = false;

    // device I32 [n_expert_used*LLAMA_MOE_CACHE_MAX_TOKENS, n_layers] that such a graph copies its routing into
    // (row routing_row), for llama_moe_cache_step() to observe; entries it did not write are -1
    ggml_tensor * routing     = nullptr;
    int32_t       routing_row = 0;

    // device policy (LLAMA_MOE_CACHE_DEVICE): I32 state of this layer (layout in llama-moecache.cpp) and the
    // view of its fill_slot part, which the cache mul_mat_ids read
    bool          device_policy = false;
    ggml_tensor * dev_state     = nullptr;
    ggml_tensor * fill_slot     = nullptr;

    // the router of this layer (the bias tensors may be null), for the host tier's guess of its routing
    ggml_tensor * gate_inp    = nullptr;
    ggml_tensor * gate_inp_b  = nullptr;
    ggml_tensor * exp_probs_b = nullptr;

    // host tier guess (LLAMA_MOE_HOST_PREDICT): the next cache layer, whose routing the graph guesses with its router, and the
    // number of experts guessed per token; nullptr when the guess is off or this is the last cache layer
    const llama_moe_cache_layer * predict_next = nullptr;
    int32_t                       predict_k    = 0;

    // host tier: > 0 when the cache chain reads a miss from a pool of this many pinned host slots; up_h, gate_h and
    // down_h have ne[2] == n_host_slots + 1 (the last slot is padding), the experts stay in up_src, gate_src, down_src
    int32_t       n_host_slots = 0;
    ggml_tensor * up_h   = nullptr;
    ggml_tensor * gate_h = nullptr;
    ggml_tensor * down_h = nullptr;
};

// build the cache for every host-resident expert layer of the model; n_host_slots > 0 asks for the host tier.
// Safe to call more than once; only the first call does work.
void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts, int32_t n_host_slots);

// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps);

// mul_mat_id over one of a reads_host_experts layer's cache tensors (up_c, gate_c or down_c) with the routed ids:
// each id reads its slot, or the expert in place from host_src (the matching up_src, gate_src or down_src),
// or from the matching host pool when the layer has the host tier
ggml_tensor * llama_moe_cache_mul_mat_id(ggml_context * ctx, const llama_moe_cache_layer & layer,
        ggml_tensor * cache, const ggml_tensor * host_src, ggml_tensor * b, ggml_tensor * ids);

// host tier: I32 [n_expert], for each routed id the host pool slot holding its expert. A CPU op that first copies
// the routed experts missing from the pool in from the mapping; the cache chain's routing copy takes it as src[4].
// guess (optional) is I32 [k, n_tokens], the experts the layer.predict_next layer may route to: the op starts reading
// those that its pool lacks and has them in the pool by the time that layer's own op runs
ggml_tensor * llama_moe_cache_host_map(ggml_context * ctx, const llama_moe_cache_layer & layer, ggml_tensor * ids, ggml_tensor * guess = nullptr);

// true when the host tier is on for the cached layers
bool llama_moe_cache_host_tier();

// ggml_backend_sched_expert_host_fn over the host pools: lets the scheduler's prefill upload of a host-resident
// up/gate/down weight copy the pool-resident experts from the pool. Does not admit or score anything.
bool llama_moe_cache_expert_host(const ggml_tensor * weight, const uint32_t * used_ids, const ggml_tensor ** pool, const int32_t ** host_slot, int32_t * n_slots, void * user_data);

// true when the host tier reads the experts from the model files with O_DIRECT
bool llama_moe_cache_direct_reads();

// ggml_backend_sched_expert_read_fn over the host tier's file reader: reads the experts of an up/gate/down weight
// into dst. False when the tier has no reader or does not know the weight.
bool llama_moe_cache_expert_read(const ggml_tensor * weight, int64_t first, int64_t n, void * dst, void * user_data);

// ggml_backend_sched_expert_src_fn over the lookahead buffers: the pinned bytes of experts [first, first + n) of an up/gate/down
// weight, once the reads that fill them are done. nullptr when the run is not buffered.
const void * llama_moe_cache_expert_src(const ggml_tensor * weight, int64_t first, int64_t n, void * user_data);

// ggml_backend_sched_expert_rows_fn over the cache: lets the scheduler's prefill upload of a
// host-resident up/gate/down weight fill the resident experts from their slots instead of over the
// host link. Entries of experts the running ubatch routes to only change in llama_moe_cache_step(),
// between graphs; llama_moe_cache_warm_from_staging() may retire unused experts mid-graph.
bool llama_moe_cache_expert_rows(const ggml_tensor * weight, const ggml_tensor ** rows, const int32_t ** expert_slot, int32_t * n_slots, void * user_data);

// ggml_backend_sched_expert_staged_fn over the cache: copies experts a prefill ubatch has already
// staged on the device into free and probation slots (LLAMA_MOE_CACHE_WARM_MAX per layer per
// ubatch, 0 = off). The new mapping is published by the next llama_moe_cache_step().
void llama_moe_cache_warm_from_staging(const ggml_tensor * weight, const ggml_tensor * staged,
        const int32_t * ids, int64_t ne0, int64_t ne1, size_t s0, size_t s1, ggml_backend_t backend, void * user_data);

// device policy: call before each ubatch of a decode. A ubatch too large for the cache matvecs stages experts
// from the host table, so it first mirrors what earlier ubatches of the same decode changed on the device
void llama_moe_cache_ubatch_begin(ggml_backend_sched_t sched, int64_t n_tokens);

// apply throttled LRU updates; call between graph executions only. Waits for sched's graph when the routing
// has to be read back from the device
void llama_moe_cache_step(ggml_backend_sched_t sched);

// bytes of the expert slot buffers on the devices (the host-side tables are not counted)
size_t llama_moe_cache_device_bytes();
